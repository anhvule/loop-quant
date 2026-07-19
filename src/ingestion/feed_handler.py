"""Websocket ingestion, warm-up, reconnection, and feed-health state.

Owns `MarketState.state`, which is *feed health only*:

    WARMUP  -> history not yet replayed; no trading
    READY   -> ticks flowing
    DEGRADED-> no tick for `staleness_timeout_sec`; Module 2 blocks new entries
    HALTED  -> no tick for `halt_timeout_sec`; Module 2 flattens and stands down

Risk-driven halts (daily loss, optimizer HALT) live in RiskManager, deliberately
NOT here -- conflating "the network hiccuped" with "we lost 3% today" would let a
recovered websocket silently re-enable trading after a risk breach.

NOTE (deliberate deviation from the blueprint stream list): we subscribe to
`@aggTrade`, not `@trade`. The dedup key the schema mandates is `agg_trade_id`,
which only the aggTrade stream carries (field `a`); it is also materially less
bandwidth for identical OHLCV.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

import websockets

from src.common.db import DB
from src.common.event_bus import EventBus
from src.common.models import Candle, MarketState, SystemState, Trade, tf_to_ms
from src.execution.exchange_adapter import ExchangeAdapter
from src.ingestion.candle_aggregator import CandleAggregator
from src.ingestion.indicator_engine import IndicatorEngine

log = logging.getLogger(__name__)

WARMUP_BARS = 500
LISTEN_KEY_KEEPALIVE_SEC = 1_800   # Binance expires an idle listenKey after 60 min


class FeedHandler:
    def __init__(self, symbol: str, tf: str, adapter: ExchangeAdapter, bus: EventBus,
                 db: DB, market: MarketState, indicators: IndicatorEngine, *,
                 staleness_timeout_sec: float = 10.0, halt_timeout_sec: float = 60.0,
                 use_user_stream: bool = True) -> None:
        self.symbol = symbol.upper()
        self.tf = tf
        self.tf_ms = tf_to_ms(tf)
        self.adapter = adapter
        self.bus = bus
        self.db = db
        self.market = market
        self.indicators = indicators
        self.agg = CandleAggregator(self.symbol, tf)
        self.staleness_timeout_sec = staleness_timeout_sec
        self.halt_timeout_sec = halt_timeout_sec
        self.use_user_stream = use_user_stream

        self._seen_ids: set[int] = set()
        self._last_id: int = -1
        self._trade_buf: list[Trade] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._listen_key: str | None = None
        self.reconnects = 0

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        await self._set_state("WARMUP")
        await self.warm_up()
        self._tasks = [
            asyncio.create_task(self._market_stream_loop(), name="ws-market"),
            asyncio.create_task(self._clock_loop(), name="clock"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
        ]
        if self.use_user_stream:
            self._tasks.append(asyncio.create_task(self._user_stream_loop(), name="ws-user"))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()
        self._flush_trades()
        if self._listen_key:
            await self.adapter.close_listen_key(self._listen_key)
            self._listen_key = None

    # -- warm-up & backfill -----------------------------------------------

    async def warm_up(self) -> None:
        """REST-fetch history and replay it through the indicators before the
        engine is allowed to trade. Nothing may enter a position until the
        indicator set has a real value -- a half-warm MACD is not a signal."""
        need = max(WARMUP_BARS, self.indicators.min_bars_required + 10)
        rows = await self.adapter.get_klines(self.symbol, self.tf, limit=min(need, 1000))
        candles = [_kline_to_candle(r) for r in rows]
        # The last kline is the in-progress bar; it has not closed, so replaying
        # it would fold a partial bar into the indicator state and then fold the
        # complete version in again a minute later.
        if candles and _is_current_bucket(candles[-1], self.tf_ms):
            candles = candles[:-1]
        self.db.upsert_candles(candles)
        self.indicators.warm_up(candles)
        if candles:
            self.agg.prev_close = candles[-1].close
            self.market.last_price = candles[-1].close
        log.info("warm-up complete: %d bars up to %s, indicators ready=%s",
                 len(candles), candles[-1].ts_open_ms if candles else None,
                 self.indicators.snapshot().ready)

    async def _backfill_gap(self) -> None:
        """After a disconnect, REST-fetch the bars we missed and replay them.
        Indicators must never see a hole: an EMA with a missing bar is silently
        wrong forever, not just for that bar."""
        last = self.db.last_candle_ts(self.symbol, self.tf)
        if last is None:
            await self.warm_up()
            return
        now = int(time.time() * 1000)
        start = last + self.tf_ms
        if start >= now - self.tf_ms:
            return   # nothing closed since we dropped
        rows = await self.adapter.get_klines_range(self.symbol, self.tf, start, now)
        candles = [_kline_to_candle(r) for r in rows]
        if candles and _is_current_bucket(candles[-1], self.tf_ms):
            candles = candles[:-1]
        if not candles:
            return
        log.warning("backfilling %d bar(s) missed during disconnect", len(candles))
        self.db.upsert_candles(candles)
        for c in candles:
            snap = self.indicators.update(c)
            await self.bus.publish("candle_closed", c)
            await self.bus.publish("indicator_update", snap)
        self.agg = CandleAggregator(self.symbol, self.tf)
        self.agg.prev_close = candles[-1].close

    # -- market stream ----------------------------------------------------

    def _streams(self) -> list[str]:
        s = self.symbol.lower()
        return [f"{s}@aggTrade", f"{s}@depth20@100ms", f"{s}@kline_{self.tf}"]

    async def _market_stream_loop(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            url = self.adapter.stream_url(self._streams())
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_queue=4096) as ws:
                    log.info("market stream connected: %s", ",".join(self._streams()))
                    delay = 1.0
                    if self.reconnects:
                        await self._backfill_gap()
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        await self._dispatch(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stop.is_set():
                    return
                self.reconnects += 1
                log.warning("market stream dropped (%s); reconnecting in %.1fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)   # exp backoff 1s..60s

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        data = msg.get("data", msg)
        etype = data.get("e")
        if etype == "aggTrade":
            await self._on_trade(data)
        elif etype == "kline":
            self._on_kline_crosscheck(data)
        elif "bids" in data and "asks" in data:
            await self._on_depth(data)

    async def _on_trade(self, d: dict[str, Any]) -> None:
        tid = int(d["a"])
        if tid <= self._last_id or tid in self._seen_ids:
            return   # duplicate or out-of-order replay after reconnect
        self._seen_ids.add(tid)
        self._last_id = max(self._last_id, tid)
        if len(self._seen_ids) > 20_000:
            # Bounded memory: ids are monotonic, so anything below the high water
            # mark is already rejected by the `tid <= self._last_id` test above.
            self._seen_ids = {i for i in self._seen_ids if i > self._last_id - 5_000}

        t = Trade(ts_ms=int(d["T"]), symbol=self.symbol, price=float(d["p"]),
                  qty=float(d["q"]), is_buyer_maker=bool(d["m"]), agg_trade_id=tid)

        # Exchange event time, never local wall clock -- our clock's drift must
        # not decide which minute a trade belongs to.
        self.market.last_price = t.price
        self.market.last_tick_ms = t.ts_ms
        if self.market.state in ("DEGRADED", "HALTED"):
            await self._set_state("READY")

        self._trade_buf.append(t)
        if len(self._trade_buf) >= 200:
            self._flush_trades()

        await self.bus.publish("trade_tick", t)
        for c in self.agg.on_trade(t):
            await self._emit_candle(c)

    async def _on_depth(self, d: dict[str, Any]) -> None:
        bids, asks = d.get("bids") or [], d.get("asks") or []
        if not bids or not asks:
            return
        self.market.best_bid = float(bids[0][0])
        self.market.best_ask = float(asks[0][0])
        self.market.bid_qty_top5 = sum(float(b[1]) for b in bids[:5])
        self.market.ask_qty_top5 = sum(float(a[1]) for a in asks[:5])
        await self.bus.publish("book_update", self.market)

    def _on_kline_crosscheck(self, d: dict[str, Any]) -> None:
        """The exchange kline stream is a consistency check, never a source of
        truth. A persistent mismatch means our trade feed is lossy."""
        k = d.get("k", {})
        if not k.get("x"):
            return
        ours = self.db.get_candles(self.symbol, self.tf, int(k["t"]), int(k["t"]))
        if not ours:
            return
        theirs_close, ours_close = float(k["c"]), ours[0].close
        if abs(theirs_close - ours_close) > 1e-9:
            log.warning("kline cross-check mismatch at %s: exchange close=%s ours=%s",
                        k["t"], theirs_close, ours_close)

    async def _emit_candle(self, c: Candle) -> None:
        self._flush_trades()
        self.db.upsert_candles([c])
        snap = self.indicators.update(c)
        if self.market.state == "WARMUP" and snap.ready:
            await self._set_state("READY")
        await self.bus.publish("candle_closed", c)
        await self.bus.publish("indicator_update", snap)

    def _flush_trades(self) -> None:
        if self._trade_buf:
            self.db.insert_trades(self._trade_buf)
            self._trade_buf.clear()

    # -- user data stream (fills / balances) -------------------------------

    async def _user_stream_loop(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                self._listen_key = await self.adapter.create_listen_key()
                url = self.adapter.raw_stream_url(self._listen_key)
                ka = asyncio.create_task(self._keepalive_loop(self._listen_key))
                try:
                    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                        log.info("user data stream connected")
                        delay = 1.0
                        async for raw in ws:
                            if self._stop.is_set():
                                break
                            await self._on_user_event(json.loads(raw))
                finally:
                    ka.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ka
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stop.is_set():
                    return
                log.warning("user stream dropped (%s); reconnecting in %.1fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def _keepalive_loop(self, key: str) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(LISTEN_KEY_KEEPALIVE_SEC)
            try:
                await self.adapter.keepalive_listen_key(key)
            except Exception as e:
                log.warning("listenKey keepalive failed: %s", e)
                return

    async def _on_user_event(self, d: dict[str, Any]) -> None:
        if d.get("e") != "executionReport" or d.get("s") != self.symbol:
            return
        # Persist every partial fill; OrderManager decides what they mean.
        if d.get("x") == "TRADE" and float(d.get("l", 0)) > 0:
            from src.common.models import Fill
            f = Fill(order_id=str(d["i"]), client_order_id=str(d.get("c", "")),
                     ts_ms=int(d["T"]), symbol=self.symbol, side=d["S"],
                     price=float(d["L"]), qty=float(d["l"]),
                     fee=float(d.get("n", 0) or 0), fee_asset=str(d.get("N") or ""))
            self.db.insert_fill(f)
        await self.bus.publish("order_filled", d)

    # -- clock & watchdog --------------------------------------------------

    async def _clock_loop(self) -> None:
        """Force-closes bars in an idle market so the strategy still evaluates
        every minute even when nothing trades."""
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
            now = int(time.time() * 1000)
            try:
                for c in self.agg.on_clock(now):
                    await self._emit_candle(c)
            except Exception:
                log.exception("clock loop error")

    async def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if self.market.state == "WARMUP" or self.market.last_tick_ms == 0:
                continue
            age = time.time() - self.market.last_tick_ms / 1000.0
            if age > self.halt_timeout_sec and self.market.state != "HALTED":
                await self._set_state("HALTED")
            elif self.staleness_timeout_sec < age <= self.halt_timeout_sec \
                    and self.market.state == "READY":
                await self._set_state("DEGRADED")

    async def _set_state(self, s: SystemState) -> None:
        if self.market.state == s:
            return
        prev = self.market.state
        self.market.state = s
        level = logging.WARNING if s in ("DEGRADED", "HALTED") else logging.INFO
        log.log(level, "feed state %s -> %s", prev, s)
        await self.bus.publish("state_changed", s)


# ---------------------------------------------------------------------------

def _kline_to_candle(r: dict[str, Any]) -> Candle:
    return Candle(ts_open_ms=r["ts_open_ms"], symbol=r["symbol"], tf=r["tf"],
                  open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                  volume=r["volume"], quote_volume=r["quote_volume"], n_trades=r["n_trades"])


def _is_current_bucket(c: Candle, tf_ms: int) -> bool:
    now = int(time.time() * 1000)
    return c.ts_open_ms == now - (now % tf_ms)
