"""Builds authoritative OHLCV bars locally from the raw trade feed.

The exchange's own kline stream is treated as a cross-check only, never as the
source of truth: our candles must be derived from exactly the trades our
indicators saw, or a reconnect could leave the two silently disagreeing.

`quote_volume` is accumulated as sum(price*qty) over the bar's raw trades. That
makes it the exact numerator of a raw-trade VWAP -- see indicator_engine.

NOTE (deliberate deviation from the blueprint signature): `on_trade` / `on_clock`
return `list[Candle]` rather than `Candle | None`. Closing a multi-minute gap has
to emit the real bar *plus* one flat bar per skipped minute, which a single-value
return cannot express without a hidden internal queue. Callers iterate the list.
"""

from __future__ import annotations

import logging

from src.common.models import Candle, Trade, tf_to_ms

log = logging.getLogger(__name__)


class CandleAggregator:
    def __init__(self, symbol: str, tf: str = "1m") -> None:
        self.symbol = symbol
        self.tf = tf
        self.tf_ms = tf_to_ms(tf)
        self._cur: dict | None = None
        self.prev_close: float | None = None
        self.dropped_late = 0

    # -- public ----------------------------------------------------------

    def on_trade(self, t: Trade) -> list[Candle]:
        if t.symbol != self.symbol:
            raise ValueError(f"trade for {t.symbol} fed to {self.symbol} aggregator")
        b = self._bucket(t.ts_ms)

        if self._cur is None:
            self._start(b, t.price)
            self._add(t)
            return []

        cur_b = self._cur["bucket"]
        if b < cur_b:
            # A tick that belongs to an already-closed bar. Rewriting a closed
            # candle would desync the indicators, so it is counted and dropped.
            self.dropped_late += 1
            if self.dropped_late in (1, 10, 100, 1000):
                log.warning("candle_aggregator: dropped %d late tick(s); latest %d ms behind bar %d",
                            self.dropped_late, cur_b - t.ts_ms, cur_b)
            return []

        if b == cur_b:
            self._add(t)
            return []

        out = self._advance_to(b)
        self._add(t)
        return out

    def on_clock(self, ts_ms: int) -> list[Candle]:
        """Force-close bars whose window has elapsed. Called on a short cadence by
        the feed handler so an idle (zero-volume) market still produces bars."""
        if self._cur is None:
            return []
        b = self._bucket(ts_ms)
        if b <= self._cur["bucket"]:
            return []
        return self._advance_to(b)

    def flush(self) -> list[Candle]:
        """Close the in-progress bar (shutdown only)."""
        return [] if self._cur is None else [self._finish()]

    @property
    def current_bucket(self) -> int | None:
        return None if self._cur is None else self._cur["bucket"]

    # -- internals -------------------------------------------------------

    def _bucket(self, ts_ms: int) -> int:
        return ts_ms - (ts_ms % self.tf_ms)

    def _start(self, bucket: int, price: float) -> None:
        self._cur = {"bucket": bucket, "o": price, "h": price, "l": price, "c": price,
                     "v": 0.0, "qv": 0.0, "n": 0}

    def _add(self, t: Trade) -> None:
        c = self._cur
        assert c is not None
        if c["n"] == 0 and c["v"] == 0.0:
            # First real trade of the bar: it defines the open. (A bar seeded by
            # the clock carries prev_close as a placeholder open.)
            c["o"] = c["h"] = c["l"] = t.price
        c["h"] = max(c["h"], t.price)
        c["l"] = min(c["l"], t.price)
        c["c"] = t.price
        c["v"] += t.qty
        c["qv"] += t.price * t.qty
        c["n"] += 1

    def _finish(self) -> Candle:
        c = self._cur
        assert c is not None
        candle = Candle(
            ts_open_ms=c["bucket"], symbol=self.symbol, tf=self.tf,
            open=c["o"], high=c["h"], low=c["l"], close=c["c"],
            volume=c["v"], quote_volume=c["qv"], n_trades=c["n"],
        )
        self.prev_close = candle.close
        self._cur = None
        return candle

    def _advance_to(self, target_bucket: int) -> list[Candle]:
        """Close the current bar, emit a flat bar for every skipped bucket, and
        open `target_bucket`."""
        assert self._cur is not None
        cur_b = self._cur["bucket"]
        out = [self._finish()]

        nb = cur_b + self.tf_ms
        gap = 0
        while nb < target_bucket:
            # Zero-volume minute: O=H=L=C=prev_close. VWAP is unaffected (it adds
            # 0/0), while RSI/MACD/ATR correctly see a flat bar rather than a hole.
            self._start(nb, self.prev_close if self.prev_close is not None else out[0].close)
            out.append(self._finish())
            nb += self.tf_ms
            gap += 1
        if gap:
            log.info("candle_aggregator: filled %d zero-volume bar(s) between %d and %d",
                     gap, cur_b, target_bucket)

        self._start(target_bucket, self.prev_close if self.prev_close is not None else out[0].close)
        return out
