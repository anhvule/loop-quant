"""Order lifecycle, protective exits, and reconciliation.

The single invariant this module exists to hold:

    **A filled entry is never left without a protective exit.**

Everything else here -- the partial-fill accounting, the OCO price nudging, the
flatten-on-rejection path, the startup reconcile -- is in service of that one
sentence. If the OCO cannot be placed for any reason, the position is closed at
market immediately rather than held and hoped over.

Position lifecycle: FLAT -> PENDING_ENTRY -> OPEN -> PENDING_EXIT -> FLAT.
Every transition is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any, Callable

from src.common.db import DB
from src.common.event_bus import EventBus
from src.common.models import ClosedTrade, Fill, MarketState, Position
from src.execution.exchange_adapter import ExchangeAdapter, ExchangeError
from src.execution.risk_manager import RiskManager

log = logging.getLogger(__name__)

# The stop-limit's limit price sits this far below its trigger. Pegging the limit
# at the trigger is how a stop gets triggered but never filled in a fast move.
STOP_LIMIT_OFFSET_BPS = 20.0
OCO_SETTLE_SEC = 1.5
TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH"}


class OrderManager:
    def __init__(self, symbol: str, adapter: ExchangeAdapter, bus: EventBus, db: DB,
                 market: MarketState, rm: RiskManager,
                 cfg_provider: Callable[[], dict[str, Any]]) -> None:
        self.symbol = symbol.upper()
        self.adapter = adapter
        self.bus = bus
        self.db = db
        self.market = market
        self.rm = rm
        self._cfg = cfg_provider

        self.position = Position(symbol=self.symbol)
        self._orders: dict[str, dict[str, Any]] = {}
        self._oco_list_id: int | None = None
        self._lock = asyncio.Lock()
        bus.subscribe("order_filled", self._on_exec_report)

    # -- state helpers -----------------------------------------------------

    def _transition(self, new: str) -> None:
        if self.position.state != new:
            log.info("position %s -> %s", self.position.state, new)
            self.position.state = new  # type: ignore[assignment]

    def _new_coid(self, suffix: str = "") -> str:
        v = self._cfg()["version"]
        return f"lq-{v}-{uuid.uuid4().hex[:12]}{suffix}"

    def _track(self, coid: str) -> dict[str, Any]:
        st = {"coid": coid, "status": "NEW", "cum_qty": 0.0, "cum_quote": 0.0,
              "order_id": "", "done": asyncio.Event(), "last_ts": 0}
        self._orders[coid] = st
        return st

    # -- execution reports -------------------------------------------------

    async def _on_exec_report(self, d: dict[str, Any]) -> None:
        if not isinstance(d, dict) or d.get("e") != "executionReport":
            return
        if d.get("s") != self.symbol:
            return
        coid = str(d.get("c", ""))
        st = self._orders.get(coid)
        if st is None and coid.startswith("lq-"):
            st = self._track(coid)
        if st is None:
            return

        st["order_id"] = str(d.get("i", ""))
        st["status"] = str(d.get("X", st["status"]))
        st["last_ts"] = int(d.get("T", 0))
        if float(d.get("l", 0) or 0) > 0:
            st["cum_qty"] = float(d.get("z", 0) or 0)
            st["cum_quote"] = float(d.get("Z", 0) or 0)

        if st["status"] in TERMINAL:
            st["done"].set()

        # A protective leg filling is the exit event.
        if d.get("S") == "SELL" and st["status"] == "FILLED" and self.position.is_open:
            reason = ("tp" if coid.endswith("-tp") else
                      "sl" if coid.endswith("-sl") else
                      "signal" if coid.endswith("-sg") else
                      "flatten" if coid.endswith("-fl") else "manual")
            avg = st["cum_quote"] / st["cum_qty"] if st["cum_qty"] > 0 else float(d.get("L", 0))
            await self._finalize_exit(avg, st["cum_qty"], reason, int(d.get("T", _now_ms())))

    # -- entry -------------------------------------------------------------

    async def enter_long(self, qty: float, mkt: MarketState, atr: float) -> Fill | None:
        async with self._lock:
            if self.position.state != "FLAT":
                log.warning("enter_long ignored: position is %s", self.position.state)
                return None
            self._transition("PENDING_ENTRY")
            try:
                fill = await self._do_entry(qty, mkt)
            except Exception:
                log.exception("entry failed; returning to FLAT")
                self._transition("FLAT")
                return None

            if fill is None or fill.qty <= 0:
                self._transition("FLAT")
                return None

            self.position = Position(
                symbol=self.symbol, state="OPEN", side="BUY", qty=fill.qty,
                entry_px=fill.price, entry_ts_ms=fill.ts_ms,
                atr_at_entry=atr, config_version=int(self._cfg()["version"]),
                trade_id=fill.client_order_id, entry_slippage_bps=fill.slippage_bps,
            )
            self.rm.open_positions = 1
            log.info("ENTERED %s qty=%.8f @ %.2f (slippage %.2f bps)",
                     self.symbol, fill.qty, fill.price, fill.slippage_bps)

        ok = await self.place_protective_oco(fill, atr)
        if not ok:
            # The invariant: unprotected is not a state we tolerate, even briefly.
            log.error("protective OCO could not be placed; flattening immediately")
            await self.flatten("oco_rejected")
            return None
        return fill

    async def _do_entry(self, qty: float, mkt: MarketState) -> Fill | None:
        cfg = self._cfg()["execution"]
        f = self.adapter.filters(self.symbol)
        intended = mkt.best_bid if mkt.best_bid > 0 else mkt.last_price

        if cfg["order_type"] == "MARKET":
            return await self._market_buy(qty, intended)

        coid = self._new_coid()
        st = self._track(coid)
        try:
            await self.adapter.place_order(self.symbol, "BUY", "LIMIT", qty=qty,
                                           price=intended, tif="GTC", client_order_id=coid)
        except ExchangeError as e:
            log.warning("limit entry rejected (%s); falling back to market", e)
            return await self._market_buy(qty, intended)

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(st["done"].wait(), timeout=float(cfg["limit_timeout_sec"]))

        if st["status"] == "FILLED" and st["cum_qty"] > 0:
            return self._fill_from(st, intended)

        # Not (fully) filled inside the window. Cancel and read the truth off the
        # cancel response -- the order may have filled during the round trip.
        cum = st["cum_qty"]
        try:
            resp = await self.adapter.cancel(self.symbol, client_order_id=coid)
            cum = float(resp.get("executedQty", cum) or cum)
            st["cum_qty"] = cum
            st["cum_quote"] = float(resp.get("cummulativeQuoteQty", st["cum_quote"]) or st["cum_quote"])
        except ExchangeError as e:
            if e.code == -2011:      # already gone: filled or cancelled already
                try:
                    o = await self.adapter.get_order(self.symbol, client_order_id=coid)
                    cum = float(o.get("executedQty", cum) or cum)
                    st["cum_qty"] = cum
                    st["cum_quote"] = float(o.get("cummulativeQuoteQty", 0) or 0)
                except ExchangeError:
                    pass
            else:
                raise

        if cum <= 0:
            log.info("limit entry unfilled after %ss; crossing with market",
                     cfg["limit_timeout_sec"])
            return await self._market_buy(qty, intended)

        if cum < qty:
            # Keep the filled portion at its real size; do NOT chase the remainder.
            # Chasing turns a passive entry into a worse-priced taker entry that
            # the RiskManager never approved.
            log.info("partial entry fill %.8f/%.8f; keeping the filled portion", cum, qty)
        return self._fill_from(st, intended)

    async def _market_buy(self, qty: float, intended: float) -> Fill | None:
        coid = self._new_coid()
        st = self._track(coid)
        try:
            resp = await self.adapter.place_order(self.symbol, "BUY", "MARKET", qty=qty,
                                                  client_order_id=coid)
        except ExchangeError as e:
            log.error("market entry rejected: %s", e)
            return None
        st["cum_qty"] = float(resp.get("executedQty", 0) or 0)
        st["cum_quote"] = float(resp.get("cummulativeQuoteQty", 0) or 0)
        st["status"] = str(resp.get("status", "FILLED"))
        st["order_id"] = str(resp.get("orderId", ""))
        if st["cum_qty"] <= 0:
            return None
        return self._fill_from(st, intended)

    def _fill_from(self, st: dict[str, Any], intended: float) -> Fill:
        avg = st["cum_quote"] / st["cum_qty"] if st["cum_qty"] > 0 else intended
        return Fill(order_id=st["order_id"], client_order_id=st["coid"],
                    ts_ms=st["last_ts"] or _now_ms(), symbol=self.symbol, side="BUY",
                    price=avg, qty=st["cum_qty"], intended_px=intended)

    # -- protective exit ---------------------------------------------------

    async def place_protective_oco(self, fill: Fill, atr: float) -> bool:
        """Sized to the ACTUAL filled qty, not the requested qty -- a partial fill
        protected at full size would leave a naked short leg."""
        f = self.adapter.filters(self.symbol)
        qty = f.round_qty(fill.qty)
        if qty <= 0:
            log.error("filled qty %.10f rounds to zero on the lot grid", fill.qty)
            return False

        stop_px, tp_px = self.rm.stops_for(fill.price, atr, self.market)
        self.position.stop_px, self.position.tp_px = stop_px, tp_px

        for attempt in (1, 2):
            tp, sl = tp_px, stop_px
            if attempt == 2:
                # Retry: snap onto the tick grid and push each leg to the correct
                # side of the current book. A LIMIT_MAKER that would cross, or a
                # stop already through the market, is rejected outright.
                tick = float(f.tick_size)
                ask = self.market.best_ask or fill.price
                bid = self.market.best_bid or fill.price
                tp = max(f.round_price(tp), f.round_price(ask + tick))
                sl = min(f.round_price(sl), f.round_price(bid - tick))
                log.warning("retrying OCO with grid-snapped prices tp=%.2f sl=%.2f", tp, sl)

            stop_limit = sl * (1.0 - STOP_LIMIT_OFFSET_BPS * 1e-4)
            base = self._new_coid()
            try:
                resp = await self.adapter.place_oco_sell(
                    self.symbol, qty, tp_price=tp, stop_price=sl,
                    stop_limit_price=stop_limit,
                    list_client_order_id=base[:36],
                    tp_client_order_id=f"{base}-tp", sl_client_order_id=f"{base}-sl",
                )
            except ExchangeError as e:
                log.error("OCO attempt %d rejected: %s", attempt, e)
                continue

            self._oco_list_id = resp.get("orderListId")
            self.position.protective_order_ids = [str(o.get("orderId"))
                                                  for o in resp.get("orders", [])]
            self._track(f"{base}-tp")
            self._track(f"{base}-sl")
            log.info("protective OCO placed: tp=%.2f sl=%.2f qty=%.8f (listId=%s)",
                     tp, sl, qty, self._oco_list_id)
            return True

        return False

    # -- exits -------------------------------------------------------------

    async def flatten(self, reason: str) -> None:
        """Cancel everything, then close at market. Used by the feed watchdog, the
        daily-loss backstop, the kill file, and the OCO-rejection path."""
        async with self._lock:
            if self.position.state == "FLAT":
                await self._cancel_all_quiet()
                return
            self._transition("PENDING_EXIT")

        await self._cancel_all_quiet()
        await asyncio.sleep(OCO_SETTLE_SEC / 3)

        qty = await self._sellable_qty()
        if qty <= 0:
            log.warning("flatten(%s): nothing sellable; marking FLAT", reason)
            self._reset_position()
            return

        coid = self._new_coid("-fl")
        st = self._track(coid)
        try:
            resp = await self.adapter.place_order(self.symbol, "SELL", "MARKET", qty=qty,
                                                  client_order_id=coid)
        except ExchangeError as e:
            log.critical("FLATTEN FAILED for %s (%s) -- position may still be open: %s",
                         self.symbol, reason, e)
            self.rm.halt(f"flatten failed: {e}")
            return

        cum = float(resp.get("executedQty", 0) or 0)
        quote = float(resp.get("cummulativeQuoteQty", 0) or 0)
        px = quote / cum if cum > 0 else self.market.last_price
        log.warning("FLATTENED %s qty=%.8f @ %.2f (reason=%s)", self.symbol, cum, px, reason)
        await self._finalize_exit(px, cum, "flatten" if reason != "signal" else "signal",
                                  _now_ms())

    async def exit_on_signal(self) -> None:
        await self.flatten("signal")

    async def _cancel_all_quiet(self) -> None:
        try:
            await self.adapter.cancel_all(self.symbol)
        except ExchangeError as e:
            log.warning("cancel_all failed: %s", e)

    async def _sellable_qty(self) -> float:
        """Exchange truth beats local bookkeeping: sell what we actually hold."""
        f = self.adapter.filters(self.symbol)
        try:
            acct = await self.adapter.get_account()
            free = next((float(b["free"]) for b in acct.get("balances", [])
                         if b["asset"] == f.base_asset), 0.0)
        except ExchangeError:
            free = self.position.qty
        return f.round_qty(min(free, self.position.qty) if self.position.qty > 0 else free)

    async def _finalize_exit(self, exit_px: float, qty: float, reason: str,
                             ts_ms: int) -> None:
        p = self.position
        if not p.is_open or qty <= 0:
            self._reset_position()
            return

        fee_bps = 10.0
        entry_notional = p.qty * p.entry_px
        exit_notional = qty * exit_px
        fees = (entry_notional + exit_notional) * fee_bps * 1e-4
        pnl_quote = exit_notional - entry_notional - fees
        pnl_pct = (pnl_quote / entry_notional * 100.0) if entry_notional > 0 else 0.0

        t = ClosedTrade(
            trade_id=p.trade_id or f"lq-{uuid.uuid4().hex[:10]}", symbol=self.symbol,
            side="BUY", entry_ts_ms=p.entry_ts_ms, exit_ts_ms=ts_ms,
            entry_px=p.entry_px, exit_px=exit_px, qty=qty,
            pnl_quote=pnl_quote, pnl_pct=pnl_pct, exit_reason=reason,  # type: ignore[arg-type]
            slippage_bps=p.entry_slippage_bps, config_version=p.config_version,
            atr_at_entry=p.atr_at_entry,
        )
        self.db.insert_closed_trade(t)
        self._reset_position()
        log.info("CLOSED %s %s pnl=%.2f (%.3f%%) reason=%s",
                 self.symbol, t.trade_id, pnl_quote, pnl_pct, reason)
        await self.bus.publish("trade_closed", t)

    def _reset_position(self) -> None:
        self._transition("FLAT")
        self.position = Position(symbol=self.symbol)
        self.rm.open_positions = 0
        self._oco_list_id = None

    # -- reconciliation ----------------------------------------------------

    async def reconcile(self) -> None:
        """Run before the strategy loop starts and after every reconnect.

        Local state ALWAYS yields to exchange truth. The dangerous case is an
        orphan: a position on the exchange that our process has no memory of
        (we crashed between the entry fill and the OCO). It gets a protective
        OCO at current-ATR stops immediately.
        """
        f = self.adapter.filters(self.symbol)
        try:
            acct = await self.adapter.get_account()
            open_orders = await self.adapter.get_open_orders(self.symbol)
        except ExchangeError as e:
            log.error("reconcile failed (%s); halting rather than guessing", e)
            self.rm.halt(f"reconcile failed: {e}")
            return

        base_free = base_locked = 0.0
        for b in acct.get("balances", []):
            if b["asset"] == f.base_asset:
                base_free, base_locked = float(b["free"]), float(b["locked"])
        base_total = base_free + base_locked
        px = self.market.last_price or 0.0
        has_position = px > 0 and base_total * px >= float(f.min_notional)

        log.info("reconcile: base=%.8f (%.2f quote), open_orders=%d, local=%s",
                 base_total, base_total * px, len(open_orders), self.position.state)

        if not has_position:
            if self.position.is_open:
                log.warning("local position but exchange is flat; trusting the exchange")
            if open_orders:
                log.warning("cancelling %d orphaned order(s) with no position", len(open_orders))
                await self._cancel_all_quiet()
            self._reset_position()
            return

        protective = [o for o in open_orders if o.get("side") == "SELL"]
        if self.position.is_open and protective:
            log.info("reconcile: position and protective orders both present; nothing to do")
            return

        # Orphan (or unprotected) position: adopt it and protect it now.
        log.warning("reconcile: adopting an unprotected position of %.8f %s",
                    base_total, f.base_asset)
        await self._cancel_all_quiet()
        qty = f.round_qty(base_free if base_free > 0 else base_total)
        if qty <= 0:
            self._reset_position()
            return

        self.position = Position(
            symbol=self.symbol, state="OPEN", side="BUY", qty=qty,
            entry_px=px, entry_ts_ms=_now_ms(), atr_at_entry=0.0,
            config_version=int(self._cfg()["version"]),
            trade_id=f"adopted-{uuid.uuid4().hex[:8]}",
        )
        self.rm.open_positions = 1
        return

    async def protect_adopted(self, atr: float) -> None:
        """Second half of adopting an orphan: needs a live ATR, which reconcile()
        may run before the indicators have warmed."""
        if not self.position.is_open or self.position.protective_order_ids:
            return
        if atr <= 0:
            log.error("cannot protect adopted position without an ATR; flattening")
            await self.flatten("adopted_no_atr")
            return
        self.position.atr_at_entry = atr
        fill = Fill(order_id="", client_order_id=self.position.trade_id, ts_ms=_now_ms(),
                    symbol=self.symbol, side="BUY", price=self.position.entry_px,
                    qty=self.position.qty, intended_px=self.position.entry_px)
        if not await self.place_protective_oco(fill, atr):
            await self.flatten("adopted_oco_rejected")


def _now_ms() -> int:
    return int(time.time() * 1000)
