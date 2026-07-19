"""Position sizing, entry gating, and the absolute risk backstops.

This is where the system says NO. Two properties matter more than anything else
in this file:

  1. It is the same code in backtest and live. `approve_entry` takes `now_ms` and
     `atr` explicitly rather than reading a clock or a global, so a simulated
     decision is bit-identical to a live one.
  2. The backstops it enforces -- max_daily_loss_pct, max_position_pct_equity,
     max_open_positions -- are NOT in bounds.json, which means the optimizer LLM
     cannot propose a change to them at all. They are the floor under the
     self-modifying loop.

NOTE (deliberate deviation from the blueprint signature): `approve_entry` also
takes `atr` and `now_ms`. Sizing and cooldown are both functions of those, and
sourcing them from ambient state would break backtest/live parity -- the exact
thing this module exists to guarantee.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.common.models import ClosedTrade, MarketState, RiskDecision, Signal
from src.execution.exchange_adapter import SymbolFilters

log = logging.getLogger(__name__)

MS_PER_DAY = 86_400_000

# An ATR below this fraction of price means the stop would sit inside the spread
# and tick noise -- every entry would be a coin flip stopped out by microstructure.
MIN_ATR_PCT = 0.0005   # 0.05%


class RiskManager:
    def __init__(self, cfg_provider: Callable[[], dict[str, Any]],
                 filters: SymbolFilters, initial_equity: float) -> None:
        self._cfg = cfg_provider
        self.filters = filters
        self.equity = float(initial_equity)

        self._day: int | None = None
        self._day_pnl_quote = 0.0
        self._day_start_equity = float(initial_equity)

        self.consecutive_stopouts = 0
        self.cooldown_until_ms = 0
        self.open_positions = 0
        self.halt_reason: str | None = None

    # -- external halts ----------------------------------------------------

    def halt(self, reason: str) -> None:
        """Hard stop from outside (optimizer T4, kill file, operator). Distinct
        from a feed DEGRADED state: this one does not clear when ticks resume."""
        if self.halt_reason != reason:
            log.error("RISK HALT engaged: %s", reason)
        self.halt_reason = reason

    def resume(self, reason: str = "") -> None:
        if self.halt_reason:
            log.warning("risk halt cleared (%s -> %s)", self.halt_reason, reason or "resumed")
        self.halt_reason = None

    # -- daily accounting --------------------------------------------------

    def _roll_day(self, now_ms: int) -> None:
        day = now_ms // MS_PER_DAY
        if self._day is None:
            self._day = day
            self._day_start_equity = self.equity
            return
        if day != self._day:
            log.info("UTC day rollover: day_pnl=%.2f -> reset; equity=%.2f",
                     self._day_pnl_quote, self.equity)
            self._day = day
            self._day_pnl_quote = 0.0
            self._day_start_equity = self.equity

    def mark_equity(self, equity: float, now_ms: int) -> None:
        self._roll_day(now_ms)
        self.equity = float(equity)

    def daily_halt_active(self, now_ms: int) -> bool:
        self._roll_day(now_ms)
        cfg = self._cfg()
        limit = self._day_start_equity * float(cfg["risk"]["max_daily_loss_pct"]) / 100.0
        return self._day_pnl_quote <= -limit

    @property
    def day_pnl_quote(self) -> float:
        return self._day_pnl_quote

    # -- sizing ------------------------------------------------------------

    def position_size(self, equity: float, atr: float, price: float) -> float:
        return self.size_detail(equity, atr, price)[0]

    def size_detail(self, equity: float, atr: float, price: float) -> tuple[str, str]:
        """Risk-based sizing capped by notional, floored to the lot grid.

        Returns (qty, binding_constraint). The caller wants to know WHICH rule
        bound: on short timeframes the stop distance is a fraction of a percent,
        risk-based sizing asks for many multiples of equity, and the notional cap
        silently becomes the only thing setting the size. `risk_per_trade_pct` is
        then inert -- see evaluation/economics.sizing_constraint.
        """
        cfg = self._cfg()["risk"]
        if atr <= 0 or price <= 0 or equity <= 0:
            return 0.0, "none"   # type: ignore[return-value]
        stop_distance = float(cfg["atr_mult_sl"]) * atr
        if stop_distance <= 0:
            return 0.0, "none"   # type: ignore[return-value]

        risk_quote = equity * float(cfg["risk_per_trade_pct"]) / 100.0
        qty = risk_quote / stop_distance
        binding = "risk"

        max_notional = equity * float(cfg["max_position_pct_equity"]) / 100.0
        if qty * price > max_notional:
            qty = max_notional / price
            binding = "notional"

        return self.filters.round_qty(qty), binding   # type: ignore[return-value]

    # -- entry gate --------------------------------------------------------

    def approve_entry(self, sig: Signal, mkt: MarketState, equity: float,
                      now_ms: int, atr: float) -> RiskDecision:
        cfg = self._cfg()
        rk, ex = cfg["risk"], cfg["execution"]
        self.mark_equity(equity, now_ms)

        def no(reason: str) -> RiskDecision:
            return RiskDecision(approved=False, reason=reason)

        if self.halt_reason:
            return no(f"risk halt active: {self.halt_reason}")
        if mkt.state != "READY":
            return no(f"feed state is {mkt.state}, not READY")
        if sig.action != "ENTER":
            return no(f"signal action is {sig.action}")
        if self.open_positions >= int(rk["max_open_positions"]):
            return no(f"already at max_open_positions={rk['max_open_positions']}")
        if self.daily_halt_active(now_ms):
            return no(f"daily loss limit hit (day_pnl={self._day_pnl_quote:.2f}, "
                      f"limit={rk['max_daily_loss_pct']}% of {self._day_start_equity:.2f})")
        if now_ms < self.cooldown_until_ms:
            return no(f"in stop-out cooldown for another {(self.cooldown_until_ms - now_ms)/1000:.0f}s")

        price = mkt.last_price
        if price <= 0:
            return no("no price")
        if atr <= 0:
            return no("no ATR")
        if atr < MIN_ATR_PCT * price:
            return no(f"ATR {atr:.6f} below {MIN_ATR_PCT:.2%} of price; stop would sit in the noise")

        spread_bps = mkt.spread_bps
        if spread_bps > float(ex["max_spread_bps"]):
            return no(f"spread {spread_bps:.1f}bps exceeds max {ex['max_spread_bps']}bps")

        qty, binding = self.size_detail(equity, atr, price)
        if qty <= 0:
            return no("computed size rounds to zero")
        if not self.filters.qty_ok(qty):
            return no(f"qty {qty} outside LOT_SIZE [{self.filters.min_qty}, {self.filters.max_qty}]")
        if not self.filters.notional_ok(qty, price):
            return no(f"notional {qty * price:.2f} below minNotional {self.filters.min_notional}")

        stop_px, tp_px = self.stops_for(price, atr, mkt)
        return RiskDecision(approved=True, qty=qty, reason="ok", stop_px=stop_px, tp_px=tp_px,
                            binding_constraint=binding)   # type: ignore[arg-type]

    def stops_for(self, entry_px: float, atr: float, mkt: MarketState | None = None) -> tuple[float, float]:
        """ATR stops, with a floor of 2x the spread.

        A stop closer than the round-trip spread is not a risk decision, it is a
        guarantee of being stopped out by the bid-ask bounce alone.
        """
        cfg = self._cfg()["risk"]
        stop_distance = float(cfg["atr_mult_sl"]) * atr
        if mkt is not None:
            spread_abs = (mkt.best_ask - mkt.best_bid) if (mkt.best_ask > 0 and mkt.best_bid > 0) else 0.0
            stop_distance = max(stop_distance, 2.0 * spread_abs)
        stop_px = entry_px - stop_distance
        tp_px = entry_px + float(cfg["atr_mult_tp"]) * atr
        return stop_px, tp_px

    # -- feedback ----------------------------------------------------------

    def on_trade_closed(self, t: ClosedTrade) -> None:
        self._roll_day(t.exit_ts_ms)
        self._day_pnl_quote += t.pnl_quote
        cfg = self._cfg()["risk"]

        if t.exit_reason == "sl":
            self.consecutive_stopouts += 1
            self.cooldown_until_ms = t.exit_ts_ms + int(cfg["cooldown_after_stopout_sec"]) * 1000
            log.info("stop-out #%d; cooldown until %d", self.consecutive_stopouts,
                     self.cooldown_until_ms)
        else:
            self.consecutive_stopouts = 0

        if self.daily_halt_active(t.exit_ts_ms):
            self.halt(f"daily loss limit breached: {self._day_pnl_quote:.2f} quote")

    def reset_day_for_test(self, now_ms: int, equity: float) -> None:
        self._day = now_ms // MS_PER_DAY
        self._day_pnl_quote = 0.0
        self._day_start_equity = equity
        self.equity = equity
