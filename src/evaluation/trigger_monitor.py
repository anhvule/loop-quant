"""Underperformance triggers and the anti-thrash gate.

Two jobs, and the second matters more than the first:

  1. Detect that the live strategy has diverged from what the config promised.
  2. Refuse to say so too often.

A self-optimizing system's characteristic failure is not missing a regime change
-- it is reacting to noise, deploying a change, reacting to the noise that change
produced, and oscillating forever. The anti-thrash rules here are hard-coded
constants rather than config keys precisely so the optimizer cannot widen its own
leash.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.common.db import DB
from src.common.models import BacktestBaseline, KPIReport, Trigger, UnderperformanceEvent

log = logging.getLogger(__name__)

MS_PER_HOUR = 3_600_000
MS_PER_DAY = 86_400_000

# --- anti-thrash constants. NOT configurable, NOT in bounds.json. ---
MIN_TRADES_SINCE_DEPLOY = 20
MIN_HOURS_SINCE_DEPLOY = 6
MAX_CYCLES_PER_24H = 4
MAX_COOLDOWN_MULT = 8.0

# Declarative trigger table, ordered most-severe first. `evaluate` returns the
# first that fires, so a drawdown breach outranks a slippage nag.
TRIGGERS: list[Trigger] = [
    Trigger("T4", "max_drawdown_pct > 1.5x baseline max_drawdown",
            scope="strategy", min_trades=5, halt_on_fire=True),
    Trigger("T1", "consecutive_stopouts >= 3",
            scope="strategy", min_trades=3),
    Trigger("T2", "bt_deviation_pct <= -10 over >= 20 trades",
            scope="strategy", min_trades=20),
    Trigger("T3", "live sharpe < 0.5x baseline sharpe over >= 30 trades",
            scope="strategy", min_trades=30),
    Trigger("T5", "avg_slippage_bps > 10 over >= 10 trades",
            scope="execution", min_trades=10),
]


class TriggerMonitor:
    def __init__(self, baseline: BacktestBaseline | None, triggers: list[Trigger],
                 db: DB, cfg_provider: Callable[[], dict[str, Any]],
                 last_deploy_ms: int = 0) -> None:
        self.baseline = baseline
        self.triggers = triggers
        self.db = db
        self._cfg = cfg_provider
        self.last_deploy_ms = last_deploy_ms

        self.cycle_in_flight = False
        self.cycle_starts: list[int] = []
        self.thrash_halt = False
        self._cooldown_mult = 1.0
        self.last_gate_reason = ""

    # -- lifecycle notifications -------------------------------------------

    def set_baseline(self, b: BacktestBaseline | None) -> None:
        self.baseline = b

    def note_deploy(self, ts_ms: int) -> None:
        """A fresh config resets the observation window: KPIs from before the
        deploy say nothing about the config running now."""
        self.last_deploy_ms = ts_ms
        self._cooldown_mult = 1.0
        log.info("trigger monitor: deploy noted at %d; observation window reset", ts_ms)

    def note_cycle_started(self, ts_ms: int) -> None:
        self.cycle_in_flight = True
        self.cycle_starts.append(ts_ms)

    def note_cycle_finished(self) -> None:
        self.cycle_in_flight = False

    def extend_cooldown(self, factor: float = 2.0) -> None:
        """Called when Phase A diagnoses 'variance'. The trigger was real but the
        evidence was noise, so back off before asking again."""
        self._cooldown_mult = min(self._cooldown_mult * factor, MAX_COOLDOWN_MULT)
        log.info("trigger cooldown extended x%.1f (now requires %d trades / %.0f hours)",
                 factor, self.min_trades_required, self.min_hours_required)

    @property
    def min_trades_required(self) -> int:
        return int(MIN_TRADES_SINCE_DEPLOY * self._cooldown_mult)

    @property
    def min_hours_required(self) -> float:
        return MIN_HOURS_SINCE_DEPLOY * self._cooldown_mult

    # -- gate ---------------------------------------------------------------

    def _cycles_in_last_24h(self, now_ms: int) -> int:
        self.cycle_starts = [t for t in self.cycle_starts if now_ms - t < MS_PER_DAY]
        return len(self.cycle_starts)

    def _gate(self, now_ms: int) -> str | None:
        """Returns a reason string if optimization is not allowed, else None."""
        if self.thrash_halt:
            return "thrash halt engaged; a human must restart"
        if self.cycle_in_flight:
            return "an optimization cycle is already in flight"

        if self._cycles_in_last_24h(now_ms) >= MAX_CYCLES_PER_24H:
            # The system has now tried to fix itself four times in a day and is
            # still unhappy. That is not a tuning problem; stop and get a human.
            self.thrash_halt = True
            log.critical("THRASH HALT: %d optimization cycles in 24h; halting for human review",
                         MAX_CYCLES_PER_24H)
            return "max optimization cycles per 24h reached"

        elapsed_h = (now_ms - self.last_deploy_ms) / MS_PER_HOUR
        if elapsed_h < self.min_hours_required:
            return (f"only {elapsed_h:.1f}h since last deploy "
                    f"(need {self.min_hours_required:.0f}h)")

        n = self.db.count_closed_trades(since_ms=self.last_deploy_ms)
        if n < self.min_trades_required:
            return f"only {n} trades since last deploy (need {self.min_trades_required})"
        return None

    def evaluate(self, r: KPIReport) -> UnderperformanceEvent | None:
        gate = self._gate(r.ts_ms)
        if gate:
            self.last_gate_reason = gate
            return None

        for t in self.triggers:
            if r.window_trades < t.min_trades:
                continue
            fired, text = self._check(t, r)
            if fired:
                log.warning("TRIGGER %s fired: %s", t.id, text)
                return UnderperformanceEvent(
                    trigger_id=t.id, scope=t.scope, condition_text=text,
                    kpi_report=r, baseline=self.baseline, fired_at_ms=r.ts_ms,
                    halt_trading=t.halt_on_fire,
                )
        return None

    def _check(self, t: Trigger, r: KPIReport) -> tuple[bool, str]:
        b = self.baseline

        if t.id == "T1":
            # Static: needs no baseline, so it stays live even with no history.
            n = r.consecutive_stopouts
            return n >= 3, f"{n} consecutive stop-outs (threshold 3)"

        if t.id == "T5":
            # Static, and execution-scoped: the strategy may be fine and the
            # fills bad. Scope tells Module 4 to touch execution params only.
            s = r.avg_slippage_bps
            if s is None:
                return False, ""
            return s > 10.0, f"average slippage {s:.1f}bps over {r.window_trades} trades (threshold 10)"

        if b is None:
            # T2/T3/T4 are all relative to the baseline. Without one they are not
            # "not fired" -- they are unevaluable, and must not silently pass.
            return False, ""

        if t.id == "T2":
            d = r.bt_deviation_pct
            if d is None:
                # Baseline expectancy ~ 0 makes the ratio meaningless (blueprint
                # edge case): skip and let T3/T4 carry.
                return False, ""
            return d <= -10.0, (f"live expectancy is {d:.1f}% below backtest "
                                f"({r.expectancy:.4f}% vs {b.expectancy:.4f}%)")

        if t.id == "T3":
            if r.sharpe is None or b.sharpe <= 0:
                return False, ""
            return r.sharpe < 0.5 * b.sharpe, (
                f"live sharpe {r.sharpe:.2f} below half of baseline {b.sharpe:.2f}")

        if t.id == "T4":
            if r.max_drawdown_pct is None or b.max_drawdown_pct <= 0:
                return False, ""
            return r.max_drawdown_pct > 1.5 * b.max_drawdown_pct, (
                f"live drawdown {r.max_drawdown_pct:.2f}% exceeds 1.5x baseline "
                f"{b.max_drawdown_pct:.2f}%")

        return False, ""

    @property
    def active_trigger_ids(self) -> list[str]:
        """Which triggers can actually fire right now. With no baseline only the
        static ones are live -- worth logging so a missing baseline is visible
        rather than looking like a quiet, healthy system."""
        if self.baseline is None:
            return ["T1", "T5"]
        return [t.id for t in self.triggers]
