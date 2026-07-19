"""Triggers and, more importantly, the anti-thrash gate.

The gate is the difference between a system that adapts and a system that
oscillates. Most of these tests are about NOT firing.
"""

from __future__ import annotations

import pytest

from src.common.models import BacktestBaseline, ClosedTrade, KPIReport
from src.evaluation.trigger_monitor import (
    MAX_CYCLES_PER_24H, MS_PER_DAY, MS_PER_HOUR, TRIGGERS, TriggerMonitor,
)

BASE = BacktestBaseline(generated_ms=0, config_version=1, window_days=30, n_trades=100,
                        expectancy=0.20, sharpe=2.0, max_drawdown_pct=5.0,
                        win_rate=0.55, profit_factor=1.6)
T0 = 100 * MS_PER_DAY   # well past the 6h gate measured from deploy 0


def _report(ts=T0, n=30, stopouts=0, dev=None, sharpe=2.0, dd=5.0, slip=1.0, expectancy=0.2):
    return KPIReport(ts_ms=ts, window_trades=n, win_rate=0.5, profit_factor=1.2,
                     expectancy=expectancy, sharpe=sharpe, max_drawdown_pct=dd,
                     avg_slippage_bps=slip, consecutive_stopouts=stopouts,
                     bt_deviation_pct=dev, equity=10_000.0, config_version=1)


def _seed_trades(db, n, since_ms=0, prefix="t"):
    """trade_id must be unique per call site -- closed_trades is INSERT OR REPLACE,
    so a repeated id silently overwrites instead of adding."""
    for i in range(n):
        db.insert_closed_trade(ClosedTrade(
            trade_id=f"{prefix}{i}", symbol="BTCUSDT", side="BUY",
            entry_ts_ms=since_ms + i, exit_ts_ms=since_ms + i + 1,
            entry_px=100.0, exit_px=101.0, qty=1.0, pnl_quote=1.0, pnl_pct=1.0,
            exit_reason="tp", slippage_bps=0.0, config_version=1, atr_at_entry=1.0))


def _mon(db, baseline=BASE, last_deploy_ms=0):
    return TriggerMonitor(baseline, TRIGGERS, db, lambda: {"version": 1},
                          last_deploy_ms=last_deploy_ms)


# -- individual triggers ------------------------------------------------------

def test_t1_fires_on_three_consecutive_stopouts(db):
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(stopouts=3))
    assert ev is not None and ev.trigger_id == "T1"
    assert ev.scope == "strategy"
    assert not ev.halt_trading


def test_t1_does_not_fire_on_two(db):
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(stopouts=2)) is None


def test_t2_fires_on_ten_percent_backtest_deviation(db):
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(dev=-12.0))
    assert ev is not None and ev.trigger_id == "T2"


def test_t2_does_not_fire_above_threshold_or_when_outperforming(db):
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(dev=-9.0)) is None
    assert _mon(db).evaluate(_report(dev=+50.0)) is None


def test_t2_needs_twenty_trades(db):
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(n=19, dev=-50.0)) is None
    assert _mon(db).evaluate(_report(n=20, dev=-50.0)) is not None


def test_t2_skipped_when_baseline_expectancy_is_zero(db):
    """Blueprint edge case: the deviation ratio is undefined near zero, so T2 must
    stand down rather than divide by ~0. dev=None models exactly that."""
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(dev=None)) is None


def test_t3_fires_below_half_baseline_sharpe(db):
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(sharpe=0.9))     # baseline 2.0 -> half is 1.0
    assert ev is not None and ev.trigger_id == "T3"
    assert _mon(db).evaluate(_report(sharpe=1.1)) is None


def test_t3_needs_thirty_trades(db):
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(n=29, sharpe=0.1)) is None


def test_t4_fires_on_excess_drawdown_and_halts(db):
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(dd=8.0))        # baseline 5.0 -> 1.5x is 7.5
    assert ev is not None and ev.trigger_id == "T4"
    assert ev.halt_trading is True                 # the only trigger that stops trading


def test_t4_outranks_other_triggers(db):
    """Severity ordering: a drawdown breach must not be masked by a stop-out run
    that happens to be reported in the same KPI snapshot."""
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(dd=99.0, stopouts=5, dev=-50.0, sharpe=0.01))
    assert ev.trigger_id == "T4"


def test_t5_fires_on_slippage_and_is_execution_scoped(db):
    """Scope matters: the strategy may be fine and the fills bad. Module 4 must
    tune execution params, not strategy weights."""
    _seed_trades(db, 30)
    ev = _mon(db).evaluate(_report(slip=12.0))
    assert ev is not None and ev.trigger_id == "T5"
    assert ev.scope == "execution"


def test_t5_needs_ten_trades(db):
    _seed_trades(db, 30)
    assert _mon(db).evaluate(_report(n=9, slip=99.0)) is None


# -- baseline availability ----------------------------------------------------

def test_without_baseline_only_static_triggers_are_active(db):
    """Blueprint edge case. A missing baseline must not make relative triggers
    silently 'pass' -- that would look like a healthy system."""
    _seed_trades(db, 30)
    m = _mon(db, baseline=None)
    assert m.active_trigger_ids == ["T1", "T5"]
    assert m.evaluate(_report(dev=-99.0, sharpe=0.0, dd=99.0)) is None   # T2/T3/T4 unevaluable
    assert m.evaluate(_report(stopouts=3)).trigger_id == "T1"            # T1 still live
    assert m.evaluate(_report(slip=50.0)).trigger_id == "T5"             # T5 still live


# -- anti-thrash gate ---------------------------------------------------------

def test_gate_requires_minimum_trades_since_deploy(db):
    _seed_trades(db, 19, prefix="a")
    m = _mon(db)
    assert m.evaluate(_report(stopouts=5)) is None
    assert "19 trades" in m.last_gate_reason
    _seed_trades(db, 1, since_ms=1000, prefix="b")
    assert m.evaluate(_report(stopouts=5)) is not None


def test_gate_requires_minimum_hours_since_deploy(db):
    deploy = T0
    _seed_trades(db, 30, since_ms=deploy)      # trades must post-date the deploy to count
    m = _mon(db, last_deploy_ms=deploy)
    assert m.evaluate(_report(ts=deploy + 5 * MS_PER_HOUR, stopouts=5)) is None
    assert "since last deploy" in m.last_gate_reason
    assert m.evaluate(_report(ts=deploy + 7 * MS_PER_HOUR, stopouts=5)) is not None


def test_gate_blocks_a_second_concurrent_cycle(db):
    _seed_trades(db, 30)
    m = _mon(db)
    assert m.evaluate(_report(stopouts=5)) is not None
    m.note_cycle_started(T0)
    assert m.evaluate(_report(stopouts=5)) is None
    assert "already in flight" in m.last_gate_reason
    m.note_cycle_finished()
    assert m.evaluate(_report(stopouts=5)) is not None


def test_four_cycles_in_24h_triggers_a_permanent_thrash_halt(db):
    """The system has tried to fix itself four times in a day and is still
    unhappy. That is not a tuning problem."""
    _seed_trades(db, 30)
    m = _mon(db)
    for i in range(MAX_CYCLES_PER_24H):
        m.note_cycle_started(T0 + i * MS_PER_HOUR)
        m.note_cycle_finished()
    assert m.evaluate(_report(ts=T0 + 5 * MS_PER_HOUR, stopouts=5)) is None
    assert m.thrash_halt is True
    # and it does not clear on its own, even a week later
    assert m.evaluate(_report(ts=T0 + 7 * MS_PER_DAY, stopouts=5)) is None
    assert "human" in m.last_gate_reason


def test_cycles_older_than_24h_do_not_count(db):
    _seed_trades(db, 30)
    m = _mon(db)
    for i in range(MAX_CYCLES_PER_24H):
        m.note_cycle_started(T0 - 2 * MS_PER_DAY + i * MS_PER_HOUR)
        m.note_cycle_finished()
    assert m.evaluate(_report(ts=T0, stopouts=5)) is not None
    assert m.thrash_halt is False


def test_deploy_resets_the_observation_window(db):
    _seed_trades(db, 30)
    m = _mon(db)
    assert m.evaluate(_report(stopouts=5)) is not None
    m.note_deploy(T0)
    # zero trades and zero hours have elapsed under the NEW config
    assert m.evaluate(_report(ts=T0 + 1000, stopouts=5)) is None


def test_variance_verdict_doubles_the_cooldown(db):
    """The primary defence against overfitting noise: when Phase A says the
    evidence was variance, ask again later with more data, not sooner."""
    _seed_trades(db, 30)
    m = _mon(db)
    assert m.min_trades_required == 20
    m.extend_cooldown(2.0)
    assert m.min_trades_required == 40
    assert m.min_hours_required == 12
    assert m.evaluate(_report(stopouts=5)) is None    # 30 trades < 40 now

    _seed_trades(db, 15, since_ms=5000, prefix="more")
    assert m.evaluate(_report(stopouts=5)) is not None


def test_cooldown_multiplier_is_capped(db):
    m = _mon(db)
    for _ in range(20):
        m.extend_cooldown(2.0)
    assert m.min_trades_required <= 20 * 8      # MAX_COOLDOWN_MULT


def test_deploy_resets_the_cooldown_multiplier(db):
    _seed_trades(db, 30)
    m = _mon(db)
    m.extend_cooldown(2.0)
    assert m.min_trades_required == 40
    m.note_deploy(0)
    assert m.min_trades_required == 20
