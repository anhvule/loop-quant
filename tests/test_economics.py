"""Trade economics: the arithmetic that decides whether a config can EVER work.

These tests pin a structural finding about the shipped configuration. Module 4's
Phase A prompt asks the LLM to compute a breakeven win rate; this module computes
it deterministically so the diagnosis rests on arithmetic rather than on the
model's estimate, and so the engine can refuse to start quietly losing money.
"""

from __future__ import annotations

import copy

import pytest

from src.evaluation.economics import sizing_constraint, trade_economics

PRICE = 50_000.0


def _atr(pct: float) -> float:
    return PRICE * pct / 100.0


def test_fee_exceeds_stop_distance_at_typical_1m_btc_volatility(cfg):
    """The core trap: fees are charged on notional, stops are measured in ATR.
    At 0.05% ATR the round-trip fee (0.20%) is larger than the whole target, so a
    'winning' trade still loses money and NO win rate rescues it."""
    e = trade_economics(cfg, atr=_atr(0.05), price=PRICE, fee_bps_per_side=10.0)
    assert e.net_win_pct < 0
    assert e.breakeven_win_rate is None
    assert not e.viable
    assert "does not even cover" in e.note


def test_breakeven_win_rate_is_unreachable_at_typical_1m_btc_volatility(cfg):
    """At 0.10% ATR -- squarely typical for BTC 1m -- the config needs an 87.5%
    win rate to break even. No mean-reversion edge delivers that."""
    e = trade_economics(cfg, atr=_atr(0.10), price=PRICE, fee_bps_per_side=10.0)
    assert e.breakeven_win_rate == pytest.approx(0.875, abs=0.005)
    assert not e.viable


def test_config_becomes_viable_only_at_much_higher_volatility(cfg):
    """Sanity on the other side: the strategy shape is fine, the timeframe is the
    problem. At 0.5% ATR (a 15m/1h bar, or a violent 1m) it needs 47.5%."""
    e = trade_economics(cfg, atr=_atr(0.5), price=PRICE, fee_bps_per_side=10.0)
    assert e.breakeven_win_rate == pytest.approx(0.475, abs=0.005)
    assert e.viable


def test_widening_stops_within_bounds_restores_viability(cfg):
    """The optimizer CAN reach a viable region without leaving bounds.json:
    sl 1.5->3.0 and tp 2.5->6.0 takes breakeven from 87.5% to ~55.6% at 0.10% ATR.
    This is why the safety framework allows those ranges."""
    wide = copy.deepcopy(cfg)
    wide["risk"]["atr_mult_sl"] = 3.0     # bounds.json max
    wide["risk"]["atr_mult_tp"] = 6.0     # bounds.json max
    e = trade_economics(wide, atr=_atr(0.10), price=PRICE, fee_bps_per_side=10.0)
    assert e.breakeven_win_rate == pytest.approx(0.556, abs=0.005)
    assert e.viable


def test_zero_fees_would_make_the_shipped_config_viable(cfg):
    """Isolates the cause: with fees removed, the same config needs only 37.5%
    (which is just the R:R of 2.5/1.5). The strategy is not broken -- the fee
    drag at this timeframe is."""
    e = trade_economics(cfg, atr=_atr(0.10), price=PRICE, fee_bps_per_side=0.0)
    assert e.breakeven_win_rate == pytest.approx(0.375, abs=0.005)
    assert e.viable


def test_gross_rr_matches_config_multiples(cfg):
    e = trade_economics(cfg, atr=_atr(0.5), price=PRICE)
    assert e.gross_rr == pytest.approx(2.5 / 1.5, rel=1e-6)


def test_sizing_constraint_reports_dead_risk_dial(cfg):
    s = sizing_constraint(cfg, atr=_atr(0.10), price=PRICE)
    assert s["binds"] == "notional"
    assert s["risk_dial_is_live"] is False
    assert s["configured_risk_per_trade_pct"] == 0.5
    assert s["effective_risk_per_trade_pct"] < 0.02
    # to make the dial live you would need a 5% stop distance
    assert s["stop_pct_needed_for_risk_sizing"] == pytest.approx(5.0)


def test_sizing_constraint_reports_live_risk_dial_when_stops_are_wide(cfg):
    s = sizing_constraint(cfg, atr=_atr(5.0), price=PRICE)   # stop = 7.5% of price
    assert s["binds"] == "risk"
    assert s["risk_dial_is_live"] is True
    assert s["effective_risk_per_trade_pct"] == pytest.approx(0.5)


def test_degenerate_inputs_do_not_raise(cfg):
    assert trade_economics(cfg, atr=0.0, price=PRICE).breakeven_win_rate is None
    assert trade_economics(cfg, atr=100.0, price=0.0).breakeven_win_rate is None
    assert sizing_constraint(cfg, atr=0.0, price=PRICE)["binds"] == "unknown"
