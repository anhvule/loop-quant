"""Signal math. These are the formulas the optimizer retunes, so their shape
(sign conventions, saturation, normalization) must be nailed down."""

from __future__ import annotations

import copy

import pytest

from src.common.models import IndicatorSnapshot, MarketState
from src.execution.signal_engine import SignalEngine, clamp


def _snap(vwap=100.0, rsi=50.0, macd_hist=0.0, atr=1.0, close=100.0):
    return IndicatorSnapshot(ts_ms=0, symbol="BTCUSDT", tf="1m", vwap=vwap, rsi=rsi,
                             macd=macd_hist, macd_signal=0.0, macd_hist=macd_hist,
                             atr=atr, close=close)


def _mkt(price=100.0, state="READY"):
    m = MarketState(symbol="BTCUSDT", last_price=price)
    m.state = state
    return m


# -- sub-signals -------------------------------------------------------------

def test_clamp_saturates():
    assert clamp(5.0) == 1.0
    assert clamp(-5.0) == -1.0
    assert clamp(0.25) == 0.25


def test_s_vwap_sign_convention_above_vwap_is_bullish():
    assert SignalEngine.s_vwap(price=101.0, vwap=100.0, band_bps=100) > 0
    assert SignalEngine.s_vwap(price=99.0, vwap=100.0, band_bps=100) < 0
    assert SignalEngine.s_vwap(price=100.0, vwap=100.0, band_bps=100) == 0.0


def test_s_vwap_saturates_at_one_band():
    # band = 100 * 15bps = 0.15 -> a 0.15 move is exactly full scale
    assert SignalEngine.s_vwap(100.15, 100.0, 15) == pytest.approx(1.0)
    assert SignalEngine.s_vwap(100.30, 100.0, 15) == pytest.approx(1.0)   # clamped
    assert SignalEngine.s_vwap(100.075, 100.0, 15) == pytest.approx(0.5)


def test_s_vwap_zero_band_is_neutral_not_infinite():
    assert SignalEngine.s_vwap(101.0, 100.0, 0) == 0.0


def test_s_rsi_is_mean_reverting():
    assert SignalEngine.s_rsi(20.0, 30, 70) == 1.0     # oversold -> buy
    assert SignalEngine.s_rsi(80.0, 30, 70) == -1.0    # overbought -> sell
    assert SignalEngine.s_rsi(50.0, 30, 70) == pytest.approx(0.0)
    assert SignalEngine.s_rsi(30.0, 30, 70) == pytest.approx(1.0)
    assert SignalEngine.s_rsi(70.0, 30, 70) == pytest.approx(-1.0)
    assert SignalEngine.s_rsi(40.0, 30, 70) == pytest.approx(0.5)


def test_s_rsi_degenerate_band_is_neutral():
    assert SignalEngine.s_rsi(50.0, 50, 50) == 0.0


def test_s_macd_normalized_by_volatility():
    # denom = 0.25*ATR; hist == 0.25*ATR -> full scale
    assert SignalEngine.s_macd(macd_hist=0.25, atr=1.0) == pytest.approx(1.0)
    assert SignalEngine.s_macd(macd_hist=0.125, atr=1.0) == pytest.approx(0.5)
    assert SignalEngine.s_macd(macd_hist=-0.25, atr=1.0) == pytest.approx(-1.0)
    # same histogram means LESS in a more volatile market
    assert SignalEngine.s_macd(0.25, atr=4.0) == pytest.approx(0.25)


def test_s_macd_zero_atr_is_neutral_not_division_by_zero():
    assert SignalEngine.s_macd(1.0, atr=0.0) == 0.0


# -- composite ---------------------------------------------------------------

def test_score_stays_within_unit_interval(cfg):
    """Weights renormalize to 1, so a saturated score can never exceed 1. If it
    could, `entry_threshold` would stop meaning the same thing across configs."""
    e = SignalEngine(lambda: cfg)
    s = e.compute(_snap(vwap=1.0, rsi=1.0, macd_hist=1e6, atr=1e-3, close=1e6), _mkt(1e6))
    assert -1.0 <= s.score <= 1.0
    assert s.score == 1.0   # exactly, not 1.0000000000000002


def test_unready_indicators_produce_hold_not_a_weak_signal(cfg):
    e = SignalEngine(lambda: cfg)
    s = e.compute(_snap(rsi=None), _mkt())
    assert s.action == "HOLD" and s.score == 0.0
    s = e.compute(_snap(atr=0.0), _mkt())          # atr 0 -> not ready
    assert s.action == "HOLD"


def test_enter_when_all_components_align_bullish(cfg):
    e = SignalEngine(lambda: cfg)
    # above VWAP, oversold RSI, positive MACD histogram
    s = e.compute(_snap(vwap=100.0, rsi=20.0, macd_hist=1.0, atr=1.0, close=100.5), _mkt(100.5))
    assert s.score == pytest.approx(1.0)
    assert s.action == "ENTER"


def test_exit_when_all_components_align_bearish(cfg):
    e = SignalEngine(lambda: cfg)
    s = e.compute(_snap(vwap=100.0, rsi=90.0, macd_hist=-1.0, atr=1.0, close=99.5), _mkt(99.5))
    assert s.score == pytest.approx(-1.0)
    assert s.action == "EXIT"


def test_neutral_market_holds(cfg):
    e = SignalEngine(lambda: cfg)
    s = e.compute(_snap(vwap=100.0, rsi=50.0, macd_hist=0.0, atr=1.0, close=100.0), _mkt(100.0))
    assert s.score == pytest.approx(0.0)
    assert s.action == "HOLD"


def test_weights_actually_weight(cfg):
    """A config that zeroes two weights must yield the third component exactly."""
    c = copy.deepcopy(cfg)
    c["strategy"]["weights"] = {"vwap": 1.0, "rsi": 0.0, "macd": 0.0}
    e = SignalEngine(lambda: c)
    s = e.compute(_snap(vwap=100.0, rsi=90.0, macd_hist=-99.0, atr=1.0, close=100.15), _mkt(100.15))
    assert s.score == pytest.approx(1.0)   # pure vwap component, bearish others ignored


def test_config_change_takes_effect_without_rebuilding_engine(cfg):
    """The engine reads config through a provider so a hot-reload lands
    immediately -- the optimizer deploys mid-session."""
    live = copy.deepcopy(cfg)
    e = SignalEngine(lambda: live)
    snap = _snap(vwap=100.0, rsi=20.0, macd_hist=1.0, atr=1.0, close=100.5)
    assert e.compute(snap, _mkt(100.5)).action == "ENTER"
    live["strategy"]["entry_threshold"] = 0.9
    assert e.compute(snap, _mkt(100.5)).action == "ENTER"   # score is 1.0, still over
    live["strategy"]["weights"] = {"vwap": 0.0, "rsi": 0.0, "macd": 1.0}
    live["strategy"]["entry_threshold"] = 0.9
    s = e.compute(_snap(vwap=100.0, rsi=20.0, macd_hist=0.05, atr=1.0, close=100.5), _mkt(100.5))
    assert s.action == "HOLD"
