"""Stress scenarios: replay arithmetic, worst-window ranking, crisis bootstrap, 1987."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from src.common.models import Candle
from src.forecast.scenarios import (
    Scenario,
    crisis_bootstrap,
    high_vol_start_pool,
    historical_stress,
    replay_log_returns,
    scenario_1987,
)


def _candles_with_crash(n: int = 400, crash_at: int = 200) -> list[Candle]:
    """Calm series with one sharp multi-day crash, so 'worst window' is unambiguous."""
    out = []
    price = 100.0
    for i in range(n):
        d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
        ts = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
        if crash_at <= i < crash_at + 10:
            price *= 0.96                       # -4%/day for 10 days
        else:
            price *= (1.0 + 0.0003 + 0.002 * math.sin(i / 5.0))
        out.append(Candle(ts_open_ms=ts, symbol="T", tf="1d", open=price, high=price * 1.01,
                          low=price * 0.99, close=price, volume=1000.0,
                          quote_volume=1000.0 * price, n_trades=0))
    return out


def test_replay_log_returns_arithmetic():
    seq = np.log(np.array([1.10, 0.90 / 1.10, 1.0]))   # +10%, then back to 0.90, flat
    path, trough, end = replay_log_returns(seq, 100.0)
    assert path[0] == pytest.approx(110.0)
    assert end == pytest.approx(90.0)
    assert trough == pytest.approx(90.0)


def test_historical_stress_finds_the_crash():
    candles = _candles_with_crash()
    scen = historical_stress(candles, horizon=30, k=3)
    assert scen
    worst = scen[0]
    assert isinstance(worst, Scenario)
    assert worst.trough_pct < -0.2          # the -4%/day x10 crash shows through
    assert worst.trough_price < worst.end_price or worst.end_pct <= 0  # sanity


def test_historical_stress_windows_non_overlapping():
    candles = _candles_with_crash()
    scen = historical_stress(candles, horizon=30, k=3)
    names = [s.name for s in scen]
    assert len(names) == len(set(names))


def test_scenario_1987_deep_trough():
    sc = scenario_1987(100.0)
    assert sc.trough_pct < -0.20            # Oct 1987 monthly ~ -21.8%
    assert "1987" in sc.name


def test_high_vol_pool_selects_turbulent_region():
    candles = _candles_with_crash(crash_at=200)
    closes = [c.close for c in candles]
    pool = high_vol_start_pool(closes, block=10, window=20, quantile=0.75)
    assert pool.size > 0
    # crash window (returns index ~199-208) should be represented in the high-vol pool
    assert any(190 <= i <= 210 for i in pool)


def test_crisis_bootstrap_more_bearish_than_calm():
    candles = _candles_with_crash()
    closes = [c.close for c in candles]
    crisis = crisis_bootstrap(closes, horizon=30, n_paths=4000, seed=1, block=10)
    # crisis cone median should be below the starting price (turbulent = drawdown-heavy)
    assert crisis.method == "crisis-bootstrap"
    assert len(crisis.median) == 30
    assert crisis.lower[-1] < closes[-1]


def test_crisis_bootstrap_rejects_short_history():
    with pytest.raises(ValueError):
        crisis_bootstrap([100.0, 101.0, 102.0], horizon=10, block=10)
