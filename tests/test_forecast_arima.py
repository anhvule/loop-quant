"""ARIMA forecaster smoke tests.

statsmodels may or may not be installed; either way `forecast_arima` must return a
valid, finite, horizon-length path (it falls back to drift-only without statsmodels
or on a non-convergent fit). These tests assert the contract, not a specific model.
"""

from __future__ import annotations

import math

import pytest

from src.forecast.arima import forecast_arima
from src.forecast.result import ForecastPath


def _trend(n: int = 180, s0: float = 100.0, daily: float = 0.002) -> list[float]:
    out = [s0]
    for i in range(1, n):
        out.append(out[-1] * (1.0 + daily + 0.005 * math.sin(i / 5.0)))
    return out


def _finite(xs) -> bool:
    return all(isinstance(x, float) and math.isfinite(x) and x > 0 for x in xs)


def test_returns_valid_horizon_length_path():
    fc = forecast_arima(_trend(), horizon=15)
    assert isinstance(fc, ForecastPath)
    assert fc.method == "arima"
    assert len(fc.median) == len(fc.lower) == len(fc.upper) == 15
    assert _finite(fc.median) and _finite(fc.lower) and _finite(fc.upper)


def test_interval_brackets_median():
    fc = forecast_arima(_trend(), horizon=10)
    for lo, mid, hi in zip(fc.lower, fc.median, fc.upper):
        assert lo <= mid <= hi


def test_drift_only_fallback_on_nonpositive_price():
    closes = _trend(120)
    closes[50] = -1.0   # forces the drift-only branch
    fc = forecast_arima(closes, horizon=8)
    assert "drift-only" in fc.note
    assert len(fc.median) == 8


def test_explicit_order_is_accepted():
    fc = forecast_arima(_trend(), horizon=6, order=(1, 1, 0))
    assert len(fc.median) == 6
    assert _finite(fc.median)


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        forecast_arima(_trend(), horizon=0)
    with pytest.raises(ValueError):
        forecast_arima([], horizon=5)
