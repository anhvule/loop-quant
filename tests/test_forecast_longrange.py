"""Long-range estimators: drift shrinkage, EWMA vol, VIX conversion, blending."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.forecast.longrange import (
    TRADING_DAYS_YEAR,
    blended_drift,
    estimate,
    ewma_sigma,
    log_returns,
    vix_forward_sigma,
)


def _series(n: int, daily: float, s0: float = 100.0) -> list[float]:
    out = [s0]
    for i in range(1, n):
        out.append(out[-1] * math.exp(daily))   # exact constant log return
    return out


def test_log_returns_constant_series():
    closes = _series(50, 0.01)
    r = log_returns(closes)
    assert r.size == 49
    assert np.allclose(r, 0.01)


def test_blended_drift_between_components():
    # Long history flat, recent window strongly up -> blend sits between.
    closes = _series(600, 0.0)[:-100] + _series(100, 0.02, s0=_series(600, 0.0)[-101])
    mu_blend, mu_recent, mu_long = blended_drift(closes, recent_window=100, w_recent=0.3)
    assert mu_recent > mu_long
    assert mu_long <= mu_blend <= mu_recent
    # w_recent=0.3 exactly reproduces the weighting
    assert mu_blend == pytest.approx(0.3 * mu_recent + 0.7 * mu_long)


def test_ewma_sigma_matches_recursion():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(300) * 0.01
    lam = 0.94
    var = float(np.var(r))
    for x in r:
        var = lam * var + (1 - lam) * x * x
    assert ewma_sigma(r, lam) == pytest.approx(math.sqrt(var))


def test_ewma_reacts_to_recent_regime():
    calm = list(np.random.default_rng(1).standard_normal(200) * 0.005)
    wild = list(np.random.default_rng(2).standard_normal(50) * 0.05)
    assert ewma_sigma(calm + wild) > ewma_sigma(wild + calm)


def test_vix_forward_sigma():
    # VIX 20 -> ~20%/sqrt(252) daily
    assert vix_forward_sigma(20.0) == pytest.approx(0.20 / math.sqrt(TRADING_DAYS_YEAR))


def test_estimate_blends_vix_5050():
    closes = _series(400, 0.0003)
    e_novix = estimate(closes)
    e_vix = estimate(closes, vix_close=30.0)
    assert e_novix.sigma_vix is None
    assert e_vix.sigma_vix == pytest.approx(vix_forward_sigma(30.0))
    assert e_vix.sigma_blend == pytest.approx(0.5 * e_vix.sigma_ewma + 0.5 * e_vix.sigma_vix)


def test_estimate_handles_short_history():
    e = estimate([100.0, 101.0, 100.5])
    assert e.n_returns == 2
    assert math.isfinite(e.mu_blend) and math.isfinite(e.sigma_blend)
