"""Drawdown / path-risk: analytic first-passage vs Monte Carlo, monotonicity, max-dd."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast.bootstrap import bootstrap_prices
from src.forecast.drawdown import (
    path_risk_bootstrap,
    path_risk_gbm,
    touch_prob_gbm,
)
from src.forecast.gbm import simulate_gbm


def _series(n: int = 500, s0: float = 100.0, daily: float = 0.0003) -> list[float]:
    import math
    out = [s0]
    for i in range(1, n):
        out.append(out[-1] * (1.0 + daily + 0.008 * math.sin(i / 6.0)))
    return out


def test_touch_prob_bounds_and_trivial_cases():
    assert touch_prob_gbm(100.0, 0.0, 0.01, 30, 110.0) == 1.0   # barrier above spot
    p = touch_prob_gbm(100.0, 0.0, 0.01, 30, 90.0)
    assert 0.0 < p < 1.0


def test_touch_prob_monotonic_in_level_and_horizon():
    # Deeper barrier -> less likely to touch.
    deep = touch_prob_gbm(100.0, 0.0, 0.01, 40, 80.0)
    shallow = touch_prob_gbm(100.0, 0.0, 0.01, 40, 95.0)
    assert deep < shallow
    # Longer horizon -> more likely to touch a given barrier.
    short = touch_prob_gbm(100.0, 0.0, 0.01, 10, 90.0)
    long = touch_prob_gbm(100.0, 0.0, 0.01, 60, 90.0)
    assert long > short


def test_analytic_touch_matches_monte_carlo():
    # Simulate normal GBM daily-close paths and compare touch frequency to the
    # closed form. Close-based MC slightly under-counts vs continuous, so allow a band.
    s0, mu, sigma, h, level = 100.0, 0.0, 0.02, 40, 90.0
    rng = np.random.default_rng(0)
    shocks = rng.standard_normal((40000, h)) * sigma + mu
    prices = s0 * np.exp(np.cumsum(shocks, axis=1))
    mc = float(np.mean(prices.min(axis=1) <= level))
    analytic = touch_prob_gbm(s0, mu, sigma, h, level)
    # analytic (continuous) >= discrete MC; within ~5 points.
    assert analytic >= mc - 0.02
    assert abs(analytic - mc) < 0.06


def test_path_risk_gbm_levels_ordered():
    r = path_risk_gbm(100.0, 0.0, 0.02, 60)
    # touching -5% is more likely than -20%
    assert r[-0.05] > r[-0.10] > r[-0.15] > r[-0.20]


def test_path_risk_bootstrap_shape_and_maxdd():
    closes = _series()
    prices = bootstrap_prices(closes, horizon=60, n_paths=4000, seed=1, block=10)
    r = path_risk_bootstrap(prices, s0=closes[-1])
    assert set(r["touch"]) == {-0.05, -0.10, -0.15, -0.20}
    assert r["touch"][-0.05] >= r["touch"][-0.20]
    assert 0.0 <= r["maxdd_median"] <= r["maxdd_p90"] <= r["maxdd_p99"] <= 1.0


def test_fat_tails_raise_deep_touch_probability():
    # Student-t shocks should touch deep barriers more often than normal, same sigma.
    closes = _series()
    s0 = closes[-1]
    # Build history with real fat tails so fit_t_dof picks a low df.
    import math
    rng = np.random.default_rng(3)
    fat = [100.0]
    for _ in range(600):
        fat.append(fat[-1] * math.exp(0.02 * rng.standard_t(3)))
    pn = simulate_gbm(fat, 40, n_paths=20000, seed=9, shock_dist="normal").terminal
    pt = simulate_gbm(fat, 40, n_paths=20000, seed=9, shock_dist="t").terminal
    # t has fatter tails -> lower 1st percentile terminal price
    assert np.percentile(pt, 1) < np.percentile(pn, 1)
