"""Block-bootstrap forecaster: determinism, shape, band ordering, drift recentering,
and the analytic-GBM helpers used alongside it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.forecast.bootstrap import block_bootstrap, bootstrap_prices
from src.forecast.gbm import (
    gbm_analytic_path,
    gbm_expected_value,
    gbm_percentile,
    gbm_prob_above,
)


def _series(n: int = 400, s0: float = 100.0, daily: float = 0.0005) -> list[float]:
    out = [s0]
    for i in range(1, n):
        out.append(out[-1] * (1.0 + daily + 0.01 * math.sin(i / 6.0)))
    return out


def test_same_seed_identical():
    closes = _series()
    a = block_bootstrap(closes, horizon=20, n_paths=3000, seed=5)
    b = block_bootstrap(closes, horizon=20, n_paths=3000, seed=5)
    assert a.median == b.median and a.lower == b.lower and a.upper == b.upper


def test_shape_and_band_ordering():
    closes = _series()
    h = 30
    fc = block_bootstrap(closes, horizon=h, n_paths=5000, seed=1)
    assert fc.days == list(range(1, h + 1))
    for i in range(h):
        assert fc.lower[i] <= fc.p25[i] <= fc.median[i] <= fc.p75[i] <= fc.upper[i]


def test_matrix_columns_match_horizon():
    closes = _series()
    m = bootstrap_prices(closes, horizon=15, n_paths=1000, seed=2)
    assert m.shape == (1000, 15)
    assert np.all(m > 0)


def test_recenter_shifts_drift_up():
    closes = _series(daily=0.0)          # ~flat history
    base = block_bootstrap(closes, horizon=25, n_paths=6000, seed=3)
    up = block_bootstrap(closes, horizon=25, n_paths=6000, seed=3, recenter_mu=0.003)
    assert up.median[-1] > base.median[-1]


def test_rejects_short_history():
    with pytest.raises(ValueError):
        block_bootstrap([100.0, 101.0], horizon=10, block=10)


# ---- analytic GBM helpers ----

def test_gbm_percentile_monotonic_in_q():
    vals = [gbm_percentile(100.0, 0.0005, 0.01, 20, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert vals == sorted(vals)


def test_gbm_prob_above_bounds_and_direction():
    p_low = gbm_prob_above(100.0, 0.0, 0.01, 30, 90.0)    # below spot -> likely
    p_high = gbm_prob_above(100.0, 0.0, 0.01, 30, 110.0)  # above spot -> less likely
    assert 0.0 <= p_high < 0.5 < p_low <= 1.0


def test_gbm_expected_value_is_lognormal_mean():
    ev = gbm_expected_value(100.0, 0.001, 0.02, 40)
    assert ev == pytest.approx(100.0 * math.exp(0.001 * 40 + 0.5 * 0.02**2 * 40))


def test_gbm_analytic_path_matches_percentile_fn():
    closes = _series()
    p = gbm_analytic_path(closes, horizon=10, mu=0.0005, sigma=0.012)
    s0 = closes[-1]
    assert p.median[-1] == pytest.approx(gbm_percentile(s0, 0.0005, 0.012, 10, 0.5))
    assert p.lower[0] == pytest.approx(gbm_percentile(s0, 0.0005, 0.012, 1, 0.05))
