"""GBM Monte Carlo forecaster: determinism, band ordering, shape, and drift tilt."""

from __future__ import annotations

import math

import pytest

from src.forecast.gbm import simulate_gbm


def _series(n: int = 200, s0: float = 100.0, daily: float = 0.001) -> list[float]:
    """A gently trending, mildly noisy price series (deterministic, no RNG)."""
    out = [s0]
    for i in range(1, n):
        wobble = 0.01 * math.sin(i / 7.0)
        out.append(out[-1] * (1.0 + daily + wobble))
    return out


def test_same_seed_is_byte_identical():
    closes = _series()
    a = simulate_gbm(closes, horizon=15, n_paths=5_000, seed=7)
    b = simulate_gbm(closes, horizon=15, n_paths=5_000, seed=7)
    assert a.median == b.median
    assert a.lower == b.lower
    assert a.upper == b.upper


def test_different_seed_differs():
    closes = _series()
    a = simulate_gbm(closes, horizon=15, n_paths=5_000, seed=7)
    b = simulate_gbm(closes, horizon=15, n_paths=5_000, seed=8)
    assert a.median != b.median


def test_shape_and_band_ordering():
    closes = _series()
    horizon = 12
    fc = simulate_gbm(closes, horizon=horizon, n_paths=8_000, seed=3)
    assert fc.days == list(range(1, horizon + 1))
    for name in ("median", "lower", "upper", "p25", "p75"):
        assert len(getattr(fc, name)) == horizon
    for i in range(horizon):
        assert fc.lower[i] <= fc.p25[i] <= fc.median[i] <= fc.p75[i] <= fc.upper[i]


def test_uncertainty_widens_with_horizon():
    closes = _series()
    fc = simulate_gbm(closes, horizon=20, n_paths=8_000, seed=5)
    spread_first = fc.upper[0] - fc.lower[0]
    spread_last = fc.upper[-1] - fc.lower[-1]
    assert spread_last > spread_first


def test_positive_tilt_raises_median():
    closes = _series()
    base = simulate_gbm(closes, horizon=10, n_paths=8_000, seed=11, drift_tilt=0.0)
    up = simulate_gbm(closes, horizon=10, n_paths=8_000, seed=11, drift_tilt=0.01)
    assert up.median[-1] > base.median[-1]


def test_flat_history_is_deterministic_line():
    closes = [50.0] * 100   # sigma == 0
    fc = simulate_gbm(closes, horizon=8, n_paths=1_000, seed=1)
    assert fc.lower == fc.median == fc.upper
    assert all(v == pytest.approx(50.0) for v in fc.median)


def test_rejects_bad_args():
    closes = _series(80)
    with pytest.raises(ValueError):
        simulate_gbm(closes, horizon=0)
    with pytest.raises(ValueError):
        simulate_gbm([], horizon=5)


# ---- Student-t shocks (Phase B) ----

def test_t_shocks_deterministic():
    closes = _series()
    a = simulate_gbm(closes, horizon=15, n_paths=4000, seed=7, shock_dist="t")
    b = simulate_gbm(closes, horizon=15, n_paths=4000, seed=7, shock_dist="t")
    assert a.median == b.median and a.lower == b.lower


def test_t_shocks_preserve_variance_but_fatten_tails():
    import numpy as np
    # A history with real fat tails so a low df is fit.
    rng = np.random.default_rng(1)
    fat = [100.0]
    for _ in range(800):
        fat.append(fat[-1] * math.exp(0.015 * rng.standard_t(3)))
    n = simulate_gbm(fat, 20, n_paths=30000, seed=2, shock_dist="normal").terminal
    t = simulate_gbm(fat, 20, n_paths=30000, seed=2, shock_dist="t").terminal
    n, t = np.array(n), np.array(t)
    # Similar central spread (variance preserved) ...
    assert abs(np.log(np.std(t)) - np.log(np.std(n))) < 0.25
    # ... but heavier extremes.
    assert np.percentile(t, 0.5) < np.percentile(n, 0.5)


def test_t_shocks_reject_bad_dist():
    with pytest.raises(ValueError):
        simulate_gbm(_series(80), horizon=5, shock_dist="cauchy")


# ---- seasonal effective sigma (Phase E) ----

def test_seasonal_effective_sigma_matches_flat_when_uniform():
    import numpy as np
    from src.forecast.gbm import gbm_analytic_path, gbm_analytic_path_seasonal
    from src.forecast.seasonal import effective_sigma
    closes = _series()
    sigma = 0.012
    eff = effective_sigma(np.full(20, sigma))     # uniform schedule -> constant sigma
    flat = gbm_analytic_path(closes, 20, mu=0.0004, sigma=sigma)
    seas = gbm_analytic_path_seasonal(closes, 20, mu=0.0004, eff_sigma=eff)
    assert seas.upper[-1] == pytest.approx(flat.upper[-1], rel=1e-9)
    assert seas.median[0] == pytest.approx(flat.median[0], rel=1e-9)
