"""Market-factor engine: alignment, beta recovery, joint correlation preservation,
drift anchoring, and determinism."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.forecast.beta import (
    align_closes,
    anchored_drifts,
    beta_stats,
    ewma_vol_path,
    joint_bootstrap_prices,
    log_return_matrix,
    recency_weights,
    standardize_matrix,
    vol_half_life_from_path,
    vol_schedule,
)


def _dates(n, start=dt.date(2024, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _series_from_rets(rets, s0=100.0):
    px = [s0]
    for r in rets:
        px.append(px[-1] * np.exp(r))
    return px


# ---- alignment ----

def test_align_closes_intersects_dates():
    d = _dates(10)
    a = {d[i]: 100.0 + i for i in range(10)}
    b = {d[i]: 50.0 + i for i in range(3, 10)}      # starts later
    dates, closes = align_closes({"A": a, "B": b})
    assert len(dates) == 7 and dates[0] == d[3]
    assert closes["A"].size == closes["B"].size == 7


def test_align_closes_raises_on_no_overlap():
    with pytest.raises(ValueError):
        align_closes({"A": {dt.date(2024, 1, 1): 1.0},
                      "B": {dt.date(2025, 1, 1): 1.0}})


def test_log_return_matrix_shape_and_order():
    d = _dates(6)
    s = {"M": {x: 100.0 * (1.01 ** i) for i, x in enumerate(d)},
         "N": {x: 50.0 * (1.02 ** i) for i, x in enumerate(d)}}
    dates, closes = align_closes(s)
    rm = log_return_matrix(dates, closes, ["M", "N"])
    assert rm.shape == (5, 2)
    assert rm[:, 1].mean() > rm[:, 0].mean()        # N compounds faster


# ---- beta ----

def test_beta_recovers_two_x_market():
    rng = np.random.default_rng(0)
    mkt = rng.standard_normal(600) * 0.01
    name = 2.0 * mkt                                 # pure 2x, no idio
    b = beta_stats("X", name, mkt)
    assert b.beta == pytest.approx(2.0, abs=0.02)
    assert b.r2 > 0.99
    assert b.idio_vol < 1e-9


def test_beta_with_idiosyncratic_noise():
    """With idio vol == market vol the EWMA fit recovers beta to ~2 standard errors.

    The EWMA half-life (~138d) means an effective sample of ~200 obs, so the standard
    error of beta is roughly (idio_vol/mkt_vol)/sqrt(200) ~ 0.07 here. Beta for a noisy
    high-idio name is genuinely imprecise -- the tolerance below reflects the estimator's
    real precision rather than pretending to more."""
    rng = np.random.default_rng(1)
    mkt = rng.standard_normal(1500) * 0.01
    idio = rng.standard_normal(1500) * 0.01
    b = beta_stats("Y", 1.5 * mkt + idio, mkt)
    assert b.beta == pytest.approx(1.5, abs=0.15)
    assert 0.3 < b.r2 < 0.95
    assert b.idio_vol == pytest.approx(0.01, rel=0.25)


def test_beta_precision_degrades_with_idio_noise():
    """Documents the limitation: heavier idio noise -> noisier beta. This is why the
    report shows r2 alongside beta."""
    rng = np.random.default_rng(7)
    mkt = rng.standard_normal(1500) * 0.01
    clean = beta_stats("C", 1.5 * mkt + rng.standard_normal(1500) * 0.002, mkt)
    noisy = beta_stats("N", 1.5 * mkt + rng.standard_normal(1500) * 0.04, mkt)
    assert clean.r2 > noisy.r2
    assert abs(clean.beta - 1.5) < abs(noisy.beta - 1.5) + 0.5


def test_beta_rejects_tiny_sample():
    with pytest.raises(ValueError):
        beta_stats("Z", np.zeros(10), np.zeros(10))


# ---- joint bootstrap ----

def _corr_matrix(n=800, rho_beta=1.8, seed=3):
    rng = np.random.default_rng(seed)
    mkt = rng.standard_normal(n) * 0.01
    a = rho_beta * mkt + rng.standard_normal(n) * 0.01
    b = 1.2 * mkt + rng.standard_normal(n) * 0.015
    return np.column_stack([mkt, a, b])


def test_joint_bootstrap_shape_and_positive():
    rm = _corr_matrix()
    s0 = np.array([100.0, 50.0, 20.0])
    pr = joint_bootstrap_prices(rm, s0, horizon=30, n_paths=2000, seed=5)
    assert pr.shape == (2000, 30, 3)
    assert np.all(pr > 0)


def test_joint_bootstrap_preserves_cross_correlation():
    """The whole point: sampled assets must keep co-moving like the history."""
    rm = _corr_matrix()
    src = np.corrcoef(rm.T)
    s0 = np.array([100.0, 50.0, 20.0])
    pr = joint_bootstrap_prices(rm, s0, horizon=60, n_paths=4000, seed=6).astype(float)
    lr = np.diff(np.log(pr), axis=1)                 # (paths, 59, 3)
    flat = lr.reshape(-1, 3)
    sim = np.corrcoef(flat.T)
    for i in range(3):
        for j in range(3):
            assert abs(sim[i, j] - src[i, j]) < 0.08


def test_joint_bootstrap_deterministic():
    rm = _corr_matrix()
    s0 = np.array([100.0, 50.0, 20.0])
    a = joint_bootstrap_prices(rm, s0, 20, n_paths=1000, seed=9)
    b = joint_bootstrap_prices(rm, s0, 20, n_paths=1000, seed=9)
    assert np.array_equal(a, b)


def test_recenter_hits_target_drift():
    rm = _corr_matrix()
    s0 = np.array([100.0, 100.0, 100.0])
    target = np.array([0.0, 0.002, -0.001])
    pr = joint_bootstrap_prices(rm, s0, 40, n_paths=6000, seed=11,
                                recenter_mu=target).astype(float)
    for k in range(3):
        realized = np.mean(np.log(pr[:, -1, k] / s0[k])) / 40
        assert realized == pytest.approx(target[k], abs=3e-4)


def test_joint_bootstrap_rejects_short_history():
    with pytest.raises(ValueError):
        joint_bootstrap_prices(np.zeros((5, 2)), np.array([1.0, 1.0]), 10, block=10)


# ---- drift anchoring ----

def test_anchored_drifts_refuse_momentum_by_default():
    stats = {"A": beta_stats("A", np.random.default_rng(2).standard_normal(400) * 0.02,
                             np.random.default_rng(3).standard_normal(400) * 0.01)}
    mu_m = 0.0005
    d = anchored_drifts(stats, mu_m)
    assert d["A"] == pytest.approx(stats["A"].beta * mu_m)   # own raw_mu ignored


def test_idio_drift_opt_in_restores_raw_mu():
    stats = {"A": beta_stats("A", np.random.default_rng(4).standard_normal(400) * 0.02,
                             np.random.default_rng(5).standard_normal(400) * 0.01)}
    d = anchored_drifts(stats, 0.0005, idio_drift=True)
    assert d["A"] == pytest.approx(stats["A"].raw_mu)


# ---- beta standard error ----

def test_beta_se_larger_when_idio_noise_larger():
    rng = np.random.default_rng(21)
    mkt = rng.standard_normal(1200) * 0.01
    clean = beta_stats("C", 1.5 * mkt + rng.standard_normal(1200) * 0.002, mkt)
    noisy = beta_stats("N", 1.5 * mkt + rng.standard_normal(1200) * 0.04, mkt)
    assert 0 < clean.se < noisy.se


def test_beta_se_brackets_truth_for_noisy_fit():
    rng = np.random.default_rng(22)
    mkt = rng.standard_normal(1500) * 0.01
    b = beta_stats("Y", 1.5 * mkt + rng.standard_normal(1500) * 0.02, mkt)
    assert abs(b.beta - 1.5) < 3 * b.se          # truth within 3 standard errors


# ---- volatility standardization ----

def _vol_regime_series(n=900, seed=5):
    """Calm first half, wild second half -- the regime shift raw resampling ignores."""
    rng = np.random.default_rng(seed)
    mkt = np.concatenate([rng.standard_normal(n // 2) * 0.004,
                          rng.standard_normal(n // 2) * 0.004])
    name = np.concatenate([2.0 * mkt[:n // 2] + rng.standard_normal(n // 2) * 0.005,
                           2.0 * mkt[n // 2:] + rng.standard_normal(n // 2) * 0.05])
    return np.column_stack([mkt, name])


def test_ewma_vol_path_tracks_regime_and_has_no_lookahead():
    rm = _vol_regime_series()
    vol, cur = ewma_vol_path(rm[:, 1])
    assert vol.size == rm.shape[0]
    assert vol[100] < vol[-1]                    # calm early, wild late
    assert cur > vol[100]
    # no lookahead: the first value cannot know about the later explosion
    calm_only, _ = ewma_vol_path(rm[:450, 1])
    assert vol[100] == pytest.approx(calm_only[100])


def test_standardize_matrix_returns_unit_scale_residuals():
    rm = _vol_regime_series()
    Z, cur = standardize_matrix(rm)
    assert Z.shape == rm.shape and cur.size == 2
    # standardized residuals have roughly unit scale in BOTH regimes
    early, late = np.std(Z[100:400, 1]), np.std(Z[600:880, 1])
    assert 0.5 < early < 2.0 and 0.5 < late < 2.0
    assert abs(np.log(late / early)) < 0.7       # regime difference largely removed


def test_volstd_bootstrap_reflects_current_not_average_vol():
    """The core fix: after a vol explosion, standardized sampling must produce a WIDER
    cone than raw resampling (which averages the calm era in)."""
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    raw = joint_bootstrap_prices(rm, s0, 60, n_paths=6000, seed=3).astype(float)
    std = joint_bootstrap_prices(rm, s0, 60, n_paths=6000, seed=3,
                                 vol_standardize=True).astype(float)
    raw_w = np.percentile(raw[:, -1, 1], 95) - np.percentile(raw[:, -1, 1], 5)
    std_w = np.percentile(std[:, -1, 1], 95) - np.percentile(std[:, -1, 1], 5)
    assert std_w > raw_w * 1.2


def test_volstd_still_hits_target_drift():
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    target = np.array([0.0, 0.001])
    pr = joint_bootstrap_prices(rm, s0, 40, n_paths=8000, seed=4,
                                recenter_mu=target, vol_standardize=True).astype(float)
    realized = np.mean(np.log(pr[:, -1, 1] / s0[1])) / 40
    assert realized == pytest.approx(target[1], abs=1e-3)


def test_volstd_preserves_cross_correlation():
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    pr = joint_bootstrap_prices(rm, s0, 60, n_paths=5000, seed=6,
                                vol_standardize=True).astype(float)
    lr = np.diff(np.log(pr), axis=1).reshape(-1, 2)
    assert np.corrcoef(lr.T)[0, 1] > 0.2          # market link survives standardization


# ---- beta uncertainty propagation ----

def test_beta_draw_widens_bands():
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    target = np.array([0.0006, 0.0012])
    rng = np.random.default_rng(0)
    base = joint_bootstrap_prices(rm, s0, 80, n_paths=6000, seed=8,
                                  recenter_mu=target).astype(float)
    drawn = joint_bootstrap_prices(rm, s0, 80, n_paths=6000, seed=8, recenter_mu=target,
                                   beta_draw=rng.normal(2.0, 0.6, 6000)).astype(float)
    w = lambda p: np.percentile(p[:, -1, 1], 95) - np.percentile(p[:, -1, 1], 5)
    assert w(drawn) > w(base)


def test_beta_draw_leaves_market_column_untouched():
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    rng = np.random.default_rng(1)
    a = joint_bootstrap_prices(rm, s0, 30, n_paths=2000, seed=9)
    b = joint_bootstrap_prices(rm, s0, 30, n_paths=2000, seed=9,
                               beta_draw=rng.normal(2.0, 0.5, 2000))
    assert np.allclose(a[:, :, 0], b[:, :, 0])


# ---- vol term structure ----

def test_vol_half_life_recovers_ar1_persistence():
    """Synthetic AR(1) log-vol with a known half-life should be recovered roughly."""
    rng = np.random.default_rng(11)
    hl_true = 60.0
    phi = 0.5 ** (1.0 / hl_true)                 # daily persistence
    lv = np.zeros(3000)
    for i in range(1, lv.size):
        lv[i] = phi * lv[i - 1] + rng.normal(0, 0.05)
    vol = np.exp(lv) * 0.02
    hl = vol_half_life_from_path(vol)
    assert hl_true * 0.4 < hl < hl_true * 2.5    # right order of magnitude


def test_vol_schedule_decays_from_current_to_long_run():
    cur = np.array([0.08])
    lr = np.array([0.05])
    hl = np.array([50.0])
    s = vol_schedule(cur, lr, hl, horizon=400)
    assert s.shape == (400, 1)
    assert s[0, 0] == pytest.approx(0.08, abs=0.002)     # day 1 ~ today's vol
    assert s[-1, 0] == pytest.approx(0.05, abs=0.002)    # far out ~ long-run
    assert np.all(np.diff(s[:, 0]) < 0)                  # monotone decay


def test_vol_schedule_flat_when_current_equals_long_run():
    s = vol_schedule(np.array([0.03]), np.array([0.03]), np.array([100.0]), 50)
    assert np.allclose(s, 0.03)


def test_term_structure_narrows_cone_vs_flat_high_vol():
    """The fix for the $12-$573 problem: decaying an elevated vol must shrink the
    long-horizon cone relative to holding today's panic vol forever."""
    rm = _vol_regime_series()                     # ends in the WILD regime
    s0 = np.array([100.0, 100.0])
    flat = joint_bootstrap_prices(rm, s0, 200, n_paths=6000, seed=12,
                                  vol_standardize=True).astype(float)
    ts = joint_bootstrap_prices(rm, s0, 200, n_paths=6000, seed=12,
                                vol_standardize=True, vol_term_structure=True).astype(float)
    w = lambda p: np.percentile(p[:, -1, 1], 95) - np.percentile(p[:, -1, 1], 5)
    assert w(ts) < w(flat)


# ---- recency weighting ----

def test_recency_weights_favour_recent_and_sum_to_one():
    w = recency_weights(1000, half_life_days=200)
    assert w.sum() == pytest.approx(1.0)
    assert w[-1] > w[0]                            # newest > oldest
    assert w[0] > 0                                # floor keeps old data alive


def test_recency_weights_floor_prevents_starvation():
    w = recency_weights(2000, half_life_days=20)   # aggressive decay
    assert w[0] > 0 and np.isfinite(w).all()


def test_recency_sampling_shifts_draws_toward_recent_history():
    """Blocks drawn under recency weighting should come from later indices on average."""
    rm = _vol_regime_series(n=1000)
    s0 = np.array([100.0, 100.0])
    # a series whose second half is far more volatile: recency-weighted sampling
    # should therefore produce a WIDER cone than uniform sampling here
    uni = joint_bootstrap_prices(rm, s0, 60, n_paths=6000, seed=15).astype(float)
    rec = joint_bootstrap_prices(rm, s0, 60, n_paths=6000, seed=15,
                                 recency_half_life=150).astype(float)
    w = lambda p: np.percentile(p[:, -1, 1], 95) - np.percentile(p[:, -1, 1], 5)
    assert w(rec) > w(uni)


def test_recency_sampling_deterministic():
    rm = _vol_regime_series()
    s0 = np.array([100.0, 100.0])
    a = joint_bootstrap_prices(rm, s0, 30, n_paths=1500, seed=16, recency_half_life=300)
    b = joint_bootstrap_prices(rm, s0, 30, n_paths=1500, seed=16, recency_half_life=300)
    assert np.array_equal(a, b)
