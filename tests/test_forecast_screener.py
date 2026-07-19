"""Screener criteria: every grade boundary, both sides.

A threshold that isn't tested at its edge is a threshold that will silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast import screener as S


def _rets(n=400, vol=0.02, seed=0):
    return np.random.default_rng(seed).normal(0.0, vol, n)


def _px_from(rets, s0=100.0):
    return s0 * np.exp(np.cumsum(rets))


# ---- 1. beta ----

def test_beta_grades_amplifier_lottery_and_below_cut():
    mkt = _rets(600, 0.01, 1)
    amp = 3.0 * mkt + np.random.default_rng(2).normal(0, 0.004, mkt.size)   # high r2
    lot = 3.0 * mkt + np.random.default_rng(3).normal(0, 0.12, mkt.size)    # tiny r2
    low = 0.5 * mkt + np.random.default_rng(4).normal(0, 0.004, mkt.size)
    assert S.c_beta(amp, mkt)[0].grade == S.PASS
    assert S.c_beta(lot, mkt)[0].grade == S.WARN
    assert S.c_beta(low, mkt)[0].grade == S.FAIL


def test_beta_threshold_text_shows_the_cutoff():
    mkt = _rets(300, 0.01, 5)
    c, _ = S.c_beta(2.5 * mkt, mkt)
    assert str(S.T_BETA) in c.threshold and str(S.T_R2) in c.threshold


# ---- 2. verifiable ----

def test_verifiable_boundary_at_min_bars():
    assert S.c_verifiable(S.MIN_VERIFY_BARS - 1, None).grade == S.FAIL
    assert S.c_verifiable(S.MIN_VERIFY_BARS, None).grade == S.WARN        # long enough, untested
    assert S.c_verifiable(S.MIN_VERIFY_BARS, {"verdict": "consistent", "cov90": .9}).grade == S.PASS


def test_verifiable_fails_on_ruled_out_calibration():
    v = {"verdict": "RULED OUT (not 90%)", "cov90": 0.74}
    assert S.c_verifiable(1500, v).grade == S.FAIL


# ---- 3. survivable ----

def test_survivable_grades_by_halving_frequency():
    flat = np.full(600, 100.0)
    c_flat, st = S.c_survivable(flat)
    assert c_flat.grade == S.PASS and st["halved"] == 0.0

    # a long steady collapse: most 126d windows lose more than half
    crash = 100.0 * np.exp(np.linspace(0, -4.0, 600))
    c_crash, st2 = S.c_survivable(crash)
    assert st2["halved"] > S.T_HALVED_SEVERE
    assert c_crash.grade == S.FAIL


def test_survivable_needs_enough_history():
    c, st = S.c_survivable(np.full(50, 10.0))
    assert c.grade == S.WARN and st == {}


# ---- 4. redundancy ----

def test_redundancy_boundaries():
    assert S.c_redundancy(S.T_CORR_DIVERSIFIER - 0.01, 5).grade == S.PASS
    assert S.c_redundancy(S.T_CORR_DIVERSIFIER, 5).grade == S.WARN
    assert S.c_redundancy(S.T_CORR_DUPLICATE, 5).grade == S.FAIL


def test_redundancy_is_info_without_peers():
    assert S.c_redundancy(None, 0).grade == S.INFO


def test_cohort_redundancy_clone_vs_independent():
    base = _rets(400, 0.02, 7)
    cohort = {"A": base, "CLONE": base * 2.0,          # perfectly correlated
              "IND": _rets(400, 0.02, 8)}
    corr = S.cohort_redundancy(cohort)
    assert corr["A"] > corr["IND"]
    assert corr["CLONE"] > 0.4          # a clone shows up as redundant


# ---- 5. liquidity ----

def test_liquidity_boundaries():
    px = np.full(120, 100.0)
    deep = np.full(120, (S.T_DOLLAR_VOL_OK / 100.0) * 2)
    thin = np.full(120, (S.T_DOLLAR_VOL_THIN / 100.0) * 0.5)
    mid = np.full(120, (S.T_DOLLAR_VOL_THIN / 100.0) * 2)
    assert S.c_liquidity(px, deep)[0].grade == S.PASS
    assert S.c_liquidity(px, mid)[0].grade == S.WARN
    assert S.c_liquidity(px, thin)[0].grade == S.FAIL


def test_liquidity_info_without_volume():
    assert S.c_liquidity(np.full(120, 10.0), None)[0].grade == S.INFO
    assert S.c_liquidity(np.full(120, 10.0), np.zeros(120))[0].grade == S.INFO


# ---- 6. vol regime ----

def test_vol_regime_flags_a_storm():
    calm = np.random.default_rng(11).normal(0, 0.005, 400)
    storm = np.concatenate([calm, np.random.default_rng(12).normal(0, 0.09, 40)])
    c_storm, st = S.c_vol_regime(_px_from(storm))
    assert st["vol_pctl"] > S.T_VOL_PCTL_STORM
    assert c_storm.grade == S.FAIL

    c_calm, st2 = S.c_vol_regime(_px_from(calm))
    assert c_calm.grade in (S.PASS, S.WARN)


def test_vol_regime_info_when_short():
    assert S.c_vol_regime(np.full(40, 10.0))[0].grade == S.INFO


# ---- 7. payoff is never graded ----

def test_payoff_is_always_info():
    _, st = S.c_survivable(_px_from(_rets(600, 0.03, 13)))
    c = S.c_payoff(st)
    assert c.grade == S.INFO
    assert "not predictive" in c.threshold
    assert S.c_payoff({}).grade == S.INFO


# ---- composite ----

def _card(grades):
    crit = [S.Criterion(k, k, 1.0, "", g, "t") for k, g in grades.items()]
    return S.Scorecard("X", crit)


def test_composite_equal_weights():
    card = _card({"beta": S.PASS, "verifiable": S.FAIL})
    assert S.composite(card) == pytest.approx(0.5)


def test_composite_respects_user_weights():
    card = _card({"beta": S.PASS, "survivable": S.FAIL})
    assert S.composite(card, {"beta": 3, "survivable": 1}) == pytest.approx(0.75)
    assert S.composite(card, {"beta": 1, "survivable": 3}) == pytest.approx(0.25)


def test_composite_zero_weight_removes_a_criterion():
    card = _card({"beta": S.PASS, "survivable": S.FAIL})
    assert S.composite(card, {"survivable": 0}) == pytest.approx(1.0)


def test_composite_excludes_info_criteria():
    card = _card({"beta": S.PASS, "payoff": S.INFO})
    assert S.composite(card) == pytest.approx(1.0)      # info must not drag the score


def test_composite_none_without_graded_criteria():
    assert S.composite(S.Scorecard("X", [])) is None


# ---- assembly ----

def test_build_scorecard_has_all_seven_and_is_json_safe():
    import json
    mkt = _rets(600, 0.01, 21)
    rets = 2.5 * mkt + np.random.default_rng(22).normal(0, 0.01, mkt.size)
    px = _px_from(rets)
    card = S.build_scorecard("X", px, px * 1.01, px * 0.99, np.full(px.size, 1e6),
                             mkt, n_shared_bars=900, avg_corr=0.2, n_peers=4,
                             verdict={"verdict": "consistent", "cov90": 0.9})
    keys = [c.key for c in card.criteria]
    assert keys == ["beta", "verifiable", "survivable", "redundancy", "liquidity",
                    "vol_regime", "payoff"]
    json.dumps(card.to_dict())


def test_build_scorecard_isolates_failure():
    card = S.build_scorecard("BAD", np.array([1.0, 2.0]), None, None, None,
                             np.array([0.01]), n_shared_bars=10)
    assert card.error and not card.criteria
