"""Parity between web/api/_screener.py and src/forecast/screener.py.

The web copy is hand-vendored. These tests make divergence impossible to miss --
otherwise the site could grade a ticker differently from the CLI that was validated.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))

import _screener as WS  # noqa: E402

from src.forecast import screener as S  # noqa: E402


def _rets(n=500, vol=0.02, seed=0):
    return np.random.default_rng(seed).normal(0.0, vol, n)


def _px(rets, s0=100.0):
    return s0 * np.exp(np.cumsum(rets))


# ---- thresholds must be identical, or the two graders disagree by construction ----

def test_thresholds_match():
    for name in ("T_BETA", "T_R2", "T_HALVED_MILD", "T_HALVED_SEVERE",
                 "T_CORR_DIVERSIFIER", "T_CORR_DUPLICATE", "T_DOLLAR_VOL_OK",
                 "T_DOLLAR_VOL_THIN", "T_VOL_PCTL_ELEVATED", "T_VOL_PCTL_STORM",
                 "MIN_VERIFY_BARS", "WINDOW"):
        assert getattr(WS, name) == getattr(S, name), name


def test_grade_constants_match():
    assert (WS.PASS, WS.WARN, WS.FAIL, WS.INFO) == (S.PASS, S.WARN, S.FAIL, S.INFO)


# ---- per-criterion parity ----

def test_beta_parity():
    mkt = _rets(600, 0.01, 1)
    nm = 2.6 * mkt + np.random.default_rng(2).normal(0, 0.01, mkt.size)
    w, _ = WS.c_beta(nm, mkt)
    s, _ = S.c_beta(nm, mkt)
    assert w["grade"] == s.grade
    assert w["value"] == pytest.approx(s.value)


def test_verifiable_parity():
    for bars, verdict in ((300, None), (900, None),
                          (900, {"verdict": "consistent", "cov90": 0.9}),
                          (900, {"verdict": "RULED OUT (not 90%)", "cov90": 0.7})):
        assert WS.c_verifiable(bars, verdict)["grade"] == S.c_verifiable(bars, verdict).grade


def test_survivable_parity():
    px = _px(_rets(700, 0.04, 3))
    w, wst = WS.c_survivable(px)
    s, sst = S.c_survivable(px)
    assert w["grade"] == s.grade
    assert w["value"] == pytest.approx(s.value)
    assert wst["halved"] == pytest.approx(sst["halved"])
    assert wst["max_dd"] == pytest.approx(sst["max_dd"])


def test_redundancy_parity():
    for c in (0.1, 0.29, 0.30, 0.44, 0.45, 0.8, None):
        assert WS.c_redundancy(c, 5)["grade"] == S.c_redundancy(c, 5).grade


def test_liquidity_parity():
    px = np.full(120, 50.0)
    for mult in (0.5, 2.0, 40.0):
        vol = np.full(120, (S.T_DOLLAR_VOL_THIN / 50.0) * mult)
        assert WS.c_liquidity(px, vol)[0]["grade"] == S.c_liquidity(px, vol)[0].grade


def test_vol_regime_parity():
    px = _px(np.concatenate([_rets(400, 0.005, 5), _rets(40, 0.08, 6)]))
    w, wst = WS.c_vol_regime(px)
    s, sst = S.c_vol_regime(px)
    assert w["grade"] == s.grade
    assert wst["vol_pctl"] == pytest.approx(sst["vol_pctl"])


def test_payoff_parity_and_always_info():
    _, st = S.c_survivable(_px(_rets(700, 0.03, 9)))
    assert WS.c_payoff(st)["grade"] == S.c_payoff(st).grade == S.INFO


def test_cohort_redundancy_parity():
    base = _rets(400, 0.02, 11)
    cohort = {"A": base, "B": base * 1.5, "C": _rets(400, 0.02, 12)}
    w = WS.cohort_redundancy(cohort)
    s = S.cohort_redundancy(cohort)
    assert set(w) == set(s)
    for k in w:
        assert w[k] == pytest.approx(s[k])


def test_composite_parity_incl_user_weights():
    mkt = _rets(600, 0.01, 13)
    nm = 2.4 * mkt + np.random.default_rng(14).normal(0, 0.01, mkt.size)
    px = _px(nm)
    wcard = WS.build_scorecard("X", px, np.full(px.size, 1e6), mkt, 900,
                               avg_corr=0.2, n_peers=4,
                               verdict={"verdict": "consistent", "cov90": 0.9})
    scard = S.build_scorecard("X", px, px * 1.01, px * 0.99, np.full(px.size, 1e6),
                              mkt, 900, avg_corr=0.2, n_peers=4,
                              verdict={"verdict": "consistent", "cov90": 0.9})
    assert [c["key"] for c in wcard["criteria"]] == [c.key for c in scard.criteria]
    assert [c["grade"] for c in wcard["criteria"]] == [c.grade for c in scard.criteria]
    for wts in ({}, {"survivable": 3}, {"beta": 0, "liquidity": 2}):
        assert WS.composite(wcard["criteria"], wts) == pytest.approx(S.composite(scard, wts))


# ---- web-specific ----

def test_scorecard_payload_is_json_safe():
    mkt = _rets(500, 0.01, 15)
    px = _px(2.2 * mkt + np.random.default_rng(16).normal(0, 0.01, mkt.size))
    card = WS.build_scorecard("X", px, np.full(px.size, 1e6), mkt, 800)
    json.dumps(card)
    assert len(card["criteria"]) == 7
    assert card["error"] == ""


def test_build_scorecard_isolates_bad_input():
    card = WS.build_scorecard("BAD", np.array([1.0, 2.0]), None, np.array([0.01]), 10)
    assert card["error"] and not card["criteria"]


def test_banner_states_no_prediction():
    assert "NONE of them predicts returns" in WS.BANNER
