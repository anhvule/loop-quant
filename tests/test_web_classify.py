"""Parity between web/api/_classify.py and src/forecast/classify.py.

The web copy is hand-vendored. These tests make divergence impossible to miss --
otherwise the site could call a ticker investable while the CLI calls it reckless.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))

import _classify as WC  # noqa: E402

from src.forecast import classify as K  # noqa: E402


def _mkt(n=300, vol=0.01, seed=0):
    r = np.random.default_rng(seed).normal(0.0, vol, n)
    return r - float(np.mean(r[-K.WIN_LONG:]))


def _name_data(mkt, beta=2.0, mu=0.0, idio=0.004, idio_last63=None, adv=50e6, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, idio, mkt.size)
    if idio_last63 is not None:
        noise[-63:] = rng.normal(0.0, idio_last63, 63)
    rets = mu + beta * mkt + noise
    closes = 100.0 * np.exp(np.cumsum(rets))
    return {"closes": closes, "volumes": adv / closes,
            "name_rets": rets, "mkt_rets": mkt}


CLEAN_FUNDAMENTALS = {"dilution_yoy": 0.01, "runway_quarters": float("inf"),
                      "revenue_ttm": 500e6, "n_revenue_quarters": 4}
DIRTY_FUNDAMENTALS = {"dilution_yoy": 0.45, "runway_quarters": 2.0,
                      "revenue_ttm": 1e6, "n_revenue_quarters": 4}


def _universe(mkt):
    u = {}
    for i in range(4):
        u[f"GOOD{i}"] = _name_data(mkt, 2.0, 0.0015 + 0.0001 * i, 0.004, seed=10 + i)
        u[f"GOOD{i}"]["fundamentals"] = CLEAN_FUNDAMENTALS
    for i in range(4):
        u[f"BAD{i}"] = _name_data(mkt, 2.0, -0.0015, 0.040, 0.070, seed=20 + i)
        u[f"BAD{i}"]["fundamentals"] = DIRTY_FUNDAMENTALS
    for i in range(2):
        u[f"MID{i}"] = _name_data(mkt, 2.0, 0.0, 0.010, seed=30 + i)
    u["THIN"] = _name_data(mkt, 2.0, 0.0015, 0.004, adv=1e6, seed=40)
    u["SLOW"] = _name_data(mkt, 0.5, 0.0015, 0.004, seed=41)
    return u


# ---- constants must be identical, or the two graders disagree by construction ----

def test_thresholds_match():
    for name in ("T_ADV_DOLLAR", "ADV_WINDOW", "T_BETA_MIN", "BETA_CONFIRM_FRAC",
                 "BETA_DIVERGE", "SMA_WINDOW", "TREYNOR_TOP_FRAC",
                 "MIN_UNIVERSE_FOR_QUARTILE", "T_IDIO_SHARE", "T_IDIO_COMPRESS",
                 "WIN_SHORT", "WIN_LONG", "MIN_BARS_CLASSIFY", "RF_DEFAULT",
                 "TRADING_DAYS", "SURV_WINDOW", "T_HALVED_MILD", "T_HALVED_SEVERE",
                 "OWN_SMA_WINDOW", "T_DILUTION_OK", "T_DILUTION_HEAVY", "T_RUNWAY_OK",
                 "T_RUNWAY_CRITICAL", "T_REVENUE_REAL", "RUNWAY_DISPLAY_CAP",
                 "DILUTION_MIN_BARS", "RECKLESS_MIN_FLAGS"):
        assert getattr(WC, name) == getattr(K, name), name


def test_grade_and_verdict_constants_match():
    assert (WC.PASS, WC.WARN, WC.FAIL, WC.INFO) == (K.PASS, K.WARN, K.FAIL, K.INFO)
    assert WC.VERDICT_ORDER == K.VERDICT_ORDER
    assert (WC.QUARTILE_RELATIVE, WC.QUARTILE_ABSOLUTE) == (K.QUARTILE_RELATIVE,
                                                            K.QUARTILE_ABSOLUTE)
    assert WC.BANNER == K.BANNER


def test_every_verdict_has_display_text():
    assert set(WC.VERDICT_TEXT) == set(K.VERDICT_ORDER)


# ---- estimator parity ----

def test_rolling_beta_parity():
    mkt = _mkt(400, 0.01, 1)
    nm = 2.6 * mkt + np.random.default_rng(2).normal(0, 0.01, mkt.size)
    for w in (K.WIN_SHORT, K.WIN_LONG):
        wb, wr = WC.rolling_ols_beta(nm, mkt, w)
        sb, sr = K.rolling_ols_beta(nm, mkt, w)
        assert wb == pytest.approx(sb) and wr == pytest.approx(sr)


def test_treynor_parity():
    mkt = _mkt(300, 0.01, 3)
    nm = _name_data(mkt, 2.0, 0.001, seed=4)["name_rets"]
    for rf in (0.0, 0.04, 0.06):
        assert WC.treynor_ratio(nm, 2.0, rf) == pytest.approx(K.treynor_ratio(nm, 2.0, rf))


# ---- per-gate parity ----

def test_liquidity_parity():
    px = np.full(120, 100.0)
    for v in (90_000.0, 100_000.0, 110_000.0, 5_000_000.0):
        w, wa = WC.g_liquidity(px, np.full(120, v))
        s, sa = K.g_liquidity(px, np.full(120, v))
        assert w["grade"] == s.grade and wa == pytest.approx(sa)
        assert w["threshold"] == s.threshold
    assert WC.g_liquidity(px, None)[0]["grade"] == K.g_liquidity(px, None)[0].grade


def test_beta_parity_across_regimes():
    mkt = _mkt(300, 0.01, 5)
    for early, last in ((1.65, 1.40), (1.40, 1.40), (2.40, 0.50), (1.30, 3.20)):
        coef = np.where(np.arange(mkt.size) >= mkt.size - 63, last, early)
        w, ws = WC.g_beta(coef * mkt, mkt)
        s, ss = K.g_beta(coef * mkt, mkt)
        assert w["grade"] == s.grade
        assert w["value"] == pytest.approx(s.value)
        for k in ("beta_63", "beta_252", "r2_63", "r2_252"):
            assert ws[k] == pytest.approx(ss[k])


def test_regime_parity():
    for px in (np.linspace(100.0, 200.0, 300), np.linspace(200.0, 100.0, 300),
               np.linspace(100.0, 90.0, 150)):
        w, won = WC.g_regime(px)
        s, son = K.g_regime(px)
        assert w["grade"] == s.grade and won == son
        assert w["display"] == s.display


def test_treynor_gate_parity_incl_the_small_universe_fallback():
    for pctl, n in ((0.80, 20), (0.75, 20), (0.70, 20), (1.0, 3), (0.5, 3)):
        for t in (0.5, -0.5):
            assert WC.g_treynor(t, pctl, n)["grade"] == K.g_treynor(t, pctl, n).grade
            assert WC.g_treynor(t, pctl, n)["threshold"] == K.g_treynor(t, pctl, n).threshold


def test_idio_parity_on_every_branch():
    for r63, r252 in ((0.55, 0.50), (0.14, 0.15), (0.10, 0.20), (0.25, 0.15),
                      (0.20, 0.33), (0.26, 0.26)):
        w, s = WC.g_idio(r63, r252), K.g_idio(r63, r252)
        assert w["grade"] == s.grade, (r63, r252)
        assert w["value"] == pytest.approx(s.value)


def test_survivable_parity():
    crash = np.full(1000, 100.0)
    crash[500:] = 40.0
    for px in (np.full(600, 100.0), crash,
               100.0 * np.exp(np.linspace(0, -4.0, 600)), np.full(100, 10.0)):
        w, wst = WC.g_survivable(px)
        s, sst = K.g_survivable(px)
        assert w["grade"] == s.grade
        assert w["value"] == pytest.approx(s.value)
        for k in sst:
            assert wst[k] == pytest.approx(sst[k])


def test_own_trend_parity():
    for px in (np.linspace(100.0, 200.0, 300), np.linspace(200.0, 100.0, 300),
               np.linspace(200.0, 100.0, 150)):
        w, wa = WC.g_own_trend(px)
        s, sa = K.g_own_trend(px)
        assert w["grade"] == s.grade and wa == sa
        assert w["display"] == s.display


def test_dilution_parity_incl_the_ipo_cap():
    for dil in (None, -0.05, 0.03, 0.10, 0.16, 1.95):
        for n_bars in (None, 100, K.DILUTION_MIN_BARS - 1, K.DILUTION_MIN_BARS, 2000):
            for rev in (None, 1e6, 3.9e9):     # unknown / pre-revenue / funded (SOFI)
                w = WC.g_dilution(dil, n_bars, rev_ttm=rev)
                s = K.g_dilution(dil, n_bars, rev_ttm=rev)
                assert w["grade"] == s.grade, (dil, n_bars, rev)
                assert w["note"] == s.note, (dil, n_bars, rev)


def test_runway_parity_incl_the_revenue_cap_and_inf_clamp():
    for q in (None, float("inf"), 20.0, 8.0, 5.0, 3.0, 0.0):
        for rev in (None, 1e6, 5e9):
            for meaningful in (True, False):     # False = lender: FCF is not burn
                w = WC.g_runway(q, rev, fcf_meaningful=meaningful)
                s = K.g_runway(q, rev, fcf_meaningful=meaningful)
                assert w["grade"] == s.grade, (q, rev, meaningful)
                assert w["value"] == pytest.approx(s.value), (q, rev, meaningful)
                assert w["note"] == s.note, (q, rev, meaningful)
            json.dumps(w)                       # inf must never reach a payload


def test_revenue_parity():
    for rev, n in ((None, 0), (1e6, 0), (5e6, 4), (50e6, 4), (0.0, 4)):
        w, s = WC.g_revenue(rev, n), K.g_revenue(rev, n)
        assert w["grade"] == s.grade and w["grade"] != WC.FAIL


# ---- whole-universe parity ----

def test_classify_universe_parity_risk_on_and_risk_off():
    mkt = _mkt(300, 0.01, 50)
    uni = _universe(mkt)
    for closes in (np.linspace(100.0, 200.0, 300), np.linspace(200.0, 100.0, 300)):
        w = WC.classify_universe(uni, closes)
        s = K.classify_universe(uni, closes)
        assert [r["symbol"] for r in w["results"]] == [r.symbol for r in s.results]
        assert [r["verdict"] for r in w["results"]] == [r.verdict for r in s.results]
        assert w["quartile_mode"] == s.quartile_mode
        assert w["n_eligible"] == s.n_eligible
        assert w["risk_on"] == s.risk_on
        for rw, rs in zip(w["results"], s.results):
            assert [g["grade"] for g in rw["gates"]] == [g.grade for g in rs.gates]
            assert rw["reasons"] == rs.reasons
            # `flags` is a list of strings; approx() only handles it by falling back to
            # equality, which is an implementation detail to lean on. Compare it plainly
            # and keep approx for the numbers it is actually for.
            assert rw["metrics"].get("flags") == rs.metrics.get("flags"), rs.symbol
            for k, v in rs.metrics.items():
                if k == "flags":
                    continue
                assert rw["metrics"][k] == pytest.approx(v), (rs.symbol, k)


def test_classify_universe_parity_with_custom_thresholds():
    mkt = _mkt(300, 0.01, 51)
    uni = _universe(mkt)
    closes = np.linspace(100.0, 200.0, 300)
    for kwargs in ({"rf": 0.06}, {"min_beta": 2.5}, {"min_adv": 20e6}):
        w = WC.classify_universe(uni, closes, **kwargs)
        s = K.classify_universe(uni, closes, **kwargs)
        assert [r["verdict"] for r in w["results"]] == [r.verdict for r in s.results]


# ---- web-specific ----

def test_payload_is_json_safe():
    mkt = _mkt(300, 0.01, 52)
    body = WC.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    json.dumps(body)          # GOOD names are self-funding: inf must have been clamped
    assert len(body["results"]) == 12
    assert body["verdict_order"][0] == WC.INVESTABLE


def test_universe_parity_without_any_fundamentals():
    """The web's real shape: no fundamentals at all. Both sides must agree that the
    three gates are INFO and that INFO does not block a verdict."""
    mkt = _mkt(300, 0.01, 55)
    uni = {k: dict(v, fundamentals=None) for k, v in _universe(mkt).items()}
    closes = np.linspace(100.0, 200.0, 300)
    w = WC.classify_universe(uni, closes)
    s = K.classify_universe(uni, closes)
    assert [r["verdict"] for r in w["results"]] == [r.verdict for r in s.results]
    for r in w["results"]:
        if r["verdict"] in ("investable", "mixed", "reckless", "stand_aside"):
            by_key = {g["key"]: g for g in r["gates"]}
            for k in ("dilution", "runway", "revenue"):
                assert by_key[k]["grade"] == WC.INFO


def test_eligible_rows_carry_all_ten_gates_in_the_same_order():
    mkt = _mkt(300, 0.01, 56)
    w = WC.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    s = K.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    for rw, rs in zip(w["results"], s.results):
        assert [g["key"] for g in rw["gates"]] == [g.key for g in rs.gates]


def test_banner_refuses_to_claim_prediction():
    assert "NOT PREDICTION" in WC.BANNER


def test_bad_ticker_becomes_an_error_row_not_an_exception():
    mkt = _mkt(300, 0.01, 53)
    short = _mkt(100, 0.01, 54)
    uni = {"OK": _name_data(mkt, 2.0, 0.001, seed=60),
           "SHORT": {"closes": 100.0 * np.exp(np.cumsum(short)),
                     "volumes": np.full(100, 1e6),
                     "name_rets": short, "mkt_rets": short}}
    body = WC.classify_universe(uni, np.linspace(100.0, 200.0, 300))
    by_sym = {r["symbol"]: r for r in body["results"]}
    assert by_sym["SHORT"]["verdict"] == WC.ERROR
    assert by_sym["OK"]["verdict"] != WC.ERROR
