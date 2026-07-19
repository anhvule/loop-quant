"""Classifier gates: every threshold, both sides.

Same discipline as test_forecast_screener.py -- a boundary that isn't tested at its
edge is a boundary that will silently drift. The end-to-end tests build a synthetic
universe where each verdict bucket is reachable by construction, so a change in the
verdict logic shows up as a failing bucket rather than a plausible-looking table.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.forecast import classify as C


def _mkt(n=300, vol=0.01, seed=0):
    """Market returns with the realized drift removed over the Treynor window.

    Uncentred noise carries a drift of its own (~0.09%/day at these seeds), which at
    beta 2 swamps the drift the fixture is trying to express and flips Treynor signs.
    Centring makes each name's Treynor follow the `mu` it was built with."""
    r = np.random.default_rng(seed).normal(0.0, vol, n)
    return r - float(np.mean(r[-C.WIN_LONG:]))


CLEAN_FUNDAMENTALS = {"dilution_yoy": 0.01, "runway_quarters": float("inf"),
                      "revenue_ttm": 500e6, "n_revenue_quarters": 4}


def _name_data(mkt, beta=2.0, mu=0.0, idio=0.004, idio_last63=None,
               adv=50e6, seed=1, fundamentals=None):
    """One name's aligned arrays. `adv` is exact: volumes are back-solved from closes."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, idio, mkt.size)
    if idio_last63 is not None:
        noise[-63:] = rng.normal(0.0, idio_last63, 63)
    rets = mu + beta * mkt + noise
    closes = 100.0 * np.exp(np.cumsum(rets))
    return {"closes": closes, "volumes": adv / closes,
            "name_rets": rets, "mkt_rets": mkt, "fundamentals": fundamentals}


def _betas(mkt, beta_early, beta_last63):
    """Name returns whose beta changes over the final 63 sessions."""
    coef = np.where(np.arange(mkt.size) >= mkt.size - 63, beta_last63, beta_early)
    return coef * mkt


# ---- rolling OLS beta ----

def test_rolling_beta_recovers_a_known_slope():
    mkt = _mkt(400, 0.01, 1)
    nm = 2.0 * mkt + np.random.default_rng(2).normal(0, 0.0005, mkt.size)
    b, r2 = C.rolling_ols_beta(nm, mkt, 252)
    assert b == pytest.approx(2.0, abs=0.02)
    assert r2 > 0.99


def test_rolling_beta_uses_only_the_trailing_window():
    mkt = _mkt(400, 0.01, 3)
    nm = _betas(mkt, beta_early=0.1, beta_last63=3.0)
    b63, _ = C.rolling_ols_beta(nm, mkt, 63)
    assert b63 == pytest.approx(3.0, abs=0.01)      # the 0.1 era must not leak in


def test_rolling_beta_rejects_short_and_mismatched_input():
    mkt = _mkt(100, 0.01, 4)
    with pytest.raises(ValueError):
        C.rolling_ols_beta(mkt, mkt, 252)
    with pytest.raises(ValueError):
        C.rolling_ols_beta(mkt[:50], mkt, 30)


# ---- gate 1: liquidity ----

def test_liquidity_boundary_is_strict():
    px = np.full(120, 100.0)
    assert C.g_liquidity(px, np.full(120, 110_000.0))[0].grade == C.PASS      # $11M
    assert C.g_liquidity(px, np.full(120, 90_000.0))[0].grade == C.FAIL       # $9M
    assert C.g_liquidity(px, np.full(120, 100_000.0))[0].grade == C.FAIL      # exactly $10M


def test_liquidity_respects_a_custom_floor():
    px = np.full(120, 100.0)
    vol = np.full(120, 110_000.0)                                   # $11M/day
    assert C.g_liquidity(px, vol, min_adv=5e6)[0].grade == C.PASS
    assert C.g_liquidity(px, vol, min_adv=20e6)[0].grade == C.FAIL


def test_liquidity_is_info_and_not_a_pass_without_volume():
    for vols in (None, np.zeros(120)):
        c, adv = C.g_liquidity(np.full(120, 10.0), vols)
        assert c.grade == C.INFO and adv is None     # INFO never clears the hard gate


def test_liquidity_uses_the_median_not_the_mean():
    px = np.full(120, 100.0)
    vol = np.full(120, 50_000.0)      # $5M/day typical
    vol[-1] = 50_000_000.0            # one enormous print
    assert C.g_liquidity(px, vol)[0].grade == C.FAIL


# ---- gate 2: rolling beta ----

def test_beta_passes_when_both_windows_are_high():
    mkt = _mkt(300, 0.01, 5)
    c, st = C.g_beta(_betas(mkt, 1.65, 1.40), mkt)
    assert c.grade == C.PASS
    assert st["beta_252"] >= C.T_BETA_MIN
    assert st["beta_63"] >= C.BETA_CONFIRM_FRAC * C.T_BETA_MIN


def test_beta_fails_below_the_cut():
    mkt = _mkt(300, 0.01, 6)
    c, st = C.g_beta(_betas(mkt, 1.40, 1.40), mkt)
    assert c.grade == C.FAIL
    assert st["beta_252"] < C.T_BETA_MIN
    assert "not high-beta" in c.note


def test_beta_fails_when_the_short_window_has_collapsed():
    mkt = _mkt(300, 0.01, 7)
    c, st = C.g_beta(_betas(mkt, 2.40, 0.50), mkt)
    assert st["beta_252"] >= C.T_BETA_MIN          # the year still looks high-beta
    assert c.grade == C.FAIL                       # but today it is not
    assert "collapsed" in c.note


def test_beta_warns_when_the_windows_diverge():
    mkt = _mkt(300, 0.01, 8)
    c, st = C.g_beta(_betas(mkt, 1.30, 3.20), mkt)
    assert c.grade == C.WARN                       # eligible, but unstable
    assert abs(st["beta_63"] - st["beta_252"]) / st["beta_252"] > C.BETA_DIVERGE


def test_beta_threshold_text_shows_the_cutoffs():
    mkt = _mkt(300, 0.01, 9)
    c, _ = C.g_beta(_betas(mkt, 2.0, 2.0), mkt, min_beta=1.5)
    assert "1.5" in c.threshold and "1.12" in c.threshold


# ---- gate 3: regime ----

def test_regime_reads_the_sma_both_ways():
    up = np.linspace(100.0, 200.0, 300)
    down = np.linspace(200.0, 100.0, 300)
    c_up, on_up = C.g_regime(up)
    c_dn, on_dn = C.g_regime(down)
    assert on_up is True and c_up.grade == C.PASS
    assert on_dn is False and c_dn.grade == C.FAIL
    assert "risk-off" in c_dn.note


def test_regime_is_info_and_permissive_when_history_is_short():
    c, risk_on = C.g_regime(np.linspace(100.0, 90.0, 150))
    assert c.grade == C.INFO and risk_on is True   # unknown must not masquerade as risk-off


# ---- gate 4: Treynor ----

def test_treynor_matches_a_hand_computation():
    rets = np.full(252, 0.001)
    # (252 * 0.001 - 0.04) / 2.0
    assert C.treynor_ratio(rets, 2.0, 0.04) == pytest.approx(0.106)


def test_treynor_needs_a_full_window_and_a_nonzero_beta():
    with pytest.raises(ValueError):
        C.treynor_ratio(np.full(100, 0.001), 2.0)
    with pytest.raises(ValueError):
        C.treynor_ratio(np.full(252, 0.001), 0.0)


def test_treynor_gate_uses_the_quartile_on_a_real_universe():
    assert C.g_treynor(1.0, 0.80, 20).grade == C.PASS
    assert C.g_treynor(1.0, 0.75, 20).grade == C.PASS      # boundary is inclusive
    assert C.g_treynor(1.0, 0.70, 20).grade == C.FAIL


def test_treynor_gate_falls_back_below_the_universe_floor():
    n = C.MIN_UNIVERSE_FOR_QUARTILE - 1
    top = C.g_treynor(0.5, 1.0, n)           # ranked first, but the rank is meaningless
    neg = C.g_treynor(-0.5, 1.0, n)
    assert top.grade == C.PASS and neg.grade == C.FAIL
    assert "too small to rank" in top.display
    assert "fallback" in top.threshold       # the fallback must be visible, not silent


# ---- gate 5: idiosyncratic share ----

def test_idio_passes_when_the_market_explains_the_name():
    assert C.g_idio(r2_63=0.55, r2_252=0.50).grade == C.PASS


def test_idio_warns_when_high_but_stable():
    c = C.g_idio(r2_63=0.14, r2_252=0.15)         # idio 86% vs 85%
    assert c.grade == C.WARN


def test_idio_fails_when_high_and_expanding():
    c = C.g_idio(r2_63=0.10, r2_252=0.20)         # idio 90% vs 80%
    assert c.grade == C.FAIL
    assert "binary-event" in c.note


def test_idio_passes_when_high_but_compressing():
    assert C.g_idio(r2_63=0.25, r2_252=0.15).grade == C.PASS   # idio 75% vs 85%


def test_a_recent_only_spike_warns_but_does_not_condemn():
    """252d idio 67% (below the cut) with 63d at 80% and climbing. An earlier rule let
    the short window declare the level and FAILed this outright -- written from one
    observation (COIN), then SOFI produced the same signature and the verdict was not
    believable. 63 observations is too thin to condemn on; it raises a flag instead."""
    c = C.g_idio(r2_63=0.20, r2_252=0.33)
    assert c.grade == C.WARN
    assert "does not corroborate" in c.note


def test_idio_fails_only_when_the_long_window_corroborates():
    # 252d idio 84% and rising to 91% -- both windows agree, which is the real
    # binary-event signature (observed on DNA).
    c = C.g_idio(r2_63=0.09, r2_252=0.16)
    assert c.grade == C.FAIL
    assert "binary-event" in c.note


def test_idio_boundary_of_the_high_share_cut():
    # idio share exactly at the cut counts as high; a hair below does not.
    assert C.g_idio(r2_63=1 - C.T_IDIO_SHARE, r2_252=1 - C.T_IDIO_SHARE).grade == C.WARN
    assert C.g_idio(r2_63=0.26, r2_252=0.26).grade == C.PASS


def test_dilution_caps_a_revenue_backed_issuer_at_warn():
    """A lender must issue equity to grow its book; a shell issues to survive. Both
    show the same share growth, so revenue decides which reading applies -- the same
    judgement g_runway already makes about negative free cash flow (the SOFI case)."""
    heavy = 0.16
    assert C.g_dilution(heavy, n_bars=2000).grade == C.FAIL            # unknown revenue
    assert C.g_dilution(heavy, n_bars=2000, rev_ttm=1e6).grade == C.FAIL   # pre-revenue
    funded = C.g_dilution(heavy, n_bars=2000, rev_ttm=3.9e9)
    assert funded.grade == C.WARN
    assert "capital formation" in funded.note
    # SPCE is untouched: years of history, and no revenue to earn either cap.
    assert C.g_dilution(1.95, n_bars=2000, rev_ttm=1.3e6).grade == C.FAIL


# ---- gate 6: survivability ----

def test_survivable_grades_by_halving_frequency():
    flat = np.full(600, 100.0)
    c_flat, st = C.g_survivable(flat)
    assert c_flat.grade == C.PASS and st["halved_126d"] == 0.0

    crash = 100.0 * np.exp(np.linspace(0, -4.0, 600))   # a long steady collapse
    c_crash, st2 = C.g_survivable(crash)
    assert st2["halved_126d"] > C.T_HALVED_SEVERE
    assert c_crash.grade == C.FAIL
    assert "chronic-collapse" in c_crash.note


def test_survivable_warns_between_the_two_cuts():
    # One 60% crash in an otherwise flat history. A constant-rate decay cannot land
    # here: every window shares the same ratio, so it halves either always or never.
    # Only the windows straddling the crash halve -- 126 of 874.
    px = np.full(1000, 100.0)
    px[500:] = 40.0
    c, st = C.g_survivable(px)
    assert C.T_HALVED_MILD <= st["halved_126d"] < C.T_HALVED_SEVERE
    assert c.grade == C.WARN


def test_survivable_warns_rather_than_passes_on_short_history():
    c, st = C.g_survivable(np.full(100, 10.0))
    # A name too young to have survived anything has not demonstrated survival, so it
    # must not collect a free PASS -- WARN blocks investable without condemning.
    assert c.grade == C.WARN and st == {}


def test_survivable_reports_drawdown_and_worst_window():
    px = np.concatenate([np.linspace(100.0, 200.0, 300), np.linspace(200.0, 20.0, 300)])
    _, st = C.g_survivable(px)
    assert st["max_dd"] == pytest.approx(0.90, abs=0.01)
    assert st["worst_window"] < -0.5


# ---- gate 7: own trend (advisory) ----

def test_own_trend_reads_the_name_sma_both_ways():
    up, _ = C.g_own_trend(np.linspace(100.0, 200.0, 300))
    down, _ = C.g_own_trend(np.linspace(200.0, 100.0, 300))
    assert up.grade == C.PASS
    assert down.grade == C.WARN


def test_own_trend_never_fails():
    """The anti-momentum rail: a downtrend withholds a blessing, it never condemns."""
    for px in (np.linspace(200.0, 1.0, 300), np.linspace(100.0, 99.0, 300),
               np.full(300, 50.0), np.linspace(1.0, 200.0, 300)):
        assert C.g_own_trend(px)[0].grade != C.FAIL


def test_own_trend_is_info_and_permissive_when_short():
    c, above = C.g_own_trend(np.linspace(200.0, 100.0, 150))
    assert c.grade == C.INFO and above is True


# ---- gate 8: dilution ----

def test_dilution_boundaries():
    assert C.g_dilution(-0.05).grade == C.PASS            # buyback
    assert C.g_dilution(C.T_DILUTION_OK).grade == C.PASS  # boundary inclusive
    assert C.g_dilution(0.10).grade == C.WARN
    assert C.g_dilution(C.T_DILUTION_HEAVY).grade == C.WARN
    assert C.g_dilution(C.T_DILUTION_HEAVY + 0.001).grade == C.FAIL


def test_dilution_is_info_when_unverified():
    c = C.g_dilution(None)
    assert c.grade == C.INFO and c.value is None
    assert "unverified" in c.note


def test_dilution_caps_recent_ipos_at_warn():
    """A share count cannot be read across a flotation: CRWV's +27% sixteen months after
    listing is what going public looks like, not a funding treadmill."""
    heavy = 0.30
    assert C.g_dilution(heavy, n_bars=C.DILUTION_MIN_BARS).grade == C.FAIL
    young = C.g_dilution(heavy, n_bars=C.DILUTION_MIN_BARS - 1)
    assert young.grade == C.WARN
    assert "IPO" in young.note
    # The cap only softens a FAIL; it never promotes a passing name.
    assert C.g_dilution(0.01, n_bars=100).grade == C.PASS
    # And a long-listed serial diluter is untouched by it.
    assert C.g_dilution(1.95, n_bars=2000).grade == C.FAIL


# ---- gate 9: runway ----

def test_runway_boundaries():
    assert C.g_runway(9.0).grade == C.PASS
    assert C.g_runway(C.T_RUNWAY_OK).grade == C.PASS
    assert C.g_runway(5.0).grade == C.WARN
    assert C.g_runway(C.T_RUNWAY_CRITICAL).grade == C.WARN
    assert C.g_runway(C.T_RUNWAY_CRITICAL - 0.1).grade == C.FAIL


def test_runway_self_funding_passes_and_is_json_safe():
    c = C.g_runway(float("inf"))
    assert c.grade == C.PASS
    assert c.value == C.RUNWAY_DISPLAY_CAP      # inf would emit invalid JSON
    json.dumps(c.to_dict())
    assert "self-funding" in c.display


def test_runway_is_info_when_unverified():
    assert C.g_runway(None).grade == C.INFO


def test_runway_is_not_assessed_where_free_cash_flow_is_meaningless():
    """A lender books loan origination as an operating outflow, so a healthy growing
    bank reads as a cash burn. Reporting no number is honest; a number measuring the
    wrong thing is not. This is 'the metric does not apply', not a sector exemption --
    an INFO cannot help a name reach `investable` that other gates would block."""
    c = C.g_runway(2.0, rev_ttm=3.9e9, fcf_meaningful=False)
    assert c.grade == C.INFO and c.value is None
    assert "does not measure burn" in c.note
    # Default stays on: only an explicit sector signal switches the gate off.
    assert C.g_runway(2.0, rev_ttm=3.9e9).grade == C.WARN
    assert C.g_runway(2.0, rev_ttm=1e6).grade == C.FAIL


def test_runway_caps_revenue_backed_burn_at_warn():
    """Burning cash on growth capex against billions in sales is a financing choice;
    burning cash with no product is a countdown. Both look like negative free cash flow."""
    critical = C.T_RUNWAY_CRITICAL - 1.0
    assert C.g_runway(critical).grade == C.FAIL                      # no revenue known
    assert C.g_runway(critical, rev_ttm=1e6).grade == C.FAIL         # pre-revenue
    funded = C.g_runway(critical, rev_ttm=5e9)
    assert funded.grade == C.WARN
    assert "financing choice" in funded.note
    # The cap only softens a FAIL; a comfortable runway is unaffected either way.
    assert C.g_runway(20.0, rev_ttm=5e9).grade == C.PASS


# ---- gate 10: revenue (advisory) ----

def test_revenue_flags_pre_revenue_without_condemning():
    assert C.g_revenue(5e6, 4).grade == C.WARN
    assert C.g_revenue(50e6, 4).grade == C.PASS
    assert C.g_revenue(None, 0).grade == C.INFO
    assert C.g_revenue(1e6, 0).grade == C.INFO       # quarters missing -> unverified


def test_revenue_never_fails():
    for rev, n in ((0.0, 4), (1.0, 4), (5e6, 1), (1e12, 4), (None, 0)):
        assert C.g_revenue(rev, n).grade != C.FAIL


# ---- verdict interactions ----

def _gates(beta=C.PASS, treynor=C.PASS, idio=C.PASS, surv=C.PASS, trend=C.PASS,
           dil=C.PASS, run=C.PASS, rev=C.PASS):
    mk = lambda k, g: C.Criterion(k, k, 1.0, "", g, "t", f"{k}-note")  # noqa: E731
    return (mk("beta", beta), mk("treynor", treynor), mk("idio", idio),
            mk("survivable", surv), mk("own_trend", trend), mk("dilution", dil),
            mk("runway", run), mk("revenue", rev))


def test_verdict_all_clean_is_investable():
    assert C._verdict(*_gates(), risk_on=True)[0] == C.INVESTABLE


def test_verdict_reckless_needs_two_corroborating_flags():
    """One gate can be wrong about one name -- a threshold sits a point away, a vendor
    number is stale. The harshest label requires corroboration."""
    v, _, flags = C._verdict(*_gates(treynor=C.FAIL, idio=C.FAIL, surv=C.FAIL),
                             risk_on=True)
    assert v == C.RECKLESS
    assert sorted(flags) == ["idio", "survivable"]


def test_verdict_a_single_flag_is_a_caveat_not_a_condemnation():
    for kw in ({"idio": C.FAIL}, {"surv": C.FAIL}, {"dil": C.FAIL}, {"run": C.FAIL}):
        v, _, flags = C._verdict(*_gates(treynor=C.FAIL, **kw), risk_on=True)
        assert v == C.MIXED, kw
        assert len(flags) == 1


def test_verdict_every_pair_of_flags_condemns_an_unpaid_name():
    keys = ["idio", "surv", "dil", "run"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            kw = {keys[i]: C.FAIL, keys[j]: C.FAIL}
            v, _, flags = C._verdict(*_gates(treynor=C.FAIL, **kw), risk_on=True)
            assert v == C.RECKLESS, kw
            assert len(flags) == 2


def test_verdict_unpaid_beta_alone_is_only_MIXED():
    v, _, flags = C._verdict(*_gates(treynor=C.FAIL), risk_on=True)
    assert v == C.MIXED and flags == []


def test_verdict_a_paid_name_is_never_reckless():
    """Deliberate asymmetry: a name that WAS paid for its beta keeps the benefit of the
    doubt however ugly its history."""
    v, _, flags = C._verdict(*_gates(surv=C.FAIL, idio=C.FAIL, dil=C.FAIL, run=C.FAIL),
                             risk_on=True)
    assert v == C.MIXED
    assert len(flags) == 4          # the evidence is still reported


def test_verdict_advisory_gates_block_but_never_condemn():
    for kw in ({"trend": C.WARN}, {"rev": C.WARN}):
        assert C._verdict(*_gates(**kw), risk_on=True)[0] == C.MIXED
    # ...and they never enter the flag set, so they cannot help condemn either
    v, _, flags = C._verdict(*_gates(treynor=C.FAIL, trend=C.FAIL, rev=C.FAIL),
                             risk_on=True)
    assert v == C.MIXED and flags == []


def test_verdict_info_does_not_block_investable_but_is_reported():
    v, reasons, _ = C._verdict(*_gates(dil=C.INFO, run=C.INFO, rev=C.INFO), risk_on=True)
    assert v == C.INVESTABLE                     # unverifiable is not bad news
    assert any("dilution" in r for r in reasons)  # but it is still said out loud


def test_verdict_risk_off_still_stands_investable_aside():
    v, _, flags = C._verdict(*_gates(), risk_on=False)
    assert v == C.STAND_ASIDE and flags == []


# ---- end to end ----

def _universe(mkt):
    """12 names engineered so every verdict bucket is reachable.

    GOOD names carry clean fundamentals so they can actually reach `investable`; the
    rest are left without any, exercising the INFO path in the same run."""
    u = {}
    for i in range(4):        # paid for their beta, market-driven
        u[f"GOOD{i}"] = _name_data(mkt, beta=2.0, mu=0.0015 + 0.0001 * i,
                                   idio=0.004, seed=10 + i,
                                   fundamentals=CLEAN_FUNDAMENTALS)
    for i in range(4):        # not paid, and their own story is taking over
        u[f"BAD{i}"] = _name_data(mkt, beta=2.0, mu=-0.0015, idio=0.040,
                                  idio_last63=0.070, seed=20 + i)
    for i in range(2):        # unremarkable middle
        u[f"MID{i}"] = _name_data(mkt, beta=2.0, mu=0.0, idio=0.010, seed=30 + i)
    u["THIN"] = _name_data(mkt, beta=2.0, mu=0.0015, idio=0.004, adv=1e6, seed=40)
    u["SLOW"] = _name_data(mkt, beta=0.5, mu=0.0015, idio=0.004, seed=41)
    return u


_OPINIONS = {C.INVESTABLE, C.MIXED, C.RECKLESS, C.STAND_ASIDE}


def test_classify_universe_reaches_every_bucket():
    mkt = _mkt(300, 0.01, 50)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    verdicts = {r.symbol: r.verdict for r in res.results}

    assert verdicts["THIN"] == C.EXCLUDED_ILLIQUID
    assert verdicts["SLOW"] == C.EXCLUDED_LOW_BETA
    assert res.quartile_mode == C.QUARTILE_RELATIVE
    assert C.INVESTABLE in verdicts.values()
    assert C.RECKLESS in verdicts.values()
    # Only names that cleared both hard gates get an opinion, and every one of them does.
    assert res.n_eligible == sum(v in _OPINIONS for v in verdicts.values())
    # A name that is not paid for its beta and is drowning in its own story is never
    # blessed -- though a 63d beta this noisy may legitimately fail the beta gate too.
    assert not any(verdicts[f"BAD{i}"] == C.INVESTABLE for i in range(4))


def test_quartile_gate_passes_only_the_top_quarter():
    mkt = _mkt(300, 0.01, 51)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    passed = [r for r in res.results
              if r.by_key("treynor") and r.by_key("treynor").grade == C.PASS]
    # Percentile rank of the k-th best of n is k/n, and the gate is inclusive at 0.75,
    # so the pass count is n - ceil(0.75n) + 1 -- derived, never hardcoded, because the
    # eligible count depends on which names survive the hard gates.
    n = res.n_eligible
    assert n >= C.MIN_UNIVERSE_FOR_QUARTILE
    assert len(passed) == n - math.ceil((1.0 - C.TREYNOR_TOP_FRAC) * n) + 1


def test_results_are_sorted_by_verdict_then_treynor():
    mkt = _mkt(300, 0.01, 52)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    order = [C.VERDICT_ORDER.index(r.verdict) for r in res.results]
    assert order == sorted(order)


def test_risk_off_stands_investable_names_aside_without_condemning_them():
    mkt = _mkt(300, 0.01, 50)
    uni = _universe(mkt)
    on = C.classify_universe(uni, np.linspace(100.0, 200.0, 300))
    off = C.classify_universe(uni, np.linspace(200.0, 100.0, 300))

    v_on = {r.symbol: r.verdict for r in on.results}
    v_off = {r.symbol: r.verdict for r in off.results}
    investable = [s for s, v in v_on.items() if v == C.INVESTABLE]
    assert investable                                     # the test needs something to move
    assert all(v_off[s] == C.STAND_ASIDE for s in investable)
    assert not any(v == C.INVESTABLE for v in v_off.values())
    # a bad name is not rehabilitated by a bad tape, and vice versa
    for s, v in v_on.items():
        if v == C.RECKLESS:
            assert v_off[s] == C.RECKLESS
    assert off.risk_on is False


def test_small_universe_uses_the_absolute_fallback():
    mkt = _mkt(300, 0.01, 53)
    uni = {"A": _name_data(mkt, beta=2.0, mu=0.0015, idio=0.004, seed=60),
           "B": _name_data(mkt, beta=2.0, mu=-0.0015, idio=0.004, seed=61),
           "C": _name_data(mkt, beta=2.0, mu=0.0020, idio=0.004, seed=62)}
    res = C.classify_universe(uni, np.linspace(100.0, 200.0, 300))
    assert res.quartile_mode == C.QUARTILE_ABSOLUTE
    grades = {r.symbol: r.by_key("treynor").grade for r in res.results}
    assert grades["A"] == C.PASS and grades["C"] == C.PASS   # both beat cash
    assert grades["B"] == C.FAIL


def test_a_bad_ticker_becomes_an_error_row_without_killing_the_run():
    mkt = _mkt(300, 0.01, 54)
    short = _mkt(100, 0.01, 55)
    uni = {"OK": _name_data(mkt, beta=2.0, mu=0.001, seed=70),
           "SHORT": {"closes": 100.0 * np.exp(np.cumsum(short)),
                     "volumes": np.full(100, 1e6),
                     "name_rets": short, "mkt_rets": short}}
    res = C.classify_universe(uni, np.linspace(100.0, 200.0, 300))
    by_sym = {r.symbol: r for r in res.results}
    assert by_sym["SHORT"].verdict == C.ERROR
    assert by_sym["SHORT"].reasons and "ValueError" in by_sym["SHORT"].reasons[0]
    assert by_sym["OK"].verdict != C.ERROR


def test_custom_thresholds_flow_through():
    mkt = _mkt(300, 0.01, 56)
    uni = {"X": _name_data(mkt, beta=1.8, mu=0.001, idio=0.004, adv=12e6, seed=80)}
    assert C.classify_universe(uni, np.linspace(100.0, 200.0, 300),
                               min_beta=2.5).results[0].verdict == C.EXCLUDED_LOW_BETA
    assert C.classify_universe(uni, np.linspace(100.0, 200.0, 300),
                               min_adv=20e6).results[0].verdict == C.EXCLUDED_ILLIQUID


def test_payload_is_json_safe_and_carries_the_banner():
    mkt = _mkt(300, 0.01, 57)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    payload = res.to_dict()
    # The GOOD names are self-funding (infinite runway); json.dumps(inf) emits invalid
    # JSON, so this asserts the gate clamped it before it reached the payload.
    json.dumps(payload)
    assert "NOT PREDICTION" in payload["banner"]
    assert payload["regime"]["key"] == "regime"
    assert len(payload["results"]) == 12


def test_eligible_rows_carry_all_ten_gates_in_order():
    mkt = _mkt(300, 0.01, 58)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    opinion = [r for r in res.results if r.verdict in _OPINIONS]
    assert opinion
    for r in opinion:
        assert [g.key for g in r.gates] == ["adv", "beta", "regime", "treynor", "idio",
                                            "survivable", "own_trend", "dilution",
                                            "runway", "revenue"]


def test_new_metrics_are_present_and_finite():
    mkt = _mkt(300, 0.01, 59)
    res = C.classify_universe(_universe(mkt), np.linspace(100.0, 200.0, 300))
    for r in res.results:
        if r.verdict not in _OPINIONS:
            continue
        for k in ("halved_126d", "max_dd", "own_sma_gap"):
            assert k in r.metrics and np.isfinite(r.metrics[k]), (r.symbol, k)
        if r.symbol.startswith("GOOD"):        # these were given fundamentals
            assert r.metrics["dilution_yoy"] == pytest.approx(0.01)
            assert r.metrics["runway_quarters"] == C.RUNWAY_DISPLAY_CAP
        else:                                   # these were not -- INFO, no metric
            assert "dilution_yoy" not in r.metrics


def test_a_pre_revenue_diluter_burning_cash_is_reckless():
    """The condemning profile: unpaid beta plus TWO corroborating flags. 600 bars so
    the IPO cap does not apply, and no revenue to earn either the dilution or runway
    cap -- which is what separates this from the SOFI case below."""
    mkt = _mkt(600, 0.01, 60)
    diluter = _name_data(mkt, beta=2.0, mu=-0.0015, idio=0.010, seed=20,
                         fundamentals={"dilution_yoy": 0.45, "runway_quarters": 2.0,
                                       "revenue_ttm": 1e6, "n_revenue_quarters": 4})
    r = C.classify_universe({"DIL": diluter}, np.linspace(100.0, 200.0, 600)).results[0]
    assert r.by_key("dilution").grade == C.FAIL
    assert r.by_key("runway").grade == C.FAIL
    assert r.by_key("treynor").grade == C.FAIL
    assert r.verdict == C.RECKLESS
    assert sorted(r.metrics["flags"]) == ["dilution", "runway"]


def test_a_revenue_backed_diluter_is_MIXED_not_reckless():
    """The SOFI shape: unpaid beta and heavy issuance, but a real business underneath.
    Both fundamentals gates recognise their blind spot, leaving no flags to corroborate."""
    mkt = _mkt(600, 0.01, 61)
    # mu=-0.002: decisively negative Treynor, but a gentle enough slide that no 126d
    # window halves (survivability stays WARN, so it contributes no flag).
    sofi = _name_data(mkt, beta=2.5, mu=-0.002, idio=0.010, seed=21,
                      fundamentals={"dilution_yoy": 0.16, "runway_quarters": 4.1,
                                    "revenue_ttm": 3.9e9, "n_revenue_quarters": 4})
    r = C.classify_universe({"SOFI": sofi}, np.linspace(100.0, 200.0, 600)).results[0]
    assert r.by_key("dilution").grade == C.WARN
    assert r.by_key("runway").grade == C.WARN
    assert r.by_key("treynor").grade == C.FAIL
    assert r.metrics["flags"] == []
    assert r.verdict == C.MIXED


def test_a_recent_ipo_is_not_condemned_for_going_public():
    """The CRWV case. Sixteen months of history, heavy apparent share growth and a
    cash-burning growth model -- but a real revenue base. Both fundamentals gates
    recognise their blind spot, so nothing FAILs and the name stays mixed."""
    mkt = _mkt(330, 0.01, 62)                   # ~16 months, under DILUTION_MIN_BARS
    ipo = _name_data(mkt, beta=2.9, mu=-0.001, idio=0.010, seed=63,
                     fundamentals={"dilution_yoy": 0.27, "runway_quarters": 2.0,
                                   "revenue_ttm": 5e9, "n_revenue_quarters": 4})
    r = C.classify_universe({"IPO": ipo}, np.linspace(100.0, 200.0, 330)).results[0]
    assert r.by_key("dilution").grade == C.WARN      # capped: cannot read across an IPO
    assert r.by_key("runway").grade == C.WARN        # capped: revenue-backed burn
    assert r.by_key("revenue").grade == C.PASS
    assert r.verdict == C.MIXED                 # withheld, not condemned


def test_a_downtrending_name_cannot_be_investable_but_is_not_reckless():
    mkt = _mkt(300, 0.01, 61)
    # A GENTLE decline: below its own 200d SMA (trend WARN) and unpaid for its beta
    # (treynor FAIL), but no 126d window halves, so nothing condemns it. Steepen this
    # to mu=-0.004 and 24% of windows halve -- at which point `reckless` is the correct
    # answer, not a bug.
    falling = _name_data(mkt, beta=2.0, mu=-0.002, idio=0.004, seed=70,
                         fundamentals=CLEAN_FUNDAMENTALS)
    res = C.classify_universe({"FALL": falling}, np.linspace(100.0, 200.0, 300))
    r = res.results[0]
    assert r.by_key("own_trend").grade == C.WARN
    assert r.verdict != C.INVESTABLE
    assert r.verdict != C.RECKLESS      # survivability and fundamentals are clean
