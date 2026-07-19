"""The DCF, the 30 checks, the snowflake, and the assembled report.

Pure inputs throughout: every test hands `company_checks` the plain dict that
`company.reduce_raw` produces, so nothing here needs pandas or a network.

The recurring assertion is the tri-state rule: a check whose inputs are missing must
grade "info", never "fail". Grading a coverage gap as a failure would quietly
condemn every ADR and recent listing.
"""

from __future__ import annotations

import json

import pytest

from src.forecast import company_checks as C
from src.forecast.company_report import build_report


def _d(**over):
    """A healthy, profitable, dividend-paying company. Override to break one thing."""
    base = {
        "symbol": "ACME", "as_of": "2026-07-19",
        "name": "Acme Corp", "sector": "Technology", "price": 60.0,
        "currency": "USD", "financial_currency": "USD",
        "market_cap": 6.0e9, "shares_outstanding": 100e6, "price_series": [],
        "trailing_pe": 18.0, "forward_pe": 15.0, "price_to_book": 2.4,
        "trailing_eps": 3.1, "forward_eps": 3.8, "peg": None,
        "target_mean": 72.0, "target_high": 90.0, "target_low": 55.0,
        "n_analysts": 12.0,
        "fiscal_years": ["2025", "2024", "2023", "2022"],
        "revenue": [400e6, 350e6, 300e6, 250e6],
        "net_income": [40e6, 30e6, 25e6, 20e6],
        "ebit": [55e6, 45e6, 38e6, 30e6],
        "interest_expense": [4e6, 4e6, 4e6, 4e6],
        "equity": [200e6, 170e6, 150e6, 130e6],
        "total_assets": [500e6, 450e6, 400e6, 360e6],
        "current_assets": [180e6, 160e6, 150e6, 140e6],
        "current_liabilities": [90e6, 85e6, 80e6, 78e6],
        "total_liabilities": [300e6, 280e6, 250e6, 230e6],
        "total_debt": [60e6, 70e6, 75e6, 80e6],
        "cash": [50e6, 40e6, 35e6, 30e6],
        "fcf": [45e6, 38e6, 30e6, 24e6],
        "ocf": [70e6, 60e6, 50e6, 42e6],
        "dividends_paid": [-12e6, -11e6, -10e6, -9e6],
        "eps_growth_1y": 0.15, "rev_growth_1y": 0.116,
        "eps_now": 3.4, "eps_next": 3.9, "rev_now": 430e6, "rev_next": 480e6,
        "dividend_yield_info": 0.02, "trailing_div_rate": 1.2,
        "payout_ratio_info": 0.30,
        "dps_by_year": {str(y): 1.0 + (y - 2015) * 0.05 for y in range(2015, 2026)},
        "roe_info": None, "officers": [], "insider_pct": 0.02,
        "institution_pct": 0.61, "top_institutions": [],
        "insider_buys_12m": 1, "insider_sells_12m": 1,
        "insider_net_shares_12m": -3000.0,
    }
    base.update(over)
    return base


def _by(section, key):
    return next(c for c in section["checks"] if c["key"] == key)


def _grade(section, key):
    return _by(section, key)["grade"]


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

def test_dcf_matches_a_hand_computed_case():
    # FCF 100, flat 10% growth for 5y, fading to 2.5% over 5 more, r=9%, 100 shares.
    d = _d(fcf=[100.0], shares_outstanding=100.0, eps_growth_1y=0.10, revenue=[])
    r = C.DISCOUNT_RATE
    fcf, pv = 100.0, 0.0
    for t in range(1, 6):
        fcf *= 1.10
        pv += fcf / (1 + r) ** t
    for j in range(1, 6):
        g = 0.10 + (C.TERMINAL_GROWTH - 0.10) * (j / 5)
        fcf *= (1 + g)
        pv += fcf / (1 + r) ** (5 + j)
    pv += fcf * (1 + C.TERMINAL_GROWTH) / (r - C.TERMINAL_GROWTH) / (1 + r) ** 10

    got = C.dcf_fair_value(d)
    assert got["fair_value"] == pytest.approx(pv / 100.0)
    assert got["growth_used"] == pytest.approx(0.10)
    assert got["growth_source"] == "analyst EPS estimate"


def test_dcf_refuses_negative_free_cash_flow():
    assert C.dcf_fair_value(_d(fcf=[-10e6, -12e6])) is None


# ---- free-cash-flow base ----

def test_fcf_base_is_the_latest_year_when_it_is_representative():
    assert C.fcf_base([45e6, 38e6, 30e6, 24e6]) == (45e6, "latest reported year")


def test_a_capex_year_is_averaged_rather_than_compounded():
    # The Amazon 2025 shape: one collapsed year after three strong ones. Growing the
    # collapsed figure for ten years would price the company at a fraction of reality.
    base, why = C.fcf_base([7.7e9, 32.9e9, 32.2e9, 20.0e9])
    assert base == pytest.approx((7.7 + 32.9 + 32.2 + 20.0) / 4 * 1e9)
    assert "average" in why and "below" in why


def test_a_short_history_is_taken_at_face_value():
    # Two years is not enough to call anything an outlier.
    assert C.fcf_base([5e6, 40e6]) == (5e6, "latest reported year")


def test_a_negative_latest_year_is_averaged_into_the_cycle():
    # One loss year among three good ones still averages positive, so it is treated
    # as part of the cycle rather than thrown away.
    base, why = C.fcf_base([-5e6, 40e6, 44e6, 48e6])
    assert base == pytest.approx((-5 + 40 + 44 + 48) / 4 * 1e6) and "average" in why


def test_a_mostly_lossmaking_history_uses_only_the_profitable_years():
    base, why = C.fcf_base([-50e6, -60e6, 20e6, 24e6])       # overall mean is negative
    assert base == pytest.approx(22e6) and "profitable years" in why


def test_fcf_base_is_none_when_every_year_lost_money():
    assert C.fcf_base([-5e6, -8e6, -9e6]) == (None, "")
    assert C.fcf_base([]) == (None, "")


def test_the_dcf_reports_which_base_it_used():
    assert C.dcf_fair_value(_d())["fcf_basis"] == "latest reported year"


def test_dcf_refuses_a_currency_mismatch():
    # Financials in EUR against a USD price would be wrong by the exchange rate.
    assert C.dcf_fair_value(_d(financial_currency="EUR")) is None


def test_dcf_refuses_without_a_share_count():
    assert C.dcf_fair_value(_d(shares_outstanding=None)) is None


def test_growth_falls_back_through_its_sources():
    assert C.growth_estimate(_d())[1] == "analyst EPS estimate"
    assert C.growth_estimate(_d(eps_growth_1y=None))[1] == "analyst revenue estimate"
    hist = C.growth_estimate(_d(eps_growth_1y=None, rev_growth_1y=None))
    assert hist[1] == "historical revenue CAGR"
    assert hist[0] == pytest.approx((400 / 250) ** (1 / 3) - 1)
    bare = C.growth_estimate(_d(eps_growth_1y=None, rev_growth_1y=None, revenue=[]))
    assert bare == (C.DEFAULT_GROWTH, "default (no estimate available)")


def test_growth_is_clamped_against_absurd_estimates():
    assert C.growth_estimate(_d(eps_growth_1y=3.0))[0] == C.GROWTH_CLAMP[1]
    assert C.growth_estimate(_d(eps_growth_1y=-0.9))[0] == C.GROWTH_CLAMP[0]


# ---------------------------------------------------------------------------
# VALUE
# ---------------------------------------------------------------------------

def test_value_flags_a_price_below_fair_value():
    fv = C.dcf_fair_value(_d())["fair_value"]
    cheap = C.value_section(_d(price=fv * 0.5))
    assert _grade(cheap, "below_fair_value") == "pass"
    assert _grade(cheap, "significantly_below_fair_value") == "pass"
    dear = C.value_section(_d(price=fv * 1.5))
    assert _grade(dear, "below_fair_value") == "fail"


def test_value_marks_a_narrow_discount_as_not_significant():
    fv = C.dcf_fair_value(_d())["fair_value"]
    s = C.value_section(_d(price=fv * 0.9))          # inside the 20% margin
    assert _grade(s, "below_fair_value") == "pass"
    assert _grade(s, "significantly_below_fair_value") == "fail"


def test_pe_boundaries():
    assert _grade(C.value_section(_d(trailing_pe=C.MARKET_PE - 0.1)),
                  "pe_vs_market") == "pass"
    assert _grade(C.value_section(_d(trailing_pe=C.MARKET_PE + 0.1)),
                  "pe_vs_market") == "fail"


def test_a_loss_maker_fails_pe_rather_than_going_unknown():
    s = C.value_section(_d(trailing_pe=None, trailing_eps=-1.2))
    c = _by(s, "pe_vs_market")
    assert c["grade"] == "fail" and "loss-making" in c["display"]


def test_pe_is_info_when_there_are_no_earnings_figures_at_all():
    s = C.value_section(_d(trailing_pe=None, trailing_eps=None))
    assert _grade(s, "pe_vs_market") == "info"


def test_peg_is_recomputed_when_the_vendor_omits_it():
    s = C.value_section(_d(peg=None, forward_pe=12.0, eps_growth_1y=0.15))
    assert _grade(s, "peg") == "pass"                # 12 / 15 = 0.8
    assert s["stats"]["peg"] == pytest.approx(0.8)


def test_peg_is_info_without_forecast_growth():
    assert _grade(C.value_section(_d(peg=None, eps_growth_1y=None,
                                     rev_growth_1y=None)), "peg") == "info"


def test_analyst_upside_needs_enough_analysts():
    assert _grade(C.value_section(_d(n_analysts=2.0)), "analyst_upside") == "info"
    assert _grade(C.value_section(_d(target_mean=61.0)), "analyst_upside") == "fail"
    assert _grade(C.value_section(_d(target_mean=80.0)), "analyst_upside") == "pass"


def test_value_is_all_info_when_nothing_is_known():
    s = C.value_section(_d(price=None, fcf=[], trailing_pe=None, trailing_eps=None,
                           peg=None, forward_pe=None, price_to_book=None,
                           eps_growth_1y=None, rev_growth_1y=None,
                           target_mean=None, n_analysts=None))
    assert s["score"] == 0 and s["n_evaluable"] == 0
    assert all(c["grade"] == "info" for c in s["checks"])


# ---------------------------------------------------------------------------
# FUTURE
# ---------------------------------------------------------------------------

def test_future_scores_growth_thresholds():
    assert C.future_section(_d(eps_growth_1y=0.18))["score"] == 6
    # Thresholds are strict: exactly 15% is not "above 15%".
    assert _grade(C.future_section(_d(eps_growth_1y=C.EPS_GROWTH_HIGH)),
                  "earnings_growth_high") == "fail"
    slow = C.future_section(_d(eps_growth_1y=0.05, rev_growth_1y=0.05))
    assert _grade(slow, "earnings_growth_positive") == "pass"
    assert _grade(slow, "earnings_growth_high") == "fail"
    assert _grade(slow, "revenue_growth_high") == "fail"


def test_future_is_unscored_without_analyst_coverage():
    s = C.future_section(_d(eps_growth_1y=None, rev_growth_1y=None,
                            eps_now=None, eps_next=None))
    assert s["score"] == 0 and s["n_evaluable"] == 0


def test_a_loss_maker_forecast_to_stay_lossy_fails_profitability():
    s = C.future_section(_d(eps_next=-0.4, eps_now=-1.0))
    assert _grade(s, "becoming_profitable") == "fail"
    assert _grade(s, "eps_improving") == "pass"     # losing less is still improving


# ---------------------------------------------------------------------------
# PAST
# ---------------------------------------------------------------------------

def test_past_scores_a_clean_record():
    s = C.past_section(_d())
    assert _grade(s, "profitable") == "pass"
    assert _grade(s, "earnings_grew") == "pass"
    assert _grade(s, "revenue_grew") == "pass"
    assert _grade(s, "roe") == "pass"               # 40/200 = 20%
    assert _grade(s, "roce_improving") == "pass"


def test_turning_profitable_counts_as_growth():
    c = _by(C.past_section(_d(net_income=[10e6, -5e6, -8e6, -9e6])), "earnings_grew")
    assert c["grade"] == "pass" and "became profitable" in c["display"]


def test_two_loss_years_is_a_fail_not_an_unknown():
    assert _grade(C.past_section(_d(net_income=[-9e6, -5e6])), "earnings_grew") == "fail"


def test_acceleration_needs_three_profitable_years():
    assert _grade(C.past_section(_d(net_income=[40e6, 30e6])),
                  "growth_accelerating") == "info"
    # 33% last year against a ~26% three-year average
    assert _grade(C.past_section(_d()), "growth_accelerating") == "pass"


def test_roe_falls_back_to_the_vendor_figure():
    s = C.past_section(_d(equity=[], roe_info=0.22))
    assert _grade(s, "roe") == "pass"


def test_roce_needs_three_years_of_history():
    short = _d(ebit=[55e6, 45e6], total_assets=[500e6, 450e6],
               current_liabilities=[90e6, 85e6])
    assert _grade(C.past_section(short), "roce_improving") == "info"


def test_past_is_unscored_without_statements():
    s = C.past_section(_d(net_income=[], revenue=[], equity=[], total_assets=[],
                          ebit=[], current_liabilities=[], roe_info=None))
    assert s["score"] == 0 and s["n_evaluable"] == 0


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

def test_health_scores_a_solid_balance_sheet():
    s = C.health_section(_d())
    assert _grade(s, "short_term_liabilities") == "pass"     # 180 > 90 due within a year
    assert _grade(s, "debt_reduction") == "pass"             # 62% -> 30% over 3 years
    assert _grade(s, "debt_coverage") == "pass"              # OCF 70 vs debt 60
    assert _grade(s, "interest_coverage") == "pass"          # 55/4 = 13.8x


def test_long_term_liability_cover_compares_against_current_assets():
    # total 300 - current 90 = 210 long-term, against 180 current assets
    assert _grade(C.health_section(_d()), "long_term_liabilities") == "fail"
    covered = _d(total_liabilities=[200e6, 190e6, 180e6, 170e6])   # 110 long-term
    assert _grade(C.health_section(covered), "long_term_liabilities") == "pass"


def test_debt_level_boundary():
    tight = _d(total_debt=[79e6], equity=[200e6])            # 39.5%
    loose = _d(total_debt=[81e6], equity=[200e6])            # 40.5%
    assert _grade(C.health_section(tight), "debt_level") == "pass"
    assert _grade(C.health_section(loose), "debt_level") == "fail"


def test_a_debt_free_company_passes_the_debt_checks_explicitly():
    s = C.health_section(_d(total_debt=[], interest_expense=[]))
    for key in ("debt_level", "debt_reduction", "debt_coverage", "interest_coverage"):
        assert _grade(s, key) == "pass", key
    assert s["stats"]["debt_free"] is True


def test_a_leveraged_company_with_no_interest_line_is_info_not_a_free_pass():
    s = C.health_section(_d(interest_expense=[]))
    assert _grade(s, "interest_coverage") == "info"


def test_health_is_unscored_without_a_balance_sheet():
    s = C.health_section(_d(current_assets=[], current_liabilities=[],
                            total_liabilities=[], total_debt=[], equity=[],
                            ocf=[], ebit=[], interest_expense=[]))
    assert s["score"] == 0 and s["n_evaluable"] == 0


# ---------------------------------------------------------------------------
# DIVIDEND
# ---------------------------------------------------------------------------

def test_a_non_payer_fails_every_dividend_check_and_is_flagged():
    s = C.dividend_section(_d(dps_by_year={}, dividend_yield_info=None,
                              trailing_div_rate=None))
    assert s["pays_dividend"] is False and s["score"] == 0
    assert all(c["grade"] == "fail" for c in s["checks"])
    assert s["series"] == {"years": [], "dps": []}


def test_yield_is_derived_from_the_trailing_rate_not_the_ambiguous_field():
    # dividendYield flipped between 0.02 and 2.0 forms across yfinance versions;
    # rate/price is unambiguous, so it wins.
    s = C.dividend_section(_d(trailing_div_rate=3.0, price=60.0,
                              dividend_yield_info=99.0))
    assert s["stats"]["yield"] == pytest.approx(0.05)
    assert _grade(s, "yield_high") == "pass"


def test_percent_form_yield_is_normalised_when_it_is_all_we_have():
    s = C.dividend_section(_d(trailing_div_rate=None, dps_by_year={},
                              dividend_yield_info=5.0))
    assert s["stats"]["yield"] == pytest.approx(0.05)


def test_a_young_payer_is_not_yet_stable():
    s = C.dividend_section(_d(dps_by_year={"2023": 1.0, "2024": 1.1, "2025": 1.2}))
    c = _by(s, "stable")
    assert c["grade"] == "fail" and "3 years" in c["display"]
    assert _grade(s, "growing") == "info"           # under five years to compare


def test_a_cut_breaks_stability():
    dps = {str(y): 1.0 for y in range(2015, 2026)}
    dps["2022"] = 0.5
    c = _by(C.dividend_section(_d(dps_by_year=dps)), "stable")
    assert c["grade"] == "fail" and "cut by more than" in c["display"]


def test_a_skipped_year_breaks_stability():
    dps = {str(y): 1.0 for y in range(2015, 2026) if y != 2020}
    c = _by(C.dividend_section(_d(dps_by_year=dps)), "stable")
    assert c["grade"] == "fail" and "skipped" in c["display"]


def test_an_unbroken_growing_record_passes():
    s = C.dividend_section(_d())
    assert _grade(s, "stable") == "pass" and _grade(s, "growing") == "pass"


def test_payout_falls_back_to_dividends_over_earnings():
    s = C.dividend_section(_d(payout_ratio_info=None))
    assert s["stats"]["payout_ratio"] == pytest.approx(12 / 40)
    assert _grade(s, "earnings_coverage") == "pass"


def test_dividends_paid_out_of_negative_cash_flow_fail():
    c = _by(C.dividend_section(_d(fcf=[-5e6])), "cash_flow_coverage")
    assert c["grade"] == "fail" and "free cash flow is" in c["display"]


def test_cash_flow_coverage_boundary():
    assert _grade(C.dividend_section(_d(dividends_paid=[-89e6], fcf=[100e6])),
                  "cash_flow_coverage") == "pass"
    assert _grade(C.dividend_section(_d(dividends_paid=[-91e6], fcf=[100e6])),
                  "cash_flow_coverage") == "fail"


# ---------------------------------------------------------------------------
# management + ownership
# ---------------------------------------------------------------------------

def test_the_ceo_is_picked_out_of_the_officer_list():
    s = C.management_section(_d(officers=[
        {"name": "A B", "title": "Chief Financial Officer", "age": 50, "pay": 1e6},
        {"name": "C D", "title": "CEO & Director", "age": 60, "pay": 5e6}]))
    assert s["ceo"]["name"] == "C D" and s["n_officers"] == 2


def test_missing_officers_are_an_empty_list_not_an_error():
    assert C.management_section(_d(officers=[])) == {
        "officers": [], "ceo": None, "n_officers": 0}


def test_insider_bias_reads_the_net_share_count():
    assert C.ownership_section(_d(insider_net_shares_12m=500.0))["insider_bias"] \
        == "buying"
    assert C.ownership_section(_d())["insider_bias"] == "selling"
    assert C.ownership_section(_d(insider_buys_12m=None,
                                  insider_sells_12m=None))["insider_bias"] is None


# ---------------------------------------------------------------------------
# snowflake
# ---------------------------------------------------------------------------

def test_snowflake_axes_are_fixed_and_score_by_passes():
    d = _d()
    sections = {"value": C.value_section(d), "future": C.future_section(d),
                "past": C.past_section(d), "health": C.health_section(d),
                "dividend": C.dividend_section(d)}
    snow = C.snowflake(sections)
    assert [a["key"] for a in snow["axes"]] == \
        ["value", "future", "past", "health", "dividend"]
    assert all(a["max"] == C.CHECKS_PER_AXIS for a in snow["axes"])
    assert snow["total"] == sum(sections[k]["score"] for k in
                                ["value", "future", "past", "health", "dividend"])
    assert snow["max"] == 30


def test_snowflake_summary_names_the_weak_axis():
    sections = {k: {"score": s} for k, s in
                [("value", 5), ("future", 5), ("past", 5), ("health", 5),
                 ("dividend", 0)]}
    assert "pays little or nothing" in C.snowflake(sections)["summary"]


def test_snowflake_summary_calls_out_a_uniformly_bad_scorecard():
    sections = {k: {"score": 1} for k, _ in C.AXES}
    assert "caution" in C.snowflake(sections)["summary"]


# ---------------------------------------------------------------------------
# assembled report
# ---------------------------------------------------------------------------

def test_report_is_serialisable_and_complete():
    r = build_report("ACME", fetcher=lambda sym, **kw: _d())
    assert r["symbol"] == "ACME"
    assert set(r["sections"]) == {"value", "future", "past", "health", "dividend",
                                  "management", "ownership"}
    assert r["snowflake"]["total"] > 0 and r["disclaimer"]
    for key, _ in C.AXES:
        for c in r["sections"][key]["checks"]:
            assert c["grade"] in ("pass", "fail", "info")
            assert c["title"] and c["threshold"]
    json.dumps(r)                                   # the API returns this untouched


def test_report_warns_about_absent_analyst_coverage():
    r = build_report("THIN", fetcher=lambda sym, **kw: _d(eps_growth_1y=None,
                                                          rev_growth_1y=None))
    assert any("No analyst forecasts" in w for w in r["warnings"])


def test_report_explains_a_missing_fair_value_by_currency():
    r = build_report("ADR", fetcher=lambda sym, **kw: _d(financial_currency="EUR"))
    assert any("reported in EUR" in w for w in r["warnings"])
    assert r["sections"]["value"]["dcf"] is None


def test_report_survives_a_ticker_with_only_a_price():
    bare = {k: v for k, v in _d().items()}
    for k in ("revenue", "net_income", "ebit", "equity", "total_assets",
              "current_assets", "current_liabilities", "total_liabilities",
              "total_debt", "cash", "fcf", "ocf", "dividends_paid",
              "interest_expense"):
        bare[k] = []
    bare.update({"eps_growth_1y": None, "rev_growth_1y": None, "eps_now": None,
                 "eps_next": None, "dps_by_year": {}, "dividend_yield_info": None,
                 "trailing_div_rate": None, "payout_ratio_info": None,
                 "trailing_pe": None, "trailing_eps": None, "peg": None,
                 "forward_pe": None, "price_to_book": None, "target_mean": None,
                 "n_analysts": None, "roe_info": None})
    r = build_report("BARE", fetcher=lambda sym, **kw: bare)
    assert r["snowflake"]["total"] == 0
    assert len(r["warnings"]) >= 3
    json.dumps(r)
