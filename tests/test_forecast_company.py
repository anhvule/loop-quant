"""Company data reduction and cache behaviour.

No test here touches the network -- the downloader is injected. The point throughout
is that patchy or drifting vendor data produces `None` (which grades "info"
downstream), never a plausible-looking wrong number.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from src.forecast import company as K

TODAY = dt.date(2026, 7, 19)


def _frame(rows: dict, cols):
    """Statement frame: rows are line items, columns are periods (newest first)."""
    return pd.DataFrame(rows, index=[pd.Timestamp(c) for c in cols]).T


def _series(pairs):
    return pd.Series([v for _, v in pairs], index=[pd.Timestamp(d) for d, _ in pairs])


def _info(**kw):
    base = {"longName": "Acme Corp", "sector": "Technology", "industry": "Widgets",
            "currency": "USD", "financialCurrency": "USD", "marketCap": 5.0e9,
            "sharesOutstanding": 100e6, "trailingPE": 18.0, "priceToBook": 2.4,
            "trailingEps": 3.1, "forwardEps": 3.8, "numberOfAnalystOpinions": 12,
            "targetMeanPrice": 70.0}
    base.update(kw)
    return base


def _kit(**over):
    years = ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"]
    raw = {
        "info": _info(),
        "prices": _series([("2026-06-28", 58.0), ("2026-07-05", 60.0)]),
        "income": _frame({"Total Revenue": [400e6, 350e6, 300e6, 250e6],
                          "Net Income": [40e6, 30e6, 25e6, 20e6],
                          "EBIT": [55e6, 45e6, 38e6, 30e6],
                          "Interest Expense": [4e6, 4e6, 4e6, 4e6]}, years),
        "balance": _frame({"Stockholders Equity": [200e6, 170e6, 150e6, 130e6],
                           "Total Assets": [500e6, 450e6, 400e6, 360e6],
                           "Current Assets": [180e6, 160e6, 150e6, 140e6],
                           "Current Liabilities": [90e6, 85e6, 80e6, 78e6],
                           "Total Liabilities Net Minority Interest":
                               [300e6, 280e6, 250e6, 230e6],
                           "Total Debt": [60e6, 70e6, 75e6, 80e6],
                           "Cash And Cash Equivalents": [50e6, 40e6, 35e6, 30e6]},
                          years),
        "cashflow": _frame({"Free Cash Flow": [45e6, 38e6, 30e6, 24e6],
                            "Operating Cash Flow": [70e6, 60e6, 50e6, 42e6],
                            "Cash Dividends Paid": [-12e6, -11e6, -10e6, -9e6]}, years),
        "dividends": _series([("2024-03-01", 0.5), ("2024-09-01", 0.5),
                              ("2025-03-01", 0.6), ("2025-09-01", 0.6),
                              ("2026-03-01", 0.7)]),
        "eps_est": pd.DataFrame({"avg": [3.4, 3.9], "growth": [0.10, 0.15]},
                                index=["0y", "+1y"]),
        "rev_est": pd.DataFrame({"avg": [430e6, 480e6], "growth": [0.075, 0.116]},
                                index=["0y", "+1y"]),
        "targets": {"current": 60.0, "mean": 72.0, "high": 90.0, "low": 55.0},
        "inst": pd.DataFrame({"Holder": ["Vanguard", "BlackRock"],
                              "pctHeld": [0.09, 0.07],
                              "Value": [450e6, 350e6]}),
        "major": pd.DataFrame({"Value": [0.02, 0.61]},
                              index=["insidersPercentHeld", "institutionsPercentHeld"]),
        "insider_tx": pd.DataFrame({
            "Shares": [1000, 4000],
            "Text": ["Purchase at price 50.00", "Sale at price 61.00"],
            "Start Date": [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")]}),
    }
    raw.update(over)
    return raw


# ---- overview + price ----

def test_reduce_produces_plain_json():
    r = K.reduce_raw(_kit(), as_of=TODAY, symbol="ACME")
    assert r["name"] == "Acme Corp" and r["sector"] == "Technology"
    assert r["price"] == 60.0                       # last weekly close, not info
    assert r["price_series"][-1] == {"d": "2026-07-05", "p": 60.0}
    assert r["market_cap"] == 5.0e9
    json.dumps(r)                                   # must be cacheable as-is


def test_price_falls_back_to_info_when_history_is_empty():
    r = K.reduce_raw(_kit(prices=None, info=_info(currentPrice=44.5)), as_of=TODAY)
    assert r["price"] == 44.5 and r["price_series"] == []


def test_summary_is_truncated_on_a_word_boundary():
    long = "word " * 400
    r = K.reduce_raw(_kit(info=_info(longBusinessSummary=long)), as_of=TODAY)
    assert len(r["summary"]) <= K.MAX_SUMMARY_CHARS + 1
    assert r["summary"].endswith("…")


def test_unknown_symbol_raises():
    with pytest.raises(K.CompanyDataError):
        K.reduce_raw({"info": {}, "prices": None}, as_of=TODAY, symbol="ZZZZ")


def test_a_price_alone_is_enough_to_report():
    # A ticker with a quote but no fundamentals still gets a (heavily "info") report.
    r = K.reduce_raw({"info": {"currentPrice": 12.0}}, as_of=TODAY, symbol="THIN")
    assert r["price"] == 12.0 and r["revenue"] == [] and r["officers"] == []


# ---- statement series ----

def test_annual_series_are_newest_first():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["revenue"] == [400e6, 350e6, 300e6, 250e6]
    assert r["net_income"][0] == 40e6
    assert r["fiscal_years"] == ["2025", "2024", "2023", "2022"]


def test_row_labels_match_across_yfinance_drift():
    years = ["2025-12-31", "2024-12-31"]
    alt = _frame({"Common Stock Equity": [210e6, 180e6],
                  "Total Assets": [500e6, 450e6]}, years)
    r = K.reduce_raw(_kit(balance=alt), as_of=TODAY)
    assert r["equity"] == [210e6, 180e6]            # not "Stockholders Equity"


def test_total_debt_is_summed_from_parts_when_absent():
    years = ["2025-12-31", "2024-12-31"]
    bal = _frame({"Stockholders Equity": [200e6, 170e6],
                  "Long Term Debt": [40e6, 50e6],
                  "Current Debt": [10e6, 12e6]}, years)
    r = K.reduce_raw(_kit(balance=bal), as_of=TODAY)
    assert r["total_debt"] == [50e6, 62e6]


def test_every_statement_missing_degrades_quietly():
    r = K.reduce_raw(_kit(income=None, balance=None, cashflow=None), as_of=TODAY)
    assert r["revenue"] == [] and r["equity"] == [] and r["fcf"] == []
    assert r["price"] == 60.0                       # one gap does not discard the rest


# ---- estimates ----

def test_estimates_are_read_by_row_and_column():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["eps_growth_1y"] == pytest.approx(0.15)
    assert r["rev_growth_1y"] == pytest.approx(0.116)
    assert r["eps_now"] == 3.4 and r["eps_next"] == 3.9


def test_missing_estimate_frames_are_none_not_zero():
    r = K.reduce_raw(_kit(eps_est=None, rev_est=None), as_of=TODAY)
    assert r["eps_growth_1y"] is None and r["eps_next"] is None


def test_targets_dict_wins_over_info():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["target_mean"] == 72.0                 # not info's 70.0
    r2 = K.reduce_raw(_kit(targets=None), as_of=TODAY)
    assert r2["target_mean"] == 70.0                # falls back to info


# ---- dividends ----

def test_dividends_sum_by_year_and_drop_the_partial_current_year():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["dps_by_year"] == {"2024": 1.0, "2025": 1.2}     # 2026 is incomplete


def test_no_dividend_history_is_an_empty_map():
    r = K.reduce_raw(_kit(dividends=None), as_of=TODAY)
    assert r["dps_by_year"] == {}


# ---- ownership + management ----

def test_major_holders_labelled_index_layout():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["insider_pct"] == pytest.approx(0.02)
    assert r["institution_pct"] == pytest.approx(0.61)


def test_major_holders_legacy_two_column_layout():
    legacy = pd.DataFrame([["1.85%", "% of Shares Held by All Insider"],
                           ["72.30%", "% of Shares Held by Institutions"]])
    r = K.reduce_raw(_kit(major=legacy), as_of=TODAY)
    assert r["insider_pct"] == pytest.approx(0.0185)
    assert r["institution_pct"] == pytest.approx(0.723)


def test_garbage_holders_frame_yields_none():
    r = K.reduce_raw(_kit(major=pd.DataFrame({"x": ["n/a"]})), as_of=TODAY)
    assert r["insider_pct"] is None and r["institution_pct"] is None


def test_top_institutions_are_capped_and_parsed():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["top_institutions"][0] == {"holder": "Vanguard", "pct": 0.09,
                                        "value": 450e6}


def test_insider_transactions_are_classified_and_netted():
    r = K.reduce_raw(_kit(), as_of=TODAY)
    assert r["insider_buys_12m"] == 1 and r["insider_sells_12m"] == 1
    assert r["insider_net_shares_12m"] == pytest.approx(-3000.0)


def test_insider_transactions_older_than_a_year_are_ignored():
    old = pd.DataFrame({"Shares": [9000], "Text": ["Sale at price 30.00"],
                        "Start Date": [pd.Timestamp("2024-01-01")]})
    r = K.reduce_raw(_kit(insider_tx=old), as_of=TODAY)
    assert r["insider_sells_12m"] == 0 and r["insider_net_shares_12m"] == 0.0


def test_officers_survive_a_wrapped_pay_field():
    info = _info(companyOfficers=[
        {"name": "Jane Roe", "title": "CEO", "age": 55, "totalPay": {"raw": 1.6e7}},
        {"name": "", "title": "ghost"},                       # dropped: no name
    ])
    r = K.reduce_raw(_kit(info=info), as_of=TODAY)
    assert r["officers"] == [{"name": "Jane Roe", "title": "CEO", "age": 55.0,
                              "pay": 1.6e7}]


# ---- fetch + cache ----

def _dl(raw=None):
    return lambda sym: raw or _kit()


def test_fetch_writes_a_cache_that_short_circuits(tmp_path):
    path = tmp_path / "c.json"
    first = K.fetch_company("ACME", path, downloader=_dl(), today=TODAY)
    assert first["cached"] is False and first["symbol"] == "ACME"

    def boom(sym):
        raise AssertionError("a fresh cache must not hit the network")

    again = K.fetch_company("acme", path, downloader=boom, today=TODAY)
    assert again["cached"] is True and again["price"] == first["price"]


def test_stale_cache_is_refetched(tmp_path):
    path = tmp_path / "c.json"
    K.fetch_company("ACME", path, downloader=_dl(), today=dt.date(2026, 7, 1))
    moved = _kit(prices=_series([("2026-07-12", 99.0)]))
    got = K.fetch_company("ACME", path, downloader=_dl(moved), today=TODAY)
    assert got["price"] == 99.0 and got["cached"] is False


def test_refresh_forces_a_refetch_even_when_fresh(tmp_path):
    path = tmp_path / "c.json"
    K.fetch_company("ACME", path, downloader=_dl(), today=TODAY)
    moved = _kit(prices=_series([("2026-07-12", 77.0)]))
    got = K.fetch_company("ACME", path, downloader=_dl(moved), refresh=True, today=TODAY)
    assert got["price"] == 77.0


def test_corrupt_cache_is_treated_as_absent(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    assert K.fetch_company("ACME", path, downloader=_dl(), today=TODAY)["price"] == 60.0


def test_empty_symbol_is_rejected(tmp_path):
    with pytest.raises(K.CompanyDataError):
        K.fetch_company("  ", tmp_path / "c.json", downloader=_dl())
