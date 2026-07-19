"""Fundamentals: derivations, frame reduction, and cache behaviour.

No test here touches the network -- the downloader is injected. The point is that
missing or malformed vendor data produces `None` (which grades INFO downstream), never
a plausible-looking wrong number.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from src.forecast import fundamentals as F


def _shares(pairs):
    """date -> count series, as yfinance's get_shares_full returns."""
    idx = [pd.Timestamp(d) for d, _ in pairs]
    return pd.Series([v for _, v in pairs], index=idx)


def _frame(rows: dict, cols):
    """Statement frame: rows are line items, columns are quarters (newest first)."""
    return pd.DataFrame(rows, index=[pd.Timestamp(c) for c in cols]).T


# ---- dilution ----

def test_dilution_is_share_growth():
    assert F.dilution_yoy(110.0, 100.0) == pytest.approx(0.10)
    assert F.dilution_yoy(90.0, 100.0) == pytest.approx(-0.10)     # buyback


def test_dilution_is_none_when_unknowable():
    assert F.dilution_yoy(None, 100.0) is None
    assert F.dilution_yoy(100.0, None) is None
    assert F.dilution_yoy(100.0, 0.0) is None
    assert F.dilution_yoy(100.0, -5.0) is None


# ---- runway ----

def test_runway_uses_only_burning_quarters():
    # One good quarter must not cancel the burn: mean of the negatives is -10.
    assert F.runway_quarters(100.0, [-10.0, -10.0, 50.0, -10.0]) == pytest.approx(10.0)


def test_runway_is_infinite_when_self_funding():
    assert F.runway_quarters(100.0, [5.0, 10.0, 2.0]) == float("inf")


def test_runway_is_none_without_inputs():
    assert F.runway_quarters(None, [-10.0]) is None
    assert F.runway_quarters(100.0, []) is None
    assert F.runway_quarters(100.0, None) is None


def test_runway_is_zero_when_cash_is_gone():
    assert F.runway_quarters(0.0, [-10.0]) == 0.0


# ---- share series reduction ----

def test_shares_picks_latest_and_closest_to_a_year_before():
    s = _shares([("2025-07-01", 100.0), ("2026-01-01", 130.0), ("2026-07-01", 150.0)])
    now, ago = F.shares_now_and_year_ago(s)
    assert now == 150.0 and ago == 100.0


def test_shares_drops_zero_and_nan_rows():
    s = _shares([("2025-07-01", 100.0), ("2026-06-01", 0.0), ("2026-07-01", 150.0)])
    now, ago = F.shares_now_and_year_ago(s)
    assert now == 150.0 and ago == 100.0        # the 0.0 row is not "the latest"


def test_shares_year_ago_is_none_when_nothing_is_near_a_year_back():
    # Oldest observation is only ~5 months before the latest: annualising that would
    # overstate dilution badly, so the gate must go unverified instead.
    s = _shares([("2026-02-01", 120.0), ("2026-07-01", 150.0)])
    now, ago = F.shares_now_and_year_ago(s)
    assert now == 150.0 and ago is None


def test_shares_handles_missing_series():
    assert F.shares_now_and_year_ago(None) == (None, None)


# ---- full reduction ----

def _full_kit():
    shares = _shares([("2025-07-01", 100e6), ("2026-07-01", 125e6)])
    balance = _frame({"Cash And Cash Equivalents": [50e6, 60e6],
                      "Other Short Term Investments": [10e6, 12e6]},
                     ["2026-06-30", "2026-03-31"])
    cashflow = _frame({"Free Cash Flow": [-20e6, -20e6, -20e6, -20e6]},
                      ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    income = _frame({"Total Revenue": [30e6, 25e6, 20e6, 15e6]},
                    ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    return shares, balance, cashflow, income


def test_reduce_frames_produces_plain_floats():
    r = F.reduce_frames(*_full_kit(), as_of=dt.date(2026, 7, 19))
    assert r["shares_now"] == 125e6 and r["shares_1y_ago"] == 100e6
    assert r["cash"] == 60e6                       # newest quarter: 50M + 10M
    assert r["fcf_quarters"] == [-20e6] * 4
    assert r["revenue_ttm"] == 90e6 and r["n_revenue_quarters"] == 4
    assert r["dilution_yoy"] == pytest.approx(0.25)
    assert r["runway_quarters"] == pytest.approx(3.0)
    json.dumps(r)                                   # must be cacheable as-is


def test_reduce_frames_matches_row_labels_loosely():
    shares, balance, cashflow, income = _full_kit()
    squashed = _frame({"FreeCashFlow": [-10e6, -10e6]}, ["2026-06-30", "2026-03-31"])
    r = F.reduce_frames(shares, balance, squashed, income)
    assert r["fcf_quarters"] == [-10e6, -10e6]      # "FreeCashFlow" == "Free Cash Flow"


def test_reduce_frames_falls_back_to_operating_cash_flow():
    shares, balance, _, income = _full_kit()
    ocf = _frame({"Operating Cash Flow": [-8e6, -8e6]}, ["2026-06-30", "2026-03-31"])
    r = F.reduce_frames(shares, balance, ocf, income)
    assert r["fcf_quarters"] == [-8e6, -8e6]


def test_sector_marks_where_free_cash_flow_is_meaningless():
    kit = _full_kit()
    bank = F.reduce_frames(*kit, sector="Financial Services")
    tech = F.reduce_frames(*kit, sector="Technology")
    unknown = F.reduce_frames(*kit)
    assert bank["sector"] == "Financial Services"
    assert bank["fcf_meaningful"] is False        # loan origination is not burn
    assert tech["fcf_meaningful"] is True
    assert unknown["fcf_meaningful"] is True      # absent sector must not disable a gate


def test_reduce_frames_survives_every_statement_missing():
    r = F.reduce_frames(None, None, None, None)
    assert r["shares_now"] is None and r["cash"] is None
    assert r["fcf_quarters"] == [] and r["revenue_ttm"] is None
    assert r["dilution_yoy"] is None and r["runway_quarters"] is None


def test_reduce_frames_keeps_revenue_when_cashflow_is_absent():
    shares, balance, _, income = _full_kit()
    r = F.reduce_frames(shares, balance, None, income)
    assert r["revenue_ttm"] == 90e6                 # one gap must not discard the rest
    assert r["runway_quarters"] is None


def test_reduce_frames_handles_cash_without_short_term_investments():
    shares, _, cashflow, income = _full_kit()
    bal = _frame({"Cash And Cash Equivalents": [40e6]}, ["2026-06-30"])
    r = F.reduce_frames(shares, bal, cashflow, income)
    assert r["cash"] == 40e6


# ---- fetch + cache ----

def _dl(kit=None):
    shares, balance, cashflow, income = kit or _full_kit()
    return lambda sym: {"shares": shares, "balance": balance,
                        "cashflow": cashflow, "income": income}


def test_fetch_writes_a_cache_that_short_circuits(tmp_path):
    path = tmp_path / "f.json"
    today = dt.date(2026, 7, 19)
    got, failed = F.fetch_fundamentals(["AAA"], path, downloader=_dl(), today=today)
    assert not failed and got["AAA"]["dilution_yoy"] == pytest.approx(0.25)

    def boom(sym):
        raise AssertionError("a fresh cache must not hit the network")

    again, _ = F.fetch_fundamentals(["AAA"], path, downloader=boom, today=today)
    assert again["AAA"] == got["AAA"]


def test_stale_cache_is_refetched(tmp_path):
    path = tmp_path / "f.json"
    F.fetch_fundamentals(["AAA"], path, downloader=_dl(), today=dt.date(2026, 7, 1))
    s = _shares([("2025-07-01", 100e6), ("2026-07-01", 200e6)])
    kit = (s,) + _full_kit()[1:]
    got, _ = F.fetch_fundamentals(["AAA"], path, downloader=_dl(kit),
                                  today=dt.date(2026, 7, 19))     # 18d > MAX_AGE_DAYS
    assert got["AAA"]["dilution_yoy"] == pytest.approx(1.0)


def test_refresh_forces_a_refetch_even_when_fresh(tmp_path):
    path = tmp_path / "f.json"
    today = dt.date(2026, 7, 19)
    F.fetch_fundamentals(["AAA"], path, downloader=_dl(), today=today)
    s = _shares([("2025-07-01", 100e6), ("2026-07-01", 300e6)])
    got, _ = F.fetch_fundamentals(["AAA"], path, downloader=_dl((s,) + _full_kit()[1:]),
                                  refresh=True, today=today)
    assert got["AAA"]["dilution_yoy"] == pytest.approx(2.0)


def test_one_bad_symbol_is_isolated(tmp_path):
    def flaky(sym):
        if sym == "BAD":
            raise OSError("no route to host")
        return _dl()(sym)

    got, failed = F.fetch_fundamentals(["AAA", "BAD", "CCC"], tmp_path / "f.json",
                                       downloader=flaky)
    assert failed == ["BAD"]
    assert set(got) == {"AAA", "CCC"}          # the run continues around the hole


def test_corrupt_cache_is_treated_as_absent(tmp_path):
    path = tmp_path / "f.json"
    path.write_text("{not json", encoding="utf-8")
    got, failed = F.fetch_fundamentals(["AAA"], path, downloader=_dl())
    assert not failed and "AAA" in got


def test_progress_callback_reports_each_symbol(tmp_path):
    seen = []
    F.fetch_fundamentals(["A", "B"], tmp_path / "f.json", downloader=_dl(),
                         progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]
