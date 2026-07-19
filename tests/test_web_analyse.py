"""The unified analysis: one fetch, two readings, optional forecast.

Network is stubbed at `_engine.fetch_prices` / `_engine.predict`, so these tests also
pin the property that motivated merging the three views in the first place: each ticker
is fetched EXACTLY ONCE, no matter how many readings consume it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))

import _analyse as A  # noqa: E402
import _classify as CL  # noqa: E402
import _engine as E  # noqa: E402


N_BARS = 800
_D0 = dt.date(2023, 1, 2)
_ALL_DATES = [_D0 + dt.timedelta(days=i) for i in range(N_BARS)]
# One shared market factor, so measured betas are real rather than noise between two
# independent random walks. Names are built from it: rets = mu + beta*market + idio.
_MKT_RETS = np.random.default_rng(7).normal(0.0, 0.01, N_BARS)
_SPY_PX = 400.0 * np.exp(np.cumsum(_MKT_RETS))

# symbol -> (beta, bars of history). Bar counts straddle the two floors on purpose:
#   YOUNG      below MIN_SCREEN_BARS  -> not scoreable at all
#   SHORTGATE  above it, below MIN_BARS_CLASSIFY -> scored but not gated
_SPEC = {"AAA": (2.2, N_BARS), "BBB": (2.6, N_BARS),
         "YOUNG": (2.0, A.MIN_SCREEN_BARS - 20),
         "SHORTGATE": (2.0, CL.MIN_BARS_CLASSIFY - 100)}


@pytest.fixture
def stub(monkeypatch):
    """Deterministic price source that counts fetches per symbol."""
    calls = {}

    def fake_fetch(symbol, rng=None):
        calls[symbol] = calls.get(symbol, 0) + 1
        if symbol == "BAD":
            raise E.DataError(f"'{symbol}' not found on Yahoo Finance (HTTP 404).")
        if symbol == "SPY":
            px, dates = _SPY_PX, _ALL_DATES
        else:
            beta, n = _SPEC.get(symbol, (2.0, N_BARS))
            r = np.random.default_rng(abs(hash(symbol)) % 1000)
            rets = 0.0008 + beta * _MKT_RETS + r.normal(0.0, 0.004, N_BARS)
            px = (100.0 * np.exp(np.cumsum(rets)))[-n:]
            dates = _ALL_DATES[-n:]          # recent tail, so it overlaps SPY
        vols = 1e9 / np.maximum(px, 1e-9)    # $1B/day: never the binding constraint
        info = {"name": symbol + " Inc", "currency": "USD", "exchange": "X"}
        return dates, px, px * 1.01, px * 0.99, vols, info

    def fake_predict(symbol, months=6, **kw):
        calls["predict:" + symbol] = calls.get("predict:" + symbol, 0) + 1
        return {"symbol": symbol, "months_requested": months, "spot": 1.0,
                "waves": {"available": False}}

    monkeypatch.setattr(E, "fetch_prices", fake_fetch)
    monkeypatch.setattr(A.E, "fetch_prices", fake_fetch)
    monkeypatch.setattr(A.E, "predict", fake_predict)
    return calls


# ---- the reason this module exists ----

def test_each_ticker_is_fetched_exactly_once(stub):
    A.analyse(["AAA", "BBB"])
    assert stub["AAA"] == 1 and stub["BBB"] == 1
    assert stub["SPY"] == 1        # the market is fetched once for the whole cohort


def test_both_readings_come_from_the_same_fetch(stub):
    body = A.analyse(["AAA", "BBB"])
    for row in body["rows"]:
        assert row["criteria"], row["symbol"]      # scorecard present
        assert row["gates"], row["symbol"]         # verdict gates present
        assert row["composite_equal"] is not None


# ---- payload shape ----

def test_payload_is_json_safe_and_complete(stub):
    body = A.analyse(["AAA", "BBB"])
    json.dumps(body)
    assert body["n_analysed"] == 2 and body["n_requested"] == 2
    assert "regime" in body and "risk_on" in body
    assert body["verdict_order"][0] == CL.INVESTABLE
    assert len(body["gate_order"]) == 10
    assert len(body["criteria_order"]) == 7


def test_banner_refuses_to_merge_the_two_readings(stub):
    body = A.analyse(["AAA"])
    assert "not merged" in body["banner"].lower()
    assert "NOT INVESTMENT ADVICE" in body["banner"]


# ---- forecast rides along with a lone ticker ----

def test_single_ticker_gets_a_forecast(stub):
    body = A.analyse(["AAA"], months=9)
    assert body["forecast"] is not None
    assert body["forecast"]["months_requested"] == 9
    assert body["rows"][0]["criteria"] and body["rows"][0]["gates"]


def test_multiple_tickers_skip_the_forecast(stub):
    body = A.analyse(["AAA", "BBB"])
    assert body["forecast"] is None
    assert "predict:AAA" not in stub          # never computed for a cohort


def test_a_failing_forecast_still_returns_the_cohort(stub, monkeypatch):
    def boom(symbol, months=6, **kw):
        raise RuntimeError("simulator exploded")

    monkeypatch.setattr(A.E, "predict", boom)
    body = A.analyse(["AAA"])
    assert body["forecast"] is None
    assert body["rows"] and body["rows"][0]["criteria"]     # the rest survives
    assert any("forecast unavailable" in f["reason"] for f in body["failures"])


# ---- partial data ----

def test_a_young_ticker_is_scored_but_not_gated(stub):
    body = A.analyse(["AAA", "SHORTGATE"])
    by_sym = {r["symbol"]: r for r in body["rows"]}
    short = by_sym["SHORTGATE"]
    assert short["verdict"] == A.INSUFFICIENT
    assert short["gates"] == []                # too short for a 252d beta
    assert short["criteria"]                   # but the scorecard still applies
    assert "252d beta" in short["reasons"][0]


def test_a_ticker_below_the_scoring_floor_is_reported_not_dropped(stub):
    body = A.analyse(["AAA", "YOUNG"])
    assert [r["symbol"] for r in body["rows"]] == ["AAA"]
    assert any(f["symbol"] == "YOUNG" for f in body["failures"])


def test_a_bad_ticker_is_isolated(stub):
    body = A.analyse(["AAA", "BAD"])
    assert [r["symbol"] for r in body["rows"]] == ["AAA"]
    assert any(f["symbol"] == "BAD" and "404" in f["reason"] for f in body["failures"])


def test_all_bad_tickers_raise_a_data_error(stub):
    with pytest.raises(E.DataError):
        A.analyse(["BAD"])


# ---- input handling ----

def test_duplicates_are_collapsed(stub):
    body = A.analyse(["AAA", "aaa", "AAA"])
    assert len(body["rows"]) == 1
    assert stub["AAA"] == 1


def test_truncation_is_reported_not_silent(stub):
    syms = ["AAA"] * 1 + [f"T{i}" for i in range(25)]
    body = A.analyse(syms, max_symbols=5)
    assert body["n_requested"] == 5
    assert len(body["truncated"]) == 21


def test_empty_input_raises(stub):
    with pytest.raises(E.DataError):
        A.analyse([])
    with pytest.raises(E.DataError):
        A.analyse(["   "])


# ---- the fetch budget: bounded time, partial results, never a silent stall ----

def test_a_slow_upstream_stops_fetching_and_reports_the_skipped(stub, monkeypatch):
    """A crawling price source must not turn into an unbounded wait: once the budget
    is spent the remaining symbols come back as failures, not as a hung request."""
    real = A.E.fetch_prices
    clock = {"t": 0.0}

    def slow(symbol, rng=None):
        clock["t"] += 40.0        # each fetch "costs" 40s of budget
        return real(symbol, rng)

    monkeypatch.setattr(A.E, "fetch_prices", slow)
    monkeypatch.setattr(A.time, "monotonic", lambda: clock["t"])

    body = A.analyse(["AAA", "BBB", "CCC", "DDD"])
    skipped = [f for f in body["failures"] if "skipped" in f["reason"]]
    assert skipped, "expected the budget to cut the cohort short"
    assert body["rows"], "partial results must still be returned"
    assert len(body["rows"]) + len(body["failures"]) == 4     # nothing vanishes


def test_the_budget_never_returns_an_empty_cohort(stub, monkeypatch):
    """The guard only trips once at least one name is in hand -- otherwise a slow first
    fetch would yield a resultless 200 instead of an honest error."""
    clock = {"t": 1000.0}         # already way past the budget
    monkeypatch.setattr(A.time, "monotonic", lambda: clock["t"])
    body = A.analyse(["AAA", "BBB"])
    assert body["rows"]           # the first name is still fetched and returned


# ---- ordering ----

def test_rows_are_ordered_by_verdict_then_score(stub):
    body = A.analyse(["AAA", "BBB", "SHORTGATE"])
    ranks = [body["verdict_order"].index(r["verdict"]) for r in body["rows"]]
    assert ranks == sorted(ranks)


def test_insufficient_history_sorts_above_hard_errors(stub):
    assert (A.VERDICT_ORDER.index(A.INSUFFICIENT)
            < A.VERDICT_ORDER.index(CL.ERROR))
    assert A.VERDICT_TEXT[A.INSUFFICIENT]
