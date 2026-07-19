"""Constituent list: parsing, caching, and what happens when the network is down.

No test here touches the network -- the downloader is injected. The point of these
tests is that a screen never silently runs on an empty or ancient universe, because
the Treynor gate would then be ranking against nothing.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.forecast import universe as U


def _tickers(n=450):
    return [f"AA{i:03d}" for i in range(n)]


def _frames(syms=None, col="Symbol"):
    syms = syms or _tickers()
    return [pd.DataFrame({"Rank": [1, 2], "Note": ["x", "y"]}),          # decoy table
            pd.DataFrame({col: syms, "Security": ["n"] * len(syms)})]


def _dl(frames):
    return lambda url: frames


# ---- parsing ----

def test_normalize_maps_class_shares_to_the_yfinance_spelling():
    assert U.normalize_ticker("BRK.B") == "BRK-B"
    assert U.normalize_ticker(" brk.b ") == "BRK-B"
    assert U.normalize_ticker("AAPL") == "AAPL"


def test_symbols_from_frames_skips_decoy_tables():
    syms = U.symbols_from_frames(_frames())
    assert len(syms) == 450 and syms[0] == "AA000"


def test_symbols_from_frames_accepts_alternative_column_names():
    for col in ("Symbol", "Ticker", "Ticker symbol"):
        assert len(U.symbols_from_frames(_frames(col=col))) == 450


def test_symbols_from_frames_rejects_an_implausibly_short_table():
    with pytest.raises(U.UniverseError):
        U.symbols_from_frames(_frames(_tickers(10)))


def test_symbols_from_frames_rejects_a_page_without_a_symbol_column():
    with pytest.raises(U.UniverseError):
        U.symbols_from_frames([pd.DataFrame({"Company": ["a"] * 500})])


def test_symbols_are_deduplicated_and_normalized():
    frames = [pd.DataFrame({"Symbol": ["BRK.B", "BRK.B"] + _tickers(450)})]
    syms = U.symbols_from_frames(frames)
    assert syms.count("BRK-B") == 1


# ---- cache behaviour ----

def test_fetch_writes_a_cache_that_reads_back(tmp_path):
    path = tmp_path / "u.csv"
    today = dt.date(2026, 7, 19)
    syms, prov = U.sp500_symbols(path, downloader=_dl(_frames()), today=today)
    assert prov == "fetched" and len(syms) == 450
    assert path.exists()

    def _boom(url):
        raise AssertionError("a fresh cache must not hit the network")

    again, prov2 = U.sp500_symbols(path, downloader=_boom, today=today)
    assert again == syms and prov2 == "cached 2026-07-19"


def test_stale_cache_triggers_a_refetch(tmp_path):
    path = tmp_path / "u.csv"
    U.sp500_symbols(path, downloader=_dl(_frames()), today=dt.date(2026, 7, 1))
    newer = _frames(_tickers(460))
    syms, prov = U.sp500_symbols(path, downloader=_dl(newer),
                                 today=dt.date(2026, 7, 19))     # 18d > MAX_AGE_DAYS
    assert prov == "fetched" and len(syms) == 460


def test_refresh_forces_a_fetch_even_when_fresh(tmp_path):
    path = tmp_path / "u.csv"
    today = dt.date(2026, 7, 19)
    U.sp500_symbols(path, downloader=_dl(_frames()), today=today)
    syms, prov = U.sp500_symbols(path, downloader=_dl(_frames(_tickers(455))),
                                 refresh=True, today=today)
    assert prov == "fetched" and len(syms) == 455


def test_failed_fetch_falls_back_to_the_cache_and_says_so(tmp_path):
    path = tmp_path / "u.csv"
    U.sp500_symbols(path, downloader=_dl(_frames()), today=dt.date(2026, 7, 1))

    def _boom(url):
        raise OSError("no route to host")

    syms, prov = U.sp500_symbols(path, downloader=_boom, today=dt.date(2026, 7, 19))
    assert len(syms) == 450                    # the run continues on known tickers
    assert prov.startswith("STALE cache") and "18d old" in prov   # but never silently


def test_failed_fetch_without_a_cache_is_a_hard_error(tmp_path):
    def _boom(url):
        raise OSError("no route to host")

    with pytest.raises(U.UniverseError) as e:
        U.sp500_symbols(tmp_path / "missing.csv", downloader=_boom)
    assert "--symbols" in str(e.value)          # the error must offer a way forward


def test_a_corrupt_cache_is_treated_as_absent(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("not,a,universe\n1,2,3\n", encoding="utf-8")
    syms, prov = U.sp500_symbols(path, downloader=_dl(_frames()))
    assert prov == "fetched" and len(syms) == 450
