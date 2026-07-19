"""Portfolio import: the shapes a real export actually arrives in.

Every case here is something a broker or Yahoo genuinely produces -- lots split across
rows, cash placeholders, exchange suffixes, a BOM, a bare pasted list. The point is
that none of them requires the user to hand-edit the file first, and that anything
unparsed is REPORTED rather than dropped.
"""

from __future__ import annotations

import pytest

from src.forecast import portfolio as P

YAHOO_HEADER = ('"Symbol","Current Price","Date","Time","Change","Open","High","Low",'
                '"Volume","Trade Date","Purchase Price","Quantity","Commission",'
                '"High Limit","Low Limit","Comment"')


def _yahoo(*rows):
    body = "\n".join('"%s",203.11,"2026/07/17","4:00pm EDT",1.2,201,204,200,1820000,'
                     '"2025/01/15",118.4,100,0,0,0,""' % r for r in rows)
    return YAHOO_HEADER + "\n" + body


# ---- the real Yahoo shape ----

def test_reads_a_yahoo_export():
    syms, skipped = P.symbols_from_csv_text(_yahoo("NVDA", "COIN", "MRVL"))
    assert syms == ["NVDA", "COIN", "MRVL"]
    assert skipped == []


def test_cash_rows_are_skipped_and_reported():
    syms, skipped = P.symbols_from_csv_text(_yahoo("NVDA", "$$CASH", "COIN"))
    assert syms == ["NVDA", "COIN"]
    assert [s["value"] for s in skipped] == ["$$CASH"]
    assert "not a tradable instrument" in skipped[0]["reason"]


def test_a_totals_row_is_skipped():
    syms, skipped = P.symbols_from_csv_text(_yahoo("NVDA", "Total"))
    assert syms == ["NVDA"] and len(skipped) == 1


def test_multiple_lots_of_one_holding_collapse():
    # Yahoo writes one row per purchase; a cohort must not be dominated by a holding
    # simply because it was bought three times.
    syms, _ = P.symbols_from_csv_text(_yahoo("NVDA", "COIN", "NVDA", "NVDA"))
    assert syms == ["NVDA", "COIN"]


def test_a_utf8_bom_is_stripped(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(_yahoo("NVDA", "COIN"), encoding="utf-8-sig")
    syms, _ = P.symbols_from_csv(p)
    assert syms == ["NVDA", "COIN"]       # not ['﻿NVDA', ...]


# ---- heading variations ----

@pytest.mark.parametrize("heading", ["Symbol", "symbol", "SYMBOL", " Ticker ",
                                     "Ticker Symbol", "Symbols"])
def test_heading_variants_are_accepted(heading):
    text = f"{heading},Qty\nNVDA,100\nCOIN,25\n"
    syms, _ = P.symbols_from_csv_text(text)
    assert syms == ["NVDA", "COIN"]


def test_symbol_column_need_not_be_first():
    text = "Quantity,Name,Ticker\n100,Nvidia Corp,NVDA\n25,Coinbase,COIN\n"
    syms, _ = P.symbols_from_csv_text(text)
    assert syms == ["NVDA", "COIN"]


# ---- exchange suffixes must survive verbatim ----

def test_exchange_suffixes_and_class_shares_pass_through_unchanged():
    """The dot is an exchange suffix here. `universe.py` rewrites BRK.B -> BRK-B for
    the Wikipedia list; doing that to a portfolio would request a different security."""
    text = "Symbol\nBHP.AX\n0700.HK\nSHOP.TO\nBRK-B\nBTC-USD\nEURUSD=X\n^GSPC\n"
    syms, skipped = P.symbols_from_csv_text(text)
    assert syms == ["BHP.AX", "0700.HK", "SHOP.TO", "BRK-B", "BTC-USD",
                    "EURUSD=X", "^GSPC"]
    assert skipped == []


# ---- headerless / pasted lists ----

def test_a_bare_list_of_tickers_works():
    syms, _ = P.symbols_from_csv_text("NVDA\nCOIN\nMRVL\n")
    assert syms == ["NVDA", "COIN", "MRVL"]


def test_a_bare_comma_separated_first_column_works():
    syms, _ = P.symbols_from_csv_text("NVDA,100\nCOIN,25\n")
    assert syms == ["NVDA", "COIN"]


def test_blank_lines_are_ignored():
    syms, skipped = P.symbols_from_csv_text("Symbol\nNVDA\n\n\nCOIN\n")
    assert syms == ["NVDA", "COIN"] and skipped == []


# ---- refusing the wrong file ----

def test_a_file_with_no_ticker_column_raises_with_guidance():
    text = "Account Name,Balance\nRetirement,10000\n"
    with pytest.raises(P.PortfolioError) as e:
        P.symbols_from_csv_text(text)
    assert "no ticker column" in str(e.value)
    assert "symbol" in str(e.value)        # the error names what it wanted


def test_junk_values_are_reported_not_silently_dropped():
    text = "Symbol\nNVDA\nSome Company Name\nCOIN\n"
    syms, skipped = P.symbols_from_csv_text(text)
    assert syms == ["NVDA", "COIN"]
    assert skipped[0]["value"] == "Some Company Name"
    assert "does not look like a ticker" in skipped[0]["reason"]


def test_empty_input_is_empty_not_an_error():
    assert P.symbols_from_csv_text("") == ([], [])
    assert P.symbols_from_csv_text("\n\n") == ([], [])


def test_looks_like_ticker_boundaries():
    for good in ("NVDA", "BRK-B", "BHP.AX", "^GSPC", "0700.HK", "EURUSD=X"):
        assert P.looks_like_ticker(good), good
    for bad in ("", "   ", "$$CASH", "Total", "Some Company Name", "A" * 13):
        assert not P.looks_like_ticker(bad), bad
