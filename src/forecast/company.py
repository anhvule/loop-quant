"""Company report data: one ticker's profile, statements, estimates and ownership,
reduced to plain JSON for the Simply-Wall-St-style report (/api/company).

Follows the same three disciplines as `fundamentals.py`, whose helpers it reuses:

  1. FETCHING IS SPLIT FROM DERIVING. `reduce_raw` is pure and unit-tested without a
     network; `fetch_company` wraps a downloader around it. yfinance imports lazily.

  2. MISSING DATA IS NOT BAD NEWS. Yahoo's coverage is patchy and its field names
     drift between yfinance versions. Every field lands as None (or an empty list)
     rather than a guess, and the checks downstream grade "info" rather than fail.

  3. Nothing here stores a DataFrame: raw pulls are reduced immediately to a small
     dict of plain floats/strings/lists, which is what gets cached and consumed.

The one hard failure is a symbol Yahoo knows nothing about (no price AND no name):
that raises `CompanyDataError`, which the API maps to a 400.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Callable

from src.common.paths import DATA_DIR
from src.forecast.fundamentals import (
    _CASH_ROWS, _FCF_ROWS, _OCF_ROWS, _REVENUE_ROWS,
    _finite, _fresh, _norm, _read_cache, _row_series, _row_value, _write_cache,
)

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "company_cache.json"
MAX_AGE_DAYS = 3      # analyst targets and price move faster than quarterly statements
MAX_YEARS = 4         # annual statement columns to keep, newest first
MAX_SUMMARY_CHARS = 600
MAX_OFFICERS = 10
MAX_HOLDERS = 10
MAX_DIVIDEND_YEARS = 12
MAX_PRICE_POINTS = 261  # ~5 years of weekly closes

# Annual-statement row labels, matched case- and space-insensitively (see
# fundamentals._row_series). Order is preference order.
_NET_INCOME_ROWS = ("net income", "net income common stockholders",
                    "net income continuous operations")
_EBIT_ROWS = ("ebit", "operating income")
_INTEREST_ROWS = ("interest expense", "interest expense non operating")
_EQUITY_ROWS = ("stockholders equity", "common stock equity",
                "total stockholder equity")
_TOTAL_ASSETS_ROWS = ("total assets",)
_CURRENT_ASSETS_ROWS = ("current assets", "total current assets")
_CURRENT_LIAB_ROWS = ("current liabilities", "total current liabilities")
_TOTAL_LIAB_ROWS = ("total liabilities net minority interest", "total liab")
_TOTAL_DEBT_ROWS = ("total debt",)
_LT_DEBT_ROWS = ("long term debt", "long term debt and capital lease obligation")
_CURRENT_DEBT_ROWS = ("current debt", "current debt and capital lease obligation")
_DIVS_PAID_ROWS = ("cash dividends paid", "common stock dividend paid")


class CompanyDataError(Exception):
    """The vendor returned nothing usable for this symbol (bad ticker, delisted)."""


# ---------------------------------------------------------------------------
# small pure pieces
# ---------------------------------------------------------------------------

def _num(info: dict, *keys: str) -> float | None:
    """First finite value among `info[key]` candidates."""
    for k in keys:
        v = _finite(info.get(k))
        if v is not None:
            return v
    return None


def _first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _text(info: dict, *keys: str) -> str | None:
    for k in keys:
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _parse_pct(v: Any) -> float | None:
    """A percentage as a fraction. Accepts '9.02%', 0.0902 and 9.02 (percent form)."""
    if isinstance(v, str):
        f = _finite(v.strip().rstrip("%"))
        return None if f is None else f / 100.0
    f = _finite(v)
    if f is None:
        return None
    return f / 100.0 if f > 1.0 else f


def _price_series(prices: Any, limit: int = MAX_PRICE_POINTS) -> list[dict]:
    """[{d: iso date, p: close}, ...] from a date-indexed close series."""
    if prices is None:
        return []
    try:
        items = list(prices.items())
    except AttributeError:
        return []
    out = []
    for k, v in items[-limit:]:
        p = _finite(v)
        d = getattr(k, "date", None)
        if p is None or p <= 0 or not callable(d):
            continue
        out.append({"d": d().isoformat(), "p": p})
    return out


def _annual(frame: Any, cands: tuple[str, ...]) -> list[float]:
    return _row_series(frame, cands, MAX_YEARS)


def _fiscal_years(*frames: Any) -> list[str]:
    """Fiscal-year labels (newest first) from the first statement that has columns."""
    for frame in frames:
        if frame is None or not hasattr(frame, "columns"):
            continue
        ys = []
        for c in list(frame.columns)[:MAX_YEARS]:
            y = getattr(c, "year", None)
            ys.append(str(y) if y else str(c)[:4])
        if ys:
            return ys
    return []


def _dps_by_year(dividends: Any, as_of: dt.date) -> dict[str, float]:
    """Dividends per share summed by calendar year. The current (partial) year is
    dropped -- it would read as a cut in every stability comparison."""
    if dividends is None:
        return {}
    try:
        items = list(dividends.items())
    except AttributeError:
        return {}
    sums: dict[str, float] = {}
    for k, v in items:
        amt = _finite(v)
        y = getattr(k, "year", None)
        if amt is None or amt <= 0 or y is None:
            continue
        sums[str(y)] = sums.get(str(y), 0.0) + amt
    sums.pop(str(as_of.year), None)
    keep = sorted(sums)[-MAX_DIVIDEND_YEARS:]
    return {y: round(sums[y], 6) for y in keep}


def _officers(info: dict, limit: int = MAX_OFFICERS) -> list[dict]:
    out = []
    for o in (info.get("companyOfficers") or [])[:limit]:
        if not isinstance(o, dict):
            continue
        pay = o.get("totalPay")
        if isinstance(pay, dict):          # some yfinance versions wrap {raw, fmt}
            pay = pay.get("raw")
        name = str(o.get("name") or "").strip()
        if not name:
            continue
        out.append({"name": name, "title": str(o.get("title") or "").strip(),
                    "age": _finite(o.get("age")), "pay": _finite(pay)})
    return out


def _holder_percents(major: Any) -> tuple[float | None, float | None]:
    """(insider %, institution %) as fractions from `major_holders`, whose layout
    drifts: newer yfinance has a labelled index + one Value column of fractions,
    older has a 2-column frame of ['0.06%', '% of Shares Held by All Insider']."""
    if major is None or not hasattr(major, "index") or not hasattr(major, "columns"):
        return None, None
    ins = inst = None
    for ridx in list(major.index):
        try:
            cells = [major.loc[ridx, c] for c in list(major.columns)]
        except (KeyError, IndexError, TypeError):
            continue
        label = " ".join(
            str(x) for x in [ridx] + [c for c in cells if isinstance(c, str)]).lower()
        val = None
        for c in cells:
            val = _parse_pct(c)
            if val is not None:
                break
        if val is None or not 0.0 <= val <= 1.0:
            continue
        if "insider" in label and ins is None:
            ins = val
        elif ("institution" in label and "float" not in label
                and "count" not in label and "number" not in label and inst is None):
            inst = val
    return ins, inst


def _top_institutions(frame: Any, limit: int = MAX_HOLDERS) -> list[dict]:
    if frame is None or not hasattr(frame, "columns") or not hasattr(frame, "index"):
        return []
    cols = {_norm(c).replace(" ", ""): c for c in frame.columns}
    hc = cols.get("holder")
    pc = cols.get("pctheld") or cols.get("%out")
    vc = cols.get("value")
    if hc is None:
        return []
    out = []
    for ridx in list(frame.index)[:limit]:
        try:
            holder = str(frame.loc[ridx, hc]).strip()
        except (KeyError, IndexError, TypeError):
            continue
        if not holder or holder.lower() == "nan":
            continue
        out.append({
            "holder": holder,
            "pct": _parse_pct(frame.loc[ridx, pc]) if pc is not None else None,
            "value": _finite(frame.loc[ridx, vc]) if vc is not None else None,
        })
    return out


def _insider_activity(tx: Any, as_of: dt.date
                      ) -> tuple[int | None, int | None, float | None]:
    """(buys, sells, net shares) over the trailing year from `insider_transactions`."""
    if tx is None or not hasattr(tx, "columns") or not hasattr(tx, "index"):
        return None, None, None
    cols = {_norm(c).replace(" ", ""): c for c in tx.columns}
    dcol, scol = cols.get("startdate"), cols.get("shares")
    tcol = cols.get("text") or cols.get("transaction")
    cutoff = as_of - dt.timedelta(days=365)
    buys = sells = 0
    net = 0.0
    for ridx in list(tx.index):
        try:
            when = tx.loc[ridx, dcol] if dcol is not None else None
        except (KeyError, IndexError, TypeError):
            continue
        d = getattr(when, "date", None)
        if callable(d) and d() < cutoff:
            continue
        txt = str(tx.loc[ridx, tcol]).lower() if tcol is not None else ""
        shares = _finite(tx.loc[ridx, scol]) if scol is not None else None
        if "purchase" in txt or "buy" in txt:
            buys += 1
            net += shares or 0.0
        elif "sale" in txt or "sell" in txt:
            sells += 1
            net -= shares or 0.0
    return buys, sells, net


# ---------------------------------------------------------------------------
# full reduction
# ---------------------------------------------------------------------------

def reduce_raw(raw: dict, as_of: dt.date | None = None, symbol: str = "") -> dict:
    """Collapse the yfinance pulls into the plain-JSON dict we cache and score."""
    as_of = as_of or dt.date.today()
    info = raw.get("info")
    if not isinstance(info, dict):
        info = {}
    income, balance, cashflow = raw.get("income"), raw.get("balance"), raw.get("cashflow")
    targets = raw.get("targets")
    if not isinstance(targets, dict):
        targets = {}

    series = _price_series(raw.get("prices"))
    price = series[-1]["p"] if series else _num(info, "currentPrice",
                                                "regularMarketPrice",
                                                "regularMarketPreviousClose")

    total_debt = _annual(balance, _TOTAL_DEBT_ROWS)
    if not total_debt:
        lt = _annual(balance, _LT_DEBT_ROWS)
        cd = _annual(balance, _CURRENT_DEBT_ROWS)
        if lt or cd:
            total_debt = [(lt[i] if i < len(lt) else 0.0)
                          + (cd[i] if i < len(cd) else 0.0)
                          for i in range(max(len(lt), len(cd)))]

    summary = _text(info, "longBusinessSummary")
    if summary and len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"

    ins_pct, inst_pct = _holder_percents(raw.get("major"))
    buys, sells, net_shares = _insider_activity(raw.get("insider_tx"), as_of)

    out = {
        "as_of": as_of.isoformat(),
        # overview
        "name": _text(info, "longName", "shortName"),
        "sector": _text(info, "sector"),
        "industry": _text(info, "industry"),
        "summary": summary,
        "website": _text(info, "website"),
        "exchange": _text(info, "fullExchangeName", "exchange"),
        "employees": _num(info, "fullTimeEmployees"),
        "currency": _text(info, "currency"),
        "financial_currency": _text(info, "financialCurrency"),
        "price": price,
        "market_cap": _num(info, "marketCap"),
        "shares_outstanding": _num(info, "sharesOutstanding"),
        "price_series": series,
        # valuation inputs
        "trailing_pe": _num(info, "trailingPE"),
        "forward_pe": _num(info, "forwardPE"),
        "price_to_book": _num(info, "priceToBook"),
        "trailing_eps": _num(info, "trailingEps"),
        "forward_eps": _num(info, "forwardEps"),
        "peg": _num(info, "trailingPegRatio", "pegRatio"),
        "target_mean": _first(_finite(targets.get("mean")),
                              _num(info, "targetMeanPrice")),
        "target_high": _first(_finite(targets.get("high")),
                              _num(info, "targetHighPrice")),
        "target_low": _first(_finite(targets.get("low")),
                             _num(info, "targetLowPrice")),
        "n_analysts": _num(info, "numberOfAnalystOpinions"),
        # annual statement series, newest first
        "fiscal_years": _fiscal_years(income, balance, cashflow),
        "revenue": _annual(income, _REVENUE_ROWS),
        "net_income": _annual(income, _NET_INCOME_ROWS),
        "ebit": _annual(income, _EBIT_ROWS),
        "interest_expense": _annual(income, _INTEREST_ROWS),
        "equity": _annual(balance, _EQUITY_ROWS),
        "total_assets": _annual(balance, _TOTAL_ASSETS_ROWS),
        "current_assets": _annual(balance, _CURRENT_ASSETS_ROWS),
        "current_liabilities": _annual(balance, _CURRENT_LIAB_ROWS),
        "total_liabilities": _annual(balance, _TOTAL_LIAB_ROWS),
        "total_debt": total_debt,
        "cash": _annual(balance, _CASH_ROWS),
        "fcf": _annual(cashflow, _FCF_ROWS),
        "ocf": _annual(cashflow, _OCF_ROWS),
        "dividends_paid": _annual(cashflow, _DIVS_PAID_ROWS),
        # analyst estimates
        "eps_growth_1y": _row_value(raw.get("eps_est"), ("+1y",), "growth"),
        "rev_growth_1y": _row_value(raw.get("rev_est"), ("+1y",), "growth"),
        "eps_now": _row_value(raw.get("eps_est"), ("0y",), "avg"),
        "eps_next": _row_value(raw.get("eps_est"), ("+1y",), "avg"),
        "rev_now": _row_value(raw.get("rev_est"), ("0y",), "avg"),
        "rev_next": _row_value(raw.get("rev_est"), ("+1y",), "avg"),
        # dividend. dividendYield flipped from fraction to percent form across
        # yfinance versions, so the raw value is kept and the effective yield is
        # derived downstream from DPS/price first, this field last.
        "dividend_yield_info": _num(info, "dividendYield"),
        "trailing_div_rate": _num(info, "trailingAnnualDividendRate",
                                  "dividendRate"),
        "payout_ratio_info": _num(info, "payoutRatio"),
        "dps_by_year": _dps_by_year(raw.get("dividends"), as_of),
        # profitability fallback when statements are missing
        "roe_info": _num(info, "returnOnEquity"),
        # management + ownership
        "officers": _officers(info),
        "insider_pct": ins_pct,
        "institution_pct": inst_pct,
        "top_institutions": _top_institutions(raw.get("inst")),
        "insider_buys_12m": buys,
        "insider_sells_12m": sells,
        "insider_net_shares_12m": net_shares,
    }

    if out["price"] is None and out["name"] is None:
        raise CompanyDataError(
            f"no data for {symbol or 'that symbol'} — check the ticker.")
    return out


# ---------------------------------------------------------------------------
# fetching + cache
# ---------------------------------------------------------------------------

def _default_downloader(symbol: str) -> dict:
    """All the pulls for one ticker off ONE yf.Ticker. Each pull is isolated: a
    missing insider table must not discard a perfectly good income statement."""
    import yfinance as yf  # noqa: PLC0415  (lazy on purpose)

    t = yf.Ticker(symbol)
    pulls = (
        ("info", lambda: t.info),
        ("prices", lambda: t.history(period="5y", interval="1wk")["Close"]),
        ("income", lambda: t.income_stmt),
        ("balance", lambda: t.balance_sheet),
        ("cashflow", lambda: t.cashflow),
        ("dividends", lambda: t.dividends),
        ("eps_est", lambda: t.earnings_estimate),
        ("rev_est", lambda: t.revenue_estimate),
        ("targets", lambda: t.analyst_price_targets),
        ("inst", lambda: t.institutional_holders),
        ("major", lambda: t.major_holders),
        ("insider_tx", lambda: t.insider_transactions),
    )
    out = {}
    for key, fn in pulls:
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001 -- one absent pull is not a failure
            log.debug("%s: %s unavailable (%s)", symbol, key, type(e).__name__)
            out[key] = None
    return out


def fetch_company(symbol: str, cache_path: Path | str = CACHE_PATH,
                  max_age_days: int = MAX_AGE_DAYS, refresh: bool = False,
                  downloader: Callable[[str], dict] | None = None,
                  today: dt.date | None = None) -> dict:
    """The reduced dict for one symbol, cache-first. `cached` marks a cache hit."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise CompanyDataError("enter a ticker.")
    path = Path(cache_path)
    today = today or dt.date.today()

    cache = _read_cache(path)
    entry = cache.get(sym)
    if entry and not refresh and _fresh(entry, today, max_age_days):
        out = dict(entry)
        out["cached"] = True
        return out

    raw = (downloader or _default_downloader)(sym)
    reduced = reduce_raw(raw, as_of=today, symbol=sym)
    reduced["symbol"] = sym
    cache[sym] = reduced
    _write_cache(path, cache)
    out = dict(reduced)
    out["cached"] = False
    return out
