"""Company fundamentals for the classifier: dilution, cash runway, revenue.

Why this exists: the price-based gates measure a *proxy*. A name whose 126-day windows
keep halving is usually a serial diluter, but "usually" is doing real work in that
sentence. This module measures the disease directly -- share count, cash against burn,
and whether an operating business exists at all.

Three disciplines, matching `universe.py`:

  1. FETCHING IS SPLIT FROM DERIVING. `dilution_yoy`, `runway_quarters` and the frame
     reducers are pure and unit-tested without a network; `fetch_fundamentals` wraps a
     downloader around them. yfinance is imported lazily so the package imports without it.

  2. MISSING DATA IS NOT BAD NEWS. Yahoo's fundamentals coverage is patchy -- ADRs,
     fresh IPOs and trusts routinely return nothing. Every derivation returns None
     rather than a number, and the gates downstream grade INFO rather than FAIL. A
     screen that condemned names for the vendor's coverage gaps would be worse than one
     that admits it does not know.

  3. ONE BAD TICKER NEVER KILLS THE RUN. Each of the four pulls is isolated, each
     symbol is isolated, and failures are returned for reporting rather than raised.

Nothing here stores a DataFrame: the raw pulls are reduced immediately to a small dict
of plain floats, which is what gets cached and what the classifier consumes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.common.paths import DATA_DIR

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "fundamentals_cache.json"
MAX_AGE_DAYS = 7          # fundamentals move quarterly; a week bounds staleness cheaply
YEAR_DAYS = 365
YEAR_TOLERANCE_DAYS = 45  # how far from "one year ago" a share observation may sit
MAX_QUARTERS = 4          # a trailing-twelve-month view

# yfinance row labels drift between versions ("Free Cash Flow" / "FreeCashFlow") and
# between filers. Match case- and space-insensitively, in preference order.
_CASH_ROWS = ("cash and cash equivalents", "cashandcashequivalents",
              "cash cash equivalents and short term investments",
              "cashcashequivalentsandshortterminvestments")
_SHORT_TERM_ROWS = ("other short term investments", "othershortterminvestments",
                    "short term investments", "shortterminvestments")
_FCF_ROWS = ("free cash flow", "freecashflow")
_OCF_ROWS = ("operating cash flow", "operatingcashflow",
             "total cash from operating activities")
_REVENUE_ROWS = ("total revenue", "totalrevenue", "operating revenue", "operatingrevenue")

# Sectors where free cash flow does not mean what the runway gate assumes. A lender
# books loan origination as an operating outflow, so a healthy growing bank reads as
# though it is burning cash. This is "the metric does not apply here", not "this sector
# gets a pass" -- the gate reports INFO rather than inventing a number.
FCF_NOT_MEANINGFUL_SECTORS = frozenset({"financial services", "financials", "financial"})


# ---------------------------------------------------------------------------
# pure derivations
# ---------------------------------------------------------------------------

def dilution_yoy(shares_now: float | None, shares_1y_ago: float | None) -> float | None:
    """Trailing-year share-count growth. Positive means holders were diluted."""
    if shares_now is None or shares_1y_ago is None:
        return None
    if shares_1y_ago <= 0 or shares_now <= 0:
        return None
    return float(shares_now / shares_1y_ago - 1.0)


def runway_quarters(cash: float | None, fcf_quarters: list[float] | None) -> float | None:
    """Quarters of cash left at the current burn rate.

    Burn is the mean of the NEGATIVE quarterly free-cash-flow readings only. Averaging
    signed values would let one good quarter cancel three bad ones and report a company
    as self-funding while it is bleeding; a company that is not burning at all is
    handled separately and explicitly.
    """
    if cash is None or not fcf_quarters:
        return None
    burns = [float(q) for q in fcf_quarters if q is not None and float(q) < 0.0]
    if not burns:
        return float("inf")                     # self-funding: no burn to run out of
    mean_burn = sum(burns) / len(burns)
    if mean_burn >= 0.0:
        return float("inf")
    if cash <= 0:
        return 0.0
    return float(cash / abs(mean_burn))


# ---------------------------------------------------------------------------
# frame reduction (pure -- takes yfinance-shaped objects, returns plain floats)
# ---------------------------------------------------------------------------

def _norm(label: Any) -> str:
    return str(label).strip().lower()


def _row_value(frame: Any, candidates: tuple[str, ...], col: Any) -> float | None:
    """First matching row's value in column `col`, matched case-insensitively."""
    if frame is None or not hasattr(frame, "index"):
        return None
    lookup = {_norm(i): i for i in frame.index}
    squashed = {_norm(i).replace(" ", ""): i for i in frame.index}
    for cand in candidates:
        key = lookup.get(cand) or squashed.get(cand.replace(" ", ""))
        if key is None:
            continue
        try:
            v = frame.loc[key, col]
        except (KeyError, IndexError, TypeError):
            continue
        return _finite(v)
    return None


def _row_series(frame: Any, candidates: tuple[str, ...], limit: int) -> list[float]:
    """First matching row across the most recent `limit` columns, newest first."""
    if frame is None or not hasattr(frame, "index") or not hasattr(frame, "columns"):
        return []
    lookup = {_norm(i): i for i in frame.index}
    squashed = {_norm(i).replace(" ", ""): i for i in frame.index}
    for cand in candidates:
        key = lookup.get(cand) or squashed.get(cand.replace(" ", ""))
        if key is None:
            continue
        out = []
        for col in list(frame.columns)[:limit]:
            v = _finite(frame.loc[key, col])
            if v is not None:
                out.append(v)
        if out:
            return out
    return []


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def shares_now_and_year_ago(shares: Any) -> tuple[float | None, float | None]:
    """(latest share count, count closest to a year before it) from a date-indexed series.

    yfinance's share series carry occasional zero and NaN rows; those are dropped before
    anything is measured. If no observation sits within `YEAR_TOLERANCE_DAYS` of one year
    before the latest, the year-ago figure is None -- an 8-month-old "year" would
    overstate annualised dilution badly.
    """
    if shares is None:
        return None, None
    try:
        items = [(k, _finite(v)) for k, v in shares.items()]
    except AttributeError:
        return None, None
    rows = []
    for k, v in items:
        if v is None or v <= 0:
            continue
        d = getattr(k, "date", None)
        rows.append((d() if callable(d) else k, v))
    if not rows:
        return None, None
    rows.sort(key=lambda r: r[0])
    latest_date, latest = rows[-1]
    target = latest_date - dt.timedelta(days=YEAR_DAYS)
    best, best_gap = None, None
    for d, v in rows[:-1]:
        gap = abs((d - target).days)
        if best_gap is None or gap < best_gap:
            best, best_gap = v, gap
    if best is None or best_gap is None or best_gap > YEAR_TOLERANCE_DAYS:
        return latest, None
    return latest, best


def reduce_frames(shares: Any, balance: Any, cashflow: Any, income: Any,
                  as_of: dt.date | None = None, sector: Any = None) -> dict:
    """Collapse the yfinance objects into the small plain-float dict we cache."""
    now, ago = shares_now_and_year_ago(shares)

    cash = None
    if balance is not None and getattr(balance, "columns", None) is not None \
            and len(balance.columns):
        col = list(balance.columns)[0]                  # newest quarter
        base = _row_value(balance, _CASH_ROWS, col)
        extra = _row_value(balance, _SHORT_TERM_ROWS, col)
        if base is not None or extra is not None:
            cash = (base or 0.0) + (extra or 0.0)

    fcf = _row_series(cashflow, _FCF_ROWS, MAX_QUARTERS)
    if not fcf:
        fcf = _row_series(cashflow, _OCF_ROWS, MAX_QUARTERS)
    rev = _row_series(income, _REVENUE_ROWS, MAX_QUARTERS)

    sector_name = str(sector).strip() if sector else None
    out = {
        "shares_now": now, "shares_1y_ago": ago,
        "cash": cash, "fcf_quarters": fcf,
        "revenue_ttm": float(sum(rev)) if rev else None,
        "n_revenue_quarters": len(rev),
        "sector": sector_name,
        "fcf_meaningful": not (sector_name or "").lower() in FCF_NOT_MEANINGFUL_SECTORS,
        "as_of": (as_of or dt.date.today()).isoformat(),
    }
    # Pre-derive so the classifier and its vendored web copy read plain numbers and
    # never need to import this module.
    out["dilution_yoy"] = dilution_yoy(now, ago)
    out["runway_quarters"] = runway_quarters(cash, fcf)
    return out


# ---------------------------------------------------------------------------
# fetching + cache
# ---------------------------------------------------------------------------

def _default_downloader(symbol: str) -> dict:
    """Pull the four statements from yfinance. Each is isolated: a missing cash-flow
    statement must not discard a perfectly good share count."""
    import yfinance as yf  # noqa: PLC0415  (lazy on purpose)

    t = yf.Ticker(symbol)
    start = dt.date.today() - dt.timedelta(days=2 * YEAR_DAYS)
    out = {}
    for key, fn in (("shares", lambda: t.get_shares_full(start=start)),
                    ("balance", lambda: t.quarterly_balance_sheet),
                    ("cashflow", lambda: t.quarterly_cashflow),
                    ("income", lambda: t.quarterly_income_stmt),
                    ("sector", lambda: (t.info or {}).get("sector"))):
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001 -- one absent statement is not a failure
            log.debug("%s: %s unavailable (%s)", symbol, key, type(e).__name__)
            out[key] = None
    return out


def _read_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    except OSError as e:  # noqa: BLE001 -- a cache we cannot write is not a run we fail
        log.warning("could not write fundamentals cache: %s", e)


def _fresh(entry: dict, today: dt.date, max_age_days: int) -> bool:
    try:
        stamped = dt.date.fromisoformat(str(entry.get("as_of", "")))
    except ValueError:
        return False
    return (today - stamped).days <= max_age_days


def fetch_fundamentals(symbols: list[str], cache_path: Path | str = CACHE_PATH,
                       max_age_days: int = MAX_AGE_DAYS, refresh: bool = False,
                       downloader: Callable[[str], dict] | None = None,
                       today: dt.date | None = None,
                       progress: Callable[[int, int], None] | None = None
                       ) -> tuple[dict[str, dict], list[str]]:
    """({symbol: reduced dict}, failed symbols). Cache-first, per-symbol isolated.

    A symbol whose fetch fails is simply absent from the returned mapping; its gates
    grade INFO downstream. Nothing raises for one bad ticker.
    """
    path = Path(cache_path)
    today = today or dt.date.today()
    cache = _read_cache(path)
    dl = downloader or _default_downloader

    out: dict[str, dict] = {}
    failed: list[str] = []
    dirty = False
    for i, sym in enumerate(symbols, start=1):
        entry = cache.get(sym)
        if entry and not refresh and _fresh(entry, today, max_age_days):
            out[sym] = entry
        else:
            try:
                raw = dl(sym)
                reduced = reduce_frames(raw.get("shares"), raw.get("balance"),
                                        raw.get("cashflow"), raw.get("income"),
                                        as_of=today, sector=raw.get("sector"))
                out[sym] = reduced
                cache[sym] = reduced
                dirty = True
            except Exception as e:  # noqa: BLE001 -- isolate, report, continue
                log.debug("%s: fundamentals fetch failed (%s)", sym, type(e).__name__)
                failed.append(sym)
        if progress:
            progress(i, len(symbols))
    if dirty:
        _write_cache(path, cache)
    return out, failed
