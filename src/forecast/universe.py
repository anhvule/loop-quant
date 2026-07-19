"""S&P 500 constituents, cached to disk.

The Treynor gate in `classify.py` is universe-relative, so a screen is only as
meaningful as the peer set behind it. Ad-hoc ticker lists make "top quartile" an
arithmetic accident; the index gives a defensible, reproducible universe.

Two honesty rails:

  * SURVIVORSHIP. This is today's membership list, applied to trailing returns. Names
    that left the index are absent, so a historical study built on it would be biased.
    The classifier only ever screens the present, which is the one use where that bias
    does not apply -- do not reuse this list for backtests without addressing it.

  * STALENESS IS REPORTED. A cached list is reused when fresh, and reused with a
    warning when the fetch fails. It is never silently regenerated as an empty list:
    an empty universe would make every name "top quartile" of nothing.

Fetching is split from parsing so the parsing is testable without a network:
`symbols_from_frames` is pure; `sp500_symbols` wraps it around a downloader.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Callable

from src.common.paths import DATA_DIR

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "loop-quant-classifier/1.0 (research tool; contact via repo)"}
CACHE_PATH = DATA_DIR / "sp500_universe.csv"
MAX_AGE_DAYS = 7
MIN_PLAUSIBLE = 400        # the index holds ~503 tickers; far fewer means a bad parse

# Yahoo writes class shares with a hyphen where Wikipedia uses a dot (BRK.B -> BRK-B).
_TICKER_COL_CANDIDATES = ("symbol", "ticker", "ticker symbol")


class UniverseError(RuntimeError):
    """No usable constituent list: the fetch failed and no cache exists."""


def normalize_ticker(sym: str) -> str:
    """Wikipedia's ticker spelling -> the spelling yfinance expects."""
    return str(sym).strip().upper().replace(".", "-")


def symbols_from_frames(frames: list[Any]) -> list[str]:
    """Extract the constituent tickers from `pandas.read_html` output (pure, no I/O).

    Wikipedia's page carries several tables and their order is not guaranteed, so the
    first table with a recognisable symbol column wins rather than a hardcoded index.
    """
    for frame in frames or []:
        cols = {str(c).strip().lower(): c for c in getattr(frame, "columns", [])}
        for cand in _TICKER_COL_CANDIDATES:
            if cand not in cols:
                continue
            syms = [normalize_ticker(v) for v in frame[cols[cand]].tolist()]
            syms = [s for s in syms if s and s.replace("-", "").isalnum()]
            if len(syms) >= MIN_PLAUSIBLE:
                return list(dict.fromkeys(syms))
    raise UniverseError(
        f"no table with >= {MIN_PLAUSIBLE} tickers found on the constituents page; "
        f"its layout may have changed."
    )


def _default_downloader(url: str) -> list[Any]:
    """Read the Wikipedia tables. Imported lazily so the package imports without
    pandas/lxml installed (tests inject frames and never hit this).

    The page is fetched by hand rather than handed to `read_html(url)`: Wikipedia
    answers urllib's default User-Agent with HTTP 403, so the request needs the same
    identifying header the web engine already sends for price data.
    """
    import io  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415  (lazy on purpose)

    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    return pd.read_html(io.StringIO(html))


def _read_cache(path: Path) -> tuple[list[str], dt.date | None]:
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    except OSError:
        return [], None
    syms = [normalize_ticker(r.get("symbol", "")) for r in rows]
    syms = [s for s in syms if s]
    fetched = None
    if rows:
        try:
            fetched = dt.date.fromisoformat((rows[0].get("fetched_on") or "").strip())
        except ValueError:
            fetched = None
    return list(dict.fromkeys(syms)), fetched


def _write_cache(path: Path, symbols: list[str], today: dt.date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "fetched_on"])
        for s in symbols:
            w.writerow([s, today.isoformat()])


def sp500_symbols(cache_path: Path | str = CACHE_PATH, max_age_days: int = MAX_AGE_DAYS,
                  refresh: bool = False, downloader: Callable[[str], list[Any]] | None = None,
                  today: dt.date | None = None) -> tuple[list[str], str]:
    """(tickers, provenance). Cache-first; the network is a fallback, not a dependency.

    `provenance` is a short human-readable string ("cached 2026-07-14", "fetched",
    "stale cache -- fetch failed: ...") that callers print, so a run is never silently
    built on a list from months ago.
    """
    path = Path(cache_path)
    today = today or dt.date.today()
    cached, fetched_on = _read_cache(path)
    fresh = (cached and fetched_on is not None
             and (today - fetched_on).days <= max_age_days)
    if fresh and not refresh:
        return cached, f"cached {fetched_on.isoformat()}"

    try:
        syms = symbols_from_frames((downloader or _default_downloader)(WIKI_URL))
    except Exception as e:  # noqa: BLE001 -- network, parse, or missing lxml
        if cached:
            age = f"{(today - fetched_on).days}d old" if fetched_on else "undated"
            log.warning("constituent fetch failed (%s); using cache", type(e).__name__)
            return cached, f"STALE cache ({age}) -- fetch failed: {type(e).__name__}"
        raise UniverseError(
            f"could not fetch the S&P 500 constituent list ({type(e).__name__}: {e}) "
            f"and no cache exists at {path}. Pass --symbols or --csv instead, or "
            f"install lxml (pip install lxml) if pandas.read_html is unavailable."
        ) from e

    _write_cache(path, syms, today)
    return syms, "fetched"
