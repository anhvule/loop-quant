"""Read a ticker list out of a broker or Yahoo Finance portfolio export.

Yahoo's export is a sixteen-column CSV built for spreadsheets, not for tools: quoted
fields, a UTF-8 BOM, one row per LOT rather than per holding, and non-tradable rows
(`$$CASH`, a totals line) mixed in with real positions. Other brokers vary the heading
("Ticker", "Ticker Symbol") and some people just paste a bare list of symbols into a
file. All of that should Just Work, because the alternative is the user hand-editing a
CSV before every run.

Two rules carried over from the rest of this project:

  1. NOTHING IS DROPPED SILENTLY. Every row that does not yield a ticker comes back in
     `skipped` with the reason, so the caller can print it. A screen that quietly
     ignores half a portfolio is worse than one that refuses to run.

  2. TICKERS ARE PASSED THROUGH UNTOUCHED apart from case. In particular the dot is
     NOT rewritten to a hyphen here -- that rule belongs to `universe.py`, where
     Wikipedia writes `BRK.B` and Yahoo wants `BRK-B`. In a portfolio export the dot
     is an exchange suffix (`BHP.AX`, `0700.HK`) and rewriting it would silently
     request the wrong instrument.

Parsing is pure and takes text, so it is testable without touching the filesystem.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

# Heading candidates, matched case- and space-insensitively.
_SYMBOL_HEADINGS = ("symbol", "ticker", "tickersymbol", "symbols", "tickers")

# Rows that are real lines in an export but never instruments.
_NOT_TICKERS = frozenset({
    "CASH", "$CASH", "$$CASH", "TOTAL", "TOTALS", "SUBTOTAL", "N/A", "NA", "--", "-",
})

# Permissive on purpose: `BRK-B`, `BHP.AX`, `0700.HK`, `BTC-USD`, `EURUSD=X`, `^GSPC`
# are all things a real portfolio contains and yfinance accepts.
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,11}$")


class PortfolioError(RuntimeError):
    """The file exists but no ticker column could be found in it."""


def _norm(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).strip().lower())


def looks_like_ticker(value: str) -> bool:
    v = str(value).strip().upper()
    return bool(v) and v not in _NOT_TICKERS and bool(_TICKER_RE.match(v))


def symbols_from_csv_text(text: str) -> tuple[list[str], list[dict]]:
    """(tickers, skipped) from the text of a portfolio export. Pure -- no I/O.

    Order is preserved and duplicates collapse to the first occurrence, so a holding
    split across several lots is analysed once rather than dominating a cohort.
    """
    lines = text.splitlines()
    if not lines:
        return [], []

    rows = list(csv.reader(lines))
    rows = [r for r in rows if any(str(c).strip() for c in r)]   # drop blank lines
    if not rows:
        return [], []

    header = [_norm(c) for c in rows[0]]
    col = next((i for i, h in enumerate(header) if h in _SYMBOL_HEADINGS), None)

    if col is None:
        # No recognisable heading. Either a bare list of symbols, or a file whose first
        # row is already data. Only treat column 0 as tickers if it actually looks like
        # one -- otherwise this is not a portfolio file and we say so.
        first = rows[0][0] if rows[0] else ""
        if not looks_like_ticker(first):
            raise PortfolioError(
                "no ticker column found. Expected a header containing one of: "
                + ", ".join(sorted(_SYMBOL_HEADINGS))
                + " -- or a file whose first column is just ticker symbols."
            )
        col, body = 0, rows          # headerless: every row is data
    else:
        body = rows[1:]

    out: list[str] = []
    seen: set[str] = set()
    skipped: list[dict] = []
    for i, row in enumerate(body, start=1):
        raw = (row[col] if col < len(row) else "").strip()
        value = raw.upper()
        if not value:
            continue                                  # padding rows are not worth noise
        if value in _NOT_TICKERS:
            skipped.append({"value": raw, "reason": "not a tradable instrument"})
            continue
        if not _TICKER_RE.match(value):
            skipped.append({"value": raw, "reason": "does not look like a ticker"})
            continue
        if value in seen:
            continue                                  # extra lots of the same holding
        seen.add(value)
        out.append(value)
    return out, skipped


def symbols_from_csv(path: Path | str) -> tuple[list[str], list[dict]]:
    """Read `path` and extract its tickers. `utf-8-sig` strips the BOM Excel/Yahoo add."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return symbols_from_csv_text(text)
