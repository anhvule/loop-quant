"""Fetch daily equity OHLCV from yfinance into the Loop Quant `candles` table.

This is the equities analogue of `scripts/seed_data.py` (which pulls crypto klines
from Binance). It maps a yfinance daily frame to `Candle` rows and upserts them via
the existing `DB.upsert_candles`, so the rest of the stack (IndicatorEngine, the
`candles` reads) works unchanged for a stock like WYFI.

Design choices:
  * Bar timestamp is the bar's calendar date at 00:00 UTC (ms). Daily bars have no
    intraday anchor, and this keeps them on a clean, reproducible grid.
  * `quote_volume` is set to volume * typical_price ((H+L+C)/3). The IndicatorEngine
    computes VWAP as quote_volume/volume, so this makes VWAP == typical price on a
    daily bar. (VWAP is near-useless on daily bars anyway -- it resets every session
    -- so the forecast's signal tilt leans on RSI/MACD, not VWAP.)
  * `n_trades` is 0: yfinance does not report it and nothing downstream needs it.

Fetching is split from mapping so the mapping is unit-testable without a network:
`candles_from_frame` is pure; `fetch_yfinance` wraps it around a downloader.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable

from src.common.models import Candle

log = logging.getLogger(__name__)

# Enough history for a stable drift/vol estimate AND to warm every indicator
# (defaults need 34 bars). 60 is the binding floor.
MIN_BARS = 60

# Column name candidates, case-insensitive, in preference order. yfinance's
# `.history()` yields Open/High/Low/Close/Volume; `.download()` may prefix "Adj".
_OHLCV = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close", "adj close"),
    "volume": ("volume",),
}


def _date_to_utc_ms(index_value: Any) -> int:
    """Normalize a pandas Timestamp / datetime / date to 00:00 UTC epoch ms.

    Uses only the calendar date, so a tz-aware NY timestamp and a tz-naive one for
    the same trading day map to the same bar -- what we want for a daily grid.
    """
    d = index_value
    # pandas Timestamp and datetime both expose year/month/day.
    y, m, day = int(d.year), int(d.month), int(d.day)
    return int(dt.datetime(y, m, day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _pick_columns(columns: Any) -> dict[str, Any]:
    """Map our canonical field -> the actual column key in the frame.

    Handles both flat columns (`"Close"`) and yfinance MultiIndex columns
    (`("Close", "WYFI")`) by matching on the first level, case-insensitively.
    """
    def key_label(col: Any) -> str:
        # MultiIndex columns arrive as tuples; the field name is level 0.
        first = col[0] if isinstance(col, tuple) else col
        return str(first).strip().lower()

    lookup: dict[str, Any] = {}
    for col in columns:
        lookup.setdefault(key_label(col), col)

    resolved: dict[str, Any] = {}
    for field, candidates in _OHLCV.items():
        for cand in candidates:
            if cand in lookup:
                resolved[field] = lookup[cand]
                break
        else:
            raise ValueError(
                f"yfinance frame is missing a '{field}' column "
                f"(have: {sorted(lookup)})"
            )
    return resolved


def candles_from_frame(symbol: str, frame: Any, tf: str = "1d") -> list[Candle]:
    """Convert a yfinance-style OHLCV DataFrame to `Candle` rows (pure, no I/O).

    Rows with a non-finite or non-positive close, or zero/NaN across OHLC, are
    skipped -- yfinance occasionally emits a trailing all-NaN row for a not-yet
    settled session.
    """
    cols = _pick_columns(frame.columns)
    out: list[Candle] = []
    for idx, row in frame.iterrows():
        try:
            o = float(row[cols["open"]])
            h = float(row[cols["high"]])
            l = float(row[cols["low"]])
            c = float(row[cols["close"]])
            v = float(row[cols["volume"]])
        except (TypeError, ValueError):
            continue
        if not all(map(_finite_pos, (o, h, l, c))):
            continue
        if v < 0 or v != v:  # NaN volume -> treat as 0 (thin/halted day)
            v = 0.0
        typical = (h + l + c) / 3.0
        out.append(Candle(
            ts_open_ms=_date_to_utc_ms(idx), symbol=symbol, tf=tf,
            open=o, high=h, low=l, close=c, volume=v,
            quote_volume=v * typical, n_trades=0,
        ))
    out.sort(key=lambda k: k.ts_open_ms)
    return out


def _finite_pos(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf")) and x > 0.0


def _default_downloader(symbol: str, days: int, interval: str) -> Any:
    """Pull history from yfinance. Imported lazily so the package imports without
    yfinance installed (tests inject a frame and never hit this)."""
    import yfinance as yf  # noqa: PLC0415  (lazy on purpose)

    # A calendar buffer so `days` *trading* bars are actually available.
    period_days = max(days, MIN_BARS) * 2 + 10
    ticker = yf.Ticker(symbol)
    frame = ticker.history(period=f"{period_days}d", interval=interval, auto_adjust=True)
    return frame


def fetch_yfinance(
    symbol: str,
    days: int = 500,
    interval: str = "1d",
    *,
    db: Any = None,
    downloader: Callable[[str, int, str], Any] | None = None,
    min_bars: int = MIN_BARS,
) -> list[Candle]:
    """Download `symbol` history, map to candles, optionally persist, and return.

    Raises `InsufficientDataError` if fewer than `min_bars` usable bars come back --
    which is the common failure for a thin/misspelled ticker, and far better than
    forecasting off a handful of prints.
    """
    dl = downloader or _default_downloader
    frame = dl(symbol, days, interval)
    if frame is None or len(frame) == 0:
        raise InsufficientDataError(
            f"yfinance returned no rows for {symbol!r}. Is the ticker correct and "
            f"does it trade? (Thinly traded or delisted symbols often return empty.)"
        )
    candles = candles_from_frame(symbol, frame, tf=interval)
    if len(candles) < min_bars:
        raise InsufficientDataError(
            f"only {len(candles)} usable daily bars for {symbol!r}; need >= {min_bars}. "
            f"Too little history to build a stable drift/volatility estimate."
        )
    if db is not None:
        db.upsert_candles(candles)
        log.info("upserted %d %s %s candles", len(candles), symbol, interval)
    return candles


class InsufficientDataError(RuntimeError):
    """Not enough usable bars to forecast responsibly."""
