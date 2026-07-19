"""yfinance -> Candle mapping and the insufficient-data guard.

Uses a hand-built pandas frame (pandas is already a project dep) so the mapping is
tested with no network and no yfinance installed. Covers flat columns, yfinance
MultiIndex columns, NaN-row skipping, and the min-bars guard.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.forecast.stock_data import (
    InsufficientDataError,
    candles_from_frame,
    fetch_yfinance,
)


def _frame(n: int, multiindex: bool = False, symbol: str = "WYFI") -> pd.DataFrame:
    idx = pd.to_datetime([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    base = 100.0
    data = {
        "Open": [base + i for i in range(n)],
        "High": [base + i + 2 for i in range(n)],
        "Low": [base + i - 1 for i in range(n)],
        "Close": [base + i + 1 for i in range(n)],
        "Volume": [1_000 + i for i in range(n)],
    }
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, [symbol]])
    return df


def test_candles_from_frame_maps_ohlcv_and_utc_midnight():
    df = _frame(3)
    candles = candles_from_frame("WYFI", df)
    assert len(candles) == 3
    c0 = candles[0]
    assert c0.symbol == "WYFI" and c0.tf == "1d"
    assert (c0.open, c0.high, c0.low, c0.close, c0.volume) == (100.0, 102.0, 99.0, 101.0, 1000.0)
    # ts is that calendar date at 00:00 UTC
    expected = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert c0.ts_open_ms == expected
    # quote_volume == volume * typical price, so VWAP == typical price downstream
    assert c0.quote_volume == pytest.approx(1000.0 * (102.0 + 99.0 + 101.0) / 3.0)


def test_candles_from_frame_handles_multiindex_columns():
    df = _frame(3, multiindex=True)
    candles = candles_from_frame("WYFI", df)
    assert len(candles) == 3
    assert candles[-1].close == 103.0


def test_candles_from_frame_skips_nan_rows():
    df = _frame(4)
    df.iloc[3, df.columns.get_loc("Close")] = float("nan")   # trailing unsettled bar
    candles = candles_from_frame("WYFI", df)
    assert len(candles) == 3


def test_fetch_yfinance_raises_on_thin_history():
    thin = _frame(5)   # far below MIN_BARS
    with pytest.raises(InsufficientDataError):
        fetch_yfinance("WYFI", days=500, downloader=lambda s, d, i: thin)


def test_fetch_yfinance_raises_on_empty():
    with pytest.raises(InsufficientDataError):
        fetch_yfinance("WYFI", days=500, downloader=lambda s, d, i: _frame(0))


def test_fetch_yfinance_persists_when_db_given(db):
    df = _frame(80)
    candles = fetch_yfinance("WYFI", days=500, db=db, downloader=lambda s, d, i: df)
    assert len(candles) == 80
    assert len(db.get_candles("WYFI", "1d")) == 80
