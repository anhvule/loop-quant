"""Outlook: calendar mapping, probability tables, seasonality, and end-to-end build."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from src.common.models import Candle
from src.forecast.outlook import (
    NYSE_HOLIDAYS_2026,
    build_outlook,
    is_trading_day,
    map_horizon_to_dates,
    monthly_seasonality,
    prob_table_from_samples,
    prob_table_gbm,
    trading_days_between,
)

_DAY_MS = 86_400_000


def _candles(n: int, end: dt.date, s0: float = 100.0) -> list[Candle]:
    """n consecutive daily bars ending on `end` (calendar days; fine for tests)."""
    out = []
    price = s0
    for i in range(n):
        d = end - dt.timedelta(days=(n - 1 - i))
        ts = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
        price *= (1.0 + 0.0004 + 0.01 * math.sin(i / 5.0))
        h, l = price * 1.01, price * 0.99
        out.append(Candle(ts_open_ms=ts, symbol="TEST", tf="1d", open=price, high=h,
                          low=l, close=price, volume=1000.0,
                          quote_volume=1000.0 * price, n_trades=0))
    return out


# ---- calendar ----

def test_is_trading_day():
    assert is_trading_day(dt.date(2026, 7, 20))       # Monday
    assert not is_trading_day(dt.date(2026, 7, 18))   # Saturday
    assert not is_trading_day(dt.date(2026, 9, 7))    # Labor Day (holiday)
    assert dt.date(2026, 9, 7) in NYSE_HOLIDAYS_2026


def test_trading_days_between_skips_weekend_and_holiday():
    # Fri 2026-09-04 -> Tue 2026-09-08, spanning Labor Day Mon 09-07.
    # Trading days after Fri: Mon(holiday,no), Tue(yes) = 1.
    assert trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 8)) == 1


def test_map_horizon_all_trading_days():
    dates = map_horizon_to_dates(dt.date(2026, 7, 17), 30)
    assert len(dates) == 30
    assert all(is_trading_day(d) for d in dates)
    assert dates == sorted(dates)


# ---- probability tables ----

def test_prob_table_gbm_symmetry_zero_drift():
    tbl = prob_table_gbm(100.0, 0.0, 0.01, 30)
    assert tbl["p_up"] == pytest.approx(0.5, abs=0.01)
    for q in (5, 25, 50, 75, 95):
        assert tbl["pctiles"][q] > 0
    ordered = [tbl["pctiles"][q] for q in (5, 25, 50, 75, 95)]
    assert ordered == sorted(ordered)


def test_prob_table_from_samples_counts():
    import numpy as np
    term = np.array([90.0, 95.0, 100.0, 105.0, 120.0])
    tbl = prob_table_from_samples(term, s0=100.0)
    assert tbl["p_up"] == pytest.approx(2 / 5)         # 105, 120
    assert tbl["p_le_5"] == pytest.approx(2 / 5)        # 90, 95
    assert tbl["p_le_10"] == pytest.approx(1 / 5)       # 90


def test_seasonality_has_months_and_hit_rate_range():
    seas = monthly_seasonality(_candles(400, dt.date(2026, 7, 17)))
    assert seas
    for m, s in seas.items():
        assert 1 <= m <= 12
        assert 0.0 <= s["hit_rate"] <= 1.0
        assert s["n"] >= 1


# ---- end to end ----

def test_build_outlook_end_to_end():
    candles = _candles(400, dt.date(2026, 7, 17))
    targets = [dt.date(2026, 9, 30), dt.date(2026, 10, 30)]
    o = build_outlook("TEST", candles, targets, dt.date(2026, 7, 17),
                      n_paths=3000, seed=7)
    assert len(o.per_date) == 2
    assert o.per_date[0].date == dt.date(2026, 9, 30)
    assert o.per_date[0].trading_day < o.per_date[1].trading_day
    assert o.horizon >= o.per_date[1].trading_day
    for df in o.per_date:
        for tbl in (df.gbm, df.boot):
            assert 0.0 <= tbl["p_up"] <= 1.0
            pc = [tbl["pctiles"][q] for q in (5, 25, 50, 75, 95)]
            assert pc == sorted(pc)
    # cones are full length and ordered
    assert len(o.gbm_path.median) == o.horizon
    assert len(o.boot_path.median) == o.horizon


def test_build_outlook_uncertainty_grows_with_horizon():
    candles = _candles(400, dt.date(2026, 7, 17))
    o = build_outlook("TEST", candles, [dt.date(2026, 9, 30), dt.date(2026, 10, 30)],
                      dt.date(2026, 7, 17), n_paths=4000, seed=7)
    sep, octo = o.per_date[0], o.per_date[1]
    sep_w = sep.gbm["pctiles"][95] - sep.gbm["pctiles"][5]
    oct_w = octo.gbm["pctiles"][95] - octo.gbm["pctiles"][5]
    assert oct_w > sep_w
