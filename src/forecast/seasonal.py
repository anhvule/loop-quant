"""Seasonal (calendar-month) volatility scaling.

Realized volatility is not constant across the year -- historically Sep/Oct run hotter
for US equities. The base simulators apply one sigma to every future day; this module
lets the GBM path scale sigma by the calendar month each simulated day falls in.

It is OFF by default and only worth enabling if the calibration backtest
(`scripts.validate_outlook`) shows it improves tail calibration -- exactly the guard
the plan requires. Multipliers are clamped to a sane band so a thin month can't blow up.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from src.common.models import Candle


def monthly_vol_multipliers(candles: list[Candle], floor: float = 0.6,
                            cap: float = 1.8) -> dict[int, float]:
    """month (1-12) -> (that month's daily vol) / (overall daily vol), clamped."""
    rets: list[float] = []
    months: list[int] = []
    prev: Candle | None = None
    for c in candles:
        if prev is not None and prev.close > 0 and c.close > 0:
            rets.append(math.log(c.close / prev.close))
            d = dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date()
            months.append(d.month)
        prev = c
    r = np.asarray(rets)
    mth = np.asarray(months)
    overall = float(r.std()) if r.size else 0.0
    out: dict[int, float] = {}
    for m in range(1, 13):
        sel = r[mth == m]
        if sel.size >= 20 and overall > 0:
            out[m] = float(min(cap, max(floor, float(sel.std()) / overall)))
        else:
            out[m] = 1.0
    return out


def sigma_schedule(dates_axis: list[dt.date], base_sigma: float,
                   multipliers: dict[int, float]) -> np.ndarray:
    """Per-day sigma over the horizon: base_sigma scaled by each day's month multiplier."""
    return np.asarray([base_sigma * multipliers.get(d.month, 1.0) for d in dates_axis])


def effective_sigma(schedule: np.ndarray) -> np.ndarray:
    """Constant-equivalent sigma to each day t: sqrt( (sum of daily variances 1..t) / t ).

    So gbm_percentile(s0, mu, effective_sigma[t-1], t, q) reproduces the seasonal
    terminal quantile exactly (variance to day t = sum of per-day variances)."""
    v = np.asarray(schedule) ** 2
    cum = np.cumsum(v)
    t = np.arange(1, v.size + 1)
    return np.sqrt(cum / t)
