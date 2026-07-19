"""ARIMA price-path forecast (statsmodels).

Fits an ARIMA model on *log* prices (so the forecast and its intervals are
multiplicative/positive once exponentiated) and projects `horizon` days forward
with a 90% prediction interval.

If `order` is None, a small AIC grid search over (p,d,q) picks the model. If every
fit fails to converge -- possible on a very short or pathological series -- we fall
back to a drift-only projection (the same GBM point/interval math), so the caller
always gets a usable path and a note explaining what happened.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np

from src.forecast.features import log_return_stats
from src.forecast.result import ForecastPath

log = logging.getLogger(__name__)

# Small, fast search space. d in {0,1} covers non-stationary log-price (d=1 is a
# random walk with drift, the usual winner for daily equity prices).
_P_RANGE = (0, 1, 2)
_D_RANGE = (0, 1)
_Q_RANGE = (0, 1, 2)

# z for a 90% two-sided normal interval, for the drift-only fallback.
_Z90 = 1.6448536269514722


def _grid_search_order(log_prices: "np.ndarray") -> tuple[int, int, int] | None:
    from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415 (lazy import)

    best_aic = math.inf
    best: tuple[int, int, int] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in _P_RANGE:
            for d in _D_RANGE:
                for q in _Q_RANGE:
                    if p == 0 and q == 0 and d == 0:
                        continue
                    try:
                        res = ARIMA(log_prices, order=(p, d, q)).fit()
                    except Exception:  # convergence/LinAlg errors: just skip
                        continue
                    aic = float(res.aic)
                    if aic < best_aic:
                        best_aic, best = aic, (p, d, q)
    return best


def _drift_only(closes: list[float], horizon: int, reason: str) -> ForecastPath:
    """GBM-style point + 90% interval from historical drift/vol, no model fit."""
    mu, sigma = log_return_stats(closes)
    s0 = float(closes[-1])
    days = list(range(1, horizon + 1))
    median, lower, upper = [], [], []
    for t in days:
        center = s0 * math.exp(mu * t)
        half = _Z90 * sigma * math.sqrt(t)   # log-space interval half-width
        median.append(center)
        lower.append(s0 * math.exp(mu * t - half))
        upper.append(s0 * math.exp(mu * t + half))
    return ForecastPath(
        method="arima", days=days, median=median, lower=lower, upper=upper,
        interval_pct=90.0, note=f"drift-only fallback ({reason}); mu={mu:.6f} sigma={sigma:.6f}",
    )


def forecast_arima(
    closes: list[float],
    horizon: int,
    order: tuple[int, int, int] | None = None,
) -> ForecastPath:
    """Fit ARIMA on log-price and project `horizon` days with a 90% interval."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not closes or closes[-1] <= 0:
        raise ValueError("closes must be non-empty with a positive last price")
    if any(c <= 0 for c in closes):
        return _drift_only(closes, horizon, "non-positive price in history")

    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415
    except ImportError:
        return _drift_only(closes, horizon, "statsmodels not installed")

    log_prices = np.log(np.asarray(closes, dtype=float))

    chosen = order or _grid_search_order(log_prices)
    if chosen is None:
        return _drift_only(closes, horizon, "no ARIMA order converged")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(log_prices, order=chosen).fit()
            fc = res.get_forecast(steps=horizon)
            mean_log = np.asarray(fc.predicted_mean, dtype=float)
            ci_log = np.asarray(fc.conf_int(alpha=0.10), dtype=float)  # 90% interval
    except Exception as e:  # noqa: BLE001
        return _drift_only(closes, horizon, f"fit failed: {type(e).__name__}")

    median = np.exp(mean_log).tolist()
    lower = np.exp(ci_log[:, 0]).tolist()
    upper = np.exp(ci_log[:, 1]).tolist()
    return ForecastPath(
        method="arima", days=list(range(1, horizon + 1)),
        median=median, lower=lower, upper=upper, interval_pct=90.0,
        note=f"ARIMA{chosen} on log-price; 90% interval",
    )
