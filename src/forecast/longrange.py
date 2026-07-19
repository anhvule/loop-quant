"""Long-range drift and volatility estimators.

The 15-day forecaster estimates a single (mu, sigma) from a trailing 500-day window.
That is fine for a few weeks but weak at 2-3 months, where drift is noise-dominated
and a constant trailing sigma ignores volatility regimes. This module supplies the
sturdier estimators the multi-month `outlook` needs:

  * drift  -- shrink a noisy recent mean toward a long-history mean.
  * vol    -- EWMA (RiskMetrics) so recent turbulence is weighted up, optionally
              anchored to the options market's expectation via ^VIX.

Everything is numpy-only and deterministic. No new dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS_YEAR = 252
RISKMETRICS_LAMBDA = 0.94   # RiskMetrics daily decay
DEFAULT_RECENT_WINDOW = 500
DEFAULT_W_RECENT = 0.30     # weight on the recent-window drift vs the long-run mean


@dataclass(frozen=True, slots=True)
class Estimates:
    """Drift/vol inputs for the long-range simulators, plus the components that
    produced them (so the report can show the user how the blend was formed)."""
    mu_recent: float          # mean daily log return over the recent window
    mu_long: float            # mean daily log return over all available history
    mu_blend: float           # shrinkage blend actually used
    sigma_ewma: float         # EWMA daily volatility
    sigma_vix: float | None   # ^VIX-implied daily volatility (None if unavailable)
    sigma_blend: float        # volatility actually used
    n_returns: int


def log_returns(closes) -> np.ndarray:
    """Daily log returns of a positive close series (non-positive prices dropped)."""
    a = np.asarray(closes, dtype=float)
    a = a[a > 0]
    if a.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.log(a))


def blended_drift(closes, recent_window: int = DEFAULT_RECENT_WINDOW,
                  w_recent: float = DEFAULT_W_RECENT) -> tuple[float, float, float]:
    """(mu_blend, mu_recent, mu_long).

    Shrinks the recent-window mean toward the full-history mean. The recent mean is
    almost pure noise at daily frequency; anchoring it to a decade of history keeps
    a 75-day projection from inheriting a spurious trend.
    """
    r = log_returns(closes)
    if r.size == 0:
        return 0.0, 0.0, 0.0
    mu_long = float(r.mean())
    recent = r[-recent_window:] if r.size > recent_window else r
    mu_recent = float(recent.mean())
    mu_blend = w_recent * mu_recent + (1.0 - w_recent) * mu_long
    return mu_blend, mu_recent, mu_long


def ewma_sigma(returns, lam: float = RISKMETRICS_LAMBDA) -> float:
    """EWMA daily volatility (RiskMetrics), zero-mean convention.

    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_{t-1}^2, seeded with the sample
    variance. Returns the final sqrt(variance).
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    var = float(np.var(r))
    for x in r:
        var = lam * var + (1.0 - lam) * x * x
    return math.sqrt(max(var, 0.0))


def vix_forward_sigma(vix_close: float) -> float:
    """Convert a ^VIX index level (annualized % implied vol) to a daily sigma."""
    return float(vix_close) / 100.0 / math.sqrt(TRADING_DAYS_YEAR)


def estimate(closes, vix_close: float | None = None,
             recent_window: int = DEFAULT_RECENT_WINDOW,
             w_recent: float = DEFAULT_W_RECENT,
             lam: float = RISKMETRICS_LAMBDA) -> Estimates:
    """Full drift + vol estimate. If `vix_close` is given, blend it 50/50 with EWMA."""
    r = log_returns(closes)
    mu_blend, mu_recent, mu_long = blended_drift(closes, recent_window, w_recent)
    s_ewma = ewma_sigma(r, lam)
    s_vix = vix_forward_sigma(vix_close) if vix_close else None
    if s_vix is not None and s_vix > 0.0:
        s_blend = 0.5 * s_ewma + 0.5 * s_vix
    else:
        s_blend = s_ewma
    return Estimates(mu_recent=mu_recent, mu_long=mu_long, mu_blend=mu_blend,
                     sigma_ewma=s_ewma, sigma_vix=s_vix, sigma_blend=s_blend,
                     n_returns=int(r.size))
