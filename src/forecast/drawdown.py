"""Intra-path drawdown risk -- "does it ever touch -X% before the target date?"

The terminal probability tables answer "where is price ON date D". They are blind to
a crash-and-recover: a -20% plunge in October that bounces back by December shows up
nowhere. For crash risk the honest question is the *first-passage* probability -- the
chance the path touches a level at ANY time up to the horizon -- and the distribution
of the worst drawdown along the way.

Two engines, same as the rest of the package:
  * GBM: closed-form first-passage (reflection principle). Exact, no sampling.
  * Bootstrap: read path minima / running max-drawdown straight off the simulated
    price matrix -- crashes present in the history are present in the paths.

Caveat: the bootstrap paths are daily closes, so an intraday spike through a level
between two closes is missed. This makes the bootstrap touch-probabilities a mild
UNDER-estimate; stated plainly in the report.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()

# Drawdown thresholds reported everywhere (fractional moves down from spot).
DRAWDOWN_LEVELS = (-0.05, -0.10, -0.15, -0.20)


def touch_prob_gbm(s0: float, mu: float, sigma: float, horizon: int, level: float) -> float:
    """P(min price over [0, horizon] <= `level`) under GBM, closed form.

    For log-price a Brownian motion with drift mu and vol sigma, and a lower barrier
    b = log(level/s0) < 0, the running-minimum law gives
        P = Phi((b - mu*h)/(sigma*sqrt(h))) + exp(2*mu*b/sigma^2) * Phi((b + mu*h)/(sigma*sqrt(h))).
    """
    if level >= s0:
        return 1.0                      # already at/above a "down" barrier
    if level <= 0.0:
        return 0.0
    b = math.log(level / s0)            # < 0
    if sigma <= 0.0:
        min_price = s0 * min(1.0, math.exp(mu * horizon))
        return 1.0 if min_price <= level else 0.0
    root = sigma * math.sqrt(horizon)
    term1 = _ND.cdf((b - mu * horizon) / root)
    # exp() can overflow for large upward drift * deep barrier; guard to [0,1] domain.
    expo = 2.0 * mu * b / (sigma * sigma)
    term2 = math.exp(expo) * _ND.cdf((b + mu * horizon) / root) if expo < 700 else 0.0
    return float(min(1.0, max(0.0, term1 + term2)))


def path_risk_gbm(s0: float, mu: float, sigma: float, horizon: int,
                  levels: tuple[float, ...] = DRAWDOWN_LEVELS) -> dict:
    """Touch probabilities at each drawdown level, closed form."""
    return {lv: touch_prob_gbm(s0, mu, sigma, horizon, s0 * (1.0 + lv)) for lv in levels}


def path_risk_bootstrap(prices: np.ndarray, s0: float,
                        levels: tuple[float, ...] = DRAWDOWN_LEVELS) -> dict:
    """Touch probabilities + max-drawdown distribution from a bootstrap price matrix.

    `prices` is (n_paths, horizon). Touch is close-based (see module caveat)."""
    n = prices.shape[0]
    mins = prices.min(axis=1)                       # worst close per path
    touch = {lv: float(np.count_nonzero(mins <= s0 * (1.0 + lv))) / n for lv in levels}

    # Max drawdown along each path, including the start (s0) in the running peak.
    full = np.concatenate([np.full((n, 1), s0), prices], axis=1)
    run_max = np.maximum.accumulate(full, axis=1)
    dd = 1.0 - full / run_max                       # fractional drawdown, >= 0
    max_dd = dd.max(axis=1)
    return {
        "touch": touch,
        "maxdd_median": float(np.percentile(max_dd, 50)),
        "maxdd_p90": float(np.percentile(max_dd, 90)),
        "maxdd_p99": float(np.percentile(max_dd, 99)),
    }
