"""Calibration statistics helpers.

A coverage rate printed without an interval invites exactly the error this project
keeps trying to avoid: reading "0.93 from 40 trials" as if it were "0.93 from 400".
These helpers make the uncertainty of a calibration estimate explicit, and adjust for
the fact that overlapping forecast windows are not independent trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRADING_DAYS_PER_MONTH = 21


@dataclass(frozen=True, slots=True)
class Rate:
    """An observed rate with a Wilson interval and an overlap-adjusted sample size."""
    k: int
    n: int
    lo: float
    hi: float
    n_eff: float

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def covers(self, target: float) -> bool:
        """Is `target` inside the interval? If yes, the data cannot rule it out."""
        return self.lo <= target <= self.hi

    def verdict(self, target: float = 0.90) -> str:
        if not self.n:
            return "no data"
        if not self.covers(target):
            return f"RULED OUT (not {target:.0%})"
        width = self.hi - self.lo
        return "consistent (weak evidence)" if width > 0.12 else "consistent"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves sensibly at small n and near 0/1,
    where the textbook normal approximation does not."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - rad) / d), min(1.0, (centre + rad) / d)


def effective_n(n: int, horizon_days: int,
                step_days: int = TRADING_DAYS_PER_MONTH) -> float:
    """Independent-trial equivalent when forecasts overlap.

    Month-end forecasts looking `horizon_days` ahead share most of their window with
    their neighbours: consecutive trials are ~step/horizon independent. A 73-day
    forecast made monthly is only ~29% as informative per trial as it looks."""
    if horizon_days <= 0:
        return float(n)
    return float(n) * min(1.0, step_days / horizon_days)


def rate(k: int, n: int, horizon_days: int = 0, z: float = 1.96) -> Rate:
    """Observed rate + interval. If `horizon_days` is given, the interval is widened
    to reflect the overlap-adjusted (smaller) effective sample."""
    n_eff = effective_n(n, horizon_days) if horizon_days else float(n)
    if n <= 0:
        return Rate(k, n, float("nan"), float("nan"), n_eff)
    # widen using n_eff while keeping the observed proportion
    k_eff = round(k * n_eff / n) if n else 0
    lo, hi = wilson(int(k_eff), int(round(n_eff)) or 1, z)
    return Rate(k, n, lo, hi, n_eff)
