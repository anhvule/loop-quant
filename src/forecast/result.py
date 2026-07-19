"""Shared result type for both forecasters.

Every method (GBM, ARIMA) returns the same shape so `report.py` can render them
uniformly. `lower`/`upper` are a symmetric central interval (default 90%: the 5th
and 95th percentiles), `median` the central estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ForecastPath:
    """A per-day price forecast over `horizon` trading days.

    All list fields have length == horizon and are aligned by index: element i is
    the forecast for trading day i+1 (day 0 is the last known close, not included).
    """
    method: str                        # "gbm" | "arima"
    days: list[int]                    # [1, 2, ..., horizon]
    median: list[float]                # central estimate per day
    lower: list[float]                 # low edge of the interval (e.g. 5th pct)
    upper: list[float]                 # high edge of the interval (e.g. 95th pct)
    interval_pct: float = 90.0         # width of [lower, upper], for labelling
    p25: list[float] = field(default_factory=list)   # optional inner band (GBM)
    p75: list[float] = field(default_factory=list)
    terminal: list[float] = field(default_factory=list)  # optional terminal draws
    note: str = ""                     # e.g. chosen ARIMA order, or fallback reason

    def __post_init__(self) -> None:
        n = len(self.days)
        for name in ("median", "lower", "upper"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"ForecastPath.{name} has length {len(getattr(self, name))}, "
                    f"expected {n} (== len(days))"
                )
