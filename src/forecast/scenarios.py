"""Stress scenarios: "what if you're right about a crash?"

Two honest ways to explore the downside, both clearly labelled as SCENARIOS, not
probabilities:

  1. Historical replays -- take the worst `horizon`-day return sequences that actually
     happened (2008, 2020, 2018, ... whatever is in the data) and re-run each from
     TODAY's price. Plus a hardcoded 1987 replay, since our data starts ~1997 and the
     October 1987 crash is the canonical case.

  2. Crisis-conditioned bootstrap -- a bootstrap cone that samples return blocks ONLY
     from historically high-volatility regimes. It answers "if the next months look
     like a turbulent period, where does price go?" -- a stressed cone next to the base.

None of this asserts a crash is coming. It quantifies the shape of one if it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.forecast.longrange import log_returns
from src.forecast.result import ForecastPath


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    trough_pct: float        # worst point along the path, vs today's price
    end_pct: float           # terminal, vs today's price
    trough_price: float
    end_price: float
    path: list[float] = field(default_factory=list)


def replay_log_returns(seq: np.ndarray, s0: float) -> tuple[list[float], float, float]:
    """Apply a historical log-return sequence from `s0`. Returns (path, trough, end)."""
    cum = np.exp(np.cumsum(seq))
    path = (s0 * cum).tolist()
    return path, float(s0 * cum.min()), float(path[-1])


def historical_stress(candles, horizon: int, k: int = 3) -> list[Scenario]:
    """The `k` worst non-overlapping `horizon`-day windows in the history, each
    replayed from today's price. Ranked by trough (deepest drawdown within the window).
    """
    closes = np.asarray([c.close for c in candles], dtype=float)
    dates = [c.ts_open_ms for c in candles]
    r = log_returns(closes)
    s0 = float(closes[-1])
    L = r.size
    if L < horizon + 1:
        return []

    import datetime as dt
    troughs = []
    for i in range(0, L - horizon + 1):
        cum = np.exp(np.cumsum(r[i:i + horizon]))
        troughs.append((float(cum.min()), i))
    troughs.sort()                                  # ascending: worst first

    picked: list[int] = []
    scenarios: list[Scenario] = []
    for trough_ratio, i in troughs:
        if any(abs(i - j) < horizon for j in picked):
            continue                                # keep windows non-overlapping
        picked.append(i)
        seq = r[i:i + horizon]
        path, trough_px, end_px = replay_log_returns(seq, s0)
        start_d = dt.datetime.fromtimestamp(dates[i] / 1000, dt.timezone.utc).date()
        scenarios.append(Scenario(
            name=f"worst window from {start_d:%Y-%m}",
            trough_pct=trough_px / s0 - 1.0, end_pct=end_px / s0 - 1.0,
            trough_price=trough_px, end_price=end_px, path=path))
        if len(scenarios) >= k:
            break
    return scenarios


def scenario_1987(s0: float) -> Scenario:
    """Approximate Aug->Dec 1987 S&P monthly returns replayed from today's price.

    Hardcoded (our daily data starts ~1997). Monthly steps, so the trough is the
    month-end low -- the intraday Oct-19-1987 low (~-20% in a day) was deeper still.
    """
    monthly = [0.035, -0.024, -0.218, -0.085, 0.073]   # Aug, Sep, Oct, Nov, Dec 1987
    price, trough, path = s0, s0, []
    for r in monthly:
        price *= (1.0 + r)
        path.append(price)
        trough = min(trough, price)
    return Scenario(name="1987 Black Monday (approx, monthly)",
                    trough_pct=trough / s0 - 1.0, end_pct=price / s0 - 1.0,
                    trough_price=trough, end_price=price, path=path)


def high_vol_start_pool(closes, block: int, window: int = 20,
                        quantile: float = 0.75) -> np.ndarray:
    """Block start indices whose trailing-`window` volatility is in the top
    `1-quantile` of history -- the 'turbulent regime' sampling pool."""
    r = log_returns(closes)
    L = r.size
    vol = np.full(L, np.nan)
    for i in range(window, L):
        vol[i] = r[i - window:i].std()
    thr = np.nanquantile(vol, quantile)
    starts = [i for i in range(0, L - block + 1)
              if not np.isnan(vol[i]) and vol[i] >= thr]
    return np.asarray(starts, dtype=int)


def crisis_bootstrap(closes, horizon: int, n_paths: int = 20_000, seed: int = 7,
                     block: int = 10, quantile: float = 0.75,
                     recenter_mu: float | None = None) -> ForecastPath:
    """Bootstrap cone sampling blocks only from high-volatility regimes."""
    r = log_returns(closes)
    if r.size < block + 1:
        raise ValueError("history too short for crisis bootstrap")
    s0 = float(np.asarray(closes, dtype=float)[-1])
    pool = high_vol_start_pool(closes, block, quantile=quantile)
    if pool.size == 0:
        raise ValueError("no high-volatility windows found")

    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    chosen = pool[rng.integers(0, pool.size, size=(n_paths, n_blocks))]
    offsets = np.arange(block)
    idx = (chosen[:, :, None] + offsets[None, None, :]).reshape(n_paths, n_blocks * block)[:, :horizon]
    sampled = r[idx]
    if recenter_mu is not None:
        sampled = sampled + (recenter_mu - float(r.mean()))
    prices = s0 * np.exp(np.cumsum(sampled, axis=1))
    pct = np.percentile(prices, [5, 25, 50, 75, 95], axis=0)
    p5, p25, p50, p75, p95 = (pct[i].tolist() for i in range(5))
    return ForecastPath(
        method="crisis-bootstrap", days=list(range(1, horizon + 1)),
        median=p50, lower=p5, upper=p95, interval_pct=90.0, p25=p25, p75=p75,
        terminal=prices[:, -1].tolist(),
        note=f"crisis bootstrap (top {(1 - quantile) * 100:.0f}% vol regime) "
             f"block={block} paths={n_paths} seed={seed}",
    )
