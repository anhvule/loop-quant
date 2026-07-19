"""Block-bootstrap Monte Carlo forecast.

GBM assumes returns are i.i.d. Normal. Real equity returns are fat-tailed, mildly
skewed, and cluster (calm and violent days come in runs). A *block* bootstrap
preserves all three: instead of drawing i.i.d. Normal shocks, it stitches the path
out of contiguous blocks of *actual historical* daily returns. A 2008-style cluster
in the data can therefore reappear intact in a simulated path.

Deterministic given `seed` (numpy default_rng), matching the repo's discipline.

`recenter_mu` optionally shifts every sampled return so the path's drift matches a
chosen estimate (e.g. the shrinkage-blended drift) while keeping the historical
*shape* -- dispersion, tails, clustering -- untouched.
"""

from __future__ import annotations

import numpy as np

from src.forecast.longrange import log_returns
from src.forecast.result import ForecastPath

_PCTS = (5.0, 25.0, 50.0, 75.0, 95.0)


def bootstrap_prices(closes, horizon: int, n_paths: int = 20_000, seed: int = 7,
                     block: int = 10, recenter_mu: float | None = None) -> np.ndarray:
    """Simulate and return the full price matrix, shape (n_paths, horizon).

    Column j-1 is the distribution of price on trading day j. Kept separate from
    `block_bootstrap` so the outlook can read intermediate-day distributions
    (e.g. end-September) off a single simulation.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if block < 1:
        raise ValueError("block must be >= 1")
    r = log_returns(closes)
    if r.size < block + 1:
        raise ValueError(f"history too short: {r.size} returns for block={block}")
    s0 = float(np.asarray(closes, dtype=float)[-1])

    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)             # ceil division
    max_start = r.size - block                  # last index where a full block fits
    starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    idx = starts[:, :, None] + offsets[None, None, :]        # (n_paths, n_blocks, block)
    idx = idx.reshape(n_paths, n_blocks * block)[:, :horizon]
    sampled = r[idx]                                          # (n_paths, horizon)

    if recenter_mu is not None:
        sampled = sampled + (recenter_mu - float(r.mean()))

    log_paths = np.cumsum(sampled, axis=1)
    return s0 * np.exp(log_paths)


def block_bootstrap(closes, horizon: int, n_paths: int = 20_000, seed: int = 7,
                    block: int = 10, recenter_mu: float | None = None) -> ForecastPath:
    """Per-day percentile bands from a block-bootstrap simulation."""
    prices = bootstrap_prices(closes, horizon, n_paths=n_paths, seed=seed,
                              block=block, recenter_mu=recenter_mu)
    pct = np.percentile(prices, _PCTS, axis=0)               # (5, horizon)
    p5, p25, p50, p75, p95 = (pct[i].tolist() for i in range(5))
    note = f"block bootstrap block={block} paths={n_paths} seed={seed}"
    if recenter_mu is not None:
        note += f" recentered mu={recenter_mu:.6f}"
    return ForecastPath(
        method="bootstrap", days=list(range(1, horizon + 1)),
        median=p50, lower=p5, upper=p95, interval_pct=90.0,
        p25=p25, p75=p75, terminal=prices[:, -1].tolist(), note=note,
    )
