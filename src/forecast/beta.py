"""Market-factor (beta) engine for high-beta single names.

Why this exists: the SPY cone is *calibrated* (90.0% coverage over 28 years), but a
single name's own momentum drift is noise -- ASTS's raw estimate annualizes to ~+50%/yr,
which no honest model should project. This module borrows the validated part and refuses
the unvalidated part:

    name return = beta * (market return) + idiosyncratic residual
                  ^ drift anchored here     ^ resampled, never "forecast"

Two disciplines are enforced:

  1. DRIFT ANCHORING. Each name's simulated drift is set to `beta * mu_market` (the
     market's shrinkage-blended drift), NOT its own recent momentum. Idiosyncratic drift
     is zero unless explicitly opted into. The raw momentum figure is still reported so
     the caller can see what was refused.

  2. JOINT SAMPLING. One block bootstrap draws the SAME historical dates for every asset
     at once, so the simulated paths inherit the real correlation matrix, fat tails, and
     -- critically -- the co-crash days. Independent per-name sims would manufacture
     diversification that does not exist in a basket of correlated spec-tech names.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_BETA_LAMBDA = 0.995   # EWMA decay for beta (half-life ~138 trading days)
DEFAULT_VOL_LAMBDA = 0.94     # RiskMetrics decay for the vol-standardization filter


@dataclass(frozen=True, slots=True)
class BetaStats:
    symbol: str
    beta: float
    idio_vol: float        # daily stdev of the residual
    r2: float              # share of variance explained by the market
    vol: float             # the name's own daily vol
    raw_mu: float          # its own momentum drift -- REPORTED, deliberately NOT used
    n: int                 # observations the fit used
    se: float = 0.0        # standard error of beta -- r2 ~0.15 means this is NOT small


def align_closes(series: dict[str, dict[dt.date, float]]) -> tuple[list[dt.date], dict[str, np.ndarray]]:
    """Restrict every symbol to the dates ALL of them share.

    The joint window is capped by the youngest listing -- callers should report its
    length, because a short window means fewer distinct blocks to resample."""
    if not series:
        raise ValueError("no series supplied")
    common = set.intersection(*(set(d) for d in series.values()))
    dates = sorted(common)
    if len(dates) < 3:
        raise ValueError(f"only {len(dates)} common dates across {list(series)}")
    return dates, {s: np.asarray([series[s][d] for d in dates], dtype=float) for s in series}


def log_return_matrix(dates: list[dt.date], closes: dict[str, np.ndarray],
                      order: list[str]) -> np.ndarray:
    """(T-1, K) matrix of daily log returns, columns in `order`."""
    cols = []
    for s in order:
        px = closes[s]
        if np.any(px <= 0):
            raise ValueError(f"{s} has non-positive prices")
        cols.append(np.diff(np.log(px)))
    return np.column_stack(cols)


def _ewma_weights(n: int, lam: float) -> np.ndarray:
    w = lam ** np.arange(n - 1, -1, -1)      # oldest .. newest
    return w / w.sum()


def beta_stats(symbol: str, name_rets: np.ndarray, mkt_rets: np.ndarray,
               lam: float = DEFAULT_BETA_LAMBDA) -> BetaStats:
    """EWMA-weighted market beta, idiosyncratic vol, and r^2.

    Recent observations are weighted up (a name's beta drifts as its story changes),
    but far more slowly than a vol estimator -- beta is a structural quantity."""
    n = name_rets.size
    if n != mkt_rets.size:
        raise ValueError("return series length mismatch")
    if n < 30:
        raise ValueError(f"{symbol}: only {n} observations; too few for a beta fit")
    w = _ewma_weights(n, lam)
    mx, my = float(w @ mkt_rets), float(w @ name_rets)
    cov = float(w @ ((mkt_rets - mx) * (name_rets - my)))
    var_m = float(w @ (mkt_rets - mx) ** 2)
    if var_m <= 0:
        raise ValueError("market variance is zero")
    beta = cov / var_m
    alpha = my - beta * mx
    resid = name_rets - (alpha + beta * mkt_rets)
    var_y = float(w @ (name_rets - my) ** 2)
    idio = float(np.sqrt(max(w @ (resid - float(w @ resid)) ** 2, 0.0)))
    r2 = 0.0 if var_y <= 0 else max(0.0, 1.0 - float(w @ (resid - float(w @ resid)) ** 2) / var_y)
    # Standard error of beta. n_eff is the EWMA effective sample size (1/sum(w^2));
    # with r2 ~0.15 on these names this error is large enough to matter, so the report
    # prints it and the simulator can sample beta from it.
    n_eff = 1.0 / float(np.sum(w ** 2))
    se = idio / (math.sqrt(n_eff) * math.sqrt(var_m)) if var_m > 0 else 0.0
    return BetaStats(symbol=symbol, beta=beta, idio_vol=idio, r2=r2,
                     vol=float(np.std(name_rets)), raw_mu=float(np.mean(name_rets)),
                     n=n, se=float(se))


# ---------------------------------------------------------------------------
# Volatility standardization
#
# The plain bootstrap resamples raw returns, so simulated dispersion equals the
# AVERAGE volatility of the sampled history -- not today's. For SPY that is benign
# (vol mean-reverts around a stable level). For names whose vol swings 3-10x it
# guarantees miscalibration: when current vol exceeds the historical average the
# bands come out too narrow, which is exactly how TE (cov90 0.74) and ASTS (0.84)
# failed validation.
#
# Fix: divide each historical return by the volatility prevailing at that time,
# resample the standardized residuals, then rescale by CURRENT volatility.
# ---------------------------------------------------------------------------

def ewma_vol_path(rets: np.ndarray, lam: float = DEFAULT_VOL_LAMBDA,
                  seed_window: int = 20) -> tuple[np.ndarray, float]:
    """Trailing EWMA vol at each observation, plus the vol prevailing NOW.

    vol[i] uses only returns strictly before i -- no lookahead, so the standardized
    series is something a forecaster could actually have computed at the time."""
    n = rets.size
    if n == 0:
        return np.empty(0), 0.0
    vol = np.empty(n)
    seed = rets[:min(seed_window, n)]
    var = float(np.var(seed)) or float(np.var(rets)) or 1e-8
    for i in range(n):
        vol[i] = math.sqrt(max(var, 1e-12))
        var = lam * var + (1.0 - lam) * float(rets[i]) ** 2
    return vol, math.sqrt(max(var, 1e-12))


def vol_half_life_from_path(vol: np.ndarray, lag: int = 21, min_hl: float = 5.0,
                            max_hl: float = 400.0) -> float:
    """Half-life (trading days) of volatility mean reversion.

    Fitted from the `lag`-day autocorrelation of log EWMA vol: if log-vol behaves like
    an AR(1) with autocorrelation a over `lag` days, its half-life is
    -lag*ln2/ln(a). Vol is persistent (ASTS ~162d) but NOT permanent -- assuming
    today's panic level lasts six months is what inflates long-horizon cones."""
    if vol.size < 60:
        return max_hl
    lv = np.log(np.maximum(vol[30:], 1e-12))
    if lv.size <= lag + 10:
        return max_hl
    a = float(np.corrcoef(lv[:-lag], lv[lag:])[0, 1])
    if not (0.0 < a < 1.0):
        return max_hl
    return float(np.clip(-lag * math.log(2.0) / math.log(a), min_hl, max_hl))


def _vol_stats(ret_matrix: np.ndarray, lam: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(standardized returns, current vol, long-run vol, half-life) per column."""
    T, K = ret_matrix.shape
    Z = np.empty_like(ret_matrix, dtype=float)
    cur, longrun, hl = np.empty(K), np.empty(K), np.empty(K)
    for k in range(K):
        vol, cur[k] = ewma_vol_path(ret_matrix[:, k], lam)
        Z[:, k] = ret_matrix[:, k] / np.maximum(vol, 1e-12)
        longrun[k] = float(np.median(vol))          # median is robust to spikes
        hl[k] = vol_half_life_from_path(vol)
    return Z, cur, longrun, hl


def standardize_matrix(ret_matrix: np.ndarray, lam: float = DEFAULT_VOL_LAMBDA
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-column vol standardization. Returns (standardized matrix, current vols).

    Columns are standardized independently but sampled jointly downstream, so the
    cross-asset correlation of the *standardized* residuals is what survives -- which
    is the more stable quantity anyway."""
    Z, cur, _, _ = _vol_stats(ret_matrix, lam)
    return Z, cur


def vol_schedule(current: np.ndarray, long_run: np.ndarray, half_life: np.ndarray,
                 horizon: int) -> np.ndarray:
    """(horizon, K) per-day volatility decaying from today's level toward long-run.

        sigma_t = long_run + (current - long_run) * 2^(-t / half_life)

    Day 1 is essentially today's vol; far out it approaches the long-run level. This
    replaces "today's vol forever", which over-widens multi-month cones for any name
    currently in a high-vol regime."""
    t = np.arange(1, horizon + 1)[:, None]
    decay = np.power(0.5, t / np.maximum(half_life[None, :], 1e-9))
    return long_run[None, :] + (current[None, :] - long_run[None, :]) * decay


def recency_weights(n_starts: int, half_life_days: float, floor: float = 0.02) -> np.ndarray:
    """Sampling weights over block start indices, favouring recent history.

    A name whose market cap grew 30x should not have its micro-cap-era blocks sampled
    at parity with last month's. `floor` keeps old regimes represented (they are real
    evidence about how violent this name can get), just not dominant."""
    ages = np.arange(n_starts - 1, -1, -1, dtype=float)   # 0 = most recent start
    w = np.power(0.5, ages / max(half_life_days, 1e-9))
    w = w + floor * w.max()
    return w / w.sum()


def joint_bootstrap_prices(ret_matrix: np.ndarray, s0: np.ndarray, horizon: int,
                           n_paths: int = 10_000, seed: int = 7, block: int = 10,
                           recenter_mu: np.ndarray | None = None,
                           vol_standardize: bool = False,
                           vol_lambda: float = DEFAULT_VOL_LAMBDA,
                           beta_draw: np.ndarray | None = None,
                           vol_term_structure: bool = False,
                           recency_half_life: float | None = None) -> np.ndarray:
    """Block bootstrap over ROWS (dates), shared across all columns (assets).

    Returns (n_paths, horizon, K) float32 prices. Because every asset uses the same
    sampled dates, the cross-asset correlation, joint fat tails, and co-crash structure
    of the history survive into the simulation.

    `recenter_mu` (length K) shifts each asset's returns so its simulated drift equals
    the supplied target -- this is where beta anchoring is applied.

    `vol_standardize=True` samples volatility-standardized residuals and rescales to
    TODAY's volatility, so the cone reflects the current regime rather than the
    historical average. Strongly recommended for names whose vol swings widely.

    `beta_draw` (n_paths,) optionally shifts each path's drift by
    (beta_i - beta_mean) * market_drift, propagating beta *estimation uncertainty*
    into the bands. Column 0 is assumed to be the market and is left untouched.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    T, K = ret_matrix.shape
    if block < 1 or T < block + 1:
        raise ValueError(f"history too short: {T} rows for block={block}")
    if s0.size != K:
        raise ValueError("s0 length must match number of columns")

    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    n_starts = T - block + 1
    if recency_half_life:
        w = recency_weights(n_starts, recency_half_life)
        starts = rng.choice(n_starts, size=(n_paths, n_blocks), p=w)
    else:
        starts = rng.integers(0, n_starts, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, n_blocks * block)[:, :horizon]

    if vol_standardize:
        Z, cur_vol, longrun_vol, hl = _vol_stats(ret_matrix, vol_lambda)
        if vol_term_structure:
            sched = vol_schedule(cur_vol, longrun_vol, hl, horizon)   # (horizon, K)
            sampled = Z[idx] * sched[None, :, :]
            base_mu = Z.mean(axis=0) * sched.mean(axis=0)
        else:
            sampled = Z[idx] * cur_vol[None, None, :]        # today's vol, held flat
            base_mu = Z.mean(axis=0) * cur_vol
    else:
        sampled = ret_matrix[idx]
        base_mu = ret_matrix.mean(axis=0)

    if recenter_mu is not None:
        sampled = sampled + (np.asarray(recenter_mu, dtype=float) - base_mu)[None, None, :]
    elif vol_standardize:
        # keep the historical drift even though scale changed
        sampled = sampled + (ret_matrix.mean(axis=0) - base_mu)[None, None, :]

    if beta_draw is not None and K > 1:
        # per-path drift perturbation from beta uncertainty (market column exempt)
        mkt_mu = float(recenter_mu[0]) if recenter_mu is not None else float(base_mu[0])
        extra = (np.asarray(beta_draw, dtype=float) - np.mean(beta_draw)) * mkt_mu
        sampled[:, :, 1:] = sampled[:, :, 1:] + extra[:, None, None]

    prices = s0[None, None, :] * np.exp(np.cumsum(sampled, axis=1))
    return prices.astype(np.float32)


def anchored_drifts(betas: dict[str, BetaStats], mu_market: float,
                    idio_drift: bool = False) -> dict[str, float]:
    """Target daily drift per name: beta * mu_market (+ its own alpha only if opted in).

    Default refuses idiosyncratic drift: over a few hundred bars a high-beta name's
    residual mean is dominated by noise, and projecting it forward is how a forecast
    turns into a fantasy."""
    out = {}
    for s, b in betas.items():
        out[s] = b.beta * mu_market + (b.raw_mu - b.beta * mu_market if idio_drift else 0.0)
    return out
