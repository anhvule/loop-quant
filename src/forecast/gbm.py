"""Monte Carlo Geometric Brownian Motion price-path forecast.

Estimate the daily log-return drift (mu) and volatility (sigma) from history, then
simulate `n_paths` forward price paths and read percentile bands off the ensemble.

Deterministic given `seed` -- `numpy.random.default_rng(seed)` produces byte-identical
draws across runs and machines, matching the repo's determinism discipline (the
backtester is byte-identical by design; so is this).

Model (discrete GBM in log space): each day's log return ~ Normal(mu + drift_tilt, sigma).
`mu` is the *empirical mean* log return, which already embeds the -0.5*sigma^2 Ito term,
so no extra correction is applied. `drift_tilt` is an additive daily-return nudge the
caller can derive from the technical signal (see scripts/forecast.py).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from src.forecast.features import log_return_stats
from src.forecast.longrange import log_returns
from src.forecast.result import ForecastPath

# Percentiles reported per day. 5/95 form the headline 90% band; 25/75 an inner band.
_PCTS = (5.0, 25.0, 50.0, 75.0, 95.0)

_ND = NormalDist()  # standard normal, for analytic (closed-form) GBM statistics

_T_DOF_FLOOR = 4.5   # keep kurtosis finite (Student-t needs df > 4)
_T_DOF_CAP = 100.0   # effectively normal above this


def fit_t_dof(returns, floor: float = _T_DOF_FLOOR, cap: float = _T_DOF_CAP) -> float:
    """Degrees of freedom for a Student-t matched to the returns' excess kurtosis.

    For t with df v, excess kurtosis = 6/(v-4)  =>  v = 4 + 6/exkurt. Thin-tailed or
    tiny samples fall back to `cap` (i.e. effectively Gaussian)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 20:
        return cap
    s = float(r.std())
    if s <= 0:
        return cap
    z = (r - r.mean()) / s
    exkurt = float(np.mean(z ** 4) - 3.0)
    if exkurt <= 0.1:
        return cap
    return float(min(cap, max(floor, 4.0 + 6.0 / exkurt)))


# ---------------------------------------------------------------------------
# Analytic GBM statistics -- exact, no Monte Carlo noise.
#
# Under GBM, log(S_h / S_0) ~ Normal(mu*h, sigma^2 * h). So terminal quantiles and
# threshold probabilities have closed forms. The long-range outlook and the
# calibration backtest use these (fast, deterministic, no sampling error) rather
# than re-running Monte Carlo at every target day.
# ---------------------------------------------------------------------------

def gbm_percentile(s0: float, mu: float, sigma: float, horizon: int, q: float) -> float:
    """The q-quantile (0<q<1) of price at `horizon` days under GBM."""
    if sigma <= 0.0:
        return s0 * math.exp(mu * horizon)
    z = _ND.inv_cdf(q)
    return s0 * math.exp(mu * horizon + z * sigma * math.sqrt(horizon))


def gbm_prob_above(s0: float, mu: float, sigma: float, horizon: int, level: float) -> float:
    """P(price at `horizon` >= `level`) under GBM."""
    if level <= 0.0:
        return 1.0
    if sigma <= 0.0:
        return 1.0 if s0 * math.exp(mu * horizon) >= level else 0.0
    z = (math.log(level / s0) - mu * horizon) / (sigma * math.sqrt(horizon))
    return 1.0 - _ND.cdf(z)


def gbm_expected_value(s0: float, mu: float, sigma: float, horizon: int) -> float:
    """E[price at horizon] = S_0 * exp(mu*h + 0.5*sigma^2*h) (lognormal mean)."""
    return s0 * math.exp(mu * horizon + 0.5 * sigma * sigma * horizon)


def gbm_analytic_path(closes: list[float], horizon: int, mu: float, sigma: float) -> ForecastPath:
    """A ForecastPath built from the closed-form quantiles (smooth 5/25/50/75/95
    bands), so it renders through the same chart/report code as the MC paths."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not closes or closes[-1] <= 0:
        raise ValueError("closes must be non-empty with a positive last price")
    s0 = float(closes[-1])
    days = list(range(1, horizon + 1))
    p5, p25, p50, p75, p95 = ([], [], [], [], [])
    for t in days:
        p5.append(gbm_percentile(s0, mu, sigma, t, 0.05))
        p25.append(gbm_percentile(s0, mu, sigma, t, 0.25))
        p50.append(gbm_percentile(s0, mu, sigma, t, 0.50))
        p75.append(gbm_percentile(s0, mu, sigma, t, 0.75))
        p95.append(gbm_percentile(s0, mu, sigma, t, 0.95))
    return ForecastPath(
        method="gbm", days=days, median=p50, lower=p5, upper=p95, interval_pct=90.0,
        p25=p25, p75=p75, terminal=[],
        note=f"analytic lognormal mu={mu:.6f} sigma={sigma:.6f}",
    )


def gbm_analytic_path_seasonal(closes: list[float], horizon: int, mu: float,
                               eff_sigma) -> ForecastPath:
    """Analytic path where each day t uses its own effective sigma (seasonal vol).

    `eff_sigma` is the day-t constant-equivalent sigma array (len horizon) from
    `seasonal.effective_sigma`, so quantiles match the seasonal terminal variance."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not closes or closes[-1] <= 0:
        raise ValueError("closes must be non-empty with a positive last price")
    if len(eff_sigma) < horizon:
        raise ValueError("eff_sigma shorter than horizon")
    s0 = float(closes[-1])
    days = list(range(1, horizon + 1))
    p5, p25, p50, p75, p95 = ([], [], [], [], [])
    for idx, t in enumerate(days):
        se = float(eff_sigma[idx])
        p5.append(gbm_percentile(s0, mu, se, t, 0.05))
        p25.append(gbm_percentile(s0, mu, se, t, 0.25))
        p50.append(gbm_percentile(s0, mu, se, t, 0.50))
        p75.append(gbm_percentile(s0, mu, se, t, 0.75))
        p95.append(gbm_percentile(s0, mu, se, t, 0.95))
    return ForecastPath(
        method="gbm", days=days, median=p50, lower=p5, upper=p95, interval_pct=90.0,
        p25=p25, p75=p75, terminal=[], note=f"seasonal-vol analytic mu={mu:.6f}",
    )


def simulate_gbm(
    closes: list[float],
    horizon: int,
    n_paths: int = 10_000,
    seed: int = 7,
    drift_tilt: float = 0.0,
    *,
    mu: float | None = None,
    sigma: float | None = None,
    shock_dist: str = "normal",
) -> ForecastPath:
    """Simulate GBM forward from the last close and return per-day percentile bands.

    Args:
        closes: historical close prices (chronological). closes[-1] is the anchor.
        horizon: number of trading days to project.
        n_paths: Monte Carlo sample size.
        seed: RNG seed -- fixes the output exactly.
        drift_tilt: additive nudge to the daily-return drift (signal tilt).
        mu, sigma: override the estimated drift/vol (else estimated from `closes`).
        shock_dist: "normal" (Gaussian shocks) or "t" (Student-t, df fit from the
            history's excess kurtosis and variance-normalized so `sigma` is preserved).
            "t" gives fatter tails -- more realistic crash probabilities.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if not closes or closes[-1] <= 0:
        raise ValueError("closes must be non-empty with a positive last price")

    est_mu, est_sigma = log_return_stats(closes)
    mu = est_mu if mu is None else mu
    sigma = est_sigma if sigma is None else sigma
    s0 = float(closes[-1])

    rng = np.random.default_rng(seed)
    daily_mu = mu + drift_tilt

    if sigma <= 0.0:
        # Degenerate (flat history): a deterministic drift line, no dispersion.
        steps = np.arange(1, horizon + 1)
        line = s0 * np.exp(daily_mu * steps)
        vals = line.tolist()
        return ForecastPath(
            method="gbm", days=list(range(1, horizon + 1)),
            median=vals, lower=list(vals), upper=list(vals),
            p25=list(vals), p75=list(vals), terminal=[float(line[-1])] * n_paths,
            note=f"sigma=0 (flat history); drift-only. mu={mu:.6f} tilt={drift_tilt:+.6f}",
        )

    # Unit-variance innovations, then scaled to sigma. log-returns cumulate in log space.
    if shock_dist == "t":
        dof = fit_t_dof(log_returns(closes))
        z = rng.standard_t(dof, size=(n_paths, horizon))
        z /= math.sqrt(dof / (dof - 2.0))         # normalize to unit variance
        tail_note = f" shocks=t(df={dof:.1f})"
    elif shock_dist == "normal":
        z = rng.standard_normal((n_paths, horizon))
        tail_note = ""
    else:
        raise ValueError(f"shock_dist must be 'normal' or 't', got {shock_dist!r}")
    shocks = z * sigma + daily_mu
    log_paths = np.cumsum(shocks, axis=1)
    prices = s0 * np.exp(log_paths)               # (n_paths, horizon)

    pct = np.percentile(prices, _PCTS, axis=0)    # (5, horizon)
    p5, p25, p50, p75, p95 = (pct[i].tolist() for i in range(5))

    return ForecastPath(
        method="gbm", days=list(range(1, horizon + 1)),
        median=p50, lower=p5, upper=p95, interval_pct=90.0,
        p25=p25, p75=p75, terminal=prices[:, -1].tolist(),
        note=(f"mu={mu:.6f} sigma={sigma:.6f} tilt={drift_tilt:+.6f} "
              f"paths={n_paths} seed={seed}{tail_note}"),
    )
