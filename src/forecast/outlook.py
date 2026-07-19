"""Multi-month probabilistic outlook.

Turns the long-range estimators + two simulators (analytic GBM and block bootstrap)
into a *date-anchored, probability-first* view: for each target date it reports the
distribution of price (percentiles) and the odds of moves the user cares about
(up, +/-5%, -10%), from both methods. Point "predictions" are deliberately demoted
to a percentile row -- at this horizon the honest output is a probability.

Also provides trading-day <-> calendar-date mapping (NYSE-aware for 2026) and a
month-of-year seasonality summary computed from the same history (shown as context,
never mixed into the simulation).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.common.models import Candle
from src.forecast.bootstrap import bootstrap_prices
from src.forecast.drawdown import DRAWDOWN_LEVELS, path_risk_bootstrap, path_risk_gbm
from src.forecast.gbm import (
    gbm_analytic_path,
    gbm_analytic_path_seasonal,
    gbm_expected_value,
    gbm_percentile,
    gbm_prob_above,
)
from src.forecast.longrange import Estimates, estimate
from src.forecast.result import ForecastPath
from src.forecast.scenarios import Scenario, crisis_bootstrap, historical_stress, scenario_1987
from src.forecast.seasonal import effective_sigma, monthly_vol_multipliers, sigma_schedule

log = logging.getLogger(__name__)

# NYSE full-day closures for 2026 (the only ones inside a mid-2026 -> Oct horizon is
# Labor Day, but the full set keeps the mapping correct for any 2026 target).
NYSE_HOLIDAYS_2026 = frozenset({
    dt.date(2026, 1, 1),    # New Year's Day
    dt.date(2026, 1, 19),   # MLK Jr. Day
    dt.date(2026, 2, 16),   # Washington's Birthday
    dt.date(2026, 4, 3),    # Good Friday
    dt.date(2026, 5, 25),   # Memorial Day
    dt.date(2026, 6, 19),   # Juneteenth
    dt.date(2026, 7, 3),    # Independence Day (observed)
    dt.date(2026, 9, 7),    # Labor Day
    dt.date(2026, 11, 26),  # Thanksgiving
    dt.date(2026, 12, 25),  # Christmas
})

_PROB_PCTS = (5, 25, 50, 75, 95)


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS_2026


def trading_days_between(start: dt.date, end: dt.date) -> int:
    """Number of trading days strictly after `start`, up to and including `end`."""
    n, d = 0, start
    while d < end:
        d += dt.timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


def map_horizon_to_dates(start: dt.date, horizon: int) -> list[dt.date]:
    """Calendar date for each of trading days 1..horizon after `start`."""
    dates, d, count = [], start, 0
    while count < horizon:
        d += dt.timedelta(days=1)
        if is_trading_day(d):
            count += 1
            dates.append(d)
    return dates


def prob_table_from_samples(terminal, s0: float) -> dict:
    """Probabilities + percentiles from a Monte Carlo terminal-price sample."""
    t = np.asarray(terminal, dtype=float)
    n = t.size
    ev = float(t.mean())
    return {
        "p_up": float(np.count_nonzero(t > s0)) / n,
        "p_ge_5": float(np.count_nonzero(t >= s0 * 1.05)) / n,
        "p_le_5": float(np.count_nonzero(t <= s0 * 0.95)) / n,
        "p_le_10": float(np.count_nonzero(t <= s0 * 0.90)) / n,
        "ev": ev,
        "ev_pct": (ev / s0 - 1.0) * 100.0,
        "pctiles": {q: float(np.percentile(t, q)) for q in _PROB_PCTS},
    }


def prob_table_gbm(s0: float, mu: float, sigma: float, horizon: int) -> dict:
    """The same table, closed-form (no sampling error) under GBM."""
    ev = gbm_expected_value(s0, mu, sigma, horizon)
    return {
        "p_up": gbm_prob_above(s0, mu, sigma, horizon, s0),
        "p_ge_5": gbm_prob_above(s0, mu, sigma, horizon, s0 * 1.05),
        "p_le_5": 1.0 - gbm_prob_above(s0, mu, sigma, horizon, s0 * 0.95),
        "p_le_10": 1.0 - gbm_prob_above(s0, mu, sigma, horizon, s0 * 0.90),
        "ev": ev,
        "ev_pct": (ev / s0 - 1.0) * 100.0,
        "pctiles": {q: gbm_percentile(s0, mu, sigma, horizon, q / 100.0) for q in _PROB_PCTS},
    }


def monthly_seasonality(candles: list[Candle]) -> dict[int, dict]:
    """Per calendar month: mean/median month-over-month return and hit rate.

    Month return is last-close-of-month over last-close-of-prior-month. Reported as
    *context* only -- it is never fed into the simulators."""
    last_close_by_ym: dict[tuple[int, int], float] = {}
    for c in candles:                       # candles are chronological
        d = dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date()
        last_close_by_ym[(d.year, d.month)] = c.close

    keys = sorted(last_close_by_ym)
    by_month: dict[int, list[float]] = defaultdict(list)
    for prev, cur in zip(keys, keys[1:]):
        p, q = last_close_by_ym[prev], last_close_by_ym[cur]
        if p > 0:
            by_month[cur[1]].append(q / p - 1.0)

    out: dict[int, dict] = {}
    for m, vals in by_month.items():
        arr = np.asarray(vals, dtype=float)
        out[m] = {
            "n": int(arr.size),
            "mean_pct": float(arr.mean() * 100.0),
            "median_pct": float(np.median(arr) * 100.0),
            "hit_rate": float(np.mean(arr > 0.0)),
        }
    return out


@dataclass(frozen=True, slots=True)
class DateForecast:
    date: dt.date
    trading_day: int
    gbm: dict            # prob_table_gbm output (terminal)
    boot: dict           # prob_table_from_samples output (terminal)
    gbm_risk: dict        # path_risk_gbm: touch prob per drawdown level (any time <= date)
    boot_risk: dict       # path_risk_bootstrap: touch probs + max-drawdown distribution


@dataclass(frozen=True, slots=True)
class Outlook:
    symbol: str
    last_close: float
    start_date: dt.date
    horizon: int
    est: Estimates
    gbm_path: ForecastPath      # analytic cone over 1..horizon (for the chart)
    boot_path: ForecastPath     # bootstrap cone over 1..horizon (for the chart)
    per_date: list[DateForecast]
    seasonality: dict[int, dict]
    headline: str = "gbm"       # which method the calibration backtest prefers
    dates_axis: list[dt.date] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    crisis_path: ForecastPath | None = None    # high-vol-regime "stressed" cone
    seasonal: bool = False


def build_outlook(symbol: str, candles: list[Candle], target_dates: list[dt.date],
                  start_date: dt.date, *, horizon: int | None = None,
                  n_paths: int = 20_000, seed: int = 7, block: int = 10,
                  w_recent: float = 0.30, lam: float = 0.94,
                  vix_close: float | None = None, headline: str = "gbm",
                  seasonal: bool = False) -> Outlook:
    """Assemble the full outlook. One bootstrap simulation covers all target dates."""
    closes = [c.close for c in candles]
    s0 = closes[-1]
    tdays = {td: trading_days_between(start_date, td) for td in sorted(target_dates)}
    if horizon is None:
        horizon = max(tdays.values())
    dates_axis = map_horizon_to_dates(start_date, horizon)

    est = estimate(closes, vix_close=vix_close, w_recent=w_recent, lam=lam)
    mu, sigma = est.mu_blend, est.sigma_blend

    # Seasonal vol: per-day effective sigma (else a flat array of the base sigma).
    if seasonal:
        eff = effective_sigma(sigma_schedule(dates_axis, sigma, monthly_vol_multipliers(candles)))
        gbm_path = gbm_analytic_path_seasonal(closes, horizon, mu, eff)
    else:
        eff = None
        gbm_path = gbm_analytic_path(closes, horizon, mu, sigma)

    def gbm_sigma(n: int) -> float:
        return float(eff[n - 1]) if eff is not None else sigma

    boot_prices = bootstrap_prices(closes, horizon, n_paths=n_paths, seed=seed,
                                   block=block, recenter_mu=mu)
    boot_path = _path_from_matrix(boot_prices, horizon, block, n_paths, seed, mu)

    per_date: list[DateForecast] = []
    for td, n in tdays.items():
        if n < 1 or n > horizon:
            log.warning("target %s maps to %d trading days, outside horizon %d; skipped",
                        td, n, horizon)
            continue
        sig_n = gbm_sigma(n)
        gbm_tbl = prob_table_gbm(s0, mu, sig_n, n)
        boot_tbl = prob_table_from_samples(boot_prices[:, n - 1], s0)
        gbm_risk = path_risk_gbm(s0, mu, sig_n, n)
        boot_risk = path_risk_bootstrap(boot_prices[:, :n], s0)
        per_date.append(DateForecast(date=td, trading_day=n, gbm=gbm_tbl, boot=boot_tbl,
                                     gbm_risk=gbm_risk, boot_risk=boot_risk))

    scenarios = historical_stress(candles, horizon, k=3)
    scenarios.append(scenario_1987(s0))
    try:
        crisis_path = crisis_bootstrap(closes, horizon, n_paths=n_paths, seed=seed,
                                       block=block, recenter_mu=None)
    except ValueError:
        crisis_path = None

    return Outlook(
        symbol=symbol, last_close=s0, start_date=start_date, horizon=horizon,
        est=est, gbm_path=gbm_path, boot_path=boot_path, per_date=per_date,
        seasonality=monthly_seasonality(candles), headline=headline,
        dates_axis=dates_axis, scenarios=scenarios, crisis_path=crisis_path,
        seasonal=seasonal,
    )


def _path_from_matrix(prices, horizon, block, n_paths, seed, mu) -> ForecastPath:
    pct = np.percentile(prices, [5, 25, 50, 75, 95], axis=0)
    p5, p25, p50, p75, p95 = (pct[i].tolist() for i in range(5))
    return ForecastPath(
        method="bootstrap", days=list(range(1, horizon + 1)),
        median=p50, lower=p5, upper=p95, interval_pct=90.0, p25=p25, p75=p75,
        terminal=prices[:, -1].tolist(),
        note=f"block bootstrap block={block} paths={n_paths} seed={seed} recentered mu={mu:.6f}",
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

DISCLAIMER = ("NOT INVESTMENT ADVICE. Calibrated probability ranges from price history "
              "only -- no macro/earnings/event modeling. Not a prediction of actual price.")


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def console_outlook(o: Outlook) -> str:
    L: list[str] = []
    L.append(f"=== {o.symbol} probabilistic outlook ===")
    L.append(DISCLAIMER)
    L.append("")
    L.append(f"as of {o.start_date}  last close {o.last_close:,.2f}  "
             f"horizon {o.horizon} trading days  ({o.est.n_returns} returns of history)")
    e = o.est
    L.append(f"drift/day : blend {e.mu_blend:+.5f}  (recent {e.mu_recent:+.5f}, "
             f"long {e.mu_long:+.5f})")
    vix = "n/a" if e.sigma_vix is None else f"{e.sigma_vix:.5f}"
    L.append(f"vol/day   : blend {e.sigma_blend:.5f}  (ewma {e.sigma_ewma:.5f}, vix {vix})"
             + ("  [seasonal-vol ON]" if o.seasonal else ""))
    L.append(f"headline method (best-calibrated): {o.headline.upper()}")
    L.append("")

    for df in o.per_date:
        L.append(f"--- {df.date}  (trading day {df.trading_day}) ---")
        L.append(f"{'':14}{'GBM':>12}{'BOOTSTRAP':>12}")
        rows = [
            ("P(up)",        "p_up"),
            ("P(>= +5%)",    "p_ge_5"),
            ("P(<= -5%)",    "p_le_5"),
            ("P(<= -10%)",   "p_le_10"),
        ]
        for label, key in rows:
            L.append(f"{label:14}{_pct(df.gbm[key]):>12}{_pct(df.boot[key]):>12}")
        L.append(f"{'exp. value':14}{df.gbm['ev']:>12,.2f}{df.boot['ev']:>12,.2f}")
        L.append(f"{'  (% move)':14}{df.gbm['ev_pct']:>+11.1f}%{df.boot['ev_pct']:>+11.1f}%")
        L.append(f"{'':14}{'--- price percentiles ---':>28}")
        for q in _PROB_PCTS:
            L.append(f"  p{q:<11}{df.gbm['pctiles'][q]:>12,.2f}{df.boot['pctiles'][q]:>12,.2f}")
        # Intra-path drawdown risk: chance of touching each level at ANY time up to date.
        L.append(f"{'':14}{'--- path risk: P(touch by date) ---':>38}")
        for lv in DRAWDOWN_LEVELS:
            label = f"touch {lv * 100:.0f}%"
            L.append(f"{label:14}{_pct(df.gbm_risk[lv]):>12}{_pct(df.boot_risk['touch'][lv]):>12}")
        br = df.boot_risk
        L.append(f"  max-drawdown (bootstrap): median {br['maxdd_median'] * 100:.1f}%  "
                 f"p90 {br['maxdd_p90'] * 100:.1f}%  p99 {br['maxdd_p99'] * 100:.1f}%")
        L.append("")

    if o.scenarios or o.crisis_path:
        L.append("STRESS SCENARIOS (replayed / regime-conditioned -- NOT probabilities):")
        for sc in o.scenarios:
            L.append(f"  {sc.name:<34} trough {sc.trough_pct * 100:+6.1f}%  "
                     f"end {sc.end_pct * 100:+6.1f}%   (-> {sc.trough_price:,.0f} / {sc.end_price:,.0f})")
        if o.crisis_path:
            cp = o.crisis_path
            L.append(f"  crisis-regime bootstrap cone at horizon: "
                     f"p5 {cp.lower[-1]:,.0f}  median {cp.median[-1]:,.0f}  p95 {cp.upper[-1]:,.0f}")
        L.append("")

    if o.seasonality:
        L.append("historical monthly seasonality (context only, not in the model):")
        L.append(f"{'month':>6}{'n':>5}{'mean':>9}{'median':>9}{'hit rate':>10}")
        for m in sorted(o.seasonality):
            s = o.seasonality[m]
            name = dt.date(2026, m, 1).strftime("%b")
            L.append(f"{name:>6}{s['n']:>5}{s['mean_pct']:>8.1f}%{s['median_pct']:>8.1f}%"
                     f"{s['hit_rate'] * 100:>9.0f}%")
    return "\n".join(L)


def write_outlook_chart(path, o: Outlook, candles: list[Candle], history_tail: int = 180):
    """History + both cones, with vertical markers at each target date."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping chart")
        return None

    tail = candles[-history_tail:] if len(candles) > history_tail else candles
    hist_x = list(range(-len(tail) + 1, 1))
    hist_y = [c.close for c in tail]
    fx = o.gbm_path.days

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(hist_x, hist_y, color="#333", lw=1.3, label=f"{o.symbol} close (history)")

    ax.fill_between(fx, o.boot_path.lower, o.boot_path.upper, color="#54a24b", alpha=0.15,
                    label="bootstrap 90%")
    ax.plot(fx, o.boot_path.median, color="#54a24b", lw=1.6, label="bootstrap median")
    ax.plot(fx, o.gbm_path.lower, color="#4c78a8", lw=1.0, ls=":", alpha=0.8)
    ax.plot(fx, o.gbm_path.upper, color="#4c78a8", lw=1.0, ls=":", alpha=0.8)
    ax.plot(fx, o.gbm_path.median, color="#4c78a8", lw=1.6, ls="--", label="GBM median (analytic)")

    ax.axvline(0, color="#999", lw=0.8)
    for df in o.per_date:
        ax.axvline(df.trading_day, color="#e45756", lw=1.0, ls="-", alpha=0.6)
        ax.text(df.trading_day, ax.get_ylim()[1], f" {df.date:%b %d}",
                color="#e45756", fontsize=8, va="top", rotation=90)

    ax.set_title(f"{o.symbol} — probabilistic outlook to {o.dates_axis[-1]:%b %d, %Y}\n{DISCLAIMER}",
                 fontsize=9)
    ax.set_xlabel("trading days (0 = last close)")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_outlook_csv(path, o: Outlook):
    """Per-target-date probability tables as CSV."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "trading_day", "method", "p_up", "p_ge_5", "p_le_5", "p_le_10",
                    "ev", "ev_pct", "p5", "p25", "p50", "p75", "p95",
                    "touch_5", "touch_10", "touch_15", "touch_20"])
        for df in o.per_date:
            for method, tbl, risk in (("gbm", df.gbm, df.gbm_risk),
                                      ("bootstrap", df.boot, df.boot_risk)):
                pc = tbl["pctiles"]
                touch = risk if method == "gbm" else risk["touch"]
                w.writerow([df.date, df.trading_day, method,
                            f"{tbl['p_up']:.4f}", f"{tbl['p_ge_5']:.4f}",
                            f"{tbl['p_le_5']:.4f}", f"{tbl['p_le_10']:.4f}",
                            f"{tbl['ev']:.2f}", f"{tbl['ev_pct']:.2f}",
                            f"{pc[5]:.2f}", f"{pc[25]:.2f}", f"{pc[50]:.2f}",
                            f"{pc[75]:.2f}", f"{pc[95]:.2f}"]
                           + [f"{touch[lv]:.4f}" for lv in DRAWDOWN_LEVELS])
    return path
