"""Walk-forward calibration backtest for the multi-month outlook.

The outlook's value rests on its bands meaning what they claim. This script tests that
honestly by re-fitting at every historical month-end on ONLY the data available then,
projecting 51/73/... days ahead, and checking where price actually went.

Reports, per method and horizon:
  * coverage90 / coverage50  -- should be ~0.90 / ~0.50
  * LOWER/UPPER tail exceedance -- share of outcomes below p5 / above p95 (target ~5%/5%).
    This is the number that matters for CRASH risk: if far more than 5% of outcomes fall
    below p5, the model's downside is too thin.
  * drawdown calibration -- predicted P(touch -10% any time) vs realized touch frequency.

Methods: gbm-normal (analytic), gbm-t (Student-t shocks), bootstrap. Recommends the
best-calibrated (overall + tail) variant to feature.

    python -m scripts.validate_outlook --symbol SPY --horizons 51,73,116

NOT INVESTMENT ADVICE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DB_PATH, ensure_dirs                 # noqa: E402
from src.forecast.bootstrap import bootstrap_prices               # noqa: E402
from src.forecast.drawdown import touch_prob_gbm                  # noqa: E402
from src.forecast.gbm import gbm_percentile, simulate_gbm         # noqa: E402
from src.forecast.longrange import estimate                       # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

MIN_TRAIN_RETURNS = 250
TOUCH_LEVEL = -0.10   # drawdown level whose calibration we report


@dataclass
class Cov:
    """Coverage + tail-exceedance accumulator for one (method, horizon)."""
    hit90: int = 0
    hit50: int = 0
    below5: int = 0
    above95: int = 0
    n: int = 0
    widths: list[float] = field(default_factory=list)

    def add(self, p5, p25, p75, p95, realized, s0):
        self.n += 1
        self.hit90 += int(p5 <= realized <= p95)
        self.hit50 += int(p25 <= realized <= p75)
        self.below5 += int(realized < p5)
        self.above95 += int(realized > p95)
        self.widths.append((p95 - p5) / s0 * 100.0)

    def rate(self, k): return getattr(self, k) / self.n if self.n else float("nan")
    @property
    def cov90(self): return self.hit90 / self.n if self.n else float("nan")
    @property
    def cov50(self): return self.hit50 / self.n if self.n else float("nan")
    @property
    def mean_width(self): return float(np.mean(self.widths)) if self.widths else float("nan")


@dataclass
class DD:
    """Drawdown-touch calibration: predicted vs realized."""
    pred_sum: float = 0.0
    realized: int = 0
    n: int = 0

    def add(self, pred, hit):
        self.pred_sum += pred
        self.realized += int(hit)
        self.n += 1

    @property
    def pred(self): return self.pred_sum / self.n if self.n else float("nan")
    @property
    def real(self): return self.realized / self.n if self.n else float("nan")


def _month_end_indices(dates: list[dt.date]) -> list[int]:
    return [i for i in range(len(dates) - 1)
            if (dates[i].year, dates[i].month) != (dates[i + 1].year, dates[i + 1].month)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward calibration incl. tails/drawdown")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--history-days", type=int, default=3600)
    ap.add_argument("--horizons", default="51,73")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--w-recent", type=float, default=0.30)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    symbol = a.symbol.upper()
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    methods = ("gbm-normal", "gbm-t", "bootstrap")
    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        try:
            candles = fetch_yfinance(symbol, days=a.history_days, interval="1d", db=db)
        except InsufficientDataError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        closes = [c.close for c in candles]
        dates = [dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date()
                 for c in candles]
        month_ends = _month_end_indices(dates)

        cov = {(m, h): Cov() for m in methods for h in horizons}
        dd = {(m, h): DD() for m in ("gbm-normal", "bootstrap") for h in horizons}

        n_evals = 0
        for i in month_ends:
            if i < MIN_TRAIN_RETURNS:
                continue
            window = closes[:i + 1]
            est = estimate(window, w_recent=a.w_recent)
            mu, sigma = est.mu_blend, est.sigma_blend
            s0 = closes[i]
            for h in horizons:
                if i + h >= len(closes):
                    continue
                realized = closes[i + h]
                realized_min = min(closes[i + 1:i + h + 1])

                # gbm-normal: analytic quantiles
                cov[("gbm-normal", h)].add(
                    gbm_percentile(s0, mu, sigma, h, 0.05), gbm_percentile(s0, mu, sigma, h, 0.25),
                    gbm_percentile(s0, mu, sigma, h, 0.75), gbm_percentile(s0, mu, sigma, h, 0.95),
                    realized, s0)

                # gbm-t: Student-t shocks (Monte Carlo)
                t_term = np.asarray(simulate_gbm(window, h, n_paths=a.paths, seed=1000 + i,
                                                 mu=mu, sigma=sigma, shock_dist="t").terminal)
                tp5, tp25, tp75, tp95 = np.percentile(t_term, [5, 25, 75, 95])
                cov[("gbm-t", h)].add(tp5, tp25, tp75, tp95, realized, s0)

                # bootstrap: full matrix (need minima for drawdown)
                prices = bootstrap_prices(window, h, n_paths=a.paths, seed=2000 + i,
                                          block=a.block, recenter_mu=mu)
                term = prices[:, -1]
                bp5, bp25, bp75, bp95 = np.percentile(term, [5, 25, 75, 95])
                cov[("bootstrap", h)].add(bp5, bp25, bp75, bp95, realized, s0)

                # drawdown-touch calibration at -10%
                lvl = s0 * (1.0 + TOUCH_LEVEL)
                hit = realized_min <= lvl
                dd[("gbm-normal", h)].add(touch_prob_gbm(s0, mu, sigma, h, lvl), hit)
                dd[("bootstrap", h)].add(float(np.mean(prices.min(axis=1) <= lvl)), hit)
                n_evals += 1

        # ---- report ----
        print(f"=== {symbol} walk-forward calibration (coverage + tails + drawdown) ===")
        print("NOT INVESTMENT ADVICE.")
        print(f"history: {len(closes)} bars {dates[0]} -> {dates[-1]}; "
              f"{n_evals} horizon-evaluations across {len(month_ends)} month-ends\n")

        print(f"{'method':>11}{'horizon':>8}{'n':>5}{'cov90':>7}{'cov50':>7}"
              f"{'<p5':>7}{'>p95':>7}{'width':>8}")
        print(f"{'':11}{'':8}{'':5}{'(.90)':>7}{'(.50)':>7}{'(.05)':>7}{'(.05)':>7}{'(%sp)':>8}")
        for h in horizons:
            for m in methods:
                x = cov[(m, h)]
                print(f"{m:>11}{h:>8}{x.n:>5}{x.cov90:>7.2f}{x.cov50:>7.2f}"
                      f"{x.rate('below5'):>7.2f}{x.rate('above95'):>7.2f}{x.mean_width:>7.1f}%")
        print()

        print(f"drawdown touch calibration at {TOUCH_LEVEL * 100:.0f}% (predicted vs realized freq):")
        print(f"{'method':>11}{'horizon':>8}{'predicted':>11}{'realized':>10}")
        for h in horizons:
            for m in ("gbm-normal", "bootstrap"):
                d = dd[(m, h)]
                print(f"{m:>11}{h:>8}{d.pred:>11.3f}{d.real:>10.3f}")
        print()

        # ---- recommendation: overall coverage + lower-tail honesty ----
        def cov_err(m): return np.mean([abs(cov[(m, h)].cov90 - 0.90)
                                        for h in horizons if cov[(m, h)].n])
        def tail_err(m): return np.mean([abs(cov[(m, h)].rate('below5') - 0.05)
                                         for h in horizons if cov[(m, h)].n])
        # Score weights the lower tail 2x (this whole exercise is about crash honesty).
        def score(m): return cov_err(m) + 2.0 * tail_err(m)

        ranked = sorted(methods, key=score)
        best = ranked[0]
        print("scores (cov_err + 2*lower_tail_err), lower is better:")
        for m in methods:
            print(f"  {m:>11}: cov_err={cov_err(m):.3f}  lower_tail_err={tail_err(m):.3f}  "
                  f"score={score(m):.3f}")
        tails_flag = "t" if best == "gbm-t" else "normal"
        head = "bootstrap" if best == "bootstrap" else "gbm"
        print(f"\nRECOMMENDATION: best-calibrated = '{best}'. "
              f"Run: scripts.outlook --headline {head}"
              + (f" --tails {tails_flag}" if head == "gbm" else ""))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
