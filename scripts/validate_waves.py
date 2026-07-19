"""Does Elliott/Fibonacci actually predict anything? Walk-forward test vs placebo.

Two experiments, both walk-forward (levels/stages computed only from data available at
the time), both scored against a null model:

1. FIB PROXIMITY TEST. At each month-end, compute Fibonacci levels. Then find the
   forward path's actual turning points. Measure the median distance from a turning
   point to its nearest fib level, and compare with PLACEBO levels -- the same number of
   levels drawn uniformly across the same price range. If fib levels mark real turning
   points, real turns should sit closer to them than to random lines.
   Reported as a skill ratio: placebo_distance / fib_distance.
       > 1 means fib beat random; ~1 means decorative; < 1 means worse than random.

2. STAGE TEST. Bucket forward returns by the reported wave stage. If the stage carries
   information, the buckets differ beyond sampling noise. Confidence intervals come from
   the same machinery as the rest of the project.

The verdict is written to data/waves_validation.json and stamped on every waves output.

    python -m scripts.validate_waves --symbols SPY,ASTS,IREN,TE

NOT INVESTMENT ADVICE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs       # noqa: E402
from src.forecast import waves as W                                # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

VERDICT_PATH = DATA_DIR / "waves_validation.json"
MIN_TRAIN = 250
N_PLACEBO = 200


def _turning_points(closes: np.ndarray, threshold: float) -> list[float]:
    """Prices at the forward window's own swing highs/lows (same detector, so the
    comparison is apples to apples)."""
    return [p.price for p in W.zigzag(closes, threshold) if p.confirmed]


def _median_nearest(points: list[float], levels: np.ndarray) -> float:
    """Median |distance| from each turning point to the nearest level, in % of price."""
    if not points or levels.size == 0:
        return float("nan")
    d = [float(np.min(np.abs(levels - p)) / max(p, 1e-9)) for p in points]
    return float(np.median(d))


def main() -> int:
    ap = argparse.ArgumentParser(description="Test Elliott/Fib against placebo levels")
    ap.add_argument("--symbols", default="SPY,ASTS,IREN,TE")
    ap.add_argument("--history-days", type=int, default=2600)
    ap.add_argument("--horizon", type=int, default=63, help="forward window, trading days")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    ensure_dirs()
    db = DB(a.db or DB_PATH)
    rng = np.random.default_rng(a.seed)
    verdicts: dict[str, dict] = {}
    try:
        print("=== Elliott / Fibonacci validation vs placebo ===")
        print("NOT INVESTMENT ADVICE.\n")
        for sym in [s.strip().upper() for s in a.symbols.split(",") if s.strip()]:
            try:
                candles = fetch_yfinance(sym, days=a.history_days, interval="1d", db=db)
            except InsufficientDataError as e:
                print(f"{sym}: SKIP ({e})")
                continue
            closes = np.asarray([c.close for c in candles], float)
            highs = np.asarray([c.high for c in candles], float)
            lows = np.asarray([c.low for c in candles], float)
            dates = [dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date()
                     for c in candles]
            month_ends = [i for i in range(len(dates) - 1)
                          if (dates[i].year, dates[i].month) != (dates[i + 1].year, dates[i + 1].month)]

            fib_d, plc_d, ratio_d, n_eval = [], [], [], 0
            stage_returns: dict[str, list[float]] = {}
            for i in month_ends:
                if i < MIN_TRAIN or i + a.horizon >= len(closes):
                    continue
                try:
                    an = W.analyse(closes[:i + 1], highs[:i + 1], lows[:i + 1])
                except ValueError:
                    continue
                lv = np.asarray([l.price for l in an["levels"]], float)
                if lv.size == 0:
                    continue
                fwd = closes[i + 1:i + a.horizon + 1]
                turns = _turning_points(fwd, an["threshold"])
                if not turns:
                    continue

                lo, hi = float(np.min(lv)), float(np.max(lv))
                if hi <= lo:
                    continue
                fd = _median_nearest(turns, lv)
                # placebo A (weak control): uniform lines across the same span
                pd_runs = [_median_nearest(turns, rng.uniform(lo, hi, lv.size))
                           for _ in range(N_PLACEBO // 20)]
                # placebo B (STRICT control): the SAME swings at random, non-fib ratios.
                # This is the control that isolates the Fibonacci numbers themselves --
                # placebo A only shows whether "levels near real structure" matter.
                rr_runs = []
                for _ in range(N_PLACEBO // 20):
                    rl = W.random_ratio_levels(an["swings"], rng)
                    if rl.size:
                        rr_runs.append(_median_nearest(turns, rl))
                if not np.isfinite(fd) or not np.isfinite(np.median(pd_runs)):
                    continue
                fib_d.append(fd)
                plc_d.append(float(np.median(pd_runs)))
                ratio_d.append(float(np.median(rr_runs)) if rr_runs else float("nan"))
                n_eval += 1

                if an["counts"]:
                    stage = an["counts"][0].stage
                    stage_returns.setdefault(stage, []).append(
                        float(closes[i + a.horizon] / closes[i] - 1.0))

            if n_eval < 8:
                print(f"{sym}: too few evaluations ({n_eval}) -- no verdict.\n")
                verdicts[sym] = {"verdict": "insufficient data", "n": n_eval}
                continue

            fib_a = np.asarray(fib_d)
            fib_m, plc_m = float(np.median(fib_a)), float(np.median(plc_d))
            se = math.sqrt(0.25 / n_eval)

            wins_a = int(np.sum(fib_a < np.asarray(plc_d)))
            z_a = (wins_a / n_eval - 0.5) / se if se > 0 else 0.0

            rr = np.asarray(ratio_d)
            valid = np.isfinite(rr)
            n_b = int(valid.sum())
            rat_m = float(np.median(rr[valid])) if n_b else float("nan")
            wins_b = int(np.sum(fib_a[valid] < rr[valid])) if n_b else 0
            se_b = math.sqrt(0.25 / n_b) if n_b else float("nan")
            z_b = (wins_b / n_b - 0.5) / se_b if n_b else 0.0

            beats_strict = n_b >= 8 and abs(z_b) > 1.96 and wins_b / n_b > 0.5
            verdict = ("Fibonacci RATIOS beat matched random ratios" if beats_strict else
                       "Fibonacci ratios NOT distinguishable from random ratios of the "
                       "same swings")

            print(f"--- {sym} ({n_eval} month-end evaluations, {a.horizon}d forward) ---")
            print(f"  median distance from a turning point to the nearest level:")
            print(f"     FIB levels                                : {fib_m * 100:.2f}%")
            print(f"     placebo A - uniform random lines (weak)   : {plc_m * 100:.2f}%   "
                  f"skill {plc_m / fib_m if fib_m > 0 else float('nan'):.2f}, "
                  f"fib closer {wins_a}/{n_eval} ({wins_a / n_eval * 100:.0f}%), z={z_a:+.2f}")
            print(f"     placebo B - SAME swings, random ratios    : {rat_m * 100:.2f}%   "
                  f"skill {rat_m / fib_m if fib_m > 0 else float('nan'):.2f}, "
                  f"fib closer {wins_b}/{n_b} ({(wins_b / n_b * 100) if n_b else 0:.0f}%), z={z_b:+.2f}")
            print(f"  VERDICT (placebo B is the one that counts): {verdict}")

            stage_lines = []
            for stage, rets in sorted(stage_returns.items(), key=lambda kv: -len(kv[1])):
                if len(rets) < 5:
                    continue
                arr = np.asarray(rets)
                se_r = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")
                stage_lines.append((stage, arr.size, float(arr.mean()), se_r))
            if stage_lines:
                print(f"  forward {a.horizon}d return by reported stage:")
                for stage, n, m, se_r in stage_lines:
                    lo_ci, hi_ci = m - 1.96 * se_r, m + 1.96 * se_r
                    sig = "" if (lo_ci <= 0 <= hi_ci) else "  <- differs from zero"
                    print(f"    {stage:<38} n={n:>3}  mean {m * 100:+6.1f}%  "
                          f"95% CI [{lo_ci * 100:+.0f}%, {hi_ci * 100:+.0f}%]{sig}")
                spread = max(s[2] for s in stage_lines) - min(s[2] for s in stage_lines)
                print(f"    spread between best and worst stage: {spread * 100:.1f}% "
                      f"(compare with the CIs above before believing it)")
            print()

            verdicts[sym] = {
                "when": dt.date.today().isoformat(), "n": n_eval, "horizon": a.horizon,
                "fib_median_pct": round(fib_m * 100, 3),
                "placebo_uniform_pct": round(plc_m * 100, 3),
                "placebo_ratio_pct": round(rat_m * 100, 3) if n_b else None,
                "skill_vs_uniform": round(plc_m / fib_m, 3) if fib_m > 0 else None,
                "skill_ratio": round(rat_m / fib_m, 3) if (n_b and fib_m > 0) else None,
                "win_rate": round(wins_b / n_b, 3) if n_b else None,
                "z": round(z_b, 2), "verdict": verdict,
            }

        VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERDICT_PATH.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
        print(f"wrote {VERDICT_PATH} -- scripts.waves stamps this on every chart.")
        print("Reminder: a skill ratio near 1.0 means the levels are DECORATIVE. "
              "That is the expected result, and reporting it is the point.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
