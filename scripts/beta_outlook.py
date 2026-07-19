"""High-beta basket outlook via a market-factor (beta) model.

Borrows SPY's *calibrated* market cone and propagates it into high-beta names through
beta, sampling all assets jointly so correlations and co-crashes survive. Each name's
drift is anchored to `beta * market drift`, NOT its own momentum (which is noise).

Design points that came out of reviewing the first version:
  * VOL STANDARDIZATION (default on). Raw-return resampling bakes in the *average*
    historical volatility; for names whose vol swings 3-10x that guarantees
    miscalibrated bands. We now sample vol-standardized residuals and rescale to
    today's volatility.
  * PER-NAME FULL HISTORY. Each name's own tables use its full overlap with SPY
    (1,100-1,700 bars) instead of the short common window the basket requires.
  * BETA UNCERTAINTY propagated into the bands (r^2 ~0.15 => beta is imprecise).
  * VALIDATION STATUS stamped onto every name from the last `--validate` run.
  * ALPHA SENSITIVITY always shown: how fast "P(up)" decays if these names do not
    earn their beta.

    python -m scripts.beta_outlook --symbols ASTS,IREN,NBIS,TE,CRWV
    python -m scripts.beta_outlook --symbols ASTS,IREN,TE --validate

NOT INVESTMENT ADVICE. Odds and scenarios only -- never a buy/sell call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                        # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs        # noqa: E402
from src.forecast.beta import (                                     # noqa: E402
    align_closes, anchored_drifts, beta_stats, joint_bootstrap_prices, log_return_matrix,
)
from src.forecast.bootstrap import bootstrap_prices                 # noqa: E402
from src.forecast.bounds import (                                   # noqa: E402
    anchored_range, bounds_ladder, reality_check, realized_windows, unmodelable_notes,
)
from src.forecast.calib import rate                                 # noqa: E402
from src.forecast.longrange import estimate                         # noqa: E402
from src.forecast.outlook import trading_days_between               # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

MARKET = "SPY"
MIN_VALIDATE_BARS = 700
STATUS_PATH = DATA_DIR / "beta_validation_status.json"
TOUCH_LEVEL = -0.20          # drawdown level whose calibration we track
DISCLAIMER = ("NOT INVESTMENT ADVICE. Calibrated odds from price history only. No launch/"
              "earnings/bitcoin/contract modeling -- those dominate these names' idio risk.")


def _load(symbols, days, db):
    out = {}
    for s in symbols:
        try:
            cs = fetch_yfinance(s, days=days, interval="1d", db=db, min_bars=120)
        except InsufficientDataError as e:
            print(f"  SKIP {s}: {e}", file=sys.stderr)
            continue
        out[s] = {dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date(): c.close
                  for c in cs}
        print(f"  {s:>5}: {len(cs):>5} bars, last {cs[-1].close:>10,.2f}")
    return out


def _month_bounds(anchor: dt.date, year: int = 2026):
    ends = [("Aug", dt.date(year, 8, 31)), ("Sep", dt.date(year, 9, 30)),
            ("Oct", dt.date(year, 10, 30)), ("Nov", dt.date(year, 11, 30)),
            ("Dec", dt.date(year, 12, 31))]
    return [(n, trading_days_between(anchor, d)) for n, d in ends if d > anchor]


def _load_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _status_line(status: dict, name: str) -> str:
    s = status.get(name)
    if not s:
        return "UNVALIDATED (run --validate)"
    return (f"last validated {s.get('when', '?')}: cov90 {s.get('cov90', float('nan')):.2f} "
            f"[{s.get('lo', float('nan')):.2f}..{s.get('hi', float('nan')):.2f}] "
            f"n={s.get('n', 0)} -> {s.get('verdict', '?')}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(a, db) -> int:
    names = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    print(f"=== high-beta basket outlook: {', '.join(names)} ===")
    print(DISCLAIMER)
    print("\nfetching:")
    series = _load([MARKET] + names, a.history_days, db)
    names = [n for n in names if n in series]
    if MARKET not in series or not names:
        print("ERROR: need the market series plus >=1 name", file=sys.stderr)
        return 2

    status = _load_status()
    mkt_all = series[MARKET]
    mkt_closes = [mkt_all[d] for d in sorted(mkt_all)]
    est_m = estimate(mkt_closes)
    mu_m = est_m.mu_blend
    print(f"\nmarket ({MARKET}) drift/day = {mu_m:+.5f}, vol/day = {est_m.sigma_blend:.5f}, "
          f"{len(mkt_closes)} bars")
    print(f"vol standardization: {'ON' if not a.raw_vol else 'OFF'}"
          f"   beta uncertainty in bands: {'ON' if not a.no_beta_uncertainty else 'OFF'}"
          f"   alpha drag: {a.alpha * 100:+.3f}%/day")

    # --- betas on each name's FULL pairwise overlap with the market ---
    pair, betas = {}, {}
    for n in names:
        dn, cn = align_closes({MARKET: mkt_all, n: series[n]})
        rmn = log_return_matrix(dn, cn, [MARKET, n])
        pair[n] = (dn, cn, rmn)
        betas[n] = beta_stats(n, rmn[:, 1], rmn[:, 0])

    drifts = anchored_drifts(betas, mu_m, idio_drift=a.idio_drift)
    for n in names:
        drifts[n] += a.alpha                       # optional alpha drag

    print(f"\n{'name':>6}{'beta':>7}{'+/-se':>8}{'r2':>6}{'vol/d':>8}{'idio/d':>8}"
          f"{'anchored mu':>13}{'raw mu':>10}{'refused':>10}{'bars':>7}")
    for n in names:
        b = betas[n]
        print(f"{n:>6}{b.beta:>7.2f}{b.se:>8.2f}{b.r2:>6.2f}{b.vol * 100:>7.1f}%"
              f"{b.idio_vol * 100:>7.1f}%{drifts[n] * 100:>12.3f}%{b.raw_mu * 100:>9.3f}%"
              f"{(b.raw_mu - drifts[n]) * 100:>9.3f}%{b.n:>7}")
    print("  ('refused' = momentum drift discarded as noise; +/-se = beta's own uncertainty)")

    anchor = max(pair[n][0][-1] for n in names)
    bounds = _month_bounds(anchor)
    if not bounds:
        print("ERROR: no target months after anchor", file=sys.stderr)
        return 2
    H = bounds[-1][1]
    rng = np.random.default_rng(a.seed)

    # --- PER-NAME sims on each name's own full history ---
    per_name_end = {}
    for n in names:
        dn, cn, rmn = pair[n]
        s0v = np.asarray([cn[MARKET][-1], cn[n][-1]])
        s0 = s0v[1]
        bd = (rng.normal(betas[n].beta, betas[n].se, size=a.paths)
              if not a.no_beta_uncertainty and betas[n].se > 0 else None)
        p = joint_bootstrap_prices(
            rmn, s0v, H, n_paths=a.paths, seed=a.seed, block=a.block,
            recenter_mu=np.asarray([mu_m, drifts[n]]),
            vol_standardize=not a.raw_vol, beta_draw=bd,
            vol_term_structure=not a.flat_vol,
            recency_half_life=None if a.no_recency else a.recency_half_life,
        )[:, :, 1].astype(float)
        per_name_end[n] = p[:, -1]

        print(f"\n--- {n}  (spot {s0:,.2f}, beta {betas[n].beta:.2f}, "
              f"{len(dn)} bars of own history) ---")
        print(f"    calibration: {_status_line(status, n)}")
        print(f"{'month':>5}{'P(up)':>8}{'med ret':>10}{'dip-10%':>9}{'dip-20%':>9}"
              f"{'dip-30%':>9}{'medDD':>8}   [cumulative dips from today]")
        prev = 0
        for mname, end in bounds:
            ref = np.full(p.shape[0], s0) if prev == 0 else p[:, prev - 1]
            mret = p[:, end - 1] / ref - 1.0
            cum = p[:, :end].min(axis=1) / s0 - 1.0
            f = lambda x: float(np.mean(cum <= x)) * 100
            print(f"{mname:>5}{np.mean(mret > 0) * 100:>7.0f}%{np.median(mret) * 100:>+9.1f}%"
                  f"{f(-0.10):>8.0f}%{f(-0.20):>8.0f}%{f(-0.30):>8.0f}%"
                  f"{-np.median(cum) * 100:>7.1f}%")
            prev = end
        ye = p[:, -1]
        print(f"  year-end: median {np.median(ye):,.2f} ({(np.median(ye) / s0 - 1) * 100:+.0f}%), "
              f"90% range [{np.percentile(ye, 5):,.2f} .. {np.percentile(ye, 95):,.2f}], "
              f"P(above today) = {np.mean(ye > s0) * 100:.0f}%")

        # realistic bounds: simulation vs what this name has ACTUALLY done, auto-checked
        prec = realized_windows(list(cn[n]), dn, H)
        checks = reality_check(ye, s0, list(cn[n]), H, tol=a.reality_tol)
        print(f"  REALISTIC BOUNDS at {H} trading days "
              f"({'no precedent data' if prec is None else f'{prec.n_windows} historical windows'})"
              f"  [auto reality-check, tol {a.reality_tol:.1f}x]:")
        print(f"    {'pct':>6}{'model':>12}{'realized':>12}{'ANCHORED':>12}   check")
        for c in checks:
            rz = "  n/a" if c.realized is None else f"{c.realized:>11,.2f}"
            print(f"    {c.q:>5.0f}%{c.model:>12,.2f}{rz}{c.anchored:>12,.2f}"
                  f"   {c.flag}")
        lo, mid, hi = anchored_range(checks)
        print(f"    => REALITY-ANCHORED: median ${mid:,.2f} ({mid / s0 - 1:+.0%}), "
              f"90% range [${lo:,.2f} .. ${hi:,.2f}] "
              f"({lo / s0 - 1:+.0%} .. {hi / s0 - 1:+.0%})")
        if prec is not None:
            print(f"    precedent: worst actual {H}d {prec.worst_mult - 1:+.0%} "
                  f"(${s0 * prec.worst_mult:,.2f}), best {prec.best_mult - 1:+.0%} "
                  f"(${s0 * prec.best_mult:,.2f}), median {prec.median_mult - 1:+.0%}; "
                  f"{prec.share_le_half:.0%} of windows lost >=50%")
        for note in unmodelable_notes(n):
            print(f"    ! {note}")

    # --- alpha sensitivity (always shown) ---
    print(f"\n=== how much of 'P(up)' is the drift ASSUMPTION? (alpha drag applied) ===")
    print(f"{'name':>6}" + "".join(f"{lbl:>12}" for lbl in
                                   ["as modeled", "-12%/yr", "-22%/yr", "-31%/yr"]))
    for n in names:
        s0 = pair[n][1][n][-1]
        lr = np.log(per_name_end[n] / s0)
        row = "".join(f"{np.mean(lr + al * H > 0) * 100:>11.0f}%"
                      for al in (0.0, -0.0005, -0.0010, -0.0015))
        print(f"{n:>6}{row}")

    # --- JOINT sim on the common window (basket + conditionals need it) ---
    order = [MARKET] + names
    dates_j, closes_j = align_closes({s: series[s] for s in order})
    rm_j = log_return_matrix(dates_j, closes_j, order)
    s0j = np.asarray([closes_j[s][-1] for s in order])
    print(f"\njoint window: {len(dates_j)} common bars ({dates_j[0]} -> {dates_j[-1]}), "
          f"capped by the youngest listing")
    if len(dates_j) < 400:
        print("  WARNING: short joint window -- correlation/tail estimates below are rough; "
              "the per-name tables above use each name's full history instead.")
    target = np.asarray([mu_m] + [drifts[n] for n in names])
    pj = joint_bootstrap_prices(rm_j, s0j, H, n_paths=a.paths, seed=a.seed + 1,
                                block=a.block, recenter_mu=target,
                                vol_standardize=not a.raw_vol,
                                vol_term_structure=not a.flat_vol,
                                recency_half_life=None if a.no_recency else a.recency_half_life
                                ).astype(float)

    corr = np.corrcoef(rm_j.T)
    print("\ncorrelation matrix (daily returns, joint window):")
    print("       " + "".join(f"{s:>7}" for s in order))
    for i, s in enumerate(order):
        print(f"{s:>6} " + "".join(f"{corr[i, j]:>7.2f}" for j in range(len(order))))

    spy_ret = pj[:, -1, 0] / s0j[0] - 1.0
    buckets = [("SPY <= -10%", spy_ret <= -0.10), ("SPY -10..0%", (spy_ret > -0.10) & (spy_ret <= 0)),
               ("SPY 0..+10%", (spy_ret > 0) & (spy_ret < 0.10)), ("SPY >= +10%", spy_ret >= 0.10)]
    print("\n=== year-end median move CONDITIONAL on the market (beta cuts both ways) ===")
    print(f"{'scenario':>13}{'P(scen)':>9}{'n':>7}" + "".join(f"{n:>9}" for n in names))
    for label, mask in buckets:
        cnt = int(mask.sum())
        if cnt < 50:
            print(f"{label:>13}{mask.mean() * 100:>8.0f}%{cnt:>7}   (too few paths to report)")
            continue
        row = "".join(f"{(np.median(pj[mask, -1, k]) / s0j[k] - 1) * 100:>8.0f}%"
                      for k in range(1, len(order)))
        print(f"{label:>13}{mask.mean() * 100:>8.0f}%{cnt:>7}{row}")
    deep = pj[:, :, 0].min(axis=1) / s0j[0] - 1.0 <= -0.15
    if deep.sum() >= 50:
        row = "".join(f"{(np.median(pj[deep, :, k].min(axis=1)) / s0j[k] - 1) * 100:>8.0f}%"
                      for k in range(1, len(order)))
        print(f"{'SPY dips -15%':>13}{deep.mean() * 100:>8.0f}%{int(deep.sum()):>7}{row}"
              f"   <- median TROUGH")

    rel = pj[:, :, 1:] / s0j[None, None, 1:]
    basket = rel.mean(axis=2)
    bt, bmin = basket[:, -1], basket.min(axis=1)
    print(f"\n=== equal-weight basket of {len(names)} names ===")
    print(f"  year-end: median {(np.median(bt) - 1) * 100:+.0f}%, 90% range "
          f"[{(np.percentile(bt, 5) - 1) * 100:+.0f}% .. {(np.percentile(bt, 95) - 1) * 100:+.0f}%], "
          f"P(up) = {np.mean(bt > 1) * 100:.0f}%")
    print(f"  cumulative dip odds: -10% {np.mean(bmin <= 0.90) * 100:.0f}%   "
          f"-20% {np.mean(bmin <= 0.80) * 100:.0f}%   -30% {np.mean(bmin <= 0.70) * 100:.0f}%   "
          f"median worst dip {(1 - np.median(bmin)) * 100:.1f}%")
    r2 = np.random.default_rng(a.seed + 2)
    indep = np.stack([rel[r2.permutation(rel.shape[0]), :, j] for j in range(rel.shape[2])], axis=2)
    print(f"  if INDEPENDENT (counterfactual): -20% dip odds "
          f"{np.mean(indep.mean(axis=2).min(axis=1) <= 0.80) * 100:.0f}% vs "
          f"{np.mean(bmin <= 0.80) * 100:.0f}% actual -> correlation is why a basket of these "
          f"does NOT diversify the crash away.")

    unval = [n for n in names if n not in status]
    if unval:
        print(f"\nCAVEAT: {', '.join(unval)} have no stored validation. Run --validate; "
              f"until then their bands are extrapolation.")
    bad = [n for n in names if status.get(n, {}).get("verdict", "").startswith("RULED OUT")]
    if bad:
        print(f"WARNING: {', '.join(bad)} FAILED calibration in the last validation run. "
              f"Treat their numbers as unreliable.")
    return 0


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------

def validate(a, db) -> int:
    names = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    print("=== walk-forward validation: naive momentum vs beta-anchored ===")
    print(DISCLAIMER)
    print("\nfetching:")
    series = _load([MARKET] + names, a.history_days, db)
    names = [n for n in names if n in series]
    horizons = [int(x) for x in a.horizons.split(",")]
    mkt = series[MARKET]
    mkt_dates = sorted(mkt)
    methods = ("naive", "beta") + (() if a.raw_vol else ("beta+volstd", "beta+volstd+ts"))
    status = _load_status()

    print(f"\n{'name':>6}{'method':>12}{'hzn':>5}{'n':>5}{'n_eff':>7}{'cov90':>7}"
          f"{'95% CI':>16}{'<p5':>7}{'touch pred':>11}{'touch real':>11}   verdict")
    for n in names:
        dates_n, closes_n = align_closes({MARKET: mkt, n: series[n]})
        if len(dates_n) < MIN_VALIDATE_BARS:
            print(f"{n:>6}  SKIPPED -- only {len(dates_n)} common bars "
                  f"(<{MIN_VALIDATE_BARS}); cannot validate, bands are extrapolation.")
            continue
        rm = log_return_matrix(dates_n, closes_n, [MARKET, n])
        px_n, px_m = closes_n[n], closes_n[MARKET]
        month_ends = [i for i in range(len(dates_n) - 1)
                      if (dates_n[i].year, dates_n[i].month) != (dates_n[i + 1].year, dates_n[i + 1].month)]
        # hit90, below5, touch_pred_sum, touch_real, n
        acc = {(m, h): [0, 0, 0.0, 0, 0] for m in methods for h in horizons}
        for i in month_ends:
            if i < 300:
                continue
            # market drift from the FULL SPY history up to this date (not the truncated
            # common window) -- matches what report() does.
            d_i = dates_n[i]
            mu_mkt = estimate([mkt[d] for d in mkt_dates if d <= d_i]).mu_blend
            b = beta_stats(n, rm[:i, 1], rm[:i, 0])
            for h in horizons:
                if i + h >= len(px_n):
                    continue
                realized = px_n[i + h]
                realized_min = float(np.min(px_n[i + 1:i + h + 1]))
                touched = realized_min <= px_n[i] * (1 + TOUCH_LEVEL)
                s0v = np.asarray([px_m[i], px_n[i]])
                sims = {}
                sims["naive"] = bootstrap_prices(
                    list(px_n[:i + 1]), h, n_paths=a.vpaths, seed=3000 + i, block=a.block,
                    recenter_mu=estimate(list(px_n[:i + 1])).mu_blend)
                sims["beta"] = joint_bootstrap_prices(
                    rm[:i], s0v, h, n_paths=a.vpaths, seed=4000 + i, block=a.block,
                    recenter_mu=np.asarray([mu_mkt, b.beta * mu_mkt]))[:, :, 1]
                if "beta+volstd" in methods:
                    sims["beta+volstd"] = joint_bootstrap_prices(
                        rm[:i], s0v, h, n_paths=a.vpaths, seed=5000 + i, block=a.block,
                        recenter_mu=np.asarray([mu_mkt, b.beta * mu_mkt]),
                        vol_standardize=True)[:, :, 1]
                    # + vol term structure and recency-weighted blocks
                    sims["beta+volstd+ts"] = joint_bootstrap_prices(
                        rm[:i], s0v, h, n_paths=a.vpaths, seed=6000 + i, block=a.block,
                        recenter_mu=np.asarray([mu_mkt, b.beta * mu_mkt]),
                        vol_standardize=True, vol_term_structure=True,
                        recency_half_life=a.recency_half_life)[:, :, 1]
                for m, mat in sims.items():
                    mat = np.asarray(mat, dtype=float)
                    p5, p95 = np.percentile(mat[:, -1], [5, 95])
                    pred_touch = float(np.mean(mat.min(axis=1) <= px_n[i] * (1 + TOUCH_LEVEL)))
                    s = acc[(m, h)]
                    s[0] += int(p5 <= realized <= p95)
                    s[1] += int(realized < p5)
                    s[2] += pred_touch
                    s[3] += int(touched)
                    s[4] += 1
        best = None
        for h in horizons:
            for m in methods:
                hit, lo5, tp, tr, cnt = acc[(m, h)]
                if not cnt:
                    continue
                r = rate(hit, cnt, horizon_days=h)
                v = r.verdict(0.90)
                print(f"{n:>6}{m:>12}{h:>5}{cnt:>5}{r.n_eff:>7.0f}{r.p:>7.2f}"
                      f"   [{r.lo:.2f}..{r.hi:.2f}]{lo5 / cnt:>7.2f}"
                      f"{tp / cnt:>11.2f}{tr / cnt:>11.2f}   {v}")
                if m != "naive" and (best is None or abs(r.p - 0.90) < abs(best[1] - 0.90)):
                    best = (m, r.p, r.lo, r.hi, cnt, v)
        if best:
            status[n] = {"when": dt.date.today().isoformat(), "method": best[0],
                         "cov90": round(best[1], 3), "lo": round(best[2], 3),
                         "hi": round(best[3], 3), "n": best[4], "verdict": best[5]}
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nwrote {STATUS_PATH} (report() stamps each name with this)")
    print("targets: cov90 ~0.90, <p5 ~0.05, touch pred ~ touch real.")
    print("n_eff adjusts for OVERLAPPING forecast windows -- the CI uses it, so a wide "
          "interval means 'weak evidence', not 'bad model'.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="High-beta basket outlook via market factor")
    ap.add_argument("--symbols", default="ASTS,IREN,NBIS,TE,CRWV")
    ap.add_argument("--history-days", type=int, default=1600)
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--vpaths", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--horizons", default="51,73")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="daily alpha drag added to every anchored drift (e.g. -0.0005)")
    ap.add_argument("--idio-drift", action="store_true",
                    help="also project each name's residual drift (OFF: it is noise)")
    ap.add_argument("--raw-vol", action="store_true",
                    help="disable volatility standardization (use raw historical returns)")
    ap.add_argument("--flat-vol", action="store_true",
                    help="hold today's vol flat instead of decaying it toward long-run")
    ap.add_argument("--no-recency", action="store_true",
                    help="sample all history uniformly instead of favouring recent regimes")
    ap.add_argument("--recency-half-life", type=float, default=500.0,
                    help="trading-day half-life for recency-weighted block sampling")
    ap.add_argument("--reality-tol", type=float, default=1.5,
                    help="flag/correct a model percentile when it exceeds the name's own "
                         "realized move at that frequency by more than this factor")
    ap.add_argument("--no-beta-uncertainty", action="store_true",
                    help="do not propagate beta's standard error into the bands")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        return validate(a, db) if a.validate else report(a, db)
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
