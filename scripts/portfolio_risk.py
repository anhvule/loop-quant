"""Portfolio concentration and drawdown risk from a holdings CSV.

Answers the question a list of tickers can actually support: how much genuine
diversification is here, and what does the whole book do in a bad tape?

    python -m scripts.portfolio_risk --csv portfolio.csv
    python -m scripts.portfolio_risk --csv portfolio.csv --horizon 126 --paths 8000

Reads `Symbol` (required) and `Quantity`/`Purchase Price` (optional). WITHOUT quantities
every position is assumed EQUAL WEIGHT -- stated in the output, because it materially
changes the answer.

Concentration is measured three ways rather than by counting tickers:
  * average pairwise correlation
  * EFFECTIVE NUMBER OF BETS  N_eff = N / (1 + (N-1) * avg_corr) -- what the position
    count is really worth once co-movement is accounted for
  * share of variance explained by the first principal component (the dominant factor
    every name is secretly loading on)

Risk is simulated jointly (shared date blocks -> real correlations and co-crash days),
then compared with a counterfactual where the same names move INDEPENDENTLY. The gap
between those two is the diversification the portfolio does not actually have.

NOT INVESTMENT ADVICE. This describes risk; it does not judge positions or recommend
any action.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs       # noqa: E402
from src.forecast.beta import (                                    # noqa: E402
    align_closes, beta_stats, joint_bootstrap_prices, log_return_matrix,
)
from src.forecast.longrange import estimate                        # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

MARKET = "SPY"
DISCLAIMER = ("NOT INVESTMENT ADVICE. Risk description only -- no judgement of individual "
              "positions, no buy/sell/hold implication. Price history cannot see earnings, "
              "launches, contracts, crypto prices or macro events.")


def read_symbols(path: Path):
    """(symbols, weights or None, notes). Weights come from Quantity x price when present."""
    rows = list(csvmod.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    syms, qty, px, notes = [], [], [], []
    for r in rows:
        s = (r.get("Symbol") or "").strip().upper()
        if not s:
            continue
        syms.append(s)
        def num(key):
            try:
                return float((r.get(key) or "").strip())
            except (TypeError, ValueError):
                return None
        qty.append(num("Quantity"))
        px.append(num("Current Price"))
    have_qty = all(q is not None and q > 0 for q in qty) and len(qty) == len(syms)
    if have_qty:
        vals = np.asarray([q * (p or 0.0) for q, p in zip(qty, px)], dtype=float)
        if vals.sum() > 0:
            notes.append("weights from Quantity x Current Price")
            return syms, vals / vals.sum(), notes
    notes.append("NO quantities in the file -> EQUAL WEIGHT assumed for every position. "
                 "Real weights would change these numbers, possibly a lot.")
    return syms, None, notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio concentration + drawdown risk")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--history-days", type=int, default=1600)
    ap.add_argument("--horizon", type=int, default=126, help="trading days (~6 months)")
    ap.add_argument("--paths", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=1000, help="paths per memory chunk")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--db", default=None)
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()

    symbols, weights, notes = read_symbols(Path(a.csv))
    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        print("=== portfolio concentration & risk ===")
        print(DISCLAIMER)
        for n in notes:
            print(f"NOTE: {n}")
        print(f"\nfetching {len(symbols)} symbols + {MARKET} ...")
        series, failed = {}, []
        for s in [MARKET] + symbols:
            try:
                cs = fetch_yfinance(s, days=a.history_days, interval="1d", db=db, min_bars=120)
            except (InsufficientDataError, Exception) as e:  # noqa: BLE001
                failed.append((s, type(e).__name__))
                continue
            series[s] = {dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date():
                         c.close for c in cs}
        names = [s for s in symbols if s in series]
        if MARKET not in series or len(names) < 2:
            print("ERROR: need the market series and >=2 usable names", file=sys.stderr)
            return 2
        if failed:
            print(f"  could not load ({len(failed)}): "
                  + ", ".join(f"{s}" for s, _ in failed))
        print(f"  usable: {len(names)} names")

        # ---- betas on each name's own full overlap with the market ----
        betas = {}
        for n in names:
            try:
                dn, cn = align_closes({MARKET: series[MARKET], n: series[n]})
                rmn = log_return_matrix(dn, cn, [MARKET, n])
                betas[n] = beta_stats(n, rmn[:, 1], rmn[:, 0])
            except Exception:  # noqa: BLE001
                betas[n] = None

        # ---- joint window ----
        order = [MARKET] + names
        dates_j, closes_j = align_closes({s: series[s] for s in order})
        rm = log_return_matrix(dates_j, closes_j, order)
        s0 = np.asarray([closes_j[s][-1] for s in order], dtype=float)
        K = len(names)
        w = np.full(K, 1.0 / K) if weights is None else np.asarray(
            [weights[symbols.index(n)] for n in names], dtype=float)
        w = w / w.sum()
        print(f"\njoint window: {len(dates_j)} shared bars ({dates_j[0]} -> {dates_j[-1]}), "
              f"capped by the youngest listing")
        if len(dates_j) < 400:
            print("  WARNING: short shared window -- correlation and tail estimates are rough.")

        # ---- concentration ----
        R = rm[:, 1:]                                   # names only
        C = np.corrcoef(R.T)
        iu = np.triu_indices(K, 1)
        avg_corr = float(np.mean(C[iu]))
        n_eff = K / (1.0 + (K - 1) * avg_corr) if avg_corr > -1 / (K - 1) else float(K)
        evals, evecs = np.linalg.eigh(C)
        order_e = np.argsort(evals)[::-1]
        evals, evecs = evals[order_e], evecs[:, order_e]
        pc1_share = float(evals[0] / evals.sum())
        pc1 = evecs[:, 0]
        if np.mean(pc1) < 0:
            pc1 = -pc1

        print("\n--- CONCENTRATION ---")
        print(f"  positions                          : {K}")
        print(f"  average pairwise correlation       : {avg_corr:.2f}")
        print(f"  EFFECTIVE number of independent bets: {n_eff:.1f}  "
              f"(you hold {K} tickers; they behave like ~{n_eff:.0f})")
        print(f"  variance explained by ONE factor   : {pc1_share * 100:.0f}%  "
              f"(first principal component)")
        loads = sorted(zip(names, pc1), key=lambda t: -abs(t[1]))
        print("  most factor-exposed names (highest loading on that single factor):")
        print("     " + ", ".join(f"{n}({l:+.2f})" for n, l in loads[:12]))
        print("  least factor-exposed (the only real diversifiers here):")
        print("     " + ", ".join(f"{n}({l:+.2f})" for n, l in loads[-8:]))

        pairs = sorted(((float(C[i, j]), names[i], names[j]) for i, j in zip(*iu)),
                       reverse=True)[:10]
        print("  most correlated pairs:")
        print("     " + ", ".join(f"{x}/{y} {c:.2f}" for c, x, y in pairs))

        bl = [(n, betas[n].beta) for n in names if betas[n]]
        hi_beta = [n for n, b in bl if b >= 2.0]
        print(f"\n  beta profile: median {np.median([b for _, b in bl]):.2f}, "
              f"{len(hi_beta)}/{len(bl)} names at beta >= 2.0")
        print("     highest: " + ", ".join(
            f"{n} {b:.1f}" for n, b in sorted(bl, key=lambda t: -t[1])[:8]))

        # ---- joint simulation, chunked to bound memory ----
        mu_m = estimate([series[MARKET][d] for d in sorted(series[MARKET])]).mu_blend
        target = np.asarray([mu_m] + [(betas[n].beta * mu_m if betas[n] else mu_m)
                                      for n in names])
        H = a.horizon
        rng_perm = np.random.default_rng(a.seed + 99)
        port_end, port_min, ind_end, ind_min, spy_end, spy_min = [], [], [], [], [], []
        done = 0
        c_i = 0
        while done < a.paths:
            npaths = min(a.chunk, a.paths - done)
            pr = joint_bootstrap_prices(rm, s0, H, n_paths=npaths, seed=a.seed + c_i,
                                        block=a.block, recenter_mu=target,
                                        vol_standardize=True, vol_term_structure=True,
                                        recency_half_life=500.0).astype(np.float32)
            rel = pr[:, :, 1:] / s0[None, None, 1:]
            port = np.einsum("nhk,k->nh", rel, w.astype(np.float32))
            port_end.append(port[:, -1]); port_min.append(port.min(axis=1))
            spy_rel = pr[:, :, 0] / s0[0]
            spy_end.append(spy_rel[:, -1]); spy_min.append(spy_rel.min(axis=1))
            # counterfactual: same marginals, correlation destroyed
            ind = np.stack([rel[rng_perm.permutation(npaths), :, k] for k in range(K)], axis=2)
            iport = np.einsum("nhk,k->nh", ind, w.astype(np.float32))
            ind_end.append(iport[:, -1]); ind_min.append(iport.min(axis=1))
            done += npaths
            c_i += 1
        cat = lambda xs: np.concatenate(xs)
        port_end, port_min = cat(port_end), cat(port_min)
        ind_end, ind_min = cat(ind_end), cat(ind_min)
        spy_end, spy_min = cat(spy_end), cat(spy_min)

        def dd(mins, lvl):
            return float(np.mean(mins <= 1 - lvl)) * 100

        print(f"\n--- PORTFOLIO RISK over {H} trading days (~{H / 21:.0f} months) ---")
        print(f"  year-ahead value: median {(np.median(port_end) - 1) * 100:+.0f}%, "
              f"90% range [{(np.percentile(port_end, 5) - 1) * 100:+.0f}% .. "
              f"{(np.percentile(port_end, 95) - 1) * 100:+.0f}%], "
              f"P(up) {np.mean(port_end > 1) * 100:.0f}%")
        print(f"  {'drawdown at any point':<26}{'ACTUAL':>10}{'if independent':>16}")
        for lvl in (0.10, 0.20, 0.30, 0.50):
            print(f"  {'  falls ' + f'{lvl * 100:.0f}%' + ' or more':<26}"
                  f"{dd(port_min, lvl):>9.0f}%{dd(ind_min, lvl):>15.0f}%")
        print(f"  {'  median worst moment':<26}{(1 - np.median(port_min)) * 100:>9.1f}%"
              f"{(1 - np.median(ind_min)) * 100:>15.1f}%")
        print("  -> the gap between the two columns is diversification the portfolio "
              "does NOT have.")

        for lo, hi, lbl in ((-1.0, -0.10, "SPY <= -10%"), (-0.10, 0.0, "SPY -10..0%"),
                            (0.0, 0.10, "SPY 0..+10%"), (0.10, 9.9, "SPY >= +10%")):
            m = (spy_end - 1 > lo) & (spy_end - 1 <= hi)
            if m.sum() < 50:
                continue
            print(f"  conditional {lbl:<13} P={m.mean() * 100:>3.0f}%  "
                  f"portfolio median {(np.median(port_end[m]) - 1) * 100:+.0f}%")
        deep = spy_min <= 0.85
        if deep.sum() >= 50:
            print(f"  if SPY dips -15% at any point (P={deep.mean() * 100:.0f}%): "
                  f"portfolio median trough {(np.median(port_min[deep]) - 1) * 100:+.0f}%")

        out = DATA_DIR / "portfolio_risk.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            wcsv = csvmod.writer(f)
            wcsv.writerow(["symbol", "weight", "beta", "r2", "vol_daily", "pc1_loading"])
            pc1_map = dict(zip(names, pc1))
            for i, n in enumerate(names):
                b = betas[n]
                wcsv.writerow([n, f"{w[i]:.4f}",
                               f"{b.beta:.3f}" if b else "", f"{b.r2:.3f}" if b else "",
                               f"{b.vol:.4f}" if b else "", f"{pc1_map[n]:.3f}"])
        print(f"\nwrote {out}")
        if not a.no_chart:
            p = _chart(DATA_DIR / "portfolio_risk.png", names, C, pc1, port_min, ind_min,
                       port_end, K, n_eff, avg_corr, pc1_share, H)
            if p:
                print(f"wrote {p}")
        if failed:
            print(f"\nCAVEAT: {len(failed)} symbols could not be loaded and are EXCLUDED: "
                  + ", ".join(s for s, _ in failed))
    finally:
        db.close()
    return 0


def _chart(path, names, C, pc1, port_min, ind_min, port_end, K, n_eff, avg_corr,
           pc1_share, H):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    im = axes[0].imshow(C, cmap="RdYlGn_r", vmin=-0.2, vmax=1.0)
    axes[0].set_title(f"correlation matrix\navg {avg_corr:.2f} — one factor explains "
                      f"{pc1_share * 100:.0f}%", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=.046)

    o = np.argsort(pc1)[::-1]
    top = o[:18]
    axes[1].barh([names[i] for i in top][::-1], [pc1[i] for i in top][::-1], color="#c23b3b")
    axes[1].set_title(f"loading on the single dominant factor\n{K} tickers behave like "
                      f"~{n_eff:.0f} independent bets", fontsize=10)
    axes[1].tick_params(labelsize=7); axes[1].grid(alpha=.2, axis="x")

    lv = [10, 20, 30, 50]
    act = [float(np.mean(port_min <= 1 - x / 100)) * 100 for x in lv]
    ind = [float(np.mean(ind_min <= 1 - x / 100)) * 100 for x in lv]
    x = np.arange(len(lv))
    axes[2].bar(x - .2, act, .4, color="#c23b3b", label="actual (correlated)")
    axes[2].bar(x + .2, ind, .4, color="#7cbf7a", label="if independent")
    for i, (av, iv) in enumerate(zip(act, ind)):
        axes[2].text(i - .2, av + 1, f"{av:.0f}%", ha="center", fontsize=8)
        axes[2].text(i + .2, iv + 1, f"{iv:.0f}%", ha="center", fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"-{v}%" for v in lv])
    axes[2].set_ylabel("probability (%)")
    axes[2].set_title(f"chance of a drawdown within {H} trading days\n"
                      "gap = diversification you do NOT have", fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.2, axis="y")

    fig.suptitle("Portfolio concentration & drawdown risk — " + DISCLAIMER, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())
