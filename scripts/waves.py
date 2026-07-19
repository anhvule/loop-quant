"""Mechanical Elliott Wave stage + Fibonacci ladders for a ticker.

    python -m scripts.waves --symbol ASTS
    python -m scripts.waves --symbol IREN --k-atr 2.5 --no-chart

Prints the current stage with RIVAL counts (flagged AMBIGUOUS when close), the fib
retracement ladder below price ("fib bottom") and extension ladder above ("fib top"),
and stamps whatever `scripts.validate_waves` concluded about whether any of it predicts.

NOT INVESTMENT ADVICE. This describes price structure; it does not forecast.
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

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs       # noqa: E402
from src.forecast import waves as W                                # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

VERDICT_PATH = DATA_DIR / "waves_validation.json"
DISCLAIMER = ("NOT INVESTMENT ADVICE. Elliott counts are subjective in practice and have "
              "no established predictive validity; these are computed mechanically so they "
              "are at least reproducible and testable.")


def _verdict(symbol: str) -> str:
    try:
        data = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "UNVALIDATED -- run `python -m scripts.validate_waves` to test these levels."
    v = data.get(symbol)
    if not v:
        return f"UNVALIDATED for {symbol} -- run `python -m scripts.validate_waves`."
    if v.get("verdict", "").startswith("insufficient"):
        return f"{symbol}: insufficient data to validate (n={v.get('n', 0)})."
    return (f"{symbol} validation ({v['when']}, n={v['n']}): {v['verdict']}. "
            f"skill ratio {v['skill_ratio']} (1.0 = no better than random lines), "
            f"fib closer in {v['win_rate'] * 100:.0f}% of windows.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Elliott stage + Fibonacci levels")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--history-days", type=int, default=1600)
    ap.add_argument("--k-atr", type=float, default=W.K_ATR,
                    help="swing filter = k x ATR%% (higher = fewer, bigger swings)")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    symbol = a.symbol.upper()
    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        try:
            candles = fetch_yfinance(symbol, days=a.history_days, interval="1d", db=db)
        except InsufficientDataError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        closes = np.asarray([c.close for c in candles], float)
        highs = np.asarray([c.high for c in candles], float)
        lows = np.asarray([c.low for c in candles], float)
        dates = [dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date()
                 for c in candles]
        an = W.analyse(closes, highs, lows, k_atr=a.k_atr)
        spot = an["spot"]

        print(f"=== {symbol} wave structure ===")
        print(DISCLAIMER)
        print(f"\n{_verdict(symbol)}\n")
        print(f"spot {spot:,.2f} as of {dates[-1]} | ATR {an['atr_pct'] * 100:.1f}%/day "
              f"| swing filter {an['threshold'] * 100:.1f}% | {len(candles)} bars")

        piv = an["pivots"]
        robust = sum(1 for p in piv if p.robust)
        print(f"pivots: {len(piv)} ({robust} robust across thresholds, "
              f"{len(piv) - robust} fragile)")
        for p in piv[-6:]:
            tag = "robust" if p.robust else "FRAGILE"
            prov = "" if p.confirmed else "  [provisional - swing still open]"
            print(f"   {dates[p.index]}  {p.kind}  {p.price:>10,.2f}   {tag}{prov}")

        counts = an["counts"]
        print("\n--- stage readings (top 3, scored) ---")
        if not counts:
            print("   no rule-valid Elliott structure in the recent pivots.")
        else:
            for i, c in enumerate(counts):
                lead = "BEST " if i == 0 else "rival"
                print(f"   {lead}  {c.label:<34} {c.stage:<40} score {c.score:.2f}")
                if c.ratios:
                    print("          " + "  ".join(f"{k}={v:.2f}" for k, v in c.ratios.items()))
            if an["ambiguous"]:
                print("   >> AMBIGUOUS: the top readings are within "
                      f"{W.AMBIGUITY_MARGIN * 100:.0f}% of each other. The stage is a "
                      "coin-flip between them; do not act on the label.")
            print("   note: RISING/FALLING is the direction the structure TRAVELS "
                  "(a falling correction = price went down). '-> ... expected' is what "
                  "the theory says comes next structurally, NOT a validated forecast.")

        def show(levels, title):
            print(f"\n--- {title} ---")
            if not levels:
                print("   (none)")
                return
            for l in levels[:8]:
                tags = []
                if l.confluent:
                    tags.append("CONFLUENT")
                if l.near_pivot:
                    tags.append("at a prior pivot")
                note = ("  <- " + ", ".join(tags)) if tags else "  (floats in empty space)"
                print(f"   {l.price:>10,.2f}  ({(l.price / spot - 1) * 100:+6.1f}%)  "
                      f"{l.label:<18} from {l.source}{note}")

        show(an["fib_below"], "FIB BOTTOM ladder (support candidates below spot)")
        show(an["fib_above"], "FIB TOP ladder (targets above spot)")
        print("\nLevels marked CONFLUENT (agreeing across swings) or sitting at a prior "
              "pivot are the only ones with even anecdotal support. Ones floating in "
              "empty space are just arithmetic on a chart.")

        if not a.no_chart:
            png = _chart(DATA_DIR / f"waves_{symbol}.png", symbol, dates, closes, an)
            if png:
                print(f"\nwrote {png}")
    finally:
        db.close()
    return 0


def _wave_labels(count):
    """Point labels for a count: the origin is 0 (or the pre-A low/high), then the
    wave that ENDS at each subsequent pivot."""
    if count.kind.startswith("impulse"):
        return ["0"] + ["1", "2", "3", "4", "5"][:len(count.points) - 1]
    return ["start"] + ["A", "B", "C"][:len(count.points) - 1]


def _chart(path, symbol, dates, closes, an, pad=60):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Zoom to the labelled structure (plus context) instead of squeezing it into the
    # right-hand edge of a long history -- otherwise the wave labels are unreadable.
    counts = an["counts"]
    if counts:
        first = min(p.index for p in counts[0].points)
        lo_i = max(0, first - pad)
    else:
        lo_i = max(0, len(closes) - 300)
    x = list(range(lo_i, len(closes)))
    y = closes[lo_i:]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.plot(x, y, color="#333", lw=1.3, label=f"{symbol} close", zorder=2)

    piv = [p for p in an["pivots"] if p.index >= lo_i]
    if piv:
        ax.plot([p.index for p in piv], [p.price for p in piv], color="#9bb8de",
                lw=1.0, ls="--", alpha=.9, label="all detected swings", zorder=3)
        for p in piv:
            ax.scatter([p.index], [p.price], s=30 if p.robust else 14,
                       color="#1a5fb4" if p.robust else "#c9d8ee",
                       marker="v" if p.kind == "H" else "^", zorder=4)

    # rival count first (behind), then the best count on top
    if len(counts) > 1:
        r = counts[1]
        rx = [p.index for p in r.points]
        ry = [p.price for p in r.points]
        ax.plot(rx, ry, color="#e39146", lw=1.8, alpha=.75, zorder=5,
                label=f"rival: {r.label} — {r.stage}")
        for lab, p in zip(_wave_labels(r), r.points):
            ax.annotate(lab, (p.index, p.price), textcoords="offset points",
                        xytext=(10, 8 if p.kind == "H" else -16), ha="center",
                        fontsize=10, color="#a35d16", alpha=.9)

    if counts:
        b = counts[0]
        bx = [p.index for p in b.points]
        by = [p.price for p in b.points]
        ax.plot(bx, by, color="#b3232c", lw=3.0, zorder=6,
                label=f"BEST: {b.label} — {b.stage}")
        ax.scatter(bx, by, s=70, color="#b3232c", zorder=7)
        for lab, p in zip(_wave_labels(b), b.points):
            ax.annotate(lab, (p.index, p.price), textcoords="offset points",
                        xytext=(0, 14 if p.kind == "H" else -22), ha="center",
                        fontsize=15, fontweight="bold", color="#b3232c", zorder=8,
                        bbox=dict(boxstyle="circle,pad=0.22", fc="white",
                                  ec="#b3232c", lw=1.4))
        if not b.complete:
            last = b.points[-1]
            ax.annotate("current wave\nstill in progress", (last.index, last.price),
                        textcoords="offset points", xytext=(28, 0), fontsize=8,
                        color="#b3232c", va="center",
                        arrowprops=dict(arrowstyle="->", color="#b3232c", lw=1))

    vis_lo, vis_hi = float(np.min(y)) * .8, float(np.max(y)) * 1.2
    for l in an["levels"]:
        if not (vis_lo <= l.price <= vis_hi):
            continue
        strong = l.confluent or l.near_pivot
        ax.axhline(l.price, color="#54a24b", ls="-" if strong else ":",
                   lw=1.3 if strong else .7, alpha=.6 if strong else .28, zorder=1)
        # Only the meaningful levels get text -- labelling all ~24 produces an
        # unreadable pile on the right edge, and the weak ones are noise anyway.
        if strong:
            ax.annotate(f"★ {l.label} {l.price:,.2f}", (x[-1], l.price), fontsize=8,
                        color="#2d6b28", va="center", fontweight="bold",
                        xytext=(6, 0), textcoords="offset points")

    ax.axhline(an["spot"], color="#666", lw=1.0, alpha=.7)
    ax.annotate(f"spot {an['spot']:,.2f}", (x[0], an["spot"]), fontsize=8,
                color="#444", va="bottom", xytext=(2, 3), textcoords="offset points")

    stage = counts[0].stage if counts else "no rule-valid structure"
    amb = "   ⚠ AMBIGUOUS — rival reading nearly ties" if an["ambiguous"] else ""
    ax.set_title(f"{symbol} — mechanical Elliott/Fibonacci\nstage: {stage}{amb}\n"
                 f"{DISCLAIMER}", fontsize=9.5)
    ax.set_xlabel(f"bar index  ({dates[lo_i]} → {dates[-1]})")
    ax.set_ylabel("price")
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(alpha=.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())
