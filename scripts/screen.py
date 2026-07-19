"""Screen high-beta candidates on measurable criteria.

    python -m scripts.screen --symbols WYFI,OKLO,IREN,MRVL,AAOI
    python -m scripts.screen --csv portfolio.csv --min-beta 2.0

Grades each ticker on seven criteria (see src/forecast/screener.py) with the thresholds
printed alongside. The composite column uses EQUAL weights by default and is a
preference ranking, not a forecast -- pass --weights to change what you care about.

NOT INVESTMENT ADVICE. None of these criteria predicts returns.
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
from src.forecast import screener as S                             # noqa: E402
from src.forecast.beta import align_closes, log_return_matrix      # noqa: E402
from src.forecast.longrange import log_returns                     # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

MARKET = "SPY"
VERDICTS = DATA_DIR / "beta_validation_status.json"
BANNER = ("NOT INVESTMENT ADVICE. These criteria measure verifiability, survivability, "
          "redundancy, liquidity and payoff shape. NONE of them predicts returns -- "
          "direction failed every predictive test in this project.")
_COLS = [("beta", "high-beta"), ("verifiable", "verifiable"), ("survivable", "survivable"),
         ("redundancy", "adds bet"), ("liquidity", "liquid"), ("vol_regime", "vol now")]
_MARK = {"pass": "OK  ", "warn": "warn", "fail": "FAIL", "info": "--  "}


def main() -> int:
    ap = argparse.ArgumentParser(description="High-beta screener")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--min-beta", type=float, default=0.0,
                    help="only report names at or above this beta")
    ap.add_argument("--history-days", type=int, default=2000)
    ap.add_argument("--weights", default="",
                    help="key=value,... e.g. survivable=3,redundancy=2")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    if a.csv:
        rows = list(csvmod.DictReader(Path(a.csv).read_text(encoding="utf-8-sig").splitlines()))
        syms += [(r.get("Symbol") or "").strip().upper() for r in rows if (r.get("Symbol") or "").strip()]
    syms = list(dict.fromkeys(syms))
    if not syms:
        print("ERROR: pass --symbols or --csv", file=sys.stderr)
        return 2

    weights = {}
    for part in a.weights.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                weights[k.strip()] = float(v)
            except ValueError:
                pass

    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        print("=== high-beta screener ===")
        print(BANNER)
        print(f"\nfetching {len(syms)} symbols + {MARKET} ...")
        series, ohlcv, failed = {}, {}, []
        for s in [MARKET] + syms:
            try:
                cs = fetch_yfinance(s, days=a.history_days, interval="1d", db=db, min_bars=120)
            except (InsufficientDataError, Exception) as e:  # noqa: BLE001
                if s != MARKET:
                    failed.append(s)
                continue
            series[s] = {dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date():
                         c.close for c in cs}
            ohlcv[s] = {"close": np.asarray([c.close for c in cs], float),
                        "high": np.asarray([c.high for c in cs], float),
                        "low": np.asarray([c.low for c in cs], float),
                        "volume": np.asarray([c.volume for c in cs], float)}
        names = [s for s in syms if s in series]
        if MARKET not in series or not names:
            print("ERROR: need the market series and >=1 name", file=sys.stderr)
            return 2
        if failed:
            print(f"  could not load: {', '.join(failed)}")

        # Align each name to the market once; reused for the beta filter and scoring.
        aligned, rets = {}, {}
        for n in names:
            try:
                dn, cn = align_closes({MARKET: series[MARKET], n: series[n]})
                aligned[n] = {"mkt_r": log_return_matrix(dn, cn, [MARKET, n])[:, 0],
                              "shared": len(dn)}
                rets[n] = log_returns(cn[n])
            except Exception:  # noqa: BLE001
                aligned[n] = None
                rets[n] = None

        # Apply --min-beta BEFORE computing redundancy, so "adds a new bet" is measured
        # against the cohort actually displayed rather than against filtered-out names.
        if a.min_beta > 0:
            kept = []
            for n in names:
                if aligned[n] is None or rets[n] is None:
                    continue
                k = min(rets[n].size, aligned[n]["mkt_r"].size)
                if k < 40:
                    continue
                b, _ = S.c_beta(rets[n][-k:], aligned[n]["mkt_r"][-k:])
                if b.value is not None and b.value >= a.min_beta:
                    kept.append(n)
            print(f"\nfiltered to beta >= {a.min_beta}: {len(kept)} of {len(names)} names")
            print("   (redundancy below is measured against these kept names only)")
            names = kept

        corr = S.cohort_redundancy({n: rets[n] for n in names if rets[n] is not None})
        verdicts = S.load_verdicts(VERDICTS)

        cards = []
        for n in names:
            if aligned[n] is None:
                cards.append(S.Scorecard(n, [], [], "no overlap with the market series"))
                continue
            o = ohlcv[n]
            cards.append(S.build_scorecard(
                n, o["close"], o["high"], o["low"], o["volume"],
                aligned[n]["mkt_r"], aligned[n]["shared"],
                avg_corr=corr.get(n), n_peers=max(0, len(corr) - 1),
                verdict=verdicts.get(n)))

        scored = [(S.composite(c, weights) or 0.0, c) for c in cards]
        scored.sort(key=lambda t: -t[0])

        wtxt = ", ".join(f"{k}={v:g}" for k, v in weights.items()) or "all equal"
        print(f"\nweights: {wtxt}  -- YOUR preference ranking, NOT a forecast\n")
        head = f"{'sym':>6}{'score':>7}  " + "".join(f"{lbl:>11}" for _, lbl in _COLS)
        print(head)
        print("-" * len(head))
        for sc, c in scored:
            if c.error:
                print(f"{c.symbol:>6}{'--':>7}  {c.error}")
                continue
            cells = ""
            for key, _ in _COLS:
                cr = c.by_key(key)
                cells += f"{_MARK.get(cr.grade, '?') if cr else '?':>11}"
            print(f"{c.symbol:>6}{sc:>7.2f}  {cells}")

        print("\n--- detail ---")
        for sc, c in scored:
            if c.error:
                continue
            print(f"\n{c.symbol}  (composite {sc:.2f})")
            for cr in c.criteria:
                print(f"   [{_MARK.get(cr.grade, '?').strip():>4}] {cr.title:<20} "
                      f"{cr.display}")
                print(f"          threshold: {cr.threshold}"
                      + (f"  |  {cr.note}" if cr.note else ""))

        out = DATA_DIR / "screen.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(["symbol", "composite"] + [k for k, _ in _COLS]
                       + [f"{k}_value" for k, _ in _COLS])
            for sc, c in scored:
                if c.error:
                    continue
                g = [(c.by_key(k).grade if c.by_key(k) else "") for k, _ in _COLS]
                v = [(c.by_key(k).value if c.by_key(k) else "") for k, _ in _COLS]
                w.writerow([c.symbol, f"{sc:.3f}"] + g + v)
        print(f"\nwrote {out}")
        print("\nReminder: the composite reflects the weights YOU chose across risk and "
              "verifiability criteria. It is not a prediction that any name will rise.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
