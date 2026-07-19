"""Multi-month probabilistic outlook for a ticker (GBM + block bootstrap).

Reframes the long-horizon question from "what price?" to "what are the odds?": for
each target date it reports P(up), P(>=+5%), P(<=-5%), P(<=-10%), expected value, and
a percentile table, from two independent methods, plus month seasonality context.

    python -m scripts.outlook --symbol SPY --dates 2026-09-30,2026-10-30 --vix-anchor
    python -m scripts.outlook --symbol SPY --dates 2026-10-30 --paths 40000 --no-chart

Pair with `scripts.validate_outlook` to check the bands are calibrated before trusting
them. NOT INVESTMENT ADVICE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs       # noqa: E402
from src.forecast import outlook as ol                            # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402


def _parse_dates(s: str) -> list[dt.date]:
    return [dt.date.fromisoformat(x.strip()) for x in s.split(",") if x.strip()]


def _fetch_vix_close() -> float | None:
    """Prefer 3-month implied vol (^VIX3M) for multi-month horizons; fall back to ^VIX."""
    for sym in ("^VIX3M", "^VIX"):
        try:
            candles = fetch_yfinance(sym, days=120, interval="1d", min_bars=5)
            if sym == "^VIX":
                print("note: using 30-day ^VIX (^VIX3M unavailable)", file=sys.stderr)
            return candles[-1].close
        except Exception:  # noqa: BLE001  try next / degrade to EWMA-only
            continue
    print("note: could not fetch ^VIX3M/^VIX; using EWMA vol only", file=sys.stderr)
    return None


# Approximate 2026 Q3-Q4 FOMC decision dates -- context only, UNVERIFIED, not modeled.
FOMC_2026_H2 = ["2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Probabilistic multi-month outlook (GBM + bootstrap)")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--dates", default="2026-09-30,2026-10-30",
                    help="comma-separated target dates YYYY-MM-DD")
    ap.add_argument("--history-days", type=int, default=2600,
                    help="calendar days of history to request (default ~10y trading)")
    ap.add_argument("--paths", type=int, default=20_000, help="Monte Carlo paths")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--block", type=int, default=10, help="bootstrap block length (days)")
    ap.add_argument("--w-recent", type=float, default=0.30,
                    help="weight on recent-window drift vs long-run mean")
    ap.add_argument("--vix-anchor", action="store_true",
                    help="blend ^VIX3M-implied vol 50/50 with EWMA")
    ap.add_argument("--seasonal-vol", action="store_true",
                    help="scale GBM vol by calendar-month seasonality (Sep/Oct run hotter)")
    ap.add_argument("--headline", choices=["gbm", "bootstrap"], default="gbm",
                    help="method to feature (run scripts.validate_outlook to choose)")
    ap.add_argument("--as-of", default=None,
                    help="anchor date YYYY-MM-DD (default: last available bar)")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    symbol = a.symbol.upper()
    targets = _parse_dates(a.dates)
    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        try:
            candles = fetch_yfinance(symbol, days=a.history_days, interval="1d", db=db)
        except InsufficientDataError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        last_bar = dt.datetime.fromtimestamp(
            candles[-1].ts_open_ms / 1000, dt.timezone.utc).date()
        start_date = dt.date.fromisoformat(a.as_of) if a.as_of else last_bar

        bad = [t for t in targets if t <= start_date]
        if bad:
            print(f"ERROR: target date(s) {bad} are not after the anchor {start_date}",
                  file=sys.stderr)
            return 2

        vix_close = _fetch_vix_close() if a.vix_anchor else None

        o = ol.build_outlook(
            symbol, candles, targets, start_date,
            n_paths=a.paths, seed=a.seed, block=a.block, w_recent=a.w_recent,
            vix_close=vix_close, headline=a.headline, seasonal=a.seasonal_vol,
        )
        print(ol.console_outlook(o))
        fomc = [d for d in FOMC_2026_H2 if start_date.isoformat() < d <= max(targets).isoformat()]
        if fomc:
            print(f"\nFOMC dates in window (context only, UNVERIFIED, not modeled): {', '.join(fomc)}")

        csv_path = ol.write_outlook_csv(DATA_DIR / f"outlook_{symbol}.csv", o)
        print(f"\nwrote {csv_path}")
        if not a.no_chart:
            png = ol.write_outlook_chart(DATA_DIR / f"outlook_{symbol}.png", o, candles)
            if png:
                print(f"wrote {png}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
