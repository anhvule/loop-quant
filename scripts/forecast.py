"""Forecast a stock's multi-day price path (GBM + ARIMA).

Pulls daily history from yfinance into the Loop Quant `candles` table, computes a
technical read with the live-engine indicators/signal, then projects the price
forward two independent ways and writes a console table, a CSV, and a PNG chart.

    python -m scripts.forecast --symbol WYFI --history-days 500 --horizon 15
    python -m scripts.forecast --symbol AAPL --horizon 10 --paths 20000 --no-chart

NOT INVESTMENT ADVICE. This is a research baseline, not a prediction of real prices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs       # noqa: E402
from src.forecast import report                                   # noqa: E402
from src.forecast.arima import forecast_arima                     # noqa: E402
from src.forecast.features import compute_features                # noqa: E402
from src.forecast.gbm import simulate_gbm                         # noqa: E402
from src.forecast.stock_data import InsufficientDataError, fetch_yfinance  # noqa: E402

# How hard the technical signal is allowed to bend the GBM drift. The tilt added to
# the daily drift is `signal_score * TILT_STRENGTH * sigma`, so at full-strength
# signal (score = +/-1) it shifts drift by a quarter of a daily stdev -- a nudge,
# not an override. Kept modest on purpose: the signal is a weak daily-bar prior.
TILT_STRENGTH = 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-day price-path forecast (GBM + ARIMA)")
    ap.add_argument("--symbol", required=True, help="ticker, e.g. WYFI or AAPL")
    ap.add_argument("--history-days", type=int, default=500,
                    help="calendar days of history to request (default 500)")
    ap.add_argument("--horizon", type=int, default=15, help="trading days to project")
    ap.add_argument("--paths", type=int, default=10_000, help="GBM Monte Carlo paths")
    ap.add_argument("--seed", type=int, default=7, help="GBM RNG seed (determinism)")
    ap.add_argument("--no-tilt", action="store_true",
                    help="disable the signal-based GBM drift tilt")
    ap.add_argument("--no-chart", action="store_true", help="skip the PNG chart")
    ap.add_argument("--db", default=None, help="SQLite path (default: project DB)")
    a = ap.parse_args()

    symbol = a.symbol.upper()
    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        try:
            candles = fetch_yfinance(symbol, days=a.history_days, interval="1d", db=db)
        except InsufficientDataError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print("Tip: verify the ticker on finance.yahoo.com, or try a liquid symbol "
                  "(e.g. AAPL) to confirm the pipeline.", file=sys.stderr)
            return 2

        feats = compute_features(candles)
        closes = [c.close for c in candles]

        drift_tilt = 0.0 if a.no_tilt else feats.signal_score * TILT_STRENGTH * feats.sigma
        gbm = simulate_gbm(closes, a.horizon, n_paths=a.paths, seed=a.seed,
                           drift_tilt=drift_tilt)
        arima = forecast_arima(closes, a.horizon)

        print(report.console_table(symbol, feats, gbm, arima))

        csv_path = report.write_csv(report.default_csv_path(DATA_DIR, symbol), gbm, arima)
        print(f"\nwrote {csv_path}")
        if not a.no_chart:
            png = report.write_chart(report.default_chart_path(DATA_DIR, symbol),
                                     symbol, candles, gbm, arima)
            if png:
                print(f"wrote {png}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
