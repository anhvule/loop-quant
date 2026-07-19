"""Classify high-beta names as investable or reckless.

    python -m scripts.classify                        # the whole S&P 500
    python -m scripts.classify --limit 50             # a quick slice of it
    python -m scripts.classify --symbols OKLO,IREN,ASTS
    python -m scripts.classify --csv "Yahoo portfolio.csv" --rf 0.045

`--csv` takes a Yahoo Finance portfolio export as downloaded -- cash rows, repeated
lots and the other fifteen columns are handled -- or any CSV with a Symbol/Ticker
column, or a plain list of tickers one per line.

Runs the five-gate stack in src/forecast/classify.py: liquidity and rolling beta are
hard eligibility gates, the market's 200d SMA is a regime overlay, and Treynor plus
idiosyncratic-share decide the label. Thresholds are printed beside every grade.

NOT INVESTMENT ADVICE. The Treynor gate ranks REALIZED return per unit of beta over the
trailing year. That is history, not a forecast.
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
from src.common.paths import DATA_DIR, DB_PATH, ensure_dirs        # noqa: E402
from src.forecast import classify as K                             # noqa: E402
from src.forecast.beta import align_closes, log_return_matrix      # noqa: E402
from src.forecast.longrange import log_returns                     # noqa: E402
from src.forecast.fundamentals import fetch_fundamentals           # noqa: E402
from src.forecast.portfolio import PortfolioError, symbols_from_csv  # noqa: E402
from src.forecast.stock_data import fetch_yfinance                 # noqa: E402
from src.forecast.universe import UniverseError, sp500_symbols     # noqa: E402

MARKET = "SPY"
PROGRESS_EVERY = 25
_MARK = {"pass": "OK  ", "warn": "warn", "fail": "FAIL", "info": "--  "}
_GATES = [("adv", "liquid"), ("beta", "high-beta"), ("regime", "regime"),
          ("treynor", "treynor"), ("idio", "mkt-drvn"), ("survivable", "surv"),
          ("own_trend", "trend"), ("dilution", "dilution"), ("runway", "runway"),
          ("revenue", "revenue")]
_CSV_COLS = ["symbol", "verdict", "adv_dollar", "beta_63", "beta_252", "r2_63", "r2_252",
             "idio_63", "idio_252", "treynor", "treynor_pctl",
             "halved_126d", "max_dd", "own_sma_gap",
             "dilution_yoy", "runway_quarters", "revenue_ttm",
             "flags", "regime", "reasons"]


def _resolve_universe(a) -> tuple[list[str], str]:
    """Explicit tickers win; otherwise fall back to the index constituents."""
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    source = "supplied"
    if a.csv:
        try:
            found, skipped = symbols_from_csv(a.csv)
        except PortfolioError as e:
            raise SystemExit(f"ERROR: {a.csv}: {e}")
        except OSError as e:
            raise SystemExit(f"ERROR: could not read {a.csv}: {e}")
        if skipped:
            head = ", ".join(f"{s['value']!r} ({s['reason']})" for s in skipped[:6])
            print(f"  ignored {len(skipped)} non-ticker row(s): {head}"
                  + (" ..." if len(skipped) > 6 else ""))
        syms += found
        source = f"{Path(a.csv).name}"
    if syms:
        return list(dict.fromkeys(syms)), source
    syms, provenance = sp500_symbols(refresh=a.refresh_universe)
    return syms, f"S&P 500 ({provenance})"


def _fetch(db, symbols: list[str], history_days: int) -> tuple[dict, dict, list[str]]:
    """(date->close per symbol, OHLCV arrays per symbol, failures)."""
    series, ohlcv, failed = {}, {}, []
    for i, s in enumerate([MARKET] + symbols, start=1):
        try:
            cs = fetch_yfinance(s, days=history_days, interval="1d", db=db, min_bars=120)
        except Exception as e:  # noqa: BLE001 -- one dead ticker must not end the sweep
            if s != MARKET:
                failed.append(s)
            else:
                raise SystemExit(f"ERROR: could not load the market series {MARKET}: {e}")
            continue
        series[s] = {dt.datetime.fromtimestamp(c.ts_open_ms / 1000, dt.timezone.utc).date():
                     c.close for c in cs}
        ohlcv[s] = {"close": np.asarray([c.close for c in cs], float),
                    "volume": np.asarray([c.volume for c in cs], float)}
        if i % PROGRESS_EVERY == 0:
            print(f"  fetched {i}/{len(symbols) + 1} ...", flush=True)
    return series, ohlcv, failed


def _build_per_name(series: dict, ohlcv: dict, names: list[str]) -> tuple[dict, list]:
    """Align every name to the market and package the arrays the classifier wants.

    Both return series must share one trading calendar, so the name's returns are taken
    from the ALIGNED closes rather than its own full history -- otherwise a name with
    extra sessions would silently regress against misaligned market days.
    """
    per_name, short = {}, []
    for n in names:
        try:
            dates, closes = align_closes({MARKET: series[MARKET], n: series[n]})
            if len(dates) < K.MIN_BARS_CLASSIFY:
                short.append((n, len(dates)))
                continue
            per_name[n] = {
                "closes": ohlcv[n]["close"],
                "volumes": ohlcv[n]["volume"],
                "name_rets": log_returns(closes[n]),
                "mkt_rets": log_return_matrix(dates, closes, [MARKET, n])[:, 0],
            }
        except Exception as e:  # noqa: BLE001
            short.append((n, f"align failed: {type(e).__name__}"))
    return per_name, short


def _row(c: K.Classification, regime: str) -> list:
    m = c.metrics
    def g(key):
        v = m.get(key)
        return "" if v is None else f"{v:.6g}"
    return [c.symbol, c.verdict, g("adv_dollar"), g("beta_63"), g("beta_252"),
            g("r2_63"), g("r2_252"),
            "" if m.get("r2_63") is None else f"{1 - m['r2_63']:.6g}",
            "" if m.get("r2_252") is None else f"{1 - m['r2_252']:.6g}",
            g("treynor"), g("treynor_pctl"),
            g("halved_126d"), g("max_dd"), g("own_sma_gap"),
            g("dilution_yoy"), g("runway_quarters"), g("revenue_ttm"),
            # The flags that earned the verdict: a three-flag reckless and a marginal
            # two read identically without them.
            "|".join(m.get("flags") or []),
            regime, "; ".join(c.reasons)]


def main() -> int:
    ap = argparse.ArgumentParser(description="High-beta investable/reckless classifier")
    ap.add_argument("--symbols", default="", help="A,B,C -- overrides the index universe")
    ap.add_argument("--csv", default="",
                    help="portfolio export (Yahoo Finance or any CSV with a Symbol/"
                         "Ticker column, or a bare list of tickers)")
    ap.add_argument("--min-adv", type=float, default=K.T_ADV_DOLLAR,
                    help="median daily dollar-volume floor")
    ap.add_argument("--min-beta", type=float, default=K.T_BETA_MIN,
                    help="252d beta cut (the 63d window must confirm it)")
    ap.add_argument("--rf", type=float, default=K.RF_DEFAULT,
                    help="flat annual risk-free rate for the Treynor ratio")
    ap.add_argument("--history-days", type=int, default=750)
    ap.add_argument("--limit", type=int, default=0, help="truncate the universe")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--no-fundamentals", action="store_true",
                    help="price-only run; the dilution/runway/revenue gates grade INFO "
                         "(much faster on a cold cache over a large universe)")
    ap.add_argument("--refresh-fundamentals", action="store_true")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    try:
        syms, provenance = _resolve_universe(a)
    except UniverseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not syms:
        print("ERROR: empty universe", file=sys.stderr)
        return 2
    dropped = 0
    if a.limit > 0 and len(syms) > a.limit:
        dropped, syms = len(syms) - a.limit, syms[:a.limit]

    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        print("=== high-beta classifier ===")
        print(K.BANNER)
        print(f"\nuniverse: {len(syms)} names from {provenance}")
        if dropped:
            print(f"  --limit dropped {dropped} names -- this is a SLICE, not a screen "
                  f"of the full universe")
        print(f"fetching {len(syms)} symbols + {MARKET} "
              f"(cached in SQLite; first run is slow) ...", flush=True)

        series, ohlcv, failed = _fetch(db, syms, a.history_days)
        names = [s for s in syms if s in series]
        if not names:
            print("ERROR: no usable names", file=sys.stderr)
            return 2
        if failed:
            head = ", ".join(failed[:12]) + (" ..." if len(failed) > 12 else "")
            print(f"  could not load {len(failed)}: {head}")

        per_name, short = _build_per_name(series, ohlcv, names)
        if short:
            head = ", ".join(f"{s}({n})" for s, n in short[:12])
            print(f"  too little shared history for {len(short)} "
                  f"(need {K.MIN_BARS_CLASSIFY} bars): {head}"
                  + (" ..." if len(short) > 12 else ""))
        if not per_name:
            print("ERROR: no name has enough shared history to classify", file=sys.stderr)
            return 2

        if a.no_fundamentals:
            print("\nfundamentals: SKIPPED (--no-fundamentals) -- the dilution, runway "
                  "and revenue gates will read 'unverified'")
        else:
            targets = sorted(per_name)
            print(f"\nfetching fundamentals for {len(targets)} names "
                  f"(cached {'refreshed' if a.refresh_fundamentals else '7d'}) ...",
                  flush=True)

            def _tick(i, n):
                if i % PROGRESS_EVERY == 0:
                    print(f"  fundamentals {i}/{n} ...", flush=True)

            funda, f_failed = fetch_fundamentals(
                targets, refresh=a.refresh_fundamentals, progress=_tick)
            for n in targets:
                per_name[n]["fundamentals"] = funda.get(n)
            if f_failed:
                head = ", ".join(f_failed[:12]) + (" ..." if len(f_failed) > 12 else "")
                print(f"  no fundamentals for {len(f_failed)}: {head} "
                      f"(their fundamentals gates read 'unverified')")

        res = K.classify_universe(per_name, ohlcv[MARKET]["close"], rf=a.rf,
                                  min_adv=a.min_adv, min_beta=a.min_beta, market=MARKET)

        regime = "risk-on" if res.risk_on else "risk-off"
        print(f"\nREGIME: {regime.upper()} -- {res.regime.display}")
        if not res.risk_on:
            print("  no name is labelled investable while the market is below its "
                  f"{K.SMA_WINDOW}d SMA; the best are 'stand aside'.")
        print(f"rank basis: {res.quartile_mode} over {res.n_eligible} eligible names "
              f"| risk-free {a.rf:.2%}")
        if res.quartile_mode == K.QUARTILE_ABSOLUTE:
            print(f"  WARNING: fewer than {K.MIN_UNIVERSE_FOR_QUARTILE} names cleared "
                  f"the hard gates, so the Treynor gate used the absolute rule "
                  f"(> 0), NOT a top-quartile rank.")

        counts = {}
        for r in res.results:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        print("\n" + "  ".join(f"{v}={counts[v]}" for v in K.VERDICT_ORDER if v in counts))

        head = (f"{'sym':>6}  {'verdict':<18}" + "".join(f"{lbl:>11}" for _, lbl in _GATES)
                + f"{'beta252':>9}{'treynor':>9}")
        print("\n" + head)
        print("-" * len(head))
        for r in res.results:
            cells = "".join(
                f"{(_MARK.get(r.by_key(k).grade, '?') if r.by_key(k) else '--  '):>11}"
                for k, _ in _GATES)
            b = r.metrics.get("beta_252")
            t = r.metrics.get("treynor")
            print(f"{r.symbol:>6}  {r.verdict:<18}{cells}"
                  f"{('' if b is None else f'{b:.2f}'):>9}"
                  f"{('' if t is None else f'{t:+.2f}'):>9}")

        print("\n--- detail ---")
        for r in res.results:
            if not r.gates:
                print(f"\n{r.symbol}  ({r.verdict})")
                for why in r.reasons:
                    print(f"          {why}")
                continue
            print(f"\n{r.symbol}  ({r.verdict})")
            for gate in r.gates:
                print(f"   [{_MARK.get(gate.grade, '?').strip():>4}] {gate.title:<26} "
                      f"{gate.display}")
                print(f"          threshold: {gate.threshold}"
                      + (f"  |  {gate.note}" if gate.note else ""))

        out = DATA_DIR / "classify.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(_CSV_COLS)
            for r in res.results:
                w.writerow(_row(r, regime))
        print(f"\nwrote {out}")
        print("\nReminder: 'investable' means the name cleared liquidity, was high-beta "
              "on both windows, was PAID for that beta in the past year, and is not "
              "dominated by its own story. It is not a prediction that it will rise.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
