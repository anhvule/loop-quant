"""Regression panel for the classifier's verdicts, run against LIVE data.

Why this exists: gate thresholds are easy to tune until the tool agrees with whoever is
looking at it. That is the failure mode -- a rule changed to rescue one name quietly
breaks the names nobody re-checked. This panel fixes the expectations up front, so any
rule change is judged on the whole set rather than the case that prompted it.

    python -m scripts.validate_verdicts
    python -m scripts.validate_verdicts --no-fundamentals

It is deliberately NOT a pytest: it needs the network and live prices, so its result
moves with the market. Treat a failure as a prompt to look, not as proof of a bug --
but never edit an expectation without a stated reason.

EXPECTATIONS ASSUME THE FULL-EVIDENCE RUN. With `--no-fundamentals` only two gates can
raise a flag (idio and survivable), so `reckless` -- which needs RECKLESS_MIN_FLAGS of
them -- effectively demands unanimity and SPCE reads `mixed` on one flag. That is the
rule behaving as designed with less evidence, not a regression; it is also why the web
app, which is price-only, is structurally more reluctant to condemn than the CLI.

Expectations are grouped by WHY they are expected, because the reasons age differently:

  structural   the name's profile would have to change materially for this to move
  contested    verdicts we have argued about; recorded so the argument is not re-run
               from memory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                       # noqa: E402
from src.common.paths import DB_PATH, ensure_dirs                  # noqa: E402
from src.forecast import classify as K                             # noqa: E402

# (symbol, expected verdict, why)
PANEL = [
    # --- structural: chronic-collapse / pre-revenue profiles ---
    ("SPCE", K.RECKLESS, "structural: ~30% of 126d windows halve, +195% dilution, "
                         "$1.3M revenue -- earns neither fundamentals cap"),
    ("BTAI", K.EXCLUDED_ILLIQUID, "structural: sub-$10M median daily dollar volume"),
    ("AKBA", K.EXCLUDED_ILLIQUID, "structural: sub-$10M median daily dollar volume"),

    # --- structural: genuine market amplifiers that were paid ---
    ("NVDA", K.INVESTABLE, "structural: buybacks, self-funding, rare halvings"),

    # --- contested: verdicts we have argued about ---
    ("SOFI", K.MIXED, "contested: unpaid beta and +16% issuance, but $3.9B "
                           "revenue -- capital formation, not a survival treadmill"),
    ("CRWV", K.MIXED, "contested: 16 months listed, so its share count still "
                           "reflects the IPO; real revenue behind the cash burn"),
    ("COIN", None, "contested: the 'high on either window' idio rule was written from "
                   "COIN alone. Whatever the corroborated rule now says is the answer "
                   "-- recorded, not asserted"),
]

MARKET = "SPY"


def main() -> int:
    ap = argparse.ArgumentParser(description="Classifier verdict regression panel")
    ap.add_argument("--no-fundamentals", action="store_true")
    ap.add_argument("--rf", type=float, default=K.RF_DEFAULT)
    a = ap.parse_args()
    if a.no_fundamentals:
        print("NOTE: --no-fundamentals leaves only two flag-capable gates, so verdicts "
              "are legitimately more cautious than these expectations assume.\n")

    # Reuse the CLI's own plumbing so the panel measures the real pipeline.
    from scripts.classify import _build_per_name, _fetch                # noqa: PLC0415
    from src.forecast.fundamentals import fetch_fundamentals            # noqa: PLC0415

    ensure_dirs()
    db = DB(DB_PATH)
    try:
        syms = [s for s, _, _ in PANEL]
        print(f"fetching {len(syms)} panel names + {MARKET} ...", flush=True)
        series, ohlcv, failed = _fetch(db, syms, 750)
        names = [s for s in syms if s in series]
        per_name, short = _build_per_name(series, ohlcv, names)
        if not a.no_fundamentals and per_name:
            print("fetching fundamentals ...", flush=True)
            funda, _f = fetch_fundamentals(sorted(per_name))
            for n in per_name:
                per_name[n]["fundamentals"] = funda.get(n)

        res = K.classify_universe(per_name, ohlcv[MARKET]["close"], rf=a.rf,
                                  market=MARKET)
        got = {r.symbol: r for r in res.results}

        print(f"\nregime: {'risk-on' if res.risk_on else 'risk-off'} | "
              f"{res.quartile_mode} basis over {res.n_eligible} eligible\n")
        head = f"{'sym':>6}  {'expected':<18} {'actual':<18} {'flags':<24} why"
        print(head)
        print("-" * 110)

        mismatches = 0
        for sym, want, why in PANEL:
            r = got.get(sym)
            actual = r.verdict if r else ("NOT FETCHED" if sym in failed else "missing")
            flags = ",".join((r.metrics.get("flags") or [])) if r else ""
            if want is None:
                mark = "note"
            elif actual == want:
                mark = "ok"
            else:
                mark = "MISMATCH"
                mismatches += 1
            print(f"{sym:>6}  {str(want or '(recorded)'):<18} {actual:<18} "
                  f"{flags:<24} [{mark}] {why}")

        if short:
            print("\ntoo little shared history: "
                  + ", ".join(f"{s}({n})" for s, n in short))
        print(f"\n{len(PANEL) - mismatches - sum(1 for _, w, _ in PANEL if w is None)}"
              f" of {sum(1 for _, w, _ in PANEL if w is not None)} expectations met"
              + (f", {mismatches} MISMATCH" if mismatches else ""))
        if mismatches:
            print("\nA mismatch is a prompt to look, not proof of a bug -- live prices "
                  "move. Do not edit an expectation without saying why in this file.")
        return 1 if mismatches else 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
