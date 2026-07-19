"""Sweep every legal single-cycle parameter move and report which ones the
sandbox would accept.

Run this BEFORE trusting the optimizer with anything. It answers two questions
that decide whether the self-optimization loop can do useful work at all:

  1. Is there ANY in-bounds, delta-capped change that beats the incumbent? If the
     answer is zero, the optimizer will trigger, diagnose, propose, get rejected,
     and eventually hit its thrash cap -- correct behaviour, but it means the fix
     lies outside the tunable set and no amount of looping will find it.

  2. Which dials are INERT? A parameter whose every legal move produces an
     identical trade list is wired to nothing. Leaving such a dial in bounds.json
     invites the optimizer to spend cycles turning it.

    python -m scripts.analyze_config_space
    python -m scripts.analyze_config_space --days 30 --fee-bps 10
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.backtester import Backtester                       # noqa: E402
from src.common.config_loader import ConfigLoader, get_path, set_path  # noqa: E402
from src.common.db import DB                                        # noqa: E402
from src.common.paths import BOUNDS_PATH, CONFIG_PATH, DB_PATH, SCHEMA_PATH  # noqa: E402
from src.evaluation.economics import sizing_constraint, trade_economics  # noqa: E402
from src.ingestion.indicator_engine import IndicatorEngine          # noqa: E402
from src.optimizer.backtest_sandbox import (                        # noqa: E402
    EXPECTANCY_IMPROVEMENT, MAX_DD_MULTIPLE, MIN_CANDIDATE_TRADES, MIN_DD_FLOOR_PCT,
)

MS_PER_DAY = 86_400_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    loader = ConfigLoader(CONFIG_PATH, SCHEMA_PATH, BOUNDS_PATH)
    cfg = loader.load()
    bounds = loader.bounds

    db = DB(a.db or DB_PATH)
    try:
        now = int(time.time() * 1000)
        candles = db.get_candles(cfg["symbol"], cfg["timeframe"], now - a.days * MS_PER_DAY, now)
    finally:
        db.close()
    if not candles:
        print("no candles. Run: python -m scripts.seed_data --days 30")
        return 2

    print(f"{len(candles)} candles of {cfg['symbol']} {cfg['timeframe']} "
          f"over the last {a.days}d\n")

    # -- economics -----------------------------------------------------------
    ind = IndicatorEngine(cfg["symbol"], cfg["timeframe"], persist=False)
    ind.warm_up(candles)
    atr = float(ind.snapshot().atr or 0)
    price = candles[-1].close
    econ = trade_economics(cfg, atr, price, a.fee_bps)
    sizing = sizing_constraint(cfg, atr, price)

    print("=" * 78)
    print("  ECONOMICS")
    print("=" * 78)
    print(f"  ATR              : {atr:.2f} ({econ.atr_pct_of_price:.4f}% of price)")
    print(f"  stop / target    : {econ.stop_pct:.4f}% / {econ.target_pct:.4f}% of price")
    print(f"  round-trip fee   : {econ.fee_pct_round_trip:.4f}% of notional")
    print(f"  net win / loss   : {econ.net_win_pct:+.4f}% / {-econ.net_loss_pct:+.4f}%")
    be = f"{econ.breakeven_win_rate:.1%}" if econ.breakeven_win_rate is not None else "UNREACHABLE"
    print(f"  breakeven WR     : {be}")
    print(f"  viable           : {econ.viable}")
    print(f"  -> {econ.note}")
    print(f"\n  sizing binds on  : {sizing['binds']}")
    print(f"  risk dial live   : {sizing['risk_dial_is_live']}")
    if not sizing["risk_dial_is_live"]:
        print(f"  -> risk.risk_per_trade_pct is INERT. Configured "
              f"{sizing['configured_risk_per_trade_pct']}%, effective "
              f"{sizing['effective_risk_per_trade_pct']:.4f}%.")

    # -- incumbent -----------------------------------------------------------
    base = Backtester(cfg, candles, fee_bps=a.fee_bps).run()
    required = base.expectancy + EXPECTANCY_IMPROVEMENT * abs(base.expectancy)
    dd_cap = max(base.max_drawdown_pct * MAX_DD_MULTIPLE, MIN_DD_FLOOR_PCT)

    print("\n" + "=" * 78)
    print("  INCUMBENT (config v%s)" % cfg["version"])
    print("=" * 78)
    print(f"  trades={base.n_trades}  expectancy={base.expectancy:.4f}%  "
          f"win={base.win_rate:.3f}  maxDD={base.max_drawdown_pct:.2f}%  "
          f"return={base.total_return_pct:.2f}%")
    print(f"\n  To pass the sandbox a candidate needs:")
    print(f"    expectancy >= {required:.4f}%   (beat incumbent by 5% of |incumbent|)")
    print(f"    maxDD      <= {dd_cap:.2f}%")
    print(f"    trades     >= {MIN_CANDIDATE_TRADES}")

    # -- sweep ---------------------------------------------------------------
    max_delta = float(bounds["max_delta_pct_per_cycle"]) / 100.0
    cands: list[tuple[str, str, dict]] = []
    for path, lim in bounds["params"].items():
        cur = get_path(cfg, path)
        if not isinstance(cur, (int, float)):
            continue
        for mult in (1 - max_delta, 1 + max_delta):
            new = max(float(lim["min"]), min(float(lim["max"]), cur * mult))
            if abs(new - cur) < 1e-9:
                continue
            c = copy.deepcopy(cfg)
            set_path(c, path, new)
            if c["risk"]["atr_mult_tp"] < 1.2 * c["risk"]["atr_mult_sl"]:
                continue   # rr_floor invariant
            cands.append((path, f"{cur} -> {round(new, 4)}", c))

    print("\n" + "=" * 78)
    print(f"  SWEEP: {len(cands)} legal single-cycle moves (+/-{bounds['max_delta_pct_per_cycle']}%)")
    print("=" * 78)
    print(f"{'parameter':<34} {'move':<20} {'trades':>7} {'expect%':>9} {'maxDD%':>8} {'pass':>5}")
    print("-" * 86)

    passing: list[str] = []
    inert: dict[str, set[str]] = {}
    for path, move, c in cands:
        r = Backtester(c, candles, fee_bps=a.fee_bps).run()
        ok = (r.n_trades >= MIN_CANDIDATE_TRADES and r.expectancy >= required
              and r.max_drawdown_pct <= dd_cap)
        if ok:
            passing.append(f"{path} {move}")
        inert.setdefault(path, set()).add(r.trade_fingerprint)
        print(f"{path:<34} {move:<20} {r.n_trades:>7} {r.expectancy:>9.4f} "
              f"{r.max_drawdown_pct:>8.2f} {'YES' if ok else '':>5}")

    dead = [p for p, fps in inert.items() if fps == {base.trade_fingerprint}]

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  {len(passing)}/{len(cands)} legal single-cycle moves pass the sandbox gate.")
    for p in passing:
        print(f"    PASS: {p}")
    if not passing:
        print("    Nothing in the tunable set beats the incumbent. The optimizer will")
        print("    trigger, diagnose, propose, be rejected, and eventually hit its")
        print("    4-cycles-per-24h thrash cap. That is the safety framework working --")
        print("    but it means the fix is OUTSIDE the tunable set (timeframe, fee tier,")
        print("    strategy shape), not inside it.")

    if dead:
        print(f"\n  INERT DIALS ({len(dead)}): every legal move produces an identical trade list.")
        for p in sorted(dead):
            print(f"    {p}")
        print("    These are in bounds.json, so the optimizer may spend cycles turning")
        print("    knobs that are wired to nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
