"""KPI formulas, shared verbatim by the backtester (Module 4's sandbox) and the
live KPITracker (Module 3).

This sharing is not a DRY nicety -- it is a correctness requirement. Trigger T2
fires on the *deviation* between live and backtest expectancy, and T3 on the
ratio of live to baseline Sharpe. If the two sides computed those numbers with
even slightly different conventions, the optimizer would chase a measurement
artifact instead of a real regime change.

Conventions (normative):
  * pnl_pct        = pnl_quote / entry_notional * 100  (return on the trade)
  * expectancy     = mean(pnl_pct) over the window
  * profit_factor  = sum(wins) / |sum(losses)|
  * equity curve   = step function of realized PnL, resampled to 1h and ffilled
  * sharpe         = mean(hourly ret)/std(hourly ret) * sqrt(24*365), crypto is 24/7
  * max_drawdown   = max peak-to-trough of that hourly curve, as a positive %
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from src.common.models import ClosedTrade

HOURS_PER_YEAR = 24 * 365
MIN_TRADES_FOR_STATS = 5   # below this, every ratio is noise; report None instead


def equity_curve_hourly(trades: Sequence[ClosedTrade], initial_equity: float,
                        start_ms: int | None = None, end_ms: int | None = None) -> pd.Series:
    """Realized-equity step function, resampled hourly.

    Only realized PnL moves the curve; open-position mark-to-market is excluded so
    that live and backtest curves are built from the same information.
    """
    ts = sorted(trades, key=lambda t: t.exit_ts_ms)
    if not ts:
        return pd.Series(dtype="float64")

    first_ms = start_ms if start_ms is not None else ts[0].entry_ts_ms
    rows: list[tuple[int, float]] = [(min(first_ms, ts[0].exit_ts_ms), float(initial_equity))]
    eq = float(initial_equity)
    for t in ts:
        eq += t.pnl_quote
        rows.append((t.exit_ts_ms, eq))
    if end_ms is not None and end_ms > rows[-1][0]:
        rows.append((end_ms, eq))

    idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    s = pd.Series([r[1] for r in rows], index=idx, dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("1h").last().ffill()


def sharpe_from_equity(curve: pd.Series) -> float | None:
    if curve is None or len(curve) < 3:
        return None
    r = curve.pct_change().dropna()
    if len(r) < 2:
        return None
    sd = float(r.std())
    if sd == 0.0 or math.isnan(sd):
        # A perfectly flat (or single-valued) curve has no risk to divide by.
        return None
    return float(r.mean()) / sd * math.sqrt(HOURS_PER_YEAR)


def max_drawdown_pct(curve: pd.Series) -> float | None:
    if curve is None or len(curve) < 2:
        return None
    peak = curve.cummax()
    dd = (curve - peak) / peak
    worst = float(dd.min())
    return abs(worst) * 100.0


def win_rate(trades: Sequence[ClosedTrade]) -> float | None:
    if not trades:
        return None
    return sum(1 for t in trades if t.pnl_quote > 0) / len(trades)


def profit_factor(trades: Sequence[ClosedTrade]) -> float | None:
    if not trades:
        return None
    wins = sum(t.pnl_quote for t in trades if t.pnl_quote > 0)
    losses = sum(-t.pnl_quote for t in trades if t.pnl_quote < 0)
    if losses == 0:
        # No losing trade in the window. Not "infinitely good" -- undefined.
        return None
    return wins / losses


def expectancy(trades: Sequence[ClosedTrade]) -> float | None:
    if not trades:
        return None
    return sum(t.pnl_pct for t in trades) / len(trades)


def avg_slippage_bps(trades: Sequence[ClosedTrade]) -> float | None:
    if not trades:
        return None
    return sum(t.slippage_bps for t in trades) / len(trades)


def consecutive_stopouts(trades: Sequence[ClosedTrade]) -> int:
    """Run length of stop-loss exits ending at the most recent trade."""
    n = 0
    for t in reversed(sorted(trades, key=lambda x: x.exit_ts_ms)):
        if t.exit_reason == "sl":
            n += 1
        else:
            break
    return n


def bt_deviation_pct(live_expectancy: float | None, bt_expectancy: float | None) -> float | None:
    """(live - backtest)/|backtest| * 100. None when the baseline expectancy is
    ~0, where the ratio is meaningless and would explode."""
    if live_expectancy is None or bt_expectancy is None:
        return None
    if abs(bt_expectancy) < 1e-9:
        return None
    return (live_expectancy - bt_expectancy) / abs(bt_expectancy) * 100.0


def total_return_pct(trades: Sequence[ClosedTrade], initial_equity: float) -> float:
    if initial_equity <= 0:
        return 0.0
    return sum(t.pnl_quote for t in trades) / initial_equity * 100.0


def summarize(trades: Sequence[ClosedTrade], initial_equity: float,
              start_ms: int | None = None, end_ms: int | None = None) -> dict:
    """Every KPI over one window. Ratios that need a minimum sample return None
    rather than a confident-looking number computed from three trades."""
    n = len(trades)
    curve = equity_curve_hourly(trades, initial_equity, start_ms, end_ms)
    enough = n >= MIN_TRADES_FOR_STATS
    return {
        "n_trades": n,
        "win_rate": win_rate(trades) if enough else None,
        "profit_factor": profit_factor(trades) if enough else None,
        "expectancy": expectancy(trades) if enough else None,
        "sharpe": sharpe_from_equity(curve) if enough else None,
        "max_drawdown_pct": max_drawdown_pct(curve) if enough else None,
        "avg_slippage_bps": avg_slippage_bps(trades) if enough else None,
        "consecutive_stopouts": consecutive_stopouts(trades),
        "total_return_pct": total_return_pct(trades, initial_equity),
        "equity": initial_equity + sum(t.pnl_quote for t in trades),
    }
