"""Backtest baseline: what the CURRENT config produced in simulation.

This is the yardstick every live KPI is measured against, so it is regenerated on
every config deploy. A stale baseline is worse than no baseline -- it would have
the optimizer chasing the gap between today's live results and a config that no
longer exists.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.backtest.backtester import INITIAL_EQUITY, Backtester
from src.common.db import DB
from src.common.models import BacktestBaseline

log = logging.getLogger(__name__)

BASELINE_WINDOW_DAYS = 30
STALE_AFTER_DAYS = 7
MS_PER_DAY = 86_400_000


def generate_baseline(cfg: dict[str, Any], db: DB, *, days: int = BASELINE_WINDOW_DAYS,
                      now_ms: int | None = None, slippage_bps: float = 0.0) -> BacktestBaseline | None:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    start = now - days * MS_PER_DAY
    candles = db.get_candles(cfg["symbol"], cfg["timeframe"], start, now)
    if not candles:
        log.warning("cannot generate baseline: no candles in the last %d days", days)
        return None

    r = Backtester(cfg, candles, slippage_bps=slippage_bps).run()
    b = BacktestBaseline(
        generated_ms=now, config_version=int(cfg["version"]), window_days=days,
        n_trades=r.n_trades, expectancy=r.expectancy, sharpe=r.sharpe,
        max_drawdown_pct=r.max_drawdown_pct, win_rate=r.win_rate,
        profit_factor=r.profit_factor,
    )
    log.info("baseline v%d: %d trades, expectancy=%.4f%%, sharpe=%.2f, maxDD=%.2f%%",
             b.config_version, b.n_trades, b.expectancy, b.sharpe, b.max_drawdown_pct)
    return b


def save_baseline(b: BacktestBaseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(b.to_dict(), indent=2), encoding="utf-8")


def load_baseline(path: Path) -> BacktestBaseline | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return BacktestBaseline(**d)
    except Exception:
        log.exception("baseline file is unreadable; treating as absent")
        return None


def is_stale(b: BacktestBaseline | None, cfg_version: int, now_ms: int | None = None) -> bool:
    """Stale if missing, older than a week, or generated for a different config."""
    if b is None:
        return True
    if b.config_version != cfg_version:
        return True
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return (now - b.generated_ms) > STALE_AFTER_DAYS * MS_PER_DAY


def ensure_baseline(cfg: dict[str, Any], db: DB, path: Path,
                    now_ms: int | None = None) -> BacktestBaseline | None:
    b = load_baseline(path)
    if not is_stale(b, int(cfg["version"]), now_ms):
        return b
    log.info("baseline missing/stale/wrong-version; regenerating")
    b = generate_baseline(cfg, db, now_ms=now_ms)
    if b is not None:
        save_baseline(b, path)
    return b
