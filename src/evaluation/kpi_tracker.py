"""Live KPI accounting.

Computes over two windows on every closed trade:
  * last 30 trades (the primary report; what the triggers read)
  * last 24 hours  (carried alongside for the optimizer bundle)

All formulas come from `metrics.py`, which the backtester also uses -- that shared
code is what makes `bt_deviation_pct` a real comparison instead of an artifact.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd

from src.common.db import DB
from src.common.event_bus import EventBus
from src.common.models import BacktestBaseline, ClosedTrade, KPIReport
from src.evaluation import metrics

log = logging.getLogger(__name__)

WINDOW_TRADES = 30
WINDOW_HOURS = 24
MS_PER_HOUR = 3_600_000


class KPITracker:
    def __init__(self, db: DB, bus: EventBus, initial_equity: float,
                 cfg_provider: Callable[[], dict[str, Any]],
                 baseline: BacktestBaseline | None = None) -> None:
        self.db = db
        self.bus = bus
        self.initial_equity = float(initial_equity)
        self._cfg = cfg_provider
        self.baseline = baseline
        self.last_report: KPIReport | None = None
        self.last_24h: dict[str, Any] | None = None
        bus.subscribe("trade_closed", self._on_trade_closed)

    def set_baseline(self, b: BacktestBaseline | None) -> None:
        self.baseline = b

    # -- event hook --------------------------------------------------------

    async def _on_trade_closed(self, t: ClosedTrade) -> None:
        r = self.compute(t.exit_ts_ms)
        self.db.insert_kpi_snapshot(r)
        self.last_report = r
        log.info("KPI n=%d win=%s expectancy=%s sharpe=%s maxDD=%s stopouts=%d dev=%s",
                 r.window_trades, _p(r.win_rate), _p(r.expectancy), _p(r.sharpe),
                 _p(r.max_drawdown_pct), r.consecutive_stopouts, _p(r.bt_deviation_pct))
        await self.bus.publish("kpi_report", r)

    def on_trade_closed(self, t: ClosedTrade) -> KPIReport:
        """Synchronous variant for tests and offline drills."""
        r = self.compute(t.exit_ts_ms)
        self.db.insert_kpi_snapshot(r)
        self.last_report = r
        return r

    # -- computation -------------------------------------------------------

    def compute(self, now_ms: int | None = None) -> KPIReport:
        now = now_ms if now_ms is not None else int(time.time() * 1000)

        window = self.db.get_recent_closed_trades(WINDOW_TRADES)
        s = metrics.summarize(window, self.initial_equity)

        h24 = self.db.get_closed_trades(since_ms=now - WINDOW_HOURS * MS_PER_HOUR)
        self.last_24h = metrics.summarize(h24, self.initial_equity)

        # Equity is the realized total across ALL history, not just the window.
        all_trades = self.db.get_closed_trades()
        equity = self.initial_equity + sum(t.pnl_quote for t in all_trades)

        dev = metrics.bt_deviation_pct(
            s["expectancy"], self.baseline.expectancy if self.baseline else None)

        return KPIReport(
            ts_ms=now,
            window_trades=s["n_trades"],
            win_rate=s["win_rate"],
            profit_factor=s["profit_factor"],
            expectancy=s["expectancy"],
            sharpe=s["sharpe"],
            max_drawdown_pct=s["max_drawdown_pct"],
            avg_slippage_bps=s["avg_slippage_bps"],
            consecutive_stopouts=s["consecutive_stopouts"],
            bt_deviation_pct=dev,
            equity=equity,
            config_version=int(self._cfg()["version"]),
        )

    def equity_curve(self, since_ms: int = 0) -> pd.Series:
        trades = self.db.get_closed_trades(since_ms=since_ms)
        return metrics.equity_curve_hourly(trades, self.initial_equity, since_ms or None)


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"
