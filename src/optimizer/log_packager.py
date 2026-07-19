"""Builds the underperformance bundle: facts, no interpretation.

This module deliberately does NOT diagnose. It assembles the evidence and hands
it over; the reasoning happens in Phase A. Mixing the two would mean the bundle's
author has already decided the answer and is selecting evidence for it.

Two inclusions carry most of the weight:

  * `config_history` -- the last 3 versions AND the KPIs each actually produced.
    Without it the optimizer cannot tell that it already tried widening the stop
    two cycles ago and it did not help, so it proposes it again forever.
  * `economics` -- the deterministic breakeven arithmetic. Phase A is asked to
    reason about breakeven win rates; giving it the computed numbers means the
    diagnosis rests on arithmetic instead of the model's mental math.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import asdict
from typing import Any

from src.common.db import DB
from src.common.models import UnderperformanceEvent
from src.evaluation.economics import sizing_constraint, trade_economics

log = logging.getLogger(__name__)

MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000
RECENT_TRADES = 50
CONFIG_HISTORY = 3


class LogPackager:
    def __init__(self, db: DB, cfg_provider, bounds: dict[str, Any]) -> None:
        self.db = db
        self._cfg = cfg_provider
        self.bounds = bounds

    def build_bundle(self, ev: UnderperformanceEvent) -> dict[str, Any]:
        cfg = self._cfg()
        now = ev.fired_at_ms or int(time.time() * 1000)

        return {
            "trigger": {
                "id": ev.trigger_id,
                "scope": ev.scope,
                "condition": ev.condition_text,
                "halted_trading": ev.halt_trading,
                "fired_at_ms": ev.fired_at_ms,
            },
            "kpi_report": ev.kpi_report.to_dict(),
            "baseline_kpis": ev.baseline.to_dict() if ev.baseline else None,
            "current_config": cfg,
            "config_history": self._config_history(),
            "recent_trades": self._recent_trades(),
            "market_regime": self._market_regime(cfg, now),
            "economics": self._economics(cfg, now),
            "bounds": self.bounds,
        }

    # -- sections ----------------------------------------------------------

    def _config_history(self) -> list[dict[str, Any]]:
        """What we already tried, and what it produced. The antidote to the
        optimizer re-proposing a change that has already failed."""
        outcomes = self.db.kpi_by_config_version()
        out = []
        for row in self.db.get_config_versions(CONFIG_HISTORY):
            v = int(row["version"])
            try:
                cfg = json.loads(row["config_json"])
            except Exception:
                cfg = None
            out.append({
                "version": v,
                "deployed_ms": row["deployed_ms"],
                "source": row["source"],
                "rolled_back_ms": row["rolled_back_ms"],
                "config": cfg,
                "realized_outcome": outcomes.get(v, {"n_trades": 0}),
            })
        return out

    def _recent_trades(self) -> list[dict[str, Any]]:
        return [
            {
                "trade_id": t.trade_id,
                "entry_ts_ms": t.entry_ts_ms, "exit_ts_ms": t.exit_ts_ms,
                "held_sec": (t.exit_ts_ms - t.entry_ts_ms) / 1000.0,
                "entry_px": round(t.entry_px, 2), "exit_px": round(t.exit_px, 2),
                "pnl_pct": round(t.pnl_pct, 4),
                "exit_reason": t.exit_reason,
                "slippage_bps": round(t.slippage_bps, 2),
                "atr_at_entry": round(t.atr_at_entry, 4),
                "atr_pct_of_entry": round(t.atr_at_entry / t.entry_px * 100, 4) if t.entry_px else None,
                "config_version": t.config_version,
            }
            for t in self.db.get_recent_closed_trades(RECENT_TRADES)
        ]

    def _market_regime(self, cfg: dict[str, Any], now: int) -> dict[str, Any]:
        symbol, tf = cfg["symbol"], cfg["timeframe"]

        c24 = self.db.get_candles(symbol, tf, now - MS_PER_DAY, now)
        c7d = self.db.get_candles(symbol, tf, now - 7 * MS_PER_DAY, now)

        vol_24h = _realized_vol_daily(c24)
        vol_7d = _realized_vol_daily(c7d)

        pairs = self.db.indicator_close_pairs(symbol, tf, now - MS_PER_DAY, now)
        above = sum(1 for vwap, close, _ in pairs if close > vwap)
        pct_above = (above / len(pairs) * 100.0) if pairs else None
        atr_ratio = (statistics.fmean([atr / close for _, close, atr in pairs if close > 0])
                     * 100.0) if pairs else None

        ret_7d = None
        if len(c7d) >= 2 and c7d[0].close > 0:
            ret_7d = (c7d[-1].close - c7d[0].close) / c7d[0].close * 100.0

        # Trend proxy: how much of the move is directional vs. noise. Near 0 = a
        # chop regime (mean reversion should work); large = a trending regime
        # (mean reversion gets run over).
        trend_proxy = None
        if ret_7d is not None and vol_7d and vol_7d > 0:
            trend_proxy = abs(ret_7d) / (vol_7d * math.sqrt(7))

        return {
            "n_candles_24h": len(c24),
            "n_candles_7d": len(c7d),
            "realized_vol_24h_pct": _r(vol_24h),
            "realized_vol_7d_avg_pct": _r(vol_7d),
            "vol_ratio_24h_vs_7d": _r(vol_24h / vol_7d) if (vol_24h and vol_7d) else None,
            "pct_time_above_vwap_24h": _r(pct_above),
            "pct_time_below_vwap_24h": _r(100.0 - pct_above) if pct_above is not None else None,
            "avg_atr_pct_of_price_24h": _r(atr_ratio),
            "avg_spread_bps_24h": _r(self.db.avg_spread_bps(symbol, now - MS_PER_DAY)),
            "return_7d_pct": _r(ret_7d),
            "trend_proxy": _r(trend_proxy),
            "trend_proxy_note": ("|7d return| / (7d daily vol * sqrt(7)). "
                                 "~<1 = chop/mean-reverting; >>1 = strong trend."),
        }

    def _economics(self, cfg: dict[str, Any], now: int) -> dict[str, Any]:
        """Deterministic breakeven arithmetic at the CURRENT volatility."""
        candles = self.db.get_candles(cfg["symbol"], cfg["timeframe"], now - MS_PER_DAY, now)
        pairs = self.db.indicator_close_pairs(cfg["symbol"], cfg["timeframe"],
                                              now - MS_PER_DAY, now)
        if not pairs:
            return {"available": False,
                    "note": "no indicator history in the last 24h to price the economics"}

        atr = statistics.fmean([a for _, _, a in pairs])
        price = statistics.fmean([c for _, c, _ in pairs])
        econ = trade_economics(cfg, atr, price, fee_bps_per_side=10.0)
        sizing = sizing_constraint(cfg, atr, price)
        return {
            "available": True,
            "assumed_atr": round(atr, 6),
            "assumed_price": round(price, 2),
            "assumed_fee_bps_per_side": 10.0,
            "trade_economics": econ.to_dict(),
            "sizing_constraint": sizing,
            "note": ("Fees are charged on notional; stops are measured in ATR. If "
                     "breakeven_win_rate is unreachable, no weight/threshold tuning "
                     "can fix it -- only wider stops, a longer timeframe, or lower fees. "
                     "If sizing_constraint.risk_dial_is_live is false, "
                     "risk.risk_per_trade_pct has NO effect on results."),
        }


# ---------------------------------------------------------------------------

def _realized_vol_daily(candles: list) -> float | None:
    """Std of 1m log returns, scaled to a daily figure, in percent."""
    if len(candles) < 30:
        return None
    rets = []
    for i in range(1, len(candles)):
        p0, p1 = candles[i - 1].close, candles[i].close
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    if len(rets) < 20:
        return None
    sd = statistics.pstdev(rets)
    return sd * math.sqrt(1440) * 100.0   # 1440 one-minute bars per day


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)
