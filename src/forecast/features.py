"""Technical read + return statistics for a candle series.

Reuses the *exact* live-engine components:
  * `IndicatorEngine.warm_up()` to fold history into VWAP/RSI/MACD/ATR, then
    `.snapshot()` for the latest values.
  * `SignalEngine.compute()` for the composite score in [-1, 1].

The strategy parameters come from `config/config.json` (the same block the live
engine uses). We do NOT run the full ConfigLoader here -- that would validate the
crypto config against schema/bounds, which is irrelevant to a read-only technical
snapshot -- we just read the strategy periods so the indicators match the signal.

Note: on daily bars VWAP resets every session (`SessionVwap` anchors to 00:00 UTC),
so `s_vwap` ~ 0. The composite score is therefore effectively RSI+MACD driven, which
is the right behaviour for a daily-bar drift tilt.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

from src.common.models import Candle, MarketState
from src.common.paths import CONFIG_PATH
from src.execution.signal_engine import SignalEngine
from src.ingestion.indicator_engine import IndicatorEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Features:
    last_close: float
    n_bars: int
    # Latest indicator snapshot (may be None if not warm -- guarded upstream).
    rsi: float | None
    macd_hist: float | None
    atr: float | None
    vwap: float | None
    # Composite VWAP/RSI/MACD score in [-1, 1]; +1 fully bullish, -1 fully bearish.
    signal_score: float
    signal_action: str
    # Daily log-return statistics over the supplied history.
    mu: float          # mean daily log return (drift)
    sigma: float       # stdev of daily log return (volatility), ddof=1


def _load_strategy_cfg(path=CONFIG_PATH) -> dict:
    """Read config.json for strategy params. Falls back to sane defaults if the
    file is absent, so the forecaster is not coupled to the crypto config existing."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "strategy": {
                "weights": {"vwap": 0.35, "rsi": 0.30, "macd": 0.35},
                "entry_threshold": 0.5, "exit_threshold": -0.3,
                "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "vwap_band_bps": 15,
            },
            "risk": {"atr_period": 14},
        }


def log_return_stats(closes: list[float]) -> tuple[float, float]:
    """(mean, stdev) of consecutive daily log returns. stdev uses ddof=1.

    Returns (0.0, 0.0) for a series too short to have >= 2 returns, and drops any
    non-positive/degenerate step (a corrupt 0 close would blow up the log)."""
    rets: list[float] = []
    for a, b in zip(closes, closes[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < 2:
        return 0.0, 0.0
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return mu, math.sqrt(var)


def compute_features(candles: list[Candle], cfg: dict | None = None) -> Features:
    """Warm indicators, read the composite signal, and compute return stats."""
    if not candles:
        raise ValueError("compute_features requires at least one candle")
    cfg = cfg or _load_strategy_cfg()
    st = cfg["strategy"]
    symbol, tf = candles[0].symbol, candles[0].tf

    engine = IndicatorEngine(
        symbol, tf,
        rsi_period=int(st["rsi"]["period"]),
        macd_fast=int(st["macd"]["fast"]),
        macd_slow=int(st["macd"]["slow"]),
        macd_signal=int(st["macd"]["signal"]),
        atr_period=int(cfg.get("risk", {}).get("atr_period", 14)),
        db=None, persist=False,
    )
    engine.warm_up(candles)
    snap = engine.snapshot()

    last_close = candles[-1].close
    mkt = MarketState(symbol=symbol, last_price=last_close)
    signal = SignalEngine(lambda: cfg).compute(snap, mkt)

    mu, sigma = log_return_stats([c.close for c in candles])
    return Features(
        last_close=last_close, n_bars=len(candles),
        rsi=snap.rsi, macd_hist=snap.macd_hist, atr=snap.atr, vwap=snap.vwap,
        signal_score=signal.score, signal_action=signal.action,
        mu=mu, sigma=sigma,
    )
