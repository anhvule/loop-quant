"""Composite VWAP/RSI/MACD score.

Deliberately a pure function of (IndicatorSnapshot, MarketState, config): the
backtester and the live engine call this exact code, so a signal that fires in
simulation fires identically in production. No clock reads, no I/O, no RNG.

Each sub-signal maps to [-1, +1] and the weighted sum stays on [-1, +1] because
weights are renormalized to sum 1 -- which is what gives `entry_threshold` a
stable meaning across configs the optimizer has retuned.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.common.config_loader import normalized_weights
from src.common.models import IndicatorSnapshot, MarketState, Signal

log = logging.getLogger(__name__)

# Below this, ATR is indistinguishable from tick noise and macd_hist/(0.25*ATR)
# explodes. Mirrors RiskManager.MIN_ATR_PCT.
_MIN_ATR_ABS = 1e-12


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class SignalEngine:
    def __init__(self, cfg_provider: Callable[[], dict[str, Any]]) -> None:
        self._cfg = cfg_provider

    # -- sub-signals (static so tests can hit them directly) ---------------

    @staticmethod
    def s_vwap(price: float, vwap: float, band_bps: float) -> float:
        """Stretched distance from session VWAP. Above VWAP = bullish (+)."""
        band = vwap * band_bps * 1e-4
        if band <= 0:
            return 0.0
        return clamp((price - vwap) / band)

    @staticmethod
    def s_rsi(rsi: float, oversold: float, overbought: float) -> float:
        """Mean-reversion component: oversold is bullish (+1), overbought bearish (-1),
        linear in between."""
        if rsi < oversold:
            return 1.0
        if rsi > overbought:
            return -1.0
        span = overbought - oversold
        if span <= 0:
            return 0.0
        return 1.0 - 2.0 * (rsi - oversold) / span

    @staticmethod
    def s_macd(macd_hist: float, atr: float) -> float:
        """Histogram normalized by volatility, so the same score means the same
        thing in a calm market and a violent one."""
        denom = 0.25 * atr
        if denom <= _MIN_ATR_ABS:
            return 0.0
        return clamp(macd_hist / denom)

    # -- composite ---------------------------------------------------------

    def compute(self, ind: IndicatorSnapshot, mkt: MarketState) -> Signal:
        cfg = self._cfg()
        st = cfg["strategy"]

        if not ind.ready:
            # A half-warm indicator set is not a weak signal, it is no signal.
            return Signal(ts_ms=ind.ts_ms, score=0.0, components={}, action="HOLD")

        price = mkt.last_price if mkt.last_price > 0 else ind.close
        if price <= 0:
            return Signal(ts_ms=ind.ts_ms, score=0.0, components={}, action="HOLD")

        w = normalized_weights(cfg)
        comp = {
            "vwap": self.s_vwap(price, float(ind.vwap), float(st["vwap_band_bps"])),
            "rsi": self.s_rsi(float(ind.rsi), float(st["rsi"]["oversold"]),
                              float(st["rsi"]["overbought"])),
            "macd": self.s_macd(float(ind.macd_hist), float(ind.atr)),
        }
        # Clamped because float error in the weight normalization can push a
        # fully-saturated score a few ulps past 1.0, and `entry_threshold` is
        # specified against a strict [-1, 1] scale.
        score = clamp(w["vwap"] * comp["vwap"] + w["rsi"] * comp["rsi"] + w["macd"] * comp["macd"])

        if score >= float(st["entry_threshold"]):
            action = "ENTER"
        elif score <= float(st["exit_threshold"]):
            action = "EXIT"
        else:
            action = "HOLD"

        return Signal(ts_ms=ind.ts_ms, score=score, components=comp, action=action)
