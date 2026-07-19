"""Incremental VWAP / RSI / MACD / ATR.

Every update is O(1) -- no window re-scans, no pandas in the hot path. The math
is pinned to explicit conventions rather than delegated to a TA library, because
the backtester and the live engine MUST agree bit-for-bit; a library version bump
that silently changes an EMA seeding rule would invalidate every baseline the
optimizer measures against.

Conventions (normative):
  * EMA        seeded with the SMA of the first `period` values (TA-Lib convention),
               then k = 2/(period+1).
  * Wilder/RMA seeded with the SMA of the first `period` values, then
               v = (v_prev*(period-1) + x)/period.
  * RSI(14)    Wilder smoothing of gains/losses; first value after 15 closes.
  * MACD(12,26,9)  macd = ema12 - ema26; signal = ema9(macd); hist = macd - signal.
  * ATR(14)    Wilder smoothing of TR; the first bar's TR is (high - low) because
               it has no previous close.
  * VWAP       session-anchored, reset at 00:00 UTC, sum(price*qty)/sum(qty).
               Fed from the candle's `quote_volume`, which the aggregator computes
               as sum(price*qty) over that bar's RAW TRADES -- so this is a true
               raw-trade VWAP, never a typical-price approximation. Accumulating
               per-bar rather than per-tick is what makes live == backtest exactly.
"""

from __future__ import annotations

import logging
from typing import Iterable

from src.common.models import Candle, IndicatorSnapshot, tf_to_ms

log = logging.getLogger(__name__)

MS_PER_DAY = 86_400_000


class Ema:
    """Exponential moving average, SMA-seeded."""

    __slots__ = ("period", "k", "_seed", "value")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self.period = period
        self.k = 2.0 / (period + 1.0)
        self._seed: list[float] = []
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) >= self.period:
                self.value = sum(self._seed) / len(self._seed)
            return self.value
        self.value = x * self.k + self.value * (1.0 - self.k)
        return self.value


class Wilder:
    """Wilder's smoothing (a.k.a. RMA), SMA-seeded."""

    __slots__ = ("period", "_seed", "value")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("Wilder period must be >= 1")
        self.period = period
        self._seed: list[float] = []
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) >= self.period:
                self.value = sum(self._seed) / len(self._seed)
            return self.value
        self.value = (self.value * (self.period - 1) + x) / self.period
        return self.value


class Rsi:
    __slots__ = ("period", "_gain", "_loss", "_prev_close", "value")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._gain = Wilder(period)
        self._loss = Wilder(period)
        self._prev_close: float | None = None
        self.value: float | None = None

    def update(self, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None
        delta = close - self._prev_close
        self._prev_close = close
        g = self._gain.update(max(delta, 0.0))
        l = self._loss.update(max(-delta, 0.0))
        if g is None or l is None:
            self.value = None
            return None
        if g == 0.0 and l == 0.0:
            # A dead-flat window: RS is 0/0, genuinely undefined. Resolve to
            # neutral, NOT to 100. Gap-filled zero-volume bars produce exactly
            # this shape, and the naive `loss == 0 -> 100` branch would turn an
            # idle market into a maximum-strength overbought reading.
            self.value = 50.0
        elif l == 0.0:
            # Only gains in the window: RS is infinite, RSI saturates at 100.
            self.value = 100.0
        elif g == 0.0:
            self.value = 0.0
        else:
            rs = g / l
            self.value = 100.0 - 100.0 / (1.0 + rs)
        return self.value


class Macd:
    __slots__ = ("_fast", "_slow", "_signal", "macd", "signal", "hist")

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("MACD fast period must be < slow period")
        self._fast = Ema(fast)
        self._slow = Ema(slow)
        self._signal = Ema(signal)
        self.macd: float | None = None
        self.signal: float | None = None
        self.hist: float | None = None

    def update(self, close: float) -> tuple[float | None, float | None, float | None]:
        f = self._fast.update(close)
        s = self._slow.update(close)
        if f is None or s is None:
            return (None, None, None)
        self.macd = f - s
        self.signal = self._signal.update(self.macd)
        self.hist = None if self.signal is None else self.macd - self.signal
        return (self.macd, self.signal, self.hist)


class Atr:
    __slots__ = ("_w", "_prev_close", "value")

    def __init__(self, period: int = 14) -> None:
        self._w = Wilder(period)
        self._prev_close: float | None = None
        self.value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        self.value = self._w.update(tr)
        return self.value


class SessionVwap:
    """Cumulative sum(price*qty)/sum(qty), reset at each 00:00 UTC boundary."""

    __slots__ = ("cum_pv", "cum_v", "session_day")

    def __init__(self) -> None:
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.session_day: int | None = None

    def add(self, pv: float, v: float, ts_ms: int) -> float | None:
        day = ts_ms // MS_PER_DAY
        if self.session_day is None or day != self.session_day:
            self.session_day = day
            self.cum_pv = 0.0
            self.cum_v = 0.0
        self.cum_pv += pv
        self.cum_v += v
        return self.value

    @property
    def value(self) -> float | None:
        # A session that has opened with only zero-volume bars has no VWAP yet.
        # Returning None (rather than a stale cross-session value) is what keeps
        # the engine from trading against a price level from yesterday.
        return None if self.cum_v <= 0 else self.cum_pv / self.cum_v


class IndicatorEngine:
    """Owns the indicator state for one (symbol, timeframe)."""

    def __init__(self, symbol: str, tf: str = "1m", *, rsi_period: int = 14,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 atr_period: int = 14, db=None, persist: bool = True) -> None:
        self.symbol = symbol
        self.tf = tf
        self.tf_ms = tf_to_ms(tf)
        self._params = dict(rsi_period=rsi_period, macd_fast=macd_fast, macd_slow=macd_slow,
                            macd_signal=macd_signal, atr_period=atr_period)
        self.db = db
        self.persist = persist
        self.bars_seen = 0
        self._last: IndicatorSnapshot | None = None
        self._reset()

    def _reset(self) -> None:
        p = self._params
        self.vwap = SessionVwap()
        self.rsi = Rsi(p["rsi_period"])
        self.macd = Macd(p["macd_fast"], p["macd_slow"], p["macd_signal"])
        self.atr = Atr(p["atr_period"])
        self.bars_seen = 0
        self._last = None

    @property
    def min_bars_required(self) -> int:
        """Bars needed before every indicator has a value (the warm-up floor)."""
        p = self._params
        rsi_ready = p["rsi_period"] + 1
        macd_ready = p["macd_slow"] + p["macd_signal"] - 1
        atr_ready = p["atr_period"]
        return max(rsi_ready, macd_ready, atr_ready)

    def warm_up(self, candles: Iterable[Candle]) -> None:
        """Replay history into a fresh state. Warm-up bars are not persisted --
        they are already in the `candles` table and re-deriving them on every
        restart would just rewrite identical rows."""
        self._reset()
        n = 0
        for c in candles:
            self._ingest(c)
            n += 1
        log.info("indicator warm-up: %d bars, ready=%s (need %d)",
                 n, self.snapshot().ready, self.min_bars_required)

    def update(self, c: Candle) -> IndicatorSnapshot:
        """Fold one closed candle into the state and return the new snapshot."""
        snap = self._ingest(c)
        if self.persist and self.db is not None:
            self.db.insert_indicator(snap)
        return snap

    def _ingest(self, c: Candle) -> IndicatorSnapshot:
        if c.symbol != self.symbol or c.tf != self.tf:
            raise ValueError(f"candle {c.symbol}/{c.tf} does not match engine "
                             f"{self.symbol}/{self.tf}")
        self.vwap.add(c.quote_volume, c.volume, c.ts_open_ms)
        self.rsi.update(c.close)
        self.macd.update(c.close)
        self.atr.update(c.high, c.low, c.close)
        self.bars_seen += 1
        self._last = IndicatorSnapshot(
            ts_ms=c.ts_open_ms,      # snapshots are labelled by the bar they close
            symbol=self.symbol, tf=self.tf,
            vwap=self.vwap.value,
            rsi=self.rsi.value,
            macd=self.macd.macd,
            macd_signal=self.macd.signal,
            macd_hist=self.macd.hist,
            atr=self.atr.value,
            close=c.close,
        )
        return self._last

    def snapshot(self) -> IndicatorSnapshot:
        if self._last is None:
            return IndicatorSnapshot(0, self.symbol, self.tf, None, None, None, None, None, None)
        return self._last
