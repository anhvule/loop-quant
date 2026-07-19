"""Indicator math is pinned against external references and hand-computed values.

If these break, every backtest baseline in the system becomes meaningless and the
optimizer starts tuning against a lie -- so they are pinned hard, not approximately.
"""

from __future__ import annotations

import pytest

from src.common.models import Candle
from src.ingestion.indicator_engine import (
    Atr, Ema, IndicatorEngine, Macd, Rsi, SessionVwap, Wilder,
)

MS_MIN = 60_000

# Wilder's worked example from "New Concepts in Technical Trading Systems",
# as reproduced in the StockCharts RSI reference table.
WILDER_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
    46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515,
    45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205, 42.6628,
    43.1314,
]
WILDER_RSI_EXPECTED = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]


def _candle(ts_ms, o, h, l, c, v=1.0, qv=None, n=1, symbol="BTCUSDT", tf="1m"):
    return Candle(ts_open_ms=ts_ms, symbol=symbol, tf=tf, open=o, high=h, low=l, close=c,
                  volume=v, quote_volume=(c * v if qv is None else qv), n_trades=n)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_matches_wilders_published_series():
    r = Rsi(14)
    got = [v for v in (r.update(c) for c in WILDER_CLOSES) if v is not None]
    assert len(got) == len(WILDER_RSI_EXPECTED)
    for i, (g, e) in enumerate(zip(got, WILDER_RSI_EXPECTED)):
        assert g == pytest.approx(e, abs=0.005), f"RSI[{i}] {g:.4f} != published {e}"


def test_rsi_needs_period_plus_one_closes():
    r = Rsi(14)
    for c in WILDER_CLOSES[:14]:
        assert r.update(c) is None
    assert r.update(WILDER_CLOSES[14]) is not None


def test_rsi_saturates_at_100_on_pure_uptrend():
    r = Rsi(14)
    v = None
    for i in range(30):
        v = r.update(100.0 + i)
    assert v == 100.0


def test_rsi_floors_at_zero_on_pure_downtrend():
    r = Rsi(14)
    v = None
    for i in range(30):
        v = r.update(100.0 - i)
    assert v == 0.0


def test_rsi_of_flat_series_is_neutral_not_overbought():
    """A dead-flat market has zero gains AND zero losses: RS is 0/0, undefined.
    It must resolve to 50 (neutral). The naive `avg_loss == 0 -> RSI 100` branch
    would report a maximally-overbought market that has not moved at all, and the
    SignalEngine would read that as a full-strength bearish signal. Gap-filled
    zero-volume bars produce exactly this series, so it is a live code path."""
    r = Rsi(14)
    v = None
    for _ in range(30):
        v = r.update(100.0)
    assert v == 50.0


def test_rsi_flat_then_rally_is_not_stuck_at_neutral():
    r = Rsi(14)
    for _ in range(20):
        r.update(100.0)
    v = None
    for i in range(1, 15):
        v = r.update(100.0 + i)
    assert v is not None and v > 50.0


# ---------------------------------------------------------------------------
# EMA / Wilder primitives
# ---------------------------------------------------------------------------

def test_ema_is_sma_seeded():
    e = Ema(3)
    assert e.update(1.0) is None
    assert e.update(2.0) is None
    assert e.update(3.0) == pytest.approx(2.0)      # SMA(1,2,3)
    assert e.update(4.0) == pytest.approx(4.0 * 0.5 + 2.0 * 0.5)  # k = 2/(3+1) = 0.5


def test_ema_of_constant_series_is_that_constant():
    e = Ema(12)
    v = None
    for _ in range(100):
        v = e.update(7.5)
    assert v == pytest.approx(7.5)


def test_wilder_is_sma_seeded_then_rma():
    w = Wilder(3)
    assert w.update(3.0) is None
    assert w.update(6.0) is None
    assert w.update(9.0) == pytest.approx(6.0)                    # SMA(3,6,9)
    assert w.update(12.0) == pytest.approx((6.0 * 2 + 12.0) / 3)  # (v*(n-1)+x)/n


def test_wilder_of_constant_series_is_that_constant():
    w = Wilder(14)
    v = None
    for _ in range(100):
        v = w.update(3.25)
    assert v == pytest.approx(3.25)


# ---------------------------------------------------------------------------
# MACD -- incremental vs an independently written batch reference
# ---------------------------------------------------------------------------

def _ref_ema(xs: list[float], n: int) -> list[float | None]:
    """Batch EMA, structured completely differently from the incremental class."""
    out: list[float | None] = [None] * len(xs)
    if len(xs) < n:
        return out
    out[n - 1] = sum(xs[:n]) / n
    k = 2.0 / (n + 1.0)
    for i in range(n, len(xs)):
        prev = out[i - 1]
        assert prev is not None
        out[i] = xs[i] * k + prev * (1.0 - k)
    return out


def _ref_macd(xs: list[float], fast=12, slow=26, signal=9):
    ef, es = _ref_ema(xs, fast), _ref_ema(xs, slow)
    macd: list[float | None] = [
        (ef[i] - es[i]) if (ef[i] is not None and es[i] is not None) else None
        for i in range(len(xs))
    ]
    dense = [m for m in macd if m is not None]
    sig_dense = _ref_ema(dense, signal)
    sig: list[float | None] = [None] * len(xs)
    j = 0
    for i, m in enumerate(macd):
        if m is not None:
            sig[i] = sig_dense[j]
            j += 1
    hist = [
        (macd[i] - sig[i]) if (macd[i] is not None and sig[i] is not None) else None
        for i in range(len(xs))
    ]
    return macd, sig, hist


def test_macd_incremental_matches_batch_reference():
    # A deterministic non-trivial series: trend + oscillation, no RNG.
    closes = [100.0 + i * 0.3 + 5.0 * ((i % 7) - 3) / 3.0 for i in range(120)]
    m = Macd(12, 26, 9)
    got = [m.update(c) for c in closes]
    ref_macd, ref_sig, ref_hist = _ref_macd(closes, 12, 26, 9)

    for i in range(len(closes)):
        g_macd, g_sig, g_hist = got[i]
        for g, r, name in ((g_macd, ref_macd[i], "macd"),
                           (g_sig, ref_sig[i], "signal"),
                           (g_hist, ref_hist[i], "hist")):
            if r is None:
                assert g is None, f"{name}[{i}] should be undefined, got {g}"
            else:
                assert g == pytest.approx(r, rel=1e-12), f"{name}[{i}] mismatch"


def test_macd_first_values_appear_at_expected_indices():
    closes = [100.0 + i for i in range(60)]
    m = Macd(12, 26, 9)
    macd_first = sig_first = None
    for i, c in enumerate(closes):
        mv, sv, _ = m.update(c)
        if mv is not None and macd_first is None:
            macd_first = i
        if sv is not None and sig_first is None:
            sig_first = i
    assert macd_first == 25          # slow EMA seeds at index slow-1
    assert sig_first == 25 + 8       # signal EMA needs 9 macd values


def test_macd_rejects_fast_ge_slow():
    with pytest.raises(ValueError):
        Macd(26, 12, 9)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def test_atr_hand_computed():
    a = Atr(2)
    assert a.update(10.0, 8.0, 9.0) is None            # TR = H-L = 2 (no prev close)
    # TR = max(12-9, |12-9|, |9-9|) = 3 -> seed = mean(2,3) = 2.5
    assert a.update(12.0, 9.0, 11.0) == pytest.approx(2.5)
    # TR = max(11-10, |11-11|, |10-11|) = 1 -> (2.5*1 + 1)/2 = 1.75
    assert a.update(11.0, 10.0, 10.5) == pytest.approx(1.75)


def test_atr_first_bar_uses_high_minus_low():
    a = Atr(1)
    assert a.update(105.0, 100.0, 102.0) == pytest.approx(5.0)


def test_atr_of_constant_range_series_is_that_range():
    a = Atr(14)
    v = None
    for i in range(50):
        base = 100.0
        v = a.update(base + 1.0, base - 1.0, base)
    assert v == pytest.approx(2.0)


def test_atr_captures_gap_through_prev_close():
    """A gap up must widen ATR even when the bar's own H-L is narrow -- this is
    the whole reason ATR uses true range and not just the bar range."""
    a = Atr(1)
    a.update(100.0, 99.0, 99.5)
    tr = a.update(110.0, 109.5, 110.0)   # H-L = 0.5, but |H - prevC| = 10.5
    assert tr == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def test_vwap_is_volume_weighted_not_average_price():
    v = SessionVwap()
    v.add(pv=100.0 * 1.0, v=1.0, ts_ms=0)      # 1 unit at 100
    v.add(pv=200.0 * 9.0, v=9.0, ts_ms=MS_MIN)  # 9 units at 200
    # simple mean would be 150; volume weighting gives 190
    assert v.value == pytest.approx(190.0)


def test_vwap_resets_at_utc_midnight():
    v = SessionVwap()
    day0 = 0
    v.add(1000.0, 10.0, day0)                    # vwap 100
    assert v.value == pytest.approx(100.0)
    v.add(3000.0, 10.0, day0 + MS_MIN)           # cum 4000/20
    assert v.value == pytest.approx(200.0)

    day1 = 86_400_000
    v.add(500.0, 5.0, day1)                      # session reset -> 500/5
    assert v.value == pytest.approx(100.0)


def test_vwap_is_none_before_any_volume_in_session():
    v = SessionVwap()
    v.add(0.0, 0.0, 0)
    assert v.value is None


def test_vwap_unaffected_by_zero_volume_bars():
    v = SessionVwap()
    v.add(1000.0, 10.0, 0)
    before = v.value
    v.add(0.0, 0.0, MS_MIN)
    assert v.value == pytest.approx(before)


# ---------------------------------------------------------------------------
# IndicatorEngine integration
# ---------------------------------------------------------------------------

def _series(n: int, start_ms: int = 0) -> list[Candle]:
    out = []
    for i in range(n):
        base = 100.0 + i * 0.2 + 3.0 * ((i % 11) - 5) / 5.0
        out.append(_candle(start_ms + i * MS_MIN, base, base + 0.5, base - 0.5, base,
                           v=2.0, qv=base * 2.0))
    return out


def test_engine_warmup_equals_streaming():
    """warm_up() over history must land in exactly the state that streaming the
    same bars one-by-one would produce. A reconnect replays history through this
    path -- if the two diverged, indicators would silently drift after every
    websocket blip."""
    candles = _series(200)
    a = IndicatorEngine("BTCUSDT", "1m", persist=False)
    a.warm_up(candles)

    b = IndicatorEngine("BTCUSDT", "1m", persist=False)
    for c in candles:
        b.update(c)

    sa, sb = a.snapshot(), b.snapshot()
    assert sa.vwap == pytest.approx(sb.vwap)
    assert sa.rsi == pytest.approx(sb.rsi)
    assert sa.macd == pytest.approx(sb.macd)
    assert sa.macd_signal == pytest.approx(sb.macd_signal)
    assert sa.macd_hist == pytest.approx(sb.macd_hist)
    assert sa.atr == pytest.approx(sb.atr)


def test_engine_not_ready_before_min_bars():
    e = IndicatorEngine("BTCUSDT", "1m", persist=False)
    need = e.min_bars_required
    assert need == 34   # max(rsi 15, macd 26+9-1 = 34, atr 14)
    for c in _series(need - 1):
        e.update(c)
    assert not e.snapshot().ready
    e.update(_candle((need - 1) * MS_MIN, 100, 100.5, 99.5, 100, v=2.0))
    assert e.snapshot().ready


def test_engine_rejects_mismatched_candle():
    e = IndicatorEngine("BTCUSDT", "1m", persist=False)
    with pytest.raises(ValueError):
        e.update(_candle(0, 1, 1, 1, 1, symbol="ETHUSDT"))
    with pytest.raises(ValueError):
        e.update(_candle(0, 1, 1, 1, 1, tf="5m"))


def test_engine_snapshot_carries_close_and_is_labelled_by_bar_open():
    e = IndicatorEngine("BTCUSDT", "1m", persist=False)
    candles = _series(50)
    snap = None
    for c in candles:
        snap = e.update(c)
    assert snap.ts_ms == candles[-1].ts_open_ms
    assert snap.close == pytest.approx(candles[-1].close)


def test_engine_reset_on_warmup_discards_prior_state():
    e = IndicatorEngine("BTCUSDT", "1m", persist=False)
    for c in _series(100):
        e.update(c)
    assert e.snapshot().ready
    e.warm_up([])                      # fresh start
    assert e.bars_seen == 0
    assert not e.snapshot().ready
