"""The aggregator is the boundary where an unreliable network feed becomes the
clean, gapless bar series the indicators assume. These tests cover that contract.
"""

from __future__ import annotations

import pytest

from src.common.models import Trade
from src.ingestion.candle_aggregator import CandleAggregator

MIN = 60_000


def _t(ts_ms: int, price: float, qty: float = 1.0, tid: int = 0, symbol="BTCUSDT") -> Trade:
    return Trade(ts_ms=ts_ms, symbol=symbol, price=price, qty=qty,
                 is_buyer_maker=False, agg_trade_id=tid)


def test_bar_closes_on_minute_boundary_with_correct_ohlcv():
    a = CandleAggregator("BTCUSDT", "1m")
    assert a.on_trade(_t(0, 100.0, 1.0, 1)) == []
    assert a.on_trade(_t(10_000, 105.0, 2.0, 2)) == []
    assert a.on_trade(_t(20_000, 95.0, 3.0, 3)) == []
    assert a.on_trade(_t(50_000, 102.0, 4.0, 4)) == []

    out = a.on_trade(_t(MIN + 1_000, 103.0, 1.0, 5))
    assert len(out) == 1
    c = out[0]
    assert c.ts_open_ms == 0
    assert (c.open, c.high, c.low, c.close) == (100.0, 105.0, 95.0, 102.0)
    assert c.volume == pytest.approx(10.0)
    assert c.n_trades == 4


def test_quote_volume_is_sum_of_price_times_qty():
    """This is the VWAP numerator -- it must be sum(p*q), never close*volume."""
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 1.0, 1))
    a.on_trade(_t(1_000, 200.0, 3.0, 2))
    c = a.on_trade(_t(MIN, 150.0, 1.0, 3))[0]
    assert c.quote_volume == pytest.approx(100.0 * 1.0 + 200.0 * 3.0)
    assert c.quote_volume != pytest.approx(c.close * c.volume)  # would be the wrong formula
    assert c.quote_volume / c.volume == pytest.approx(175.0)     # the bar's true VWAP


def test_late_tick_is_dropped_not_backfilled():
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 1.0, 1))
    a.on_trade(_t(MIN + 1_000, 110.0, 1.0, 2))   # closes bar 0, opens bar 1
    assert a.dropped_late == 0

    out = a.on_trade(_t(30_000, 999.0, 5.0, 3))  # belongs to the closed bar
    assert out == []
    assert a.dropped_late == 1
    # the reopened bar must be untouched by the late tick
    closed = a.on_trade(_t(2 * MIN, 111.0, 1.0, 4))[0]
    assert closed.high == 110.0
    assert closed.volume == pytest.approx(1.0)


def test_gap_is_filled_with_flat_zero_volume_bars():
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 2.0, 1))
    out = a.on_trade(_t(3 * MIN + 5_000, 120.0, 1.0, 2))

    assert len(out) == 3                      # bar 0 (real) + bars 1,2 (flat)
    assert out[0].ts_open_ms == 0
    assert out[0].close == 100.0

    for i, c in enumerate(out[1:], start=1):
        assert c.ts_open_ms == i * MIN
        assert c.open == c.high == c.low == c.close == 100.0
        assert c.volume == 0.0
        assert c.quote_volume == 0.0
        assert c.n_trades == 0

    # bars are contiguous with no holes
    assert [c.ts_open_ms for c in out] == [0, MIN, 2 * MIN]


def test_on_clock_closes_idle_bars():
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 1.0, 1))
    assert a.on_clock(30_000) == []            # still inside bar 0
    out = a.on_clock(MIN + 1)
    assert len(out) == 1 and out[0].ts_open_ms == 0
    assert a.current_bucket == MIN


def test_on_clock_before_any_trade_is_noop():
    a = CandleAggregator("BTCUSDT", "1m")
    assert a.on_clock(10 * MIN) == []


def test_clock_and_trade_produce_the_same_bar_sequence():
    """Whether a bar is closed by the clock or by the next trade must not change
    the emitted series -- otherwise indicator state depends on feed timing."""
    trades = [_t(i * 7_000, 100.0 + (i % 5), 1.0, i) for i in range(40)]

    a = CandleAggregator("BTCUSDT", "1m")
    by_trade = []
    for t in trades:
        by_trade.extend(a.on_trade(t))

    b = CandleAggregator("BTCUSDT", "1m")
    by_clock = []
    for t in trades:
        by_clock.extend(b.on_clock(t.ts_ms))
        by_clock.extend(b.on_trade(t))

    assert [c.ts_open_ms for c in by_trade] == [c.ts_open_ms for c in by_clock]
    assert [c.close for c in by_trade] == [c.close for c in by_clock]
    assert [c.volume for c in by_trade] == [c.volume for c in by_clock]


def test_trade_after_clock_seeded_bar_defines_the_open():
    """A bar the clock opened carries prev_close as a placeholder; the first real
    trade must claim the open, or every post-idle bar reports a phantom wick."""
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 1.0, 1))
    a.on_clock(MIN + 1)                       # closes bar 0, opens bar 1 at 100.0
    a.on_trade(_t(MIN + 30_000, 130.0, 1.0, 2))
    c = a.on_clock(2 * MIN + 1)[0]
    assert c.open == 130.0
    assert c.high == 130.0
    assert c.low == 130.0


def test_flush_emits_partial_bar():
    a = CandleAggregator("BTCUSDT", "1m")
    a.on_trade(_t(0, 100.0, 1.0, 1))
    out = a.flush()
    assert len(out) == 1 and out[0].n_trades == 1
    assert a.flush() == []


def test_wrong_symbol_rejected():
    a = CandleAggregator("BTCUSDT", "1m")
    with pytest.raises(ValueError):
        a.on_trade(_t(0, 1.0, 1.0, 1, symbol="ETHUSDT"))
