"""Backtester correctness.

The optimizer accepts or rejects a config on ~5% expectancy differences measured
here. That makes two things load-bearing:
  * determinism -- a jittery backtest turns the accept/reject gate into a coin flip
  * no lookahead -- a backtest that peeks is a machine for generating configs that
    work only in simulation, which is exactly the failure mode a self-optimizing
    system amplifies.
"""

from __future__ import annotations

import copy
import math

import pytest

from src.backtest.backtester import Backtester, run_backtest
from src.common.models import Candle, Position
from src.execution.exchange_adapter import SymbolFilters

MIN = 60_000
SYMBOL = "BTCUSDT"


def _c(i, o, h, l, c, v=10.0, qv=None):
    return Candle(ts_open_ms=i * MIN, symbol=SYMBOL, tf="1m", open=o, high=h, low=l, close=c,
                  volume=v, quote_volume=(qv if qv is not None else c * v), n_trades=10)


def oscillating(n: int, base: float = 1_000.0, amp_pct: float = 3.0,
                period: int = 24) -> list[Candle]:
    """A deterministic series with enough volatility to clear MIN_ATR_PCT and
    produce real entries/exits. No RNG anywhere -- determinism starts here."""
    out = []
    prev = base
    for i in range(n):
        mid = base * (1.0 + amp_pct / 100.0 * math.sin(2 * math.pi * i / period))
        o = prev
        c = mid
        h = max(o, c) * 1.002
        l = min(o, c) * 0.998
        out.append(_c(i, o, h, l, c))
        prev = c
    return out


def _pos(entry=1000.0, stop=990.0, tp=1020.0, qty=1.0):
    return Position(symbol=SYMBOL, state="OPEN", side="BUY", qty=qty, entry_px=entry,
                    entry_ts_ms=0, stop_px=stop, tp_px=tp, atr_at_entry=10.0,
                    config_version=1, trade_id="t-0")


# -- determinism --------------------------------------------------------------

def test_backtester_is_deterministic(cfg):
    candles = oscillating(600)
    a = Backtester(cfg, candles).run()
    b = Backtester(cfg, candles).run()
    assert a.trade_fingerprint == b.trade_fingerprint
    assert a.n_trades == b.n_trades
    assert a.expectancy == b.expectancy
    assert a.sharpe == b.sharpe
    assert a.max_drawdown_pct == b.max_drawdown_pct


def test_backtester_produces_trades_on_a_volatile_series(cfg):
    """If this stops producing trades the other tests silently pass on empty sets."""
    r = Backtester(cfg, oscillating(600)).run()
    assert r.n_trades > 0


def test_identical_config_reruns_are_byte_identical_json(cfg):
    candles = oscillating(300)
    import json
    a = json.dumps(Backtester(cfg, candles).run().to_dict(), sort_keys=True)
    b = json.dumps(Backtester(cfg, candles).run().to_dict(), sort_keys=True)
    assert a == b


def test_different_config_gives_different_result(cfg):
    """Guards against the gate silently comparing a config to itself."""
    candles = oscillating(600)
    base = Backtester(cfg, candles).run()
    tweaked = copy.deepcopy(cfg)
    tweaked["strategy"]["entry_threshold"] = 0.85
    other = Backtester(tweaked, candles).run()
    assert base.trade_fingerprint != other.trade_fingerprint


# -- lookahead ---------------------------------------------------------------

def test_no_lookahead_entry_bar_range_cannot_close_the_position(cfg):
    """A position opened at bar i's CLOSE must not be closed by bar i's own
    high/low -- that range was not knowable at the moment of entry."""
    c = copy.deepcopy(cfg)
    c["strategy"]["entry_threshold"] = 0.2      # trade often
    bt = Backtester(c, oscillating(400))
    r = bt.run()
    assert r.n_trades > 0
    for t in bt.trades:
        assert t.exit_ts_ms > t.entry_ts_ms, "a trade exited on its own entry bar"
        # entry and exit are at least one full bar apart
        assert t.exit_ts_ms - t.entry_ts_ms >= MIN


def test_open_position_at_end_of_window_is_not_counted(cfg):
    """Marking an open position to the last close would book a result the live
    system never realized, and inflate the baseline the optimizer measures against."""
    candles = oscillating(600)
    bt = Backtester(cfg, candles)
    bt.run()
    last_close_ts = candles[-1].ts_open_ms + MIN - 1
    for t in bt.trades:
        assert t.exit_ts_ms <= last_close_ts
        assert t.exit_reason in ("tp", "sl", "signal")   # never a synthetic mark-out


# -- fill model ---------------------------------------------------------------

def test_stop_and_target_in_one_bar_assumes_the_stop_filled_first(cfg):
    """OHLC cannot reveal the intrabar path. Assuming the good side would
    manufacture profit that never existed."""
    bt = Backtester(cfg, [])
    bar = _c(1, o=1000.0, h=1030.0, l=985.0, c=1010.0)   # touches BOTH tp(1020) and stop(990)
    px, reason = bt._check_stop_tp(_pos(), bar)
    assert reason == "sl"
    assert px == pytest.approx(990.0)


def test_stop_gap_through_fills_at_the_open_not_the_stop_price(cfg):
    """A bar that opens below the stop never offered the stop price."""
    bt = Backtester(cfg, [])
    bar = _c(1, o=950.0, h=955.0, l=940.0, c=945.0)      # gapped under stop 990
    px, reason = bt._check_stop_tp(_pos(), bar)
    assert reason == "sl"
    assert px == pytest.approx(950.0)                    # the open, i.e. worse than the stop


def test_target_fills_at_limit_price_when_bar_trades_through(cfg):
    bt = Backtester(cfg, [])
    bar = _c(1, o=1005.0, h=1030.0, l=1000.0, c=1025.0)  # crosses tp 1020 intrabar
    px, reason = bt._check_stop_tp(_pos(), bar)
    assert reason == "tp"
    assert px == pytest.approx(1020.0)                   # a resting limit fills at its price


def test_target_gap_up_fills_better_at_the_open(cfg):
    bt = Backtester(cfg, [])
    bar = _c(1, o=1040.0, h=1050.0, l=1035.0, c=1045.0)  # gapped past tp 1020
    px, reason = bt._check_stop_tp(_pos(), bar)
    assert reason == "tp"
    assert px == pytest.approx(1040.0)


def test_untouched_bar_returns_no_exit(cfg):
    bt = Backtester(cfg, [])
    assert bt._check_stop_tp(_pos(), _c(1, 1000.0, 1015.0, 995.0, 1005.0)) is None


def test_slippage_worsens_both_entry_and_stop(cfg):
    bt = Backtester(cfg, [], slippage_bps=100.0)   # 1%
    bar = _c(1, o=1000.0, h=1005.0, l=985.0, c=1000.0)
    px, reason = bt._check_stop_tp(_pos(), bar)
    assert reason == "sl"
    assert px == pytest.approx(990.0 * 0.99)      # filled below the stop


# -- fees ---------------------------------------------------------------------

def test_fees_reduce_pnl(cfg):
    candles = oscillating(600)
    free = Backtester(cfg, candles, fee_bps=0.0).run()
    paid = Backtester(cfg, candles, fee_bps=10.0).run()
    assert paid.n_trades == free.n_trades          # fees change PnL, not the signal
    assert paid.expectancy < free.expectancy


def test_fee_drag_is_charged_on_both_legs(cfg):
    bt = Backtester(cfg, [], fee_bps=100.0)        # 1% per side
    pos = _pos(entry=1000.0, qty=1.0)
    t = bt._close(pos, exit_px=1000.0, reason="signal", exit_ts=MIN, seq=0)
    # flat price round trip at 1% per side -> ~2% of notional lost
    assert t.pnl_quote == pytest.approx(-20.0, rel=1e-6)
    assert t.pnl_pct == pytest.approx(-2.0, rel=1e-6)


def test_pnl_pct_is_return_on_entry_notional(cfg):
    bt = Backtester(cfg, [], fee_bps=0.0)
    t = bt._close(_pos(entry=1000.0, qty=2.0), exit_px=1100.0, reason="tp", exit_ts=MIN, seq=0)
    assert t.pnl_quote == pytest.approx(200.0)
    assert t.pnl_pct == pytest.approx(10.0)        # 200 / (1000*2) * 100


# -- edges --------------------------------------------------------------------

def test_empty_candles_yield_empty_result(cfg):
    r = Backtester(cfg, []).run()
    assert r.n_trades == 0
    assert r.expectancy == 0.0
    assert r.trade_fingerprint == Backtester(cfg, []).run().trade_fingerprint


def test_series_shorter_than_warmup_produces_no_trades(cfg):
    r = Backtester(cfg, oscillating(20)).run()     # min_bars_required is 34
    assert r.n_trades == 0


def test_flat_series_produces_no_trades(cfg):
    """A zero-volatility market must not trade: ATR is below the noise floor."""
    flat = [_c(i, 1000.0, 1000.0, 1000.0, 1000.0) for i in range(200)]
    assert Backtester(cfg, flat).run().n_trades == 0


def test_run_backtest_helper_matches_class(cfg):
    candles = oscillating(300)
    assert run_backtest(cfg, candles).trade_fingerprint == Backtester(cfg, candles).run().trade_fingerprint
