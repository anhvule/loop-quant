"""RiskManager is where the system says NO. Every one of these tests describes a
way the engine could lose money that the gate must prevent."""

from __future__ import annotations

import copy

import pytest

from src.common.models import ClosedTrade, MarketState, Signal
from src.execution.exchange_adapter import SymbolFilters
from src.execution.risk_manager import MS_PER_DAY, RiskManager

FILTERS = SymbolFilters.default("BTCUSDT")
EQUITY = 10_000.0


def _rm(cfg, equity=EQUITY):
    return RiskManager(lambda: cfg, FILTERS, equity)


def _sig(action="ENTER", score=0.9):
    return Signal(ts_ms=0, score=score, components={}, action=action)


def _mkt(price=50_000.0, spread_bps=2.0, state="READY"):
    m = MarketState(symbol="BTCUSDT", last_price=price)
    half = price * spread_bps * 1e-4 / 2
    m.best_bid, m.best_ask = price - half, price + half
    m.state = state
    return m


def _closed(pnl_quote=-50.0, reason="sl", ts=0, entry_px=50_000.0, qty=0.01):
    return ClosedTrade(trade_id="t", symbol="BTCUSDT", side="BUY", entry_ts_ms=ts - 1000,
                       exit_ts_ms=ts, entry_px=entry_px, exit_px=entry_px + pnl_quote / qty,
                       qty=qty, pnl_quote=pnl_quote,
                       pnl_pct=pnl_quote / (entry_px * qty) * 100,
                       exit_reason=reason, slippage_bps=0.0, config_version=1,
                       atr_at_entry=100.0)


# -- sizing ------------------------------------------------------------------

def test_position_size_risks_the_configured_fraction_when_risk_sizing_binds(cfg):
    """Risk sizing only binds when the stop is wide relative to price. Here the
    stop is 15% of price, so the notional cap is not reached."""
    rm = _rm(cfg)
    atr, price = 100.0, 1_000.0        # stop = 1.5*100 = 150 = 15% of price
    qty, binding = rm.size_detail(EQUITY, atr, price)
    assert binding == "risk"
    assert qty == pytest.approx(50.0 / 150.0, rel=1e-3)
    # the loss if the stop is hit is exactly the amount we said we would risk
    assert qty * (1.5 * atr) == pytest.approx(EQUITY * 0.005, rel=1e-2)


def test_notional_cap_binds_at_realistic_1m_btc_volatility(cfg):
    """THE FINDING, pinned as a test.

    On a 1m BTC bar ATR is ~0.05-0.15% of price, so a 1.5*ATR stop is ~0.1-0.2%.
    Risk-based sizing then wants a position worth many multiples of equity, and
    `max_position_pct_equity` becomes the only rule setting the size. The result:
    `risk_per_trade_pct` is INERT -- and it is a dial bounds.json lets the
    optimizer tune. Effective risk lands near 0.015% of equity, not the 0.5% the
    config claims.

    This test does not assert that the situation is desirable. It asserts that it
    is what the shipped config does, so the fact cannot silently change.
    """
    rm = _rm(cfg)
    price = 50_000.0
    atr = price * 0.001                # 0.10% of price -- typical 1m BTC
    qty, binding = rm.size_detail(EQUITY, atr, price)

    assert binding == "notional"
    assert qty * price == pytest.approx(EQUITY * 0.10, rel=1e-3)   # pinned to the 10% cap

    effective_risk = qty * (1.5 * atr)
    assert effective_risk / EQUITY * 100 < 0.02        # ~0.015%, not the configured 0.5%

    # And the dial is provably dead: doubling risk_per_trade_pct changes nothing.
    c2 = copy.deepcopy(cfg)
    c2["risk"]["risk_per_trade_pct"] = 1.0
    assert _rm(c2).size_detail(EQUITY, atr, price)[0] == qty


def test_position_size_capped_by_max_position_pct_equity(cfg):
    """With a tiny ATR, risk-based sizing wants an enormous position. The notional
    cap is what stops a low-volatility regime from becoming 50x leverage."""
    rm = _rm(cfg)
    qty = rm.position_size(EQUITY, atr=0.01, price=50_000.0)
    notional = qty * 50_000.0
    assert notional <= EQUITY * 0.10 + 1e-6      # max_position_pct_equity = 10%


def test_position_size_floors_to_lot_grid_never_rounds_up(cfg):
    rm = _rm(cfg)
    qty = rm.position_size(EQUITY, atr=137.7, price=50_000.0)
    step = float(FILTERS.step_size)
    assert abs((qty / step) - round(qty / step)) < 1e-6   # on the grid
    raw = (EQUITY * 0.005) / (1.5 * 137.7)
    assert qty <= raw + 1e-12                              # never rounded up


def test_position_size_degenerate_inputs_are_zero(cfg):
    rm = _rm(cfg)
    assert rm.position_size(EQUITY, atr=0.0, price=50_000.0) == 0.0
    assert rm.position_size(EQUITY, atr=100.0, price=0.0) == 0.0
    assert rm.position_size(0.0, atr=100.0, price=50_000.0) == 0.0


# -- entry gate --------------------------------------------------------------

def test_happy_path_approves(cfg):
    rm = _rm(cfg)
    d = rm.approve_entry(_sig(), _mkt(), EQUITY, now_ms=0, atr=100.0)
    assert d.approved, d.reason
    assert d.qty > 0
    assert d.stop_px < 50_000.0 < d.tp_px


def test_rejects_when_feed_not_ready(cfg):
    rm = _rm(cfg)
    for state in ("WARMUP", "DEGRADED", "HALTED"):
        d = rm.approve_entry(_sig(), _mkt(state=state), EQUITY, 0, 100.0)
        assert not d.approved and "READY" in d.reason


def test_rejects_non_enter_signal(cfg):
    rm = _rm(cfg)
    assert not rm.approve_entry(_sig("HOLD"), _mkt(), EQUITY, 0, 100.0).approved
    assert not rm.approve_entry(_sig("EXIT"), _mkt(), EQUITY, 0, 100.0).approved


def test_rejects_wide_spread(cfg):
    rm = _rm(cfg)
    d = rm.approve_entry(_sig(), _mkt(spread_bps=50.0), EQUITY, 0, 100.0)
    assert not d.approved and "spread" in d.reason


def test_rejects_degenerate_atr(cfg):
    """A stop inside the tick noise is not a stop; it is a coin flip that pays fees."""
    rm = _rm(cfg)
    price = 50_000.0
    d = rm.approve_entry(_sig(), _mkt(price), EQUITY, 0, atr=price * 0.0001)
    assert not d.approved and "noise" in d.reason


def test_rejects_when_already_at_max_open_positions(cfg):
    rm = _rm(cfg)
    rm.open_positions = 1
    d = rm.approve_entry(_sig(), _mkt(), EQUITY, 0, 100.0)
    assert not d.approved and "max_open_positions" in d.reason


def test_rejects_below_min_notional(cfg):
    """Tiny equity -> a size the exchange would reject. Better to decline than to
    fire an order that bounces."""
    rm = _rm(cfg, equity=50.0)
    d = rm.approve_entry(_sig(), _mkt(), 50.0, 0, 100.0)
    assert not d.approved
    assert "minNotional" in d.reason or "rounds to zero" in d.reason


def test_external_halt_blocks_everything(cfg):
    rm = _rm(cfg)
    rm.halt("optimizer T4")
    d = rm.approve_entry(_sig(), _mkt(), EQUITY, 0, 100.0)
    assert not d.approved and "halt" in d.reason
    rm.resume()
    assert rm.approve_entry(_sig(), _mkt(), EQUITY, 0, 100.0).approved


# -- stop-out cooldown --------------------------------------------------------

def test_stopout_starts_cooldown(cfg):
    rm = _rm(cfg)
    t0 = 1_000_000
    rm.on_trade_closed(_closed(pnl_quote=-50.0, reason="sl", ts=t0))
    assert rm.consecutive_stopouts == 1

    d = rm.approve_entry(_sig(), _mkt(), EQUITY, t0 + 10_000, 100.0)
    assert not d.approved and "cooldown" in d.reason

    # cooldown_after_stopout_sec = 300
    d = rm.approve_entry(_sig(), _mkt(), EQUITY, t0 + 301_000, 100.0)
    assert d.approved, d.reason


def test_consecutive_stopouts_reset_on_a_non_stop_exit(cfg):
    rm = _rm(cfg)
    rm.on_trade_closed(_closed(reason="sl", ts=1000))
    rm.on_trade_closed(_closed(reason="sl", ts=2000))
    assert rm.consecutive_stopouts == 2
    rm.on_trade_closed(_closed(pnl_quote=+80.0, reason="tp", ts=3000))
    assert rm.consecutive_stopouts == 0


# -- daily loss backstop ------------------------------------------------------

def test_daily_loss_limit_halts_trading(cfg):
    """max_daily_loss_pct = 3% of 10000 = 300 quote."""
    rm = _rm(cfg)
    rm.mark_equity(EQUITY, 0)
    assert not rm.daily_halt_active(0)

    rm.on_trade_closed(_closed(pnl_quote=-200.0, reason="sl", ts=1000))
    assert not rm.daily_halt_active(1000)

    rm.on_trade_closed(_closed(pnl_quote=-150.0, reason="sl", ts=2000))
    assert rm.daily_halt_active(2000)
    assert rm.halt_reason is not None

    d = rm.approve_entry(_sig(), _mkt(), EQUITY, 3000, 100.0)
    assert not d.approved


def test_daily_pnl_resets_at_utc_midnight(cfg):
    rm = _rm(cfg)
    rm.mark_equity(EQUITY, 0)
    rm.on_trade_closed(_closed(pnl_quote=-250.0, reason="sl", ts=1000))
    assert rm.day_pnl_quote == pytest.approx(-250.0)

    rm.mark_equity(EQUITY - 250.0, MS_PER_DAY + 1)   # next UTC day
    assert rm.day_pnl_quote == pytest.approx(0.0)


def test_daily_limit_measured_against_day_start_equity(cfg):
    """The limit must not shrink as the day's losses shrink equity, or the
    threshold would chase itself downward."""
    rm = _rm(cfg)
    rm.mark_equity(EQUITY, 0)
    rm.on_trade_closed(_closed(pnl_quote=-290.0, reason="sl", ts=1000))
    rm.mark_equity(EQUITY - 290.0, 1000)
    assert not rm.daily_halt_active(1000)          # 290 < 300 = 3% of the DAY-START equity


# -- stops --------------------------------------------------------------------

def test_stops_are_atr_multiples(cfg):
    rm = _rm(cfg)
    stop, tp = rm.stops_for(entry_px=50_000.0, atr=100.0, mkt=None)
    assert stop == pytest.approx(50_000.0 - 1.5 * 100.0)
    assert tp == pytest.approx(50_000.0 + 2.5 * 100.0)


def test_stop_never_tighter_than_twice_the_spread(cfg):
    """A stop inside the round-trip spread is hit by the bid-ask bounce alone."""
    c = copy.deepcopy(cfg)
    c["risk"]["atr_mult_sl"] = 0.5
    rm = _rm(c)
    m = _mkt(price=50_000.0, spread_bps=20.0)   # spread = 100 quote
    stop, _ = rm.stops_for(50_000.0, atr=1.0, mkt=m)   # atr stop would be 0.5 quote
    spread_abs = m.best_ask - m.best_bid
    assert 50_000.0 - stop >= 2.0 * spread_abs - 1e-9


def test_reward_exceeds_risk_under_shipped_config(cfg):
    rm = _rm(cfg)
    stop, tp = rm.stops_for(50_000.0, 100.0, None)
    risk, reward = 50_000.0 - stop, tp - 50_000.0
    assert reward / risk == pytest.approx(2.5 / 1.5)
    assert reward > risk
