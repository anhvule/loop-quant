"""Trade economics: does this config have a reachable breakeven win rate?

This module exists because of a structural trap in short-timeframe crypto: the
round-trip fee is charged on NOTIONAL, while the stop and target are measured in
ATR. On a 1m BTC bar, ATR is roughly 0.05-0.15% of price, so a 1.5*ATR stop is
~15bps -- while a 10bps taker fee each way is 20bps round trip. The fee is then
LARGER than the entire stop distance, and no win rate can save the strategy.

Module 4's Phase A prompt asks the LLM to reason about exactly this. Computing it
here, deterministically, means the diagnosis is grounded in arithmetic rather than
in whatever the model estimates -- and it lets the engine refuse to start quietly
losing money.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Above this, the strategy needs a win rate that no realistic edge delivers.
UNREACHABLE_WIN_RATE = 0.75


@dataclass(frozen=True, slots=True)
class TradeEconomics:
    atr_pct_of_price: float
    stop_pct: float           # stop distance as % of price
    target_pct: float         # target distance as % of price
    fee_pct_round_trip: float
    gross_rr: float           # target/stop before fees
    net_win_pct: float        # what a winner actually nets, after fees
    net_loss_pct: float       # what a loser actually costs, after fees (positive number)
    breakeven_win_rate: float | None
    viable: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trade_economics(cfg: dict[str, Any], atr: float, price: float,
                    fee_bps_per_side: float = 10.0) -> TradeEconomics:
    """Net-of-fee economics for one round trip at the given volatility."""
    rk = cfg["risk"]
    if price <= 0 or atr <= 0:
        return TradeEconomics(0, 0, 0, 0, 0, 0, 0, None, False, "no ATR or price")

    stop_pct = float(rk["atr_mult_sl"]) * atr / price * 100.0
    target_pct = float(rk["atr_mult_tp"]) * atr / price * 100.0
    fee_pct = 2.0 * fee_bps_per_side / 100.0     # bps -> %, both legs

    net_win = target_pct - fee_pct
    net_loss = stop_pct + fee_pct                # a loser pays the stop AND the fees

    gross_rr = target_pct / stop_pct if stop_pct > 0 else 0.0

    if net_win <= 0:
        return TradeEconomics(
            atr / price * 100.0, stop_pct, target_pct, fee_pct, gross_rr,
            net_win, net_loss, None, False,
            "a winning trade does not even cover the round-trip fee; no win rate is enough",
        )

    breakeven = net_loss / (net_loss + net_win)
    viable = breakeven < UNREACHABLE_WIN_RATE
    if viable:
        note = f"needs a {breakeven:.1%} win rate to break even"
    else:
        note = (f"needs a {breakeven:.1%} win rate to break even -- fees ({fee_pct:.3f}%) "
                f"are large relative to the stop ({stop_pct:.3f}%); widen stops, use a "
                f"longer timeframe, or reduce fees")
    return TradeEconomics(atr / price * 100.0, stop_pct, target_pct, fee_pct, gross_rr,
                          net_win, net_loss, breakeven, viable, note)


def sizing_constraint(cfg: dict[str, Any], atr: float, price: float) -> dict[str, Any]:
    """Which sizing rule actually binds -- and therefore whether
    `risk_per_trade_pct` does anything at all.

    Risk-based sizing binds only when
        stop_distance / price >= risk_per_trade_pct / max_position_pct_equity
    Below that, the notional cap decides the size and `risk_per_trade_pct` is
    inert -- an optimizer that tunes it would be turning a dial wired to nothing.
    """
    rk = cfg["risk"]
    risk_pct = float(rk["risk_per_trade_pct"])
    max_pos_pct = float(rk["max_position_pct_equity"])
    if price <= 0 or atr <= 0 or max_pos_pct <= 0:
        return {"binds": "unknown", "risk_dial_is_live": False}

    stop_pct = float(rk["atr_mult_sl"]) * atr / price * 100.0
    required_stop_pct = risk_pct / max_pos_pct * 100.0
    binds = "risk" if stop_pct >= required_stop_pct else "notional"

    effective_risk_pct = (risk_pct if binds == "risk"
                          else max_pos_pct * stop_pct / 100.0)
    return {
        "binds": binds,
        "risk_dial_is_live": binds == "risk",
        "stop_pct_of_price": stop_pct,
        "stop_pct_needed_for_risk_sizing": required_stop_pct,
        "configured_risk_per_trade_pct": risk_pct,
        "effective_risk_per_trade_pct": effective_risk_pct,
    }
