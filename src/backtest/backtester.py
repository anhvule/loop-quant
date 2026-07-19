"""Deterministic replay engine. Shared by baseline generation and the optimizer's
sandbox.

Two properties are non-negotiable:

  * **Same code as live.** It drives the real `SignalEngine` and `RiskManager`.
    A backtest that used a reimplementation of the strategy would validate the
    reimplementation, not the thing that trades.
  * **Deterministic.** No RNG, no clock reads, no dict-ordering dependence. Same
    (config, candles) -> byte-identical trade list, enforced by
    `test_backtester.py::test_backtester_is_deterministic`. The optimizer accepts
    or rejects configs on ~5% expectancy differences; a jittery backtest would
    make that gate meaningless.

Fill modelling is deliberately pessimistic:
  * entry     = bar close, crossed by slippage (taker)
  * stop hit  = stop price, or the bar OPEN if the bar gapped straight through it
  * target hit= limit price exactly (or the open if it gapped past, which is better)
  * SL and TP both touched in one bar -> assume the STOP filled first. OHLC cannot
    tell us the intrabar path, and assuming the good side would manufacture
    profits that do not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from src.common.models import (
    BacktestResult, Candle, ClosedTrade, MarketState, Position, tf_to_ms,
)
from src.evaluation import metrics
from src.execution.exchange_adapter import SymbolFilters
from src.execution.risk_manager import RiskManager
from src.execution.signal_engine import SignalEngine
from src.ingestion.indicator_engine import IndicatorEngine

log = logging.getLogger(__name__)

INITIAL_EQUITY = 10_000.0
TAKER_FEE_BPS = 10.0        # Binance spot taker, no BNB discount
DEFAULT_SPREAD_BPS = 2.0    # modelled book width when no live book is available


class Backtester:
    def __init__(self, cfg: dict[str, Any], candles: Sequence[Candle], *,
                 filters: SymbolFilters | None = None,
                 initial_equity: float = INITIAL_EQUITY,
                 fee_bps: float = TAKER_FEE_BPS,
                 slippage_bps: float = 0.0,
                 spread_bps: float = DEFAULT_SPREAD_BPS) -> None:
        self.cfg = cfg
        self.candles = list(candles)
        self.symbol = cfg["symbol"]
        self.tf = cfg["timeframe"]
        self.tf_ms = tf_to_ms(self.tf)
        self.filters = filters or SymbolFilters.default(self.symbol)
        self.initial_equity = float(initial_equity)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        self.trades: list[ClosedTrade] = []

    # -- main loop ---------------------------------------------------------

    def run(self) -> BacktestResult:
        cfg = self.cfg
        st, rk = cfg["strategy"], cfg["risk"]

        ind = IndicatorEngine(
            self.symbol, self.tf,
            rsi_period=int(st["rsi"]["period"]),
            macd_fast=int(st["macd"]["fast"]), macd_slow=int(st["macd"]["slow"]),
            macd_signal=int(st["macd"]["signal"]), atr_period=int(rk["atr_period"]),
            persist=False,
        )
        engine = SignalEngine(lambda: cfg)
        rm = RiskManager(lambda: cfg, self.filters, self.initial_equity)
        market = MarketState(symbol=self.symbol, state="READY")

        equity = self.initial_equity
        pos: Position | None = None
        seq = 0
        self.trades = []

        for c in self.candles:
            close_ts = c.ts_open_ms + self.tf_ms - 1

            # 1. Exits first: this bar can only close a position opened on an
            #    EARLIER bar. Checking the entry bar's own high/low would be
            #    lookahead -- we would be using a range we had not seen yet.
            if pos is not None:
                hit = self._check_stop_tp(pos, c)
                if hit is not None:
                    px, reason = hit
                    t = self._close(pos, px, reason, close_ts, seq)
                    seq += 1
                    self.trades.append(t)
                    equity += t.pnl_quote
                    rm.on_trade_closed(t)
                    rm.open_positions = 0
                    pos = None

            # 2. Fold the bar into the indicators.
            snap = ind.update(c)

            # 3. Synthesize the market state as of this bar's close.
            market.last_price = c.close
            half = c.close * self.spread_bps * 1e-4 / 2.0
            market.best_bid = c.close - half
            market.best_ask = c.close + half
            market.bid_qty_top5 = market.ask_qty_top5 = 1e9   # depth is not modelled
            market.last_tick_ms = close_ts
            rm.mark_equity(equity, close_ts)

            sig = engine.compute(snap, market)

            # 4. Signal exit (taker, crosses the spread).
            if pos is not None and sig.action == "EXIT":
                px = c.close * (1.0 - self.slippage_bps * 1e-4)
                t = self._close(pos, px, "signal", close_ts, seq)
                seq += 1
                self.trades.append(t)
                equity += t.pnl_quote
                rm.on_trade_closed(t)
                rm.open_positions = 0
                pos = None

            # 5. Entry.
            if pos is None:
                d = rm.approve_entry(sig, market, equity, close_ts, float(snap.atr or 0.0))
                if d.approved:
                    fill_px = c.close * (1.0 + self.slippage_bps * 1e-4)
                    pos = Position(
                        symbol=self.symbol, state="OPEN", side="BUY", qty=d.qty,
                        entry_px=fill_px, entry_ts_ms=close_ts,
                        stop_px=d.stop_px, tp_px=d.tp_px,
                        atr_at_entry=float(snap.atr or 0.0),
                        config_version=int(cfg["version"]),
                        trade_id=f"bt-{seq:06d}",
                        entry_slippage_bps=self.slippage_bps,
                    )
                    rm.open_positions = 1

        # An open position at the end of the window is NOT counted as a trade.
        # Marking it to the last close would book an unrealized result that the
        # live system would never have recorded.
        return self._result()

    # -- fill model --------------------------------------------------------

    def _check_stop_tp(self, pos: Position, c: Candle) -> tuple[float, str] | None:
        hit_sl = c.low <= pos.stop_px
        hit_tp = c.high >= pos.tp_px

        if hit_sl:
            # Gapped straight through the stop -> we fill at the open, not at the
            # stop price we hoped for.
            raw = min(pos.stop_px, c.open)
            return (raw * (1.0 - self.slippage_bps * 1e-4), "sl")
        if hit_tp:
            # A resting limit fills at its price, or better if the bar gapped past it.
            raw = pos.tp_px if c.open < pos.tp_px else c.open
            return (raw, "tp")
        return None

    def _close(self, pos: Position, exit_px: float, reason: str, exit_ts: int,
               seq: int) -> ClosedTrade:
        entry_notional = pos.qty * pos.entry_px
        exit_notional = pos.qty * exit_px
        fees = (entry_notional + exit_notional) * self.fee_bps * 1e-4
        pnl_quote = exit_notional - entry_notional - fees
        pnl_pct = (pnl_quote / entry_notional * 100.0) if entry_notional > 0 else 0.0
        return ClosedTrade(
            trade_id=pos.trade_id or f"bt-{seq:06d}", symbol=pos.symbol, side="BUY",
            entry_ts_ms=pos.entry_ts_ms, exit_ts_ms=exit_ts,
            entry_px=pos.entry_px, exit_px=exit_px, qty=pos.qty,
            pnl_quote=pnl_quote, pnl_pct=pnl_pct, exit_reason=reason,  # type: ignore[arg-type]
            slippage_bps=pos.entry_slippage_bps,
            config_version=pos.config_version, atr_at_entry=pos.atr_at_entry,
        )

    # -- results -----------------------------------------------------------

    def _result(self) -> BacktestResult:
        start = self.candles[0].ts_open_ms if self.candles else None
        end = (self.candles[-1].ts_open_ms + self.tf_ms) if self.candles else None
        s = metrics.summarize(self.trades, self.initial_equity, start, end)
        series = [t.pnl_pct for t in self.trades]
        return BacktestResult(
            n_trades=s["n_trades"],
            expectancy=_f(s["expectancy"]),
            sharpe=_f(s["sharpe"]),
            max_drawdown_pct=_f(s["max_drawdown_pct"]),
            win_rate=_f(s["win_rate"]),
            profit_factor=_f(s["profit_factor"]),
            total_return_pct=_f(s["total_return_pct"]),
            pnl_pct_std=(statistics.stdev(series) if len(series) > 1 else 0.0),
            pnl_pct_series=series,
            trade_fingerprint=self.fingerprint(),
        )

    def fingerprint(self) -> str:
        """Stable hash of the trade list -- the determinism test's assertion."""
        h = hashlib.sha256()
        for t in self.trades:
            h.update(f"{t.entry_ts_ms}|{t.exit_ts_ms}|{t.entry_px:.8f}|{t.exit_px:.8f}"
                     f"|{t.qty:.8f}|{t.exit_reason}|{t.pnl_quote:.8f}\n".encode())
        return h.hexdigest()


def _f(x: float | None) -> float:
    return 0.0 if x is None else float(x)


def run_backtest(cfg: dict[str, Any], candles: Sequence[Candle], **kw: Any) -> BacktestResult:
    return Backtester(cfg, candles, **kw).run()


# ---------------------------------------------------------------------------
# CLI -- this is the entrypoint the sandbox runs in a subprocess
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Loop Quant deterministic backtester")
    ap.add_argument("--config", required=True, help="path to a config.json to evaluate")
    ap.add_argument("--db", default=None, help="sqlite path (default: paths.DB_PATH)")
    ap.add_argument("--start-ms", type=int, default=0)
    ap.add_argument("--end-ms", type=int, default=2**63 - 1)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--fee-bps", type=float, default=TAKER_FEE_BPS)
    ap.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY)
    ap.add_argument("--out", default=None, help="write result JSON here (else stdout)")
    a = ap.parse_args(argv)

    from src.common.db import DB
    from src.common.paths import DB_PATH

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    db = DB(a.db or DB_PATH)
    try:
        candles = db.get_candles(cfg["symbol"], cfg["timeframe"], a.start_ms, a.end_ms)
    finally:
        db.close()

    if not candles:
        payload = {"error": "no candles in range", "n_trades": 0}
        _emit(payload, a.out)
        return 2

    r = Backtester(cfg, candles, initial_equity=a.initial_equity, fee_bps=a.fee_bps,
                   slippage_bps=a.slippage_bps).run()
    payload = r.to_dict()
    payload["trade_fingerprint"] = r.trade_fingerprint
    payload["window"] = {"start_ms": candles[0].ts_open_ms, "end_ms": candles[-1].ts_open_ms,
                         "n_candles": len(candles)}
    payload["config_version"] = cfg["version"]
    _emit(payload, a.out)
    return 0


def _emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
