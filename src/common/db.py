"""SQLite access layer (WAL mode).

All persistence lives here. The DDL below is normative -- Module 3 and Module 4
both read these tables directly, and `closed_trades.config_version` is the join
key that lets the optimizer attribute performance to the config that caused it.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.common.models import Candle, ClosedTrade, Fill, IndicatorSnapshot, KPIReport, Trade

DDL = """
CREATE TABLE IF NOT EXISTS trades_raw (
  ts_ms INTEGER, symbol TEXT, price REAL, qty REAL,
  is_buyer_maker INTEGER, agg_trade_id INTEGER,
  PRIMARY KEY(symbol, agg_trade_id)
);
CREATE INDEX IF NOT EXISTS ix_trades_raw_ts ON trades_raw(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS candles (
  ts_open_ms INTEGER, symbol TEXT, tf TEXT,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  quote_volume REAL, n_trades INTEGER,
  PRIMARY KEY(symbol, tf, ts_open_ms)
);

CREATE TABLE IF NOT EXISTS indicators (
  ts_ms INTEGER, symbol TEXT, tf TEXT,
  vwap REAL, rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL, atr REAL,
  PRIMARY KEY(symbol, tf, ts_ms)
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
  ts_ms INTEGER, symbol TEXT,
  best_bid REAL, best_ask REAL, bid_qty_top5 REAL, ask_qty_top5 REAL,
  PRIMARY KEY(symbol, ts_ms)
);

CREATE TABLE IF NOT EXISTS fills (
  order_id TEXT, client_order_id TEXT, ts_ms INTEGER, symbol TEXT,
  side TEXT, price REAL, qty REAL, fee REAL, fee_asset TEXT
);
CREATE INDEX IF NOT EXISTS ix_fills_order ON fills(order_id);

CREATE TABLE IF NOT EXISTS closed_trades (
  trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT,
  entry_ts_ms INTEGER, exit_ts_ms INTEGER, entry_px REAL, exit_px REAL, qty REAL,
  pnl_quote REAL, pnl_pct REAL, exit_reason TEXT,
  slippage_bps REAL, config_version INTEGER, atr_at_entry REAL
);
CREATE INDEX IF NOT EXISTS ix_closed_exit_ts ON closed_trades(exit_ts_ms);
CREATE INDEX IF NOT EXISTS ix_closed_cfgver ON closed_trades(config_version);

CREATE TABLE IF NOT EXISTS kpi_snapshots (
  ts_ms INTEGER PRIMARY KEY, window_trades INTEGER, win_rate REAL, profit_factor REAL,
  expectancy REAL, sharpe REAL, max_drawdown_pct REAL, avg_slippage_bps REAL,
  consecutive_stopouts INTEGER, bt_deviation_pct REAL, equity REAL, config_version INTEGER
);

CREATE TABLE IF NOT EXISTS config_versions (
  version INTEGER PRIMARY KEY, deployed_ms INTEGER, config_json TEXT,
  source TEXT, verdict_json TEXT, rolled_back_ms INTEGER
);
"""


class DB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(DDL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- writes -----------------------------------------------------------

    def insert_trades(self, trades: Iterable[Trade]) -> int:
        rows = [(t.ts_ms, t.symbol, t.price, t.qty, int(t.is_buyer_maker), t.agg_trade_id)
                for t in trades]
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                "INSERT OR IGNORE INTO trades_raw(ts_ms,symbol,price,qty,is_buyer_maker,agg_trade_id)"
                " VALUES (?,?,?,?,?,?)", rows)
        return len(rows)

    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [(c.ts_open_ms, c.symbol, c.tf, c.open, c.high, c.low, c.close,
                 c.volume, c.quote_volume, c.n_trades) for c in candles]
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO candles"
                "(ts_open_ms,symbol,tf,open,high,low,close,volume,quote_volume,n_trades)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def insert_indicator(self, s: IndicatorSnapshot) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO indicators"
                "(ts_ms,symbol,tf,vwap,rsi,macd,macd_signal,macd_hist,atr) VALUES (?,?,?,?,?,?,?,?,?)",
                (s.ts_ms, s.symbol, s.tf, s.vwap, s.rsi, s.macd, s.macd_signal, s.macd_hist, s.atr))

    def insert_book_snapshot(self, ts_ms: int, symbol: str, bid: float, ask: float,
                             bid_q: float, ask_q: float) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO orderbook_snapshots"
                "(ts_ms,symbol,best_bid,best_ask,bid_qty_top5,ask_qty_top5) VALUES (?,?,?,?,?,?)",
                (ts_ms, symbol, bid, ask, bid_q, ask_q))

    def insert_fill(self, f: Fill) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO fills(order_id,client_order_id,ts_ms,symbol,side,price,qty,fee,fee_asset)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (f.order_id, f.client_order_id, f.ts_ms, f.symbol, f.side, f.price, f.qty,
                 f.fee, f.fee_asset))

    def insert_closed_trade(self, t: ClosedTrade) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO closed_trades(trade_id,symbol,side,entry_ts_ms,exit_ts_ms,"
                "entry_px,exit_px,qty,pnl_quote,pnl_pct,exit_reason,slippage_bps,config_version,"
                "atr_at_entry) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.trade_id, t.symbol, t.side, t.entry_ts_ms, t.exit_ts_ms, t.entry_px, t.exit_px,
                 t.qty, t.pnl_quote, t.pnl_pct, t.exit_reason, t.slippage_bps, t.config_version,
                 t.atr_at_entry))

    def insert_kpi_snapshot(self, r: KPIReport) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO kpi_snapshots(ts_ms,window_trades,win_rate,profit_factor,"
                "expectancy,sharpe,max_drawdown_pct,avg_slippage_bps,consecutive_stopouts,"
                "bt_deviation_pct,equity,config_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.ts_ms, r.window_trades, r.win_rate, r.profit_factor, r.expectancy, r.sharpe,
                 r.max_drawdown_pct, r.avg_slippage_bps, r.consecutive_stopouts,
                 r.bt_deviation_pct, r.equity, r.config_version))

    def insert_config_version(self, version: int, deployed_ms: int, config_json: str,
                              source: str, verdict_json: str | None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO config_versions"
                "(version,deployed_ms,config_json,source,verdict_json,rolled_back_ms)"
                " VALUES (?,?,?,?,?,NULL)",
                (version, deployed_ms, config_json, source, verdict_json))

    def mark_config_rolled_back(self, version: int, ts_ms: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE config_versions SET rolled_back_ms=? WHERE version=?", (ts_ms, version))

    # -- reads ------------------------------------------------------------

    def get_candles(self, symbol: str, tf: str, start_ms: int = 0,
                    end_ms: int = 2**63 - 1, limit: int | None = None) -> list[Candle]:
        q = ("SELECT * FROM candles WHERE symbol=? AND tf=? AND ts_open_ms>=? AND ts_open_ms<=?"
             " ORDER BY ts_open_ms ASC")
        args: list[Any] = [symbol, tf, start_ms, end_ms]
        if limit is not None:
            q += " LIMIT ?"
            args.append(limit)
        return [_row_to_candle(r) for r in self.conn.execute(q, args)]

    def get_last_candles(self, symbol: str, tf: str, n: int) -> list[Candle]:
        rows = self.conn.execute(
            "SELECT * FROM candles WHERE symbol=? AND tf=? ORDER BY ts_open_ms DESC LIMIT ?",
            (symbol, tf, n)).fetchall()
        return [_row_to_candle(r) for r in reversed(rows)]

    def last_candle_ts(self, symbol: str, tf: str) -> int | None:
        r = self.conn.execute(
            "SELECT MAX(ts_open_ms) AS m FROM candles WHERE symbol=? AND tf=?",
            (symbol, tf)).fetchone()
        return r["m"] if r and r["m"] is not None else None

    def candle_span(self, symbol: str, tf: str) -> tuple[int, int] | None:
        r = self.conn.execute(
            "SELECT MIN(ts_open_ms) AS a, MAX(ts_open_ms) AS b FROM candles WHERE symbol=? AND tf=?",
            (symbol, tf)).fetchone()
        if not r or r["a"] is None:
            return None
        return int(r["a"]), int(r["b"])

    def get_closed_trades(self, limit: int | None = None, since_ms: int = 0,
                          config_version: int | None = None, ascending: bool = True) -> list[ClosedTrade]:
        q = "SELECT * FROM closed_trades WHERE exit_ts_ms>=?"
        args: list[Any] = [since_ms]
        if config_version is not None:
            q += " AND config_version=?"
            args.append(config_version)
        q += f" ORDER BY exit_ts_ms {'ASC' if ascending else 'DESC'}"
        if limit is not None:
            q += " LIMIT ?"
            args.append(limit)
        return [_row_to_closed_trade(r) for r in self.conn.execute(q, args)]

    def get_recent_closed_trades(self, n: int) -> list[ClosedTrade]:
        """Chronological order, newest n."""
        rows = self.get_closed_trades(limit=n, ascending=False)
        return list(reversed(rows))

    def count_closed_trades(self, since_ms: int = 0) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) AS n FROM closed_trades WHERE exit_ts_ms>=?", (since_ms,)).fetchone()
        return int(r["n"])

    def kpi_by_config_version(self) -> dict[int, dict[str, Any]]:
        """Realized outcome per config version -- what the optimizer sees as
        'what did the last 3 configs actually produce'."""
        rows = self.conn.execute(
            "SELECT config_version AS v, COUNT(*) AS n, AVG(pnl_pct) AS expectancy,"
            " SUM(CASE WHEN pnl_quote>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) AS win_rate,"
            " SUM(pnl_quote) AS pnl_quote"
            " FROM closed_trades GROUP BY config_version").fetchall()
        return {int(r["v"]): {"n_trades": int(r["n"]), "expectancy": r["expectancy"],
                              "win_rate": r["win_rate"], "pnl_quote": r["pnl_quote"]}
                for r in rows}

    def get_config_versions(self, limit: int = 3) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM config_versions ORDER BY version DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def last_deploy_ms(self) -> int | None:
        r = self.conn.execute("SELECT MAX(deployed_ms) AS m FROM config_versions").fetchone()
        return int(r["m"]) if r and r["m"] is not None else None

    def indicator_close_pairs(self, symbol: str, tf: str, start_ms: int,
                              end_ms: int) -> list[tuple[float, float, float]]:
        """(vwap, close, atr) joined bar-by-bar -- the raw material for the
        market-regime section of the optimizer bundle."""
        rows = self.conn.execute(
            "SELECT i.vwap AS vwap, c.close AS close, i.atr AS atr"
            " FROM indicators i JOIN candles c"
            "   ON i.ts_ms=c.ts_open_ms AND i.symbol=c.symbol AND i.tf=c.tf"
            " WHERE i.symbol=? AND i.tf=? AND i.ts_ms>=? AND i.ts_ms<=?"
            "   AND i.vwap IS NOT NULL AND i.atr IS NOT NULL"
            " ORDER BY i.ts_ms ASC", (symbol, tf, start_ms, end_ms)).fetchall()
        return [(r["vwap"], r["close"], r["atr"]) for r in rows]

    def avg_spread_bps(self, symbol: str, since_ms: int) -> float | None:
        r = self.conn.execute(
            "SELECT AVG((best_ask-best_bid)/((best_ask+best_bid)/2)*10000.0) AS s"
            " FROM orderbook_snapshots WHERE symbol=? AND ts_ms>=?"
            "   AND best_bid>0 AND best_ask>0", (symbol, since_ms)).fetchone()
        return float(r["s"]) if r and r["s"] is not None else None


def _row_to_candle(r: sqlite3.Row) -> Candle:
    return Candle(
        ts_open_ms=int(r["ts_open_ms"]), symbol=r["symbol"], tf=r["tf"],
        open=r["open"], high=r["high"], low=r["low"], close=r["close"],
        volume=r["volume"], quote_volume=r["quote_volume"], n_trades=int(r["n_trades"]))


def _row_to_closed_trade(r: sqlite3.Row) -> ClosedTrade:
    return ClosedTrade(
        trade_id=r["trade_id"], symbol=r["symbol"], side=r["side"],
        entry_ts_ms=int(r["entry_ts_ms"]), exit_ts_ms=int(r["exit_ts_ms"]),
        entry_px=r["entry_px"], exit_px=r["exit_px"], qty=r["qty"],
        pnl_quote=r["pnl_quote"], pnl_pct=r["pnl_pct"], exit_reason=r["exit_reason"],
        slippage_bps=r["slippage_bps"], config_version=int(r["config_version"]),
        atr_at_entry=r["atr_at_entry"])
