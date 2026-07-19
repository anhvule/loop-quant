"""Seed the candle table with history so the backtester/baseline/sandbox have
something real to chew on.

Sources:
  * `binance`   -- download real 1m klines from Binance's PUBLIC market-data
                   endpoint. Read-only and unauthenticated.
  * `synthetic` -- deterministic generated bars, for running the drill offline.

On the production host: this script is the one place that talks to api.binance.com,
and it does so with NO API KEY AND NO SECRET. ExchangeAdapter._sign raises without
a secret, so every signed endpoint -- i.e. everything that could place an order --
is unreachable from this process by construction. Downloading public candles is
not trading; the testnet-only guard exists to keep ORDERS off production, and it
still does.

    python -m scripts.seed_data --days 30
    python -m scripts.seed_data --source synthetic --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.db import DB                                    # noqa: E402
from src.common.models import Candle                            # noqa: E402
from src.common.paths import DB_PATH, ensure_dirs              # noqa: E402
from src.execution.exchange_adapter import ExchangeAdapter     # noqa: E402

PUBLIC_REST = "https://api.binance.com"
MS_PER_DAY = 86_400_000


async def seed_binance(symbol: str, tf: str, days: int, db: DB) -> int:
    now = int(time.time() * 1000)
    start = now - days * MS_PER_DAY
    adapter = ExchangeAdapter(
        api_key="", api_secret="",          # unauthenticated: signed calls cannot be made
        rest_base=PUBLIC_REST, ws_base="wss://stream.binance.com",
        require_testnet=False,              # read-only public market data only
    )
    await adapter.start()
    try:
        print(f"downloading {days}d of {symbol} {tf} klines from {PUBLIC_REST} ...")
        rows = await adapter.get_klines_range(symbol, tf, start, now)
    finally:
        await adapter.close()

    candles = [Candle(ts_open_ms=r["ts_open_ms"], symbol=symbol, tf=tf, open=r["open"],
                      high=r["high"], low=r["low"], close=r["close"], volume=r["volume"],
                      quote_volume=r["quote_volume"], n_trades=r["n_trades"])
               for r in rows]
    return db.upsert_candles(candles)


def seed_synthetic(symbol: str, tf: str, days: int, db: DB, seed_price: float = 60_000.0) -> int:
    """Deterministic pseudo-market. No RNG: a sum of sines with incommensurate
    periods gives trend, chop, and volatility clustering while staying exactly
    reproducible across runs and machines."""
    n = days * 1440
    now = int(time.time() * 1000)
    start = (now - n * 60_000) // 60_000 * 60_000

    candles = []
    prev = seed_price
    for i in range(n):
        drift = 0.00002 * i
        wave = (0.010 * math.sin(i / 90.0)
                + 0.004 * math.sin(i / 17.0)
                + 0.002 * math.sin(i / 5.0))
        vol_cluster = 1.0 + 0.8 * math.sin(i / 720.0) ** 2
        c = seed_price * (1.0 + drift + wave * vol_cluster)
        o = prev
        rng = abs(c - o) + seed_price * 0.0004 * vol_cluster
        h = max(o, c) + rng * 0.4
        l = min(o, c) - rng * 0.4
        v = 5.0 + 4.0 * abs(math.sin(i / 33.0))
        # quote_volume must be sum(price*qty); approximate with the bar's mean price
        qv = v * (o + h + l + c) / 4.0
        candles.append(Candle(ts_open_ms=start + i * 60_000, symbol=symbol, tf=tf,
                              open=o, high=h, low=l, close=c, volume=v,
                              quote_volume=qv, n_trades=20))
        prev = c
    return db.upsert_candles(candles)


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the Loop Quant candle table")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1m")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--source", choices=["binance", "synthetic"], default="binance")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    ensure_dirs()
    db = DB(a.db or DB_PATH)
    try:
        if a.source == "binance":
            n = asyncio.run(seed_binance(a.symbol, a.tf, a.days, db))
        else:
            n = seed_synthetic(a.symbol, a.tf, a.days, db)

        span = db.candle_span(a.symbol, a.tf)
        print(f"seeded {n} candles ({a.source})")
        if span:
            import datetime as dt
            f = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")
            print(f"span: {f(span[0])} -> {f(span[1])} UTC")
            print(f"total rows for {a.symbol}/{a.tf}: "
                  f"{len(db.get_candles(a.symbol, a.tf))}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
