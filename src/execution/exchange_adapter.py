"""Binance Spot **Testnet** REST/WS adapter.

Every byte of exchange I/O goes through this class. v1 is testnet-only by
construction: the base URLs are hardcoded and `require_testnet` refuses to
construct against production hosts. Going live is a deliberate, reviewable edit
here -- not an environment variable someone can flip by accident.

Filter handling is the un-glamorous part that decides whether orders are accepted
at all: Binance rejects any price not on the tick grid and any qty not on the lot
grid, so we floor to the grid before every send.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any
from urllib.parse import urlencode

import aiohttp

log = logging.getLogger(__name__)

TESTNET_REST = "https://testnet.binance.vision"
TESTNET_WS = "wss://stream.testnet.binance.vision"

# Production hosts, listed only so the guard can refuse them.
_PROD_HOSTS = ("api.binance.com", "stream.binance.com", "api1.binance.com",
               "api2.binance.com", "api3.binance.com", "fapi.binance.com")

RECV_WINDOW_MS = 5_000


class ExchangeError(RuntimeError):
    def __init__(self, status: int, code: int | None, msg: str) -> None:
        super().__init__(f"binance {status} code={code}: {msg}")
        self.status = status
        self.code = code
        self.msg = msg


@dataclass(frozen=True, slots=True)
class SymbolFilters:
    """The exchange's hard constraints for one symbol."""
    symbol: str
    tick_size: Decimal        # PRICE_FILTER.tickSize
    step_size: Decimal        # LOT_SIZE.stepSize
    min_qty: Decimal          # LOT_SIZE.minQty
    max_qty: Decimal          # LOT_SIZE.maxQty
    min_notional: Decimal     # NOTIONAL/MIN_NOTIONAL.minNotional
    base_asset: str = ""
    quote_asset: str = ""

    def round_price(self, px: float) -> float:
        """Snap to the tick grid (nearest; a price off-grid is rejected outright)."""
        d = Decimal(str(px)).quantize(self.tick_size, rounding=ROUND_HALF_UP)
        # quantize to a tick that is not a power of ten (e.g. 0.05) still needs a snap
        steps = (Decimal(str(px)) / self.tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        d = steps * self.tick_size
        return float(d)

    def round_qty(self, qty: float) -> float:
        """FLOOR to the lot grid -- never round a size up, that would silently
        take more risk than the RiskManager approved."""
        steps = (Decimal(str(qty)) / self.step_size).to_integral_value(rounding=ROUND_DOWN)
        return float(steps * self.step_size)

    def qty_ok(self, qty: float) -> bool:
        return float(self.min_qty) <= qty <= float(self.max_qty)

    def notional_ok(self, qty: float, price: float) -> bool:
        return qty * price >= float(self.min_notional)

    @staticmethod
    def from_exchange_info(info: dict[str, Any]) -> "SymbolFilters":
        f = {x["filterType"]: x for x in info["filters"]}
        lot = f.get("LOT_SIZE", {})
        price = f.get("PRICE_FILTER", {})
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        return SymbolFilters(
            symbol=info["symbol"],
            tick_size=Decimal(price.get("tickSize", "0.01")),
            step_size=Decimal(lot.get("stepSize", "0.00001")),
            min_qty=Decimal(lot.get("minQty", "0.00001")),
            max_qty=Decimal(lot.get("maxQty", "9000")),
            min_notional=Decimal(notional.get("minNotional", "10")),
            base_asset=info.get("baseAsset", ""),
            quote_asset=info.get("quoteAsset", ""),
        )

    @staticmethod
    def default(symbol: str = "BTCUSDT") -> "SymbolFilters":
        """Offline default used by the backtester and tests. Mirrors testnet
        BTCUSDT at the time of writing; the live path always overwrites this
        from exchangeInfo."""
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"),
                             step_size=Decimal("0.00001"), min_qty=Decimal("0.00001"),
                             max_qty=Decimal("9000"), min_notional=Decimal("10"),
                             base_asset="BTC", quote_asset="USDT")


class ExchangeAdapter:
    def __init__(self, api_key: str = "", api_secret: str = "", *,
                 rest_base: str = TESTNET_REST, ws_base: str = TESTNET_WS,
                 require_testnet: bool = True) -> None:
        if require_testnet and any(h in rest_base or h in ws_base for h in _PROD_HOSTS):
            raise ValueError(
                "refusing to construct an ExchangeAdapter against a production Binance host. "
                "v1 is testnet-only; going live is a deliberate code change, not a config flip."
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_base = rest_base.rstrip("/")
        self.ws_base = ws_base.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._filters: dict[str, SymbolFilters] = {}
        self._time_offset_ms = 0

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> "ExchangeAdapter":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {},
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError("ExchangeAdapter.start() was not called")
        return self._session

    # -- signing / transport ---------------------------------------------

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_secret:
            raise RuntimeError("signed endpoint requires an API secret")
        p = dict(params)
        p["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        p.setdefault("recvWindow", RECV_WINDOW_MS)
        qs = urlencode(p, doseq=True)
        p["signature"] = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return p

    async def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                       signed: bool = False, max_retries: int = 5) -> Any:
        params = dict(params or {})
        url = f"{self.rest_base}{path}"
        delay = 1.0
        for attempt in range(max_retries):
            send = self._sign(params) if signed else params
            try:
                async with self.session.request(method, url, params=send) as r:
                    text = await r.text()
                    if r.status == 200:
                        return await _json(text)

                    body = await _json_safe(text)
                    code = body.get("code") if isinstance(body, dict) else None
                    msg = body.get("msg", text) if isinstance(body, dict) else text

                    # 429 = rate limited, 418 = IP auto-banned for ignoring 429s.
                    # Honour Retry-After exactly; hammering a 418 extends the ban.
                    if r.status in (429, 418):
                        wait = float(r.headers.get("Retry-After", delay))
                        log.warning("rate limited (%s) on %s; sleeping %.1fs", r.status, path, wait)
                        await asyncio.sleep(wait)
                        delay = min(delay * 2, 60.0)
                        continue
                    if r.status >= 500:
                        log.warning("binance %s on %s; retry in %.1fs", r.status, path, delay)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    # 4xx: a client error. Retrying an identical bad request is
                    # pointless, so surface it immediately.
                    raise ExchangeError(r.status, code, str(msg))
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise ExchangeError(0, None, f"network error: {e}") from e
                log.warning("network error on %s (%s); retry in %.1fs", path, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise ExchangeError(0, None, f"exhausted retries on {path}")

    # -- public data ------------------------------------------------------

    async def sync_time(self) -> int:
        """Align our clock to the exchange's. A drifting local clock produces
        -1021 'timestamp outside recvWindow' rejections on every signed call."""
        t0 = int(time.time() * 1000)
        r = await self._request("GET", "/api/v3/time")
        self._time_offset_ms = int(r["serverTime"]) - t0
        log.info("exchange time offset: %+d ms", self._time_offset_ms)
        return self._time_offset_ms

    async def load_filters(self, symbol: str) -> SymbolFilters:
        r = await self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})
        syms = r.get("symbols", [])
        if not syms:
            raise ExchangeError(0, None, f"exchangeInfo returned no entry for {symbol}")
        f = SymbolFilters.from_exchange_info(syms[0])
        self._filters[symbol] = f
        log.info("filters %s: tick=%s step=%s minNotional=%s",
                 symbol, f.tick_size, f.step_size, f.min_notional)
        return f

    def filters(self, symbol: str) -> SymbolFilters:
        return self._filters.get(symbol) or SymbolFilters.default(symbol)

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500,
                         start_ms: int | None = None, end_ms: int | None = None) -> list[dict]:
        p: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_ms is not None:
            p["startTime"] = start_ms
        if end_ms is not None:
            p["endTime"] = end_ms
        rows = await self._request("GET", "/api/v3/klines", p)
        return [_kline_row(r, symbol, interval) for r in rows]

    async def get_klines_range(self, symbol: str, interval: str, start_ms: int,
                               end_ms: int) -> list[dict]:
        """Page through a range 1000 bars at a time (the REST cap)."""
        out: list[dict] = []
        cursor = start_ms
        step_ms = {"1m": 60_000, "5m": 300_000}[interval]
        while cursor <= end_ms:
            batch = await self.get_klines(symbol, interval, 1000, cursor, end_ms)
            if not batch:
                break
            out.extend(batch)
            cursor = batch[-1]["ts_open_ms"] + step_ms
            if len(batch) < 1000:
                break
            await asyncio.sleep(0.12)   # stay well inside the weight budget
        return out

    async def get_book_ticker(self, symbol: str) -> dict:
        return await self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})

    # -- signed: account & orders -----------------------------------------

    async def get_account(self) -> dict:
        return await self._request("GET", "/api/v3/account", signed=True)

    async def get_equity_quote(self, symbol: str, mark_price: float) -> float:
        """Account equity denominated in the quote asset (free+locked base marked
        to `mark_price`, plus free+locked quote)."""
        f = self.filters(symbol)
        acct = await self.get_account()
        base = quote = 0.0
        for b in acct.get("balances", []):
            total = float(b["free"]) + float(b["locked"])
            if b["asset"] == f.base_asset:
                base = total
            elif b["asset"] == f.quote_asset:
                quote = total
        return quote + base * mark_price

    async def get_open_orders(self, symbol: str) -> list[dict]:
        return await self._request("GET", "/api/v3/openOrders", {"symbol": symbol}, signed=True)

    async def place_order(self, symbol: str, side: str, type_: str, *,
                          qty: float | None = None, price: float | None = None,
                          tif: str | None = None, client_order_id: str | None = None,
                          quote_qty: float | None = None) -> dict:
        p: dict[str, Any] = {"symbol": symbol, "side": side, "type": type_}
        f = self.filters(symbol)
        if qty is not None:
            p["quantity"] = _fmt(f.round_qty(qty), f.step_size)
        if quote_qty is not None:
            p["quoteOrderQty"] = f"{quote_qty:.8f}".rstrip("0").rstrip(".")
        if price is not None:
            p["price"] = _fmt(f.round_price(price), f.tick_size)
        if tif:
            p["timeInForce"] = tif
        if client_order_id:
            p["newClientOrderId"] = client_order_id
        p["newOrderRespType"] = "FULL"
        return await self._request("POST", "/api/v3/order", p, signed=True)

    async def cancel(self, symbol: str, order_id: str | int | None = None,
                     client_order_id: str | None = None) -> dict:
        p: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            p["orderId"] = order_id
        elif client_order_id is not None:
            p["origClientOrderId"] = client_order_id
        else:
            raise ValueError("cancel needs order_id or client_order_id")
        return await self._request("DELETE", "/api/v3/order", p, signed=True)

    async def cancel_all(self, symbol: str) -> list[dict]:
        try:
            return await self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol},
                                       signed=True)
        except ExchangeError as e:
            if e.code == -2011:   # "Unknown order sent" == nothing open. Not an error.
                return []
            raise

    async def get_order(self, symbol: str, order_id: str | int | None = None,
                        client_order_id: str | None = None) -> dict:
        p: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            p["orderId"] = order_id
        elif client_order_id is not None:
            p["origClientOrderId"] = client_order_id
        else:
            raise ValueError("get_order needs order_id or client_order_id")
        return await self._request("GET", "/api/v3/order", p, signed=True)

    async def place_oco_sell(self, symbol: str, qty: float, tp_price: float,
                             stop_price: float, stop_limit_price: float,
                             list_client_order_id: str | None = None,
                             tp_client_order_id: str | None = None,
                             sl_client_order_id: str | None = None) -> dict:
        """Protective exit for a long: LIMIT_MAKER take-profit above, STOP_LOSS_LIMIT below.

        `stop_limit_price` is set below `stop_price` by the caller so the stop
        still fills through a fast move -- a stop-limit pegged at the trigger is
        how you end up triggered but unfilled in a gap.
        """
        f = self.filters(symbol)
        p: dict[str, Any] = {
            "symbol": symbol,
            "side": "SELL",
            "quantity": _fmt(f.round_qty(qty), f.step_size),
            "aboveType": "LIMIT_MAKER",
            "abovePrice": _fmt(f.round_price(tp_price), f.tick_size),
            "belowType": "STOP_LOSS_LIMIT",
            "belowStopPrice": _fmt(f.round_price(stop_price), f.tick_size),
            "belowPrice": _fmt(f.round_price(stop_limit_price), f.tick_size),
            "belowTimeInForce": "GTC",
        }
        if list_client_order_id:
            p["listClientOrderId"] = list_client_order_id
        # Tagging each leg lets the OrderManager read the exit reason ('tp' vs 'sl')
        # straight off the execution report instead of inferring it from price.
        if tp_client_order_id:
            p["aboveClientOrderId"] = tp_client_order_id
        if sl_client_order_id:
            p["belowClientOrderId"] = sl_client_order_id
        try:
            return await self._request("POST", "/api/v3/orderList/oco", p, signed=True)
        except ExchangeError as e:
            if e.status == 404 or e.code == -1121:
                raise
            # Older testnet builds only expose the legacy OCO endpoint.
            if e.code in (-1102, -1104, -1128):
                log.warning("orderList/oco rejected (%s); falling back to legacy order/oco", e.msg)
                legacy = {
                    "symbol": symbol, "side": "SELL",
                    "quantity": p["quantity"],
                    "price": p["abovePrice"],
                    "stopPrice": p["belowStopPrice"],
                    "stopLimitPrice": p["belowPrice"],
                    "stopLimitTimeInForce": "GTC",
                }
                if list_client_order_id:
                    legacy["listClientOrderId"] = list_client_order_id
                if tp_client_order_id:
                    legacy["limitClientOrderId"] = tp_client_order_id
                if sl_client_order_id:
                    legacy["stopClientOrderId"] = sl_client_order_id
                return await self._request("POST", "/api/v3/order/oco", legacy, signed=True)
            raise

    async def cancel_order_list(self, symbol: str, order_list_id: int) -> dict:
        return await self._request("DELETE", "/api/v3/orderList",
                                   {"symbol": symbol, "orderListId": order_list_id}, signed=True)

    # -- user data stream -------------------------------------------------

    async def create_listen_key(self) -> str:
        r = await self._request("POST", "/api/v3/userDataStream")
        return r["listenKey"]

    async def keepalive_listen_key(self, key: str) -> None:
        await self._request("PUT", "/api/v3/userDataStream", {"listenKey": key})

    async def close_listen_key(self, key: str) -> None:
        try:
            await self._request("DELETE", "/api/v3/userDataStream", {"listenKey": key})
        except ExchangeError:
            pass

    def stream_url(self, streams: list[str]) -> str:
        return f"{self.ws_base}/stream?streams=" + "/".join(streams)

    def raw_stream_url(self, stream: str) -> str:
        return f"{self.ws_base}/ws/{stream}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(x: float, step: Decimal) -> str:
    """Format to exactly the precision the filter implies. Binance rejects both
    excess precision and scientific notation."""
    exp = -step.as_tuple().exponent
    return f"{x:.{max(exp, 0)}f}"


def _kline_row(r: list, symbol: str, interval: str) -> dict:
    return {
        "ts_open_ms": int(r[0]), "symbol": symbol, "tf": interval,
        "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
        "volume": float(r[5]), "ts_close_ms": int(r[6]), "quote_volume": float(r[7]),
        "n_trades": int(r[8]),
    }


async def _json(text: str) -> Any:
    import json
    return json.loads(text)


async def _json_safe(text: str) -> Any:
    import json
    try:
        return json.loads(text)
    except Exception:
        return {"msg": text}
