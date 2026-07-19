"""Minimal asyncio pub/sub bus.

Design rules:
  * A subscriber that raises never takes down the publisher or its siblings --
    the exception is logged and the remaining handlers still run. A crashed
    KPI subscriber must not stop the risk manager from seeing a fill.
  * Handlers run sequentially in subscription order, so an ordering-sensitive
    chain (e.g. position bookkeeping before KPI accounting) is expressible by
    subscribing in the right order.
  * Both sync and async handlers are supported.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable, Awaitable

log = logging.getLogger(__name__)

Handler = Callable[[Any], None] | Callable[[Any], Awaitable[None]]

# Canonical topic names. Publishing an unknown topic raises -- typos in a topic
# string would otherwise silently route messages into the void.
TOPICS: frozenset[str] = frozenset({
    "trade_tick",
    "book_update",
    "candle_closed",
    "indicator_update",
    "signal",
    "order_filled",
    "trade_closed",
    "kpi_report",
    "underperformance",
    "config_deployed",
    "state_changed",
})


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._published: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Handler) -> None:
        if topic not in TOPICS:
            raise ValueError(f"unknown topic {topic!r}; add it to EventBus.TOPICS")
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if handler in self._subs.get(topic, []):
            self._subs[topic].remove(handler)

    async def publish(self, topic: str, payload: Any) -> None:
        if topic not in TOPICS:
            raise ValueError(f"unknown topic {topic!r}; add it to EventBus.TOPICS")
        self._published[topic] += 1
        for handler in list(self._subs[topic]):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "event_bus: handler %s failed on topic %r",
                    getattr(handler, "__qualname__", repr(handler)), topic,
                )

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._published)
