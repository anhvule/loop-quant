"""Structured JSONL logging to logs/engine.log plus a human-readable console line."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(record.created * 1000),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_path: Path, level: int = logging.INFO, console: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(JsonlFormatter())
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                                          datefmt="%H:%M:%S"))
        root.addHandler(ch)

    # The websockets/aiohttp DEBUG streams drown the engine's own signal.
    for noisy in ("websockets", "websockets.client", "aiohttp", "asyncio", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def jlog(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Log with structured fields attached to the JSONL record."""
    logger.log(level, msg, extra={"extra_fields": fields})
