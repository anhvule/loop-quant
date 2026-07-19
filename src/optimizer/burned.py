"""Registry of configurations that were tried and failed.

Without this, a rolled-back config can be re-proposed on the next cycle: the same
KPIs are still bad, the same diagnosis follows, the same fix looks attractive, and
the system loops. Burning a config for a week breaks that cycle.

Keyed by a hash of the TUNABLE values only -- not the whole document -- so a
version-number bump doesn't disguise an identical proposal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BURN_DAYS = 7
MS_PER_DAY = 86_400_000


def config_fingerprint(cfg: dict[str, Any], bounds: dict[str, Any]) -> str:
    from src.common.config_loader import get_path
    tunables = {p: get_path(cfg, p) for p in sorted(bounds.get("params", {}).keys())}
    blob = json.dumps(tunables, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class BurnedRegistry:
    def __init__(self, path: Path, bounds: dict[str, Any]) -> None:
        self.path = Path(path)
        self.bounds = bounds
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("burned.json unreadable; starting empty")
            self._entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")

    def burn(self, cfg: dict[str, Any], reason: str, now_ms: int | None = None) -> str:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        fp = config_fingerprint(cfg, self.bounds)
        self._entries[fp] = {"burned_ms": now, "reason": reason,
                             "version": cfg.get("version"),
                             "expires_ms": now + BURN_DAYS * MS_PER_DAY}
        self._save()
        log.warning("burned config fingerprint %s (v%s): %s", fp, cfg.get("version"), reason)
        return fp

    def is_burned(self, cfg: dict[str, Any], now_ms: int | None = None) -> dict[str, Any] | None:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        fp = config_fingerprint(cfg, self.bounds)
        e = self._entries.get(fp)
        if not e:
            return None
        if now >= e.get("expires_ms", 0):
            # The burn has aged out. Market regimes change; a config that failed a
            # week ago is allowed to be tried again.
            del self._entries[fp]
            self._save()
            return None
        return e

    def prune(self, now_ms: int | None = None) -> int:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        before = len(self._entries)
        self._entries = {k: v for k, v in self._entries.items()
                         if now < v.get("expires_ms", 0)}
        if len(self._entries) != before:
            self._save()
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
