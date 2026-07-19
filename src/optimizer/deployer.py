"""Staged rollout, shadow monitoring, and rollback.

Passing the sandbox earns a config a *trial*, not a coronation. Simulation is a
model; the shadow window is where the model meets the market. If the first 15
live trades under a new config come in materially below what the sandbox
promised, it is reverted automatically and the parameter set is burned.

SHADOW THRESHOLD -- an interpretation the blueprint leaves ambiguous:

  The spec says roll back when shadow expectancy < (sandbox expectancy - 1.5σ).
  Read literally with σ = per-trade standard deviation, the band is enormous
  (per-trade PnL is far more dispersed than the mean of 15 of them) and rollback
  would essentially never fire. σ here is therefore the STANDARD ERROR of the
  15-trade mean, σ/sqrt(n) -- the dispersion of the quantity actually being
  compared. That makes the test meaningful instead of decorative.

Note on open positions: a hot-reload deliberately does not touch a live position's
stops. They were computed from the ATR at entry and are already resting on the
exchange as an OCO. Re-deriving them under new parameters mid-trade would move a
stop the trader never agreed to.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from src.common.db import DB
from src.common.event_bus import EventBus
from src.common.models import BacktestBaseline, KPIReport, SandboxVerdict
from src.evaluation.baseline import save_baseline
from src.optimizer.burned import BurnedRegistry

log = logging.getLogger(__name__)

SHADOW_TRADES = 15
SHADOW_SIGMA_MULT = 1.5


class Deployer:
    def __init__(self, config_path: Path, versions_dir: Path, baseline_path: Path,
                 db: DB, bus: EventBus, loader, burned: BurnedRegistry,
                 on_deploy: Callable[[int], None] | None = None) -> None:
        self.config_path = Path(config_path)
        self.versions_dir = Path(versions_dir)
        self.baseline_path = Path(baseline_path)
        self.db = db
        self.bus = bus
        self.loader = loader
        self.burned = burned
        self.on_deploy = on_deploy

        self.shadow_version: int | None = None
        self.shadow_expected: float = 0.0
        self.shadow_std: float = 0.0
        self.shadow_started_ms: int = 0
        self.rollback_count = 0

    # -- deploy ------------------------------------------------------------

    def deploy(self, candidate: dict[str, Any], verdict: SandboxVerdict,
               now_ms: int | None = None, source: str = "optimizer") -> int:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        current = self.loader.current()
        new_version = int(current["version"]) + 1

        cfg = dict(candidate)
        cfg["version"] = new_version

        # 1 + 2: archive, then atomically swap. os.replace is atomic on both POSIX
        # and Windows, so a crash mid-write cannot leave a truncated config.json
        # that the engine would refuse to load on restart.
        self._archive(cfg, now)
        self._atomic_write(cfg)

        self.db.insert_config_version(
            new_version, now, json.dumps(cfg), source,
            json.dumps(verdict.to_dict()) if verdict else None)

        # 3: hot-reload. If the new config somehow fails the load gate, the loader
        # keeps the previous one live and raises -- we then revert the file.
        try:
            self.loader.reload()
        except Exception:
            log.exception("deployed config failed the load gate; reverting the file")
            self._atomic_write(current)
            raise

        log.warning("DEPLOYED config v%d (source=%s): %s", new_version, source,
                    self._diff_text(current, cfg))

        # 4: the baseline must describe the config that is now running.
        if verdict and verdict.candidate:
            b = BacktestBaseline(
                generated_ms=now, config_version=new_version, window_days=30,
                n_trades=verdict.candidate.n_trades,
                expectancy=verdict.candidate.expectancy,
                sharpe=verdict.candidate.sharpe,
                max_drawdown_pct=verdict.candidate.max_drawdown_pct,
                win_rate=verdict.candidate.win_rate,
                profit_factor=verdict.candidate.profit_factor,
            )
            save_baseline(b, self.baseline_path)

        # 5: begin the trial.
        if verdict and verdict.candidate:
            self.shadow_version = new_version
            self.shadow_expected = verdict.candidate.expectancy
            self.shadow_std = verdict.candidate.pnl_pct_std
            self.shadow_started_ms = now
            log.info("shadow window open for v%d: expecting ~%.4f%%/trade (std %.4f) "
                     "over the next %d trades",
                     new_version, self.shadow_expected, self.shadow_std, SHADOW_TRADES)

        if self.on_deploy:
            self.on_deploy(new_version)
        return new_version

    # -- shadow ------------------------------------------------------------

    def check_shadow(self, r: KPIReport) -> bool:
        """Returns True if a rollback was triggered."""
        if self.shadow_version is None:
            return False

        trades = self.db.get_closed_trades(config_version=self.shadow_version)
        if len(trades) < SHADOW_TRADES:
            return False

        window = trades[:SHADOW_TRADES]
        observed = statistics.fmean([t.pnl_pct for t in window])

        if self.shadow_std > 0:
            se = self.shadow_std / math.sqrt(SHADOW_TRADES)
        else:
            # The sandbox showed no dispersion at all (degenerate). Fall back to the
            # live window's own dispersion rather than trusting a zero-width band.
            live_std = statistics.stdev([t.pnl_pct for t in window]) if len(window) > 1 else 0.0
            se = live_std / math.sqrt(SHADOW_TRADES)

        floor = self.shadow_expected - SHADOW_SIGMA_MULT * se
        passed = observed >= floor

        log.info("shadow check v%d: observed %.4f%%/trade vs floor %.4f%% "
                 "(expected %.4f%%, se %.4f) -> %s",
                 self.shadow_version, observed, floor, self.shadow_expected, se,
                 "PASS" if passed else "ROLLBACK")

        version = self.shadow_version
        self.shadow_version = None   # the trial is over either way

        if passed:
            log.info("config v%d survived its shadow window", version)
            return False

        self.rollback(version - 1,
                      f"shadow window: {observed:.4f}%/trade over {SHADOW_TRADES} trades, "
                      f"below the {floor:.4f}% floor implied by the sandbox")
        return True

    # -- rollback ----------------------------------------------------------

    def rollback(self, to_version: int, reason: str, now_ms: int | None = None) -> int:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        bad = self.loader.current()
        bad_version = int(bad["version"])

        target = self._load_version(to_version)
        if target is None:
            log.critical("cannot roll back to v%d: no archived file. Halting rather than "
                         "leaving a known-bad config live.", to_version)
            raise FileNotFoundError(f"no archived config for version {to_version}")

        # Burn the bad parameter set BEFORE writing, so even a crash between here
        # and the next cycle leaves the rejection recorded.
        self.burned.burn(bad, reason, now)
        self.db.mark_config_rolled_back(bad_version, now)

        # Roll forward to the old content under a NEW version number rather than
        # reusing the old one: version numbers stay monotonic, closed_trades keeps
        # an unambiguous join key, and the audit trail shows the reversion happened.
        restored = dict(target)
        restored["version"] = bad_version + 1

        self._archive(restored, now)
        self._atomic_write(restored)
        self.db.insert_config_version(restored["version"], now, json.dumps(restored),
                                      f"rollback_from_v{bad_version}", None)
        self.loader.reload()

        self.rollback_count += 1
        self.shadow_version = None
        log.warning("ROLLED BACK v%d -> v%d (content of v%d): %s",
                    bad_version, restored["version"], to_version, reason)

        if self.on_deploy:
            self.on_deploy(restored["version"])
        return restored["version"]

    # -- file plumbing -----------------------------------------------------

    def _archive(self, cfg: dict[str, Any], now_ms: int) -> Path:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        p = self.versions_dir / f"config_v{cfg['version']}_{now_ms}.json"
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return p

    def _atomic_write(self, cfg: dict[str, Any]) -> None:
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        os.replace(tmp, self.config_path)

    def _load_version(self, version: int) -> dict[str, Any] | None:
        matches = sorted(self.versions_dir.glob(f"config_v{version}_*.json"))
        if matches:
            try:
                return json.loads(matches[-1].read_text(encoding="utf-8"))
            except Exception:
                log.exception("archived config v%d is unreadable", version)
        # Fall back to the DB copy -- the archive directory could have been pruned.
        for row in self.db.get_config_versions(50):
            if int(row["version"]) == version:
                try:
                    return json.loads(row["config_json"])
                except Exception:
                    pass
        return None

    def _diff_text(self, old: dict[str, Any], new: dict[str, Any]) -> str:
        from src.common.config_loader import flatten_paths
        a, b = flatten_paths(old), flatten_paths(new)
        parts = [f"{k}: {a[k]} -> {b[k]}" for k in sorted(b) if k in a and a[k] != b[k]]
        return "; ".join(parts) or "(no tunable changes)"
