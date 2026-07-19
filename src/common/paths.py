"""Canonical filesystem paths. Every module resolves paths through here so the
backtester subprocess and the live engine can never disagree about where the
database or the config lives.
"""

from __future__ import annotations

import os
from pathlib import Path

# loop_quant/src/common/paths.py -> loop_quant/
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
SCHEMA_PATH = CONFIG_DIR / "config.schema.json"
BOUNDS_PATH = CONFIG_DIR / "bounds.json"
VERSIONS_DIR = CONFIG_DIR / "versions"
BASELINE_PATH = CONFIG_DIR / "baseline.json"
BURNED_PATH = CONFIG_DIR / "burned.json"
STRESS_WINDOWS_PATH = CONFIG_DIR / "stress_windows.json"
FILTERS_PATH = CONFIG_DIR / "exchange_filters.json"

DATA_DIR = ROOT / "data"
DB_PATH = Path(os.environ.get("LOOPQUANT_DB", DATA_DIR / "loopquant.db"))

LOGS_DIR = ROOT / "logs"
ENGINE_LOG = LOGS_DIR / "engine.log"
TRADES_LOG = LOGS_DIR / "trades.jsonl"
OPTIMIZER_AUDIT = LOGS_DIR / "optimizer_audit.jsonl"
HUMAN_REVIEW_QUEUE = LOGS_DIR / "human_review_queue.jsonl"
KILL_FILE = LOGS_DIR / "KILL"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, VERSIONS_DIR, DATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
