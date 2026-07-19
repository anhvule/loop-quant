"""Pytest bootstrap: put the project root on sys.path so `import src.*` works
without an editable install, and give every test an isolated DB."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import ConfigLoader  # noqa: E402
from src.common.db import DB  # noqa: E402
from src.common.paths import BOUNDS_PATH, CONFIG_PATH, SCHEMA_PATH  # noqa: E402


@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_PATH, SCHEMA_PATH, BOUNDS_PATH)


@pytest.fixture
def cfg(loader: ConfigLoader) -> dict:
    return loader.load()


@pytest.fixture
def db(tmp_path: Path) -> DB:
    d = DB(tmp_path / "test.db")
    yield d
    d.close()
