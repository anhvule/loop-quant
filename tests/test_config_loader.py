"""The config gate is the load-bearing safety component: if a bad config can
reach the engine, every downstream guarantee is void. These tests pin that gate.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.common.config_loader import (
    ConfigError, ConfigLoader, eval_invariant, get_path, normalized_weights,
    set_path, validate, validate_bounds, validate_schema,
)


def test_shipped_config_is_valid(loader, cfg):
    assert validate(cfg, loader.schema, loader.bounds) == []
    assert cfg["version"] == 1
    assert cfg["symbol"] == "BTCUSDT"


def test_weights_are_renormalized_to_one(cfg):
    w = normalized_weights(cfg)
    assert pytest.approx(sum(w.values()), abs=1e-12) == 1.0
    # a config whose weights sum to 3 must produce the same normalized vector
    scaled = copy.deepcopy(cfg)
    for k in scaled["strategy"]["weights"]:
        scaled["strategy"]["weights"][k] *= 3.0
    assert normalized_weights(scaled) == pytest.approx(w)


def test_zero_weights_rejected(cfg, loader):
    bad = copy.deepcopy(cfg)
    bad["strategy"]["weights"] = {"vwap": 0.0, "rsi": 0.0, "macd": 0.0}
    errs = validate(bad, loader.schema, loader.bounds)
    assert any("weights_nonzero" in e for e in errs)
    with pytest.raises(ValueError):
        normalized_weights(bad)


@pytest.mark.parametrize("path,value", [
    ("risk.atr_mult_sl", 9.0),        # above hard max 3.0
    ("risk.atr_mult_sl", 0.1),        # below hard min 0.5
    ("risk.risk_per_trade_pct", 4.0),  # above hard max 1.0
    ("strategy.entry_threshold", 0.95),
    ("execution.max_spread_bps", 50),
    ("strategy.rsi.oversold", 45),
])
def test_out_of_bounds_values_rejected(cfg, loader, path, value):
    bad = copy.deepcopy(cfg)
    set_path(bad, path, value)
    errs = validate(bad, loader.schema, loader.bounds)
    assert errs, f"{path}={value} should have been rejected"


def test_rr_floor_invariant(cfg, loader):
    bad = copy.deepcopy(cfg)
    bad["risk"]["atr_mult_tp"] = 1.0  # sl is 1.5 -> tp < 1.2*sl
    errs = validate_bounds(bad, loader.bounds)
    assert any("rr_floor" in e for e in errs)


def test_threshold_ordering_invariant(cfg, loader):
    bad = copy.deepcopy(cfg)
    bad["strategy"]["entry_threshold"] = 0.2
    bad["strategy"]["exit_threshold"] = -0.0
    bad["strategy"]["entry_threshold"] = -0.0  # entry == exit
    errs = validate_bounds(bad, loader.bounds)
    assert any("threshold_ordering" in e for e in errs)


def test_schema_rejects_unknown_keys(cfg, loader):
    bad = copy.deepcopy(cfg)
    bad["strategy"]["backdoor"] = True
    assert validate_schema(bad, loader.schema)


def test_schema_rejects_missing_section(cfg, loader):
    bad = copy.deepcopy(cfg)
    del bad["risk"]
    assert validate_schema(bad, loader.schema)


def test_set_path_refuses_new_keys(cfg):
    with pytest.raises(KeyError):
        set_path(cfg, "risk.brand_new_dial", 1.0)
    with pytest.raises(KeyError):
        set_path(cfg, "nonexistent.section.key", 1.0)


def test_eval_invariant_blocks_code_execution(cfg):
    # An attacker-supplied bounds.json must not be able to reach the interpreter.
    with pytest.raises(ValueError):
        eval_invariant("__import__('os').system('echo pwned')", cfg)
    with pytest.raises(ValueError):
        eval_invariant("open('x','w')", cfg)


def test_eval_invariant_arithmetic(cfg):
    assert eval_invariant("risk.atr_mult_tp >= 1.2 * risk.atr_mult_sl", cfg) is True
    assert eval_invariant("risk.atr_mult_tp < risk.atr_mult_sl", cfg) is False
    assert eval_invariant("strategy.rsi.overbought > strategy.rsi.oversold + 10", cfg) is True


def test_eval_invariant_unknown_path_raises(cfg):
    with pytest.raises(ValueError):
        eval_invariant("risk.not_a_real_param > 1", cfg)


def test_loader_refuses_bad_config_and_keeps_previous(tmp_path, loader, cfg):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    cl = ConfigLoader(p, loader.schema_path, loader.bounds_path)
    good = cl.load()
    assert good["version"] == 1

    bad = copy.deepcopy(cfg)
    bad["risk"]["atr_mult_sl"] = 99.0
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ConfigError):
        cl.reload()
    # the previously loaded good config must still be live
    assert cl.current()["risk"]["atr_mult_sl"] == 1.5


def test_loader_rejects_malformed_json(tmp_path, loader):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    cl = ConfigLoader(p, loader.schema_path, loader.bounds_path)
    with pytest.raises(ConfigError):
        cl.load()


def test_bounds_never_lists_the_absolute_backstops(loader):
    """These dials are what stand between a bad optimizer cycle and a blown
    account. If a future edit adds them to bounds.json, this test must fail."""
    forbidden = {
        "risk.max_daily_loss_pct",
        "risk.max_position_pct_equity",
        "risk.max_open_positions",
        "symbol",
        "version",
        "strategy.rsi.period",
        "strategy.macd.fast",
        "strategy.macd.slow",
        "strategy.macd.signal",
        "risk.atr_period",
    }
    assert forbidden.isdisjoint(loader.bounds["params"].keys())
