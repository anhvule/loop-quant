"""The validator is the last deterministic thing between an LLM's opinion and a
running trading config. Every test here is an attack it must survive.
"""

from __future__ import annotations

import copy

import pytest

from src.common.models import OptimizationProposal, ProposedChange
from src.optimizer.burned import BurnedRegistry, config_fingerprint
from src.optimizer.proposal_validator import ProposalValidator


def _v(loader, burned=None):
    return ProposalValidator(loader.schema, loader.bounds, burned)


def _prop(*changes, abstain=False, plan="compare 30d expectancy"):
    return OptimizationProposal(
        proposed_changes=[ProposedChange(p, o, n, "because") for p, o, n in changes],
        expected_impact=None, simulation_plan=plan, abstain=abstain)


# -- happy path ---------------------------------------------------------------

def test_valid_single_change_passes(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8)), cfg, loader.bounds)
    assert r.ok, r.errors
    assert r.candidate_config["risk"]["atr_mult_sl"] == 1.8
    assert r.candidate_config["risk"]["atr_mult_tp"] == 2.5   # untouched
    assert cfg["risk"]["atr_mult_sl"] == 1.5                  # original not mutated


def test_valid_three_changes_pass(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8),
                         ("risk.atr_mult_tp", 2.5, 3.0),
                         ("strategy.entry_threshold", 0.5, 0.6)), cfg, loader.bounds)
    assert r.ok, r.errors


# -- the whitelist (blueprint check 7) ----------------------------------------

@pytest.mark.parametrize("path,new", [
    ("risk.max_daily_loss_pct", 50.0),        # the account backstop
    ("risk.max_position_pct_equity", 100.0),  # the leverage backstop
    ("risk.max_open_positions", 5),
    ("strategy.rsi.period", 7),               # would invalidate every baseline
    ("strategy.macd.fast", 5),
    ("risk.atr_period", 5),
    ("version", 99),
    ("symbol", "ETHUSDT"),
])
def test_non_whitelisted_paths_are_refused(loader, cfg, path, new):
    """These are the dials that stand between a bad cycle and a blown account.
    They are absent from bounds.params, therefore immutable BY CONSTRUCTION."""
    v = _v(loader)
    r = v.validate(_prop((path, None, new)), cfg, loader.bounds)
    assert not r.ok
    assert any("not a tunable parameter" in e for e in r.errors)


def test_new_keys_cannot_be_invented(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.secret_leverage", None, 10.0)), cfg, loader.bounds)
    assert not r.ok
    assert any("not a tunable parameter" in e for e in r.errors)


# -- bounds (check 4) ---------------------------------------------------------

@pytest.mark.parametrize("path,old,new", [
    ("risk.atr_mult_sl", 1.5, 1.87),          # would be fine on delta, but...
])
def test_within_bounds_and_delta_passes(loader, cfg, path, old, new):
    assert _v(loader).validate(_prop((path, old, new)), cfg, loader.bounds).ok


def test_value_outside_hard_bounds_is_rejected(loader, cfg):
    """atr_mult_sl max is 3.0. Walk there in legal 25% steps and the bound still
    holds at the end."""
    v = _v(loader)
    c = copy.deepcopy(cfg)
    c["risk"]["atr_mult_sl"] = 2.9
    c["risk"]["atr_mult_tp"] = 5.0
    r = v.validate(_prop(("risk.atr_mult_sl", 2.9, 3.5)), c, loader.bounds)
    assert not r.ok
    assert any("outside hard limit" in e or "3.0" in e for e in r.errors)


# -- delta cap (check 5) ------------------------------------------------------

def test_delta_larger_than_cap_is_rejected(loader, cfg):
    """25% per cycle. 1.5 -> 2.5 is 67%: no single cycle may make that jump, even
    though 2.5 is inside the bounds."""
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 2.5)), cfg, loader.bounds)
    assert not r.ok
    assert any("per cycle" in e for e in r.errors)


def test_delta_exactly_at_cap_is_allowed(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.875)), cfg, loader.bounds)  # exactly +25%
    assert r.ok, r.errors


def test_delta_from_zero_uses_the_bound_span(loader, cfg):
    """A relative cap is undefined at zero. exit_threshold can legitimately be 0."""
    c = copy.deepcopy(cfg)
    c["strategy"]["exit_threshold"] = 0.0
    v = _v(loader)
    # span is 0.9; 25% of that is 0.225
    assert v.validate(_prop(("strategy.exit_threshold", 0.0, -0.2)), c, loader.bounds).ok
    r = v.validate(_prop(("strategy.exit_threshold", 0.0, -0.5)), c, loader.bounds)
    assert not r.ok
    assert any("bound span" in e for e in r.errors)


# -- change count (check 6) ---------------------------------------------------

def test_more_than_three_changes_is_rejected(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.6),
                         ("risk.atr_mult_tp", 2.5, 2.6),
                         ("strategy.entry_threshold", 0.5, 0.55),
                         ("strategy.vwap_band_bps", 15, 16)), cfg, loader.bounds)
    assert not r.ok
    assert any("max is 3" in e for e in r.errors)


def test_duplicate_paths_are_rejected(loader, cfg):
    """Otherwise last-write-wins and the change count is a lie."""
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.6),
                         ("risk.atr_mult_sl", 1.5, 1.7)), cfg, loader.bounds)
    assert not r.ok
    assert any("duplicate" in e for e in r.errors)


# -- invariants ---------------------------------------------------------------

def test_change_violating_the_rr_invariant_is_rejected(loader, cfg):
    """Each value is individually in bounds and within the delta cap, but the pair
    breaks tp >= 1.2*sl. Per-field checks alone would let this through."""
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.875),
                         ("risk.atr_mult_tp", 2.5, 2.0)), cfg, loader.bounds)
    assert not r.ok
    assert any("rr_floor" in e for e in r.errors)


def test_rsi_band_ordering_invariant_fires_on_a_violating_config(loader, cfg):
    """The invariant works, tested where it can actually fire.

    It is unreachable through any single proposal: bounds cap oversold at 40 and
    floor overbought at 60, so the minimum legal gap is 20 -- already wider than
    the invariant's 10. That is bounds + delta cap doing their job, and the
    invariant is a backstop for a future edit that widens those ranges.
    """
    from src.common.config_loader import validate_bounds
    bad = copy.deepcopy(cfg)
    bad["strategy"]["rsi"]["oversold"] = 45
    bad["strategy"]["rsi"]["overbought"] = 50
    errs = validate_bounds(bad, loader.bounds)
    assert any("rsi_band_ordering" in e for e in errs)


def test_rsi_band_cannot_be_inverted_by_any_single_legal_proposal(loader, cfg):
    """The complementary half: push both RSI bands as far toward each other as one
    cycle allows and the band is still well-ordered."""
    v = _v(loader)
    r = v.validate(_prop(("strategy.rsi.oversold", 30, 37.5),      # +25%, the cap
                         ("strategy.rsi.overbought", 70, 60.0)),   # -14%, hits the bound
                   cfg, loader.bounds)
    assert r.ok, r.errors
    c = r.candidate_config
    assert c["strategy"]["rsi"]["overbought"] > c["strategy"]["rsi"]["oversold"] + 10


# -- stale reasoning ----------------------------------------------------------

def test_wrong_old_value_is_rejected(loader, cfg):
    """If the model thinks sl is 2.0 when it is 1.5, its arithmetic was done
    against a config that does not exist -- the new value may be legal and still
    be the answer to the wrong question."""
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 2.0, 1.8)), cfg, loader.bounds)
    assert not r.ok
    assert any("stale config" in e for e in r.errors)


# -- degenerate payloads ------------------------------------------------------

def test_no_op_is_rejected(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.5)), cfg, loader.bounds)
    assert not r.ok
    assert any("no-op" in e for e in r.errors)


def test_empty_change_list_without_abstain_is_rejected(loader, cfg):
    assert not _v(loader).validate(_prop(), cfg, loader.bounds).ok


def test_abstain_with_changes_is_contradictory(loader, cfg):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8), abstain=True), cfg, loader.bounds)
    assert not r.ok
    assert any("contradictory" in e for e in r.errors)


def test_abstain_alone_yields_no_candidate(loader, cfg):
    r = _v(loader).validate(_prop(abstain=True), cfg, loader.bounds)
    assert not r.ok and r.candidate_config is None


@pytest.mark.parametrize("bad", ["1.8", None, True, [1.8], {"v": 1.8}])
def test_non_numeric_values_are_rejected(loader, cfg, bad):
    v = _v(loader)
    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, bad)), cfg, loader.bounds)
    assert not r.ok
    assert any("numeric" in e for e in r.errors)


# -- tool-input schema (check 1) ----------------------------------------------

def test_tool_input_schema_rejects_unknown_path(loader):
    v = _v(loader)
    raw = {"proposed_changes": [{"path": "risk.max_daily_loss_pct", "old": 3.0, "new": 50.0,
                                 "rationale": "x"}],
           "simulation_plan": "p", "abstain": False}
    assert v.validate_tool_input(raw, "strategy")


def test_tool_input_schema_rejects_too_many_changes(loader):
    v = _v(loader)
    raw = {"proposed_changes": [
        {"path": "risk.atr_mult_sl", "old": 1.5, "new": 1.6, "rationale": "x"},
        {"path": "risk.atr_mult_tp", "old": 2.5, "new": 2.6, "rationale": "x"},
        {"path": "strategy.entry_threshold", "old": 0.5, "new": 0.55, "rationale": "x"},
        {"path": "strategy.vwap_band_bps", "old": 15, "new": 16, "rationale": "x"},
    ], "simulation_plan": "p", "abstain": False}
    assert v.validate_tool_input(raw, "strategy")


def test_tool_input_schema_rejects_non_dict(loader):
    assert _v(loader).validate_tool_input("just a string", "strategy")
    assert _v(loader).validate_tool_input(None, "strategy")


def test_execution_scope_schema_hides_strategy_weights(loader):
    """HARD RULE 5 enforced structurally: while diagnosing a fills problem, the
    model cannot even name a strategy weight."""
    v = _v(loader)
    raw = {"proposed_changes": [{"path": "strategy.weights.rsi", "old": 0.3, "new": 0.4,
                                 "rationale": "x"}],
           "simulation_plan": "p", "abstain": False}
    assert v.validate_tool_input(raw, "execution")      # rejected under execution scope
    assert not v.validate_tool_input(raw, "strategy")   # allowed under strategy scope


def test_valid_tool_input_passes_schema(loader):
    raw = {"proposed_changes": [{"path": "risk.atr_mult_sl", "old": 1.5, "new": 1.8,
                                 "rationale": "wider stop"}],
           "expected_impact": {"kpi": "consecutive_stopouts", "direction": "down",
                               "mechanism": "fewer noise stop-outs"},
           "simulation_plan": "compare 30d", "abstain": False}
    assert _v(loader).validate_tool_input(raw, "strategy") == []


# -- burned configs -----------------------------------------------------------

def test_burned_config_is_rejected(loader, cfg, tmp_path):
    reg = BurnedRegistry(tmp_path / "burned.json", loader.bounds)
    v = _v(loader, burned=reg)

    r = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8)), cfg, loader.bounds)
    assert r.ok
    reg.burn(r.candidate_config, "shadow window failed", now_ms=1000)

    r2 = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8)), cfg, loader.bounds, now_ms=2000)
    assert not r2.ok
    assert any("burned" in e for e in r2.errors)


def test_burn_expires_after_a_week(loader, cfg, tmp_path):
    """Regimes change; a config that failed last week deserves another look."""
    reg = BurnedRegistry(tmp_path / "burned.json", loader.bounds)
    v = _v(loader, burned=reg)
    cand = v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8)), cfg, loader.bounds).candidate_config
    reg.burn(cand, "shadow failed", now_ms=0)

    day8 = 8 * 86_400_000
    assert v.validate(_prop(("risk.atr_mult_sl", 1.5, 1.8)), cfg, loader.bounds, now_ms=day8).ok


def test_burn_fingerprint_ignores_version_number(loader, cfg):
    """A version bump must not disguise an identical parameter set."""
    a = copy.deepcopy(cfg)
    b = copy.deepcopy(cfg)
    b["version"] = 99
    assert config_fingerprint(a, loader.bounds) == config_fingerprint(b, loader.bounds)

    c = copy.deepcopy(cfg)
    c["risk"]["atr_mult_sl"] = 1.8
    assert config_fingerprint(a, loader.bounds) != config_fingerprint(c, loader.bounds)
