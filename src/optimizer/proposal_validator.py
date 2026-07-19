"""Deterministic gate. Zero trust in LLM output.

The operating assumption here is not that the model is malicious -- it is that the
model is *fallible in ways that are expensive*, and that the difference between a
research toy and a system you would leave running is whether something mechanical
checks its work.

Checks run in the blueprint's order:
  1. parses against the tool JSON schema
  2. every path exists in the current config (no new keys, no deletions)
  3. the RESULTING config validates against config.schema.json
  4. each new value sits inside bounds.json's absolute min/max
  5. relative delta per parameter <= max_delta_pct_per_cycle
  6. at most `max_params_per_cycle` changes
  7. forbidden paths untouched (anything not whitelisted in bounds.params)

Plus three that experience says you need:
  * `old` must match reality -- a mismatch means the model reasoned about a stale
    config, which invalidates its arithmetic even if the new value is legal
  * no-ops are rejected -- they burn a deploy cycle and a shadow window for nothing
  * burned configs are rejected -- see burned.py
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import jsonschema

from src.common.config_loader import get_path, set_path, validate as validate_config
from src.common.models import OptimizationProposal, ValidationResult
from src.optimizer.burned import BurnedRegistry
from src.optimizer.prompt_builder import proposal_tool

log = logging.getLogger(__name__)

_MISSING = object()


class ProposalValidator:
    def __init__(self, schema: dict[str, Any], bounds: dict[str, Any],
                 burned: BurnedRegistry | None = None) -> None:
        self.schema = schema
        self.bounds = bounds
        self.burned = burned

    # -- step 1: the tool payload itself -----------------------------------

    def validate_tool_input(self, raw: Any, scope: str = "strategy") -> list[str]:
        """Re-validate against the tool schema locally. The API enforces this too,
        but a mock client, an SDK change, or a future non-forced call must not be
        able to slip an unshaped payload past us."""
        if not isinstance(raw, dict):
            return [f"tool input is {type(raw).__name__}, expected an object"]
        tool = proposal_tool(self.bounds, scope)
        v = jsonschema.Draft7Validator(tool["input_schema"])
        return [f"tool schema: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in sorted(v.iter_errors(raw), key=lambda e: list(e.absolute_path))]

    # -- steps 2-7 ---------------------------------------------------------

    def validate(self, p: OptimizationProposal, current: dict[str, Any],
                 bounds: dict[str, Any], now_ms: int | None = None) -> ValidationResult:
        errors: list[str] = []
        changes = p.proposed_changes

        if p.abstain:
            if changes:
                errors.append("abstain=true but proposed_changes is non-empty; contradictory")
            else:
                errors.append("proposal abstained; no config change to validate")
            return ValidationResult(False, errors, None)

        if not changes:
            return ValidationResult(False, ["no changes proposed and abstain is false"], None)

        # (6) count
        max_params = int(bounds.get("max_params_per_cycle", 3))
        if len(changes) > max_params:
            errors.append(f"{len(changes)} changes proposed; max is {max_params}")

        # duplicate paths -- last-write-wins would make the change count a lie
        paths = [c.path for c in changes]
        dupes = {p_ for p_ in paths if paths.count(p_) > 1}
        if dupes:
            errors.append(f"duplicate paths in proposal: {sorted(dupes)}")

        allowed = set(bounds.get("params", {}).keys())
        max_delta = float(bounds.get("max_delta_pct_per_cycle", 25))

        for ch in changes:
            # (7) whitelist. Anything not in bounds.params is immutable BY
            # CONSTRUCTION -- including max_daily_loss_pct, max_position_pct_equity,
            # max_open_positions, symbol, version, and every indicator period.
            if ch.path not in allowed:
                errors.append(f"{ch.path!r} is not a tunable parameter; refusing")
                continue

            # (2) must already exist
            cur = get_path(current, ch.path)
            if cur is _MISSING:
                errors.append(f"{ch.path!r} does not exist in the current config")
                continue

            # types
            if isinstance(ch.new, bool) or not isinstance(ch.new, (int, float)):
                errors.append(f"{ch.path}: new value must be numeric, got {ch.new!r}")
                continue
            if not isinstance(cur, (int, float)):
                errors.append(f"{ch.path}: current value is not numeric ({cur!r})")
                continue

            # `old` must match reality
            if ch.old is not None and isinstance(ch.old, (int, float)):
                if abs(float(ch.old) - float(cur)) > 1e-9:
                    errors.append(
                        f"{ch.path}: proposal says the current value is {ch.old} but it is "
                        f"{cur}; the model reasoned about a stale config"
                    )
                    continue

            # (5) per-cycle delta cap
            lim = bounds["params"][ch.path]
            lo, hi = float(lim["min"]), float(lim["max"])
            if abs(float(cur)) > 1e-9:
                delta_pct = abs(float(ch.new) - float(cur)) / abs(float(cur)) * 100.0
                if delta_pct > max_delta + 1e-9:
                    errors.append(
                        f"{ch.path}: {cur} -> {ch.new} is a {delta_pct:.1f}% move; "
                        f"max is {max_delta}% per cycle"
                    )
            else:
                # A relative cap is undefined at zero, so fall back to a fraction
                # of the parameter's own legal span.
                span = hi - lo
                if span > 0 and abs(float(ch.new)) > span * max_delta / 100.0 + 1e-9:
                    errors.append(
                        f"{ch.path}: current value is 0; a move to {ch.new} exceeds "
                        f"{max_delta}% of the bound span ({span})"
                    )

        if errors:
            return ValidationResult(False, errors, None)

        # Build the candidate. set_path refuses to create keys, so a path that
        # slipped through above still cannot introduce structure.
        candidate = copy.deepcopy(current)
        try:
            for ch in changes:
                set_path(candidate, ch.path, ch.new)
        except KeyError as e:
            return ValidationResult(False, [f"failed to apply change: {e}"], None)

        # (3) + (4) + invariants, via the exact same gate the engine uses at load.
        cfg_errors = validate_config(candidate, self.schema, bounds)
        if cfg_errors:
            return ValidationResult(False, cfg_errors, None)

        if candidate == current:
            return ValidationResult(False, ["proposal is a no-op"], None)

        if self.burned is not None:
            b = self.burned.is_burned(candidate, now_ms)
            if b:
                return ValidationResult(
                    False,
                    [f"this exact parameter set was burned at {b['burned_ms']} "
                     f"({b['reason']}); it stays rejected until {b['expires_ms']}"],
                    None,
                )

        return ValidationResult(True, [], candidate)
