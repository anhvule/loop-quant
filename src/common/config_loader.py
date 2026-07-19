"""Config loading, validation, and hot-reload.

This module is the chokepoint that makes the self-optimizing loop safe: *every*
path into a running config -- startup, hot-reload, and optimizer deploy -- goes
through `validate()`. Bounds are re-checked here on load, so even a config.json
that was corrupted by hand (or by a bug in the Deployer) cannot start the engine.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Callable

import jsonschema

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dotted-path helpers (shared with ProposalValidator)
# ---------------------------------------------------------------------------

_MISSING = object()


def get_path(doc: dict[str, Any], path: str) -> Any:
    """`get_path(cfg, 'risk.atr_mult_sl')` -> value, or _MISSING sentinel."""
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def has_path(doc: dict[str, Any], path: str) -> bool:
    return get_path(doc, path) is not _MISSING


def set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    """Mutates `doc` in place. Refuses to create new keys -- the optimizer may
    only move existing dials, never invent them."""
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"cannot set {path!r}: {part!r} does not exist")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"cannot set {path!r}: leaf does not exist")
    node[parts[-1]] = value


def flatten_paths(doc: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if k.startswith("_"):
            continue
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_paths(v, p + "."))
        else:
            out[p] = v
    return out


# ---------------------------------------------------------------------------
# safe invariant expression evaluation
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.Constant, ast.Name, ast.Attribute, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    ast.And, ast.Or, ast.USub, ast.Not,
)


def _dotted_name(node: ast.AST) -> str | None:
    """Rebuild `a.b.c` from the Attribute/Name chain the parser produces."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def eval_invariant(expr: str, cfg: dict[str, Any]) -> bool:
    """Evaluate a bounds.json invariant against a config.

    Deliberately NOT `eval()`: the node whitelist means a malformed or hostile
    bounds.json can only ever compute arithmetic over config values -- no calls,
    no imports, no attribute escapes.
    """
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax {type(node).__name__} in invariant: {expr!r}")

    def _ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = _dotted_name(node)
            if name is None:
                raise ValueError(f"unresolvable name in invariant: {expr!r}")
            val = get_path(cfg, name)
            if val is _MISSING:
                raise ValueError(f"invariant references unknown config path {name!r}")
            return val
        if isinstance(node, ast.UnaryOp):
            v = _ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            return not v
        if isinstance(node, ast.BinOp):
            l, r = _ev(node.left), _ev(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return l + r
            if isinstance(op, ast.Sub):
                return l - r
            if isinstance(op, ast.Mult):
                return l * r
            if isinstance(op, ast.Div):
                return l / r
        if isinstance(node, ast.BoolOp):
            vals = [_ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = _ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = _ev(comp)
                ok = (
                    left > right if isinstance(op, ast.Gt) else
                    left >= right if isinstance(op, ast.GtE) else
                    left < right if isinstance(op, ast.Lt) else
                    left <= right if isinstance(op, ast.LtE) else
                    left == right if isinstance(op, ast.Eq) else
                    left != right if isinstance(op, ast.NotEq) else
                    None
                )
                if ok is None:
                    raise ValueError(f"unsupported comparison in invariant: {expr!r}")
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(f"unsupported node {type(node).__name__} in invariant: {expr!r}")

    return bool(_ev(tree))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_schema(cfg: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft7Validator(schema)
    return [
        f"schema: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(cfg), key=lambda e: list(e.absolute_path))
    ]


def validate_bounds(cfg: dict[str, Any], bounds: dict[str, Any]) -> list[str]:
    """Absolute min/max per whitelisted parameter + cross-field invariants."""
    errors: list[str] = []
    for path, lim in bounds.get("params", {}).items():
        val = get_path(cfg, path)
        if val is _MISSING:
            errors.append(f"bounds: config is missing bounded parameter {path!r}")
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            errors.append(f"bounds: {path!r} must be numeric, got {type(val).__name__}")
            continue
        lo, hi = lim["min"], lim["max"]
        if not (lo <= val <= hi):
            errors.append(f"bounds: {path}={val} outside hard limit [{lo}, {hi}]")

    for inv in bounds.get("invariants", []):
        try:
            if not eval_invariant(inv["expr"], cfg):
                errors.append(f"invariant {inv['id']!r} violated ({inv['expr']}): {inv['message']}")
        except ValueError as e:
            errors.append(f"invariant {inv.get('id', '?')!r} could not be evaluated: {e}")
    return errors


def validate(cfg: dict[str, Any], schema: dict[str, Any], bounds: dict[str, Any]) -> list[str]:
    """Full gate. Schema first (structure), then bounds (safety). Bounds are only
    meaningful on a structurally valid doc, so short-circuit."""
    errs = validate_schema(cfg, schema)
    if errs:
        return errs
    return validate_bounds(cfg, bounds)


def normalized_weights(cfg: dict[str, Any]) -> dict[str, float]:
    """Strategy weights renormalized to sum 1.0.

    The optimizer proposes weights independently and they will rarely sum to 1;
    normalizing here means the score stays on [-1, 1] and `entry_threshold`
    keeps a stable meaning across configs.
    """
    w = cfg["strategy"]["weights"]
    total = float(w["vwap"]) + float(w["rsi"]) + float(w["macd"])
    if total <= 0:
        raise ValueError("strategy weights sum to zero; no signal could ever fire")
    return {k: float(w[k]) / total for k in ("vwap", "rsi", "macd")}


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    pass


class ConfigLoader:
    """Owns the in-memory current config and notifies listeners on hot-reload."""

    def __init__(self, config_path: Path, schema_path: Path, bounds_path: Path) -> None:
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.bounds_path = Path(bounds_path)
        self._cfg: dict[str, Any] | None = None
        self._schema: dict[str, Any] | None = None
        self._bounds: dict[str, Any] | None = None
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # -- static docs ------------------------------------------------------
    @property
    def schema(self) -> dict[str, Any]:
        if self._schema is None:
            self._schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        return self._schema

    @property
    def bounds(self) -> dict[str, Any]:
        if self._bounds is None:
            self._bounds = json.loads(self.bounds_path.read_text(encoding="utf-8"))
        return self._bounds

    # -- live config ------------------------------------------------------
    def load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ConfigError(f"config not found: {self.config_path}") from None
        except json.JSONDecodeError as e:
            raise ConfigError(f"config is not valid JSON: {e}") from None

        errors = validate(raw, self.schema, self.bounds)
        if errors:
            raise ConfigError(
                f"refusing to load {self.config_path.name} (v{raw.get('version', '?')}):\n  - "
                + "\n  - ".join(errors)
            )
        normalized_weights(raw)  # raises early on a degenerate weight vector
        self._cfg = raw
        return raw

    def current(self) -> dict[str, Any]:
        if self._cfg is None:
            return self.load()
        return self._cfg

    def reload(self) -> dict[str, Any]:
        """Re-read from disk. On failure the previous good config stays live --
        a bad deploy must never leave the engine config-less mid-position."""
        prev = self._cfg
        try:
            cfg = self.load()
        except ConfigError:
            self._cfg = prev
            raise
        log.info("config reloaded: version=%s", cfg["version"])
        for fn in self._listeners:
            try:
                fn(cfg)
            except Exception:
                log.exception("config listener failed")
        return cfg

    def on_reload(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    # -- convenience ------------------------------------------------------
    @property
    def version(self) -> int:
        return int(self.current()["version"])

    @property
    def symbol(self) -> str:
        return str(self.current()["symbol"])

    def get(self, path: str, default: Any = None) -> Any:
        v = get_path(self.current(), path)
        return default if v is _MISSING else v
