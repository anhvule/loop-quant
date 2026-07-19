"""Two-phase prompt architecture and the tool schemas that constrain the output.

The central safety idea: **the LLM never emits code, only data**, and the shape of
that data is pinned by a forced tool schema rather than by asking nicely in prose.

The schemas do real work beyond validation:
  * `path` is an ENUM built from bounds.json, so a parameter outside the whitelist
    is not merely rejected downstream -- it is unrepresentable in the output.
  * When the trigger is execution-scoped, that enum is filtered further, so the
    model cannot reach strategy weights at all while diagnosing a fills problem.
  * `new` is typed `number`, so a string or an expression cannot arrive.
  * `maxItems: 3` encodes the change-count cap structurally.

ProposalValidator re-checks every one of these anyway. The schema makes the right
answer easy; the validator makes the wrong answer impossible.
"""

from __future__ import annotations

from typing import Any

# --- Phase A -----------------------------------------------------------------

DIAGNOSIS_SYSTEM = """\
You are a quantitative strategy analyst. You will receive a performance bundle \
from a live paper-trading system. Diagnose WHY the strategy underperformed.

Ground every claim in the numbers provided; cite the specific bundle fields you \
relied on. Distinguish clearly between:
  * strategy failure -- the signal is wrong for this market regime
  * execution failure -- the signal is fine but fills are bad (slippage, spread)
  * variance -- the sample is too small to conclude anything
  * mixed -- more than one of the above

The bundle contains a precomputed `economics` section with the deterministic \
breakeven arithmetic. Use it. In particular:
  * If `economics.trade_economics.viable` is false, the configuration cannot be \
profitable at the current volatility no matter what the win rate is, because the \
round-trip fee is large relative to the stop distance. Say so plainly.
  * If `economics.sizing_constraint.risk_dial_is_live` is false, then \
risk.risk_per_trade_pct currently has NO effect on results and must not be \
treated as a lever.

You may include simple mathematical reasoning (expectancy math, the breakeven win \
rate implied by the current R:R = atr_mult_tp/atr_mult_sl, fee drag per round trip).

Be conservative about `variance`. If the trigger fired on a small sample, or the \
observed effect is within the range that ordinary luck would produce over this \
many trades, classify it as variance -- that verdict stops the cycle and costs \
nothing but a delay. A wrong 'strategy' verdict causes a real config change.

Do NOT propose parameter values in this phase. Diagnose only.\
"""

# --- Phase B -----------------------------------------------------------------

PROPOSAL_SYSTEM = """\
You are tuning the configuration of a live paper-trading system in response to a \
diagnosis you just produced. Propose the MINIMAL parameter change that addresses it.

HARD RULES (violations are rejected automatically):
  1. Only keys present in the provided config may change. No new keys, no deletions.
  2. Every value must lie inside the provided bounds.
  3. Change at most 3 parameters. Fewer is better.
  4. No parameter may move more than `bounds.max_delta_pct_per_cycle` percent of \
its current value in one cycle.
  5. If the diagnosis is execution-scoped, touch only execution/risk parameters -- \
never strategy weights.
  6. Respect the invariants listed in `bounds.invariants` (e.g. atr_mult_tp must \
stay at least 1.2x atr_mult_sl).

For each change, give the quantitative rationale -- for example the new R:R and the \
breakeven win rate it implies, compared against the observed win rate. "It might \
help" is not a rationale.

Read `config_history` before proposing. It shows the last few configurations AND \
what each actually produced. If a change of this kind was already tried and did not \
work, do not propose it again; say so and try something else, or abstain.

If `economics.sizing_constraint.risk_dial_is_live` is false, do NOT propose a change \
to risk.risk_per_trade_pct: it is currently wired to nothing and the change would be \
a no-op that wastes a deployment cycle.

Also emit a `simulation_plan`: the specific backtest comparison that would confirm \
or refute your change.

You may set `abstain: true` with an empty change list. Abstaining is the correct \
answer when no in-bounds change would address the diagnosis -- for example when the \
economics are structurally unviable and the fix lies outside the tunable set \
(timeframe, fee tier, instrument). Say so in `simulation_plan`. An honest abstention \
is worth far more than a plausible-looking change to a parameter that cannot help.\
"""


def diagnosis_tool() -> dict[str, Any]:
    return {
        "name": "submit_diagnosis",
        "description": "Record the diagnosis of why the strategy underperformed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "The reasoning, grounded in specific bundle fields.",
                },
                "failure_class": {
                    "type": "string",
                    "enum": ["strategy", "execution", "variance", "mixed"],
                },
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "supporting_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Dotted bundle paths, e.g. 'kpi_report.win_rate'.",
                            },
                        },
                        "required": ["claim", "supporting_fields"],
                    },
                },
                "breakeven_analysis": {
                    "type": "object",
                    "properties": {
                        "rr_ratio": {"type": "number"},
                        "required_win_rate": {
                            "type": "number",
                            "description": "Fraction in [0,1], net of fees.",
                        },
                        "actual_win_rate": {"type": "number"},
                    },
                    "required": ["rr_ratio", "required_win_rate", "actual_win_rate"],
                },
            },
            "required": ["diagnosis", "failure_class", "evidence"],
        },
    }


def proposal_tool(bounds: dict[str, Any], scope: str = "strategy") -> dict[str, Any]:
    """Build the Phase B tool schema, with the tunable-path enum baked in.

    Filtering the enum by scope is a structural enforcement of HARD RULE 5: when
    the trigger is about slippage, strategy weights are not merely discouraged --
    they are absent from the schema.
    """
    paths = sorted(bounds.get("params", {}).keys())
    if scope == "execution":
        paths = [p for p in paths if p.startswith(("execution.", "risk."))]

    max_params = int(bounds.get("max_params_per_cycle", 3))
    return {
        "name": "submit_proposal",
        "description": "Propose bounded parameter changes to config.json.",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposed_changes": {
                    "type": "array",
                    "maxItems": max_params,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "enum": paths,
                                "description": "Dotted config path. Only these may change.",
                            },
                            "old": {"type": "number", "description": "Current value."},
                            "new": {"type": "number", "description": "Proposed value."},
                            "rationale": {
                                "type": "string",
                                "description": "Quantitative justification, not a vibe.",
                            },
                        },
                        "required": ["path", "old", "new", "rationale"],
                    },
                },
                "expected_impact": {
                    "type": "object",
                    "properties": {
                        "kpi": {"type": "string"},
                        "direction": {"type": "string", "enum": ["up", "down"]},
                        "mechanism": {"type": "string"},
                    },
                    "required": ["kpi", "direction", "mechanism"],
                },
                "simulation_plan": {
                    "type": "string",
                    "description": "The backtest comparison that would confirm or refute this.",
                },
                "abstain": {
                    "type": "boolean",
                    "description": "True = recommend no change. Correct when nothing in bounds helps.",
                },
            },
            "required": ["proposed_changes", "simulation_plan", "abstain"],
        },
    }


def render_diagnosis_messages(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    import json
    return [{
        "role": "user",
        "content": (
            "Here is the performance bundle from the live system. Diagnose why it "
            "underperformed, then call submit_diagnosis.\n\n<bundle>\n"
            + json.dumps(bundle, indent=2, default=str)
            + "\n</bundle>"
        ),
    }]


def render_proposal_messages(bundle: dict[str, Any], diagnosis: dict[str, Any],
                             validation_errors: list[str] | None = None) -> list[dict[str, Any]]:
    import json
    text = (
        "Bundle:\n<bundle>\n" + json.dumps(bundle, indent=2, default=str) + "\n</bundle>\n\n"
        "Your Phase A diagnosis:\n<diagnosis>\n" + json.dumps(diagnosis, indent=2, default=str)
        + "\n</diagnosis>\n\n"
        "Propose the minimal in-bounds config change that addresses this diagnosis, "
        "then call submit_proposal."
    )
    if validation_errors:
        # The retry path: tell it exactly what was wrong rather than letting it
        # guess. One retry only -- see OptimizerCycle.
        text += (
            "\n\nYOUR PREVIOUS PROPOSAL WAS REJECTED by the deterministic validator "
            "for these reasons:\n  - " + "\n  - ".join(validation_errors)
            + "\n\nProduce a corrected proposal that satisfies every hard rule, or "
              "abstain if no valid change addresses the diagnosis."
        )
    return [{"role": "user", "content": text}]
