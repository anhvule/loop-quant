"""The optimization cycle: bundle -> diagnose -> propose -> validate -> sandbox -> deploy.

Every path through this function terminates in an audit record, including the
paths where nothing happens. "The optimizer looked and decided not to act" is a
result, and a system that only logs its actions makes its inaction invisible.

The single most important branch is the variance off-ramp in Phase A. A
self-optimizing trader's characteristic failure is not missing a real regime
change -- it is confidently fixing noise. Stopping the cycle when the evidence is
thin costs a delay; acting on it costs a real config change and a shadow window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.common.models import (
    Diagnosis, OptimizationProposal, UnderperformanceEvent, ValidationResult,
)
from src.optimizer import prompt_builder as pb
from src.optimizer.backtest_sandbox import BacktestSandbox, SandboxError
from src.optimizer.log_packager import LogPackager
from src.optimizer.proposal_validator import ProposalValidator

log = logging.getLogger(__name__)

TEMPERATURE = 0.2
MAX_PROPOSAL_ATTEMPTS = 2   # one shot + one retry with the validator's errors


class OptimizerCycle:
    def __init__(self, packager: LogPackager, llm, validator: ProposalValidator,
                 sandbox: BacktestSandbox, deployer, monitor, loader,
                 audit_path: Path, human_review_path: Path,
                 stress_windows: list[dict[str, Any]] | None = None) -> None:
        self.packager = packager
        self.llm = llm
        self.validator = validator
        self.sandbox = sandbox
        self.deployer = deployer
        self.monitor = monitor
        self.loader = loader
        self.audit_path = Path(audit_path)
        self.human_review_path = Path(human_review_path)
        self.stress_windows = stress_windows or []

    # -- main ---------------------------------------------------------------

    async def run(self, ev: UnderperformanceEvent) -> dict[str, Any]:
        now = ev.fired_at_ms or _now_ms()
        self.monitor.note_cycle_started(now)

        rec: dict[str, Any] = {
            "cycle_started_ms": now,
            "trigger": {"id": ev.trigger_id, "scope": ev.scope,
                        "condition": ev.condition_text, "halt": ev.halt_trading},
            "config_version_at_start": int(self.loader.current()["version"]),
            "outcome": "unknown",
        }

        try:
            bundle = self.packager.build_bundle(ev)
            rec["bundle_hash"] = _hash(bundle)
            rec["bundle_summary"] = _bundle_summary(bundle)

            # ---- Phase A: diagnosis ----
            diag_raw, diag = await self._diagnose(bundle, rec)
            if diag is None:
                return self._finish(rec, "phase_a_failed")

            if diag.failure_class == "variance":
                # The trigger was real; the evidence was not. Back off and let more
                # data accumulate rather than tuning against luck.
                self.monitor.extend_cooldown(2.0)
                rec["decision"] = ("no action: diagnosed as variance; trigger cooldown "
                                   "extended 2x")
                log.warning("optimizer: variance verdict, no action. %s", diag.diagnosis[:200])
                return self._finish(rec, "no_action_variance")

            # ---- Phase B: proposal ----
            proposal, vres = await self._propose(bundle, diag_raw, ev.scope, rec)
            if proposal is not None and proposal.abstain:
                rec["decision"] = "no action: model abstained"
                log.warning("optimizer: model abstained. plan=%s", proposal.simulation_plan[:300])
                return self._finish(rec, "no_action_abstained")
            if vres is None or not vres.ok or vres.candidate_config is None:
                return self._finish(rec, "rejected_validation")

            # ---- sandbox ----
            current = self.loader.current()
            try:
                verdict = self.sandbox.run(vres.candidate_config, current, now_ms=now,
                                           stress_windows=self.stress_windows)
            except SandboxError as e:
                rec["sandbox_error"] = str(e)
                log.error("sandbox failed: %s", e)
                return self._finish(rec, "sandbox_error")

            rec["sandbox_verdict"] = verdict.to_dict()
            if not verdict.passed:
                rec["decision"] = f"rejected by sandbox: {'; '.join(verdict.reasons)}"
                return self._finish(rec, "rejected_sandbox")

            # ---- deploy ----
            new_version = self.deployer.deploy(vres.candidate_config, verdict, now_ms=now)
            self.monitor.note_deploy(now)
            rec["deployed_version"] = new_version
            rec["decision"] = f"deployed v{new_version}"
            return self._finish(rec, "deployed")

        except Exception as e:
            log.exception("optimizer cycle crashed")
            rec["error"] = f"{type(e).__name__}: {e}"
            return self._finish(rec, "error")
        finally:
            self.monitor.note_cycle_finished()

    # -- phases -------------------------------------------------------------

    async def _diagnose(self, bundle: dict[str, Any],
                        rec: dict[str, Any]) -> tuple[dict[str, Any] | None, Diagnosis | None]:
        tool = pb.diagnosis_tool()
        messages = pb.render_diagnosis_messages(bundle)
        rec["phase_a"] = {"system": pb.DIAGNOSIS_SYSTEM, "tool": tool["name"],
                          "temperature": TEMPERATURE}
        try:
            resp = await self.llm.call_tool(pb.DIAGNOSIS_SYSTEM, messages, tool, TEMPERATURE)
        except Exception as e:
            rec["phase_a"]["error"] = f"{type(e).__name__}: {e}"
            log.error("phase A call failed: %s", e)
            return None, None

        rec["phase_a"]["raw_response"] = resp.tool_input
        rec["phase_a"]["usage"] = resp.usage
        try:
            diag = Diagnosis.from_tool_input(resp.tool_input)
        except Exception as e:
            rec["phase_a"]["error"] = f"malformed diagnosis: {e}"
            log.error("phase A returned a malformed diagnosis: %s", e)
            return resp.tool_input, None

        log.info("phase A: failure_class=%s -- %s", diag.failure_class, diag.diagnosis[:200])
        return resp.tool_input, diag

    async def _propose(self, bundle: dict[str, Any], diag_raw: dict[str, Any], scope: str,
                       rec: dict[str, Any]) -> tuple[OptimizationProposal | None,
                                                     ValidationResult | None]:
        tool = pb.proposal_tool(self.validator.bounds, scope)
        current = self.loader.current()
        errors: list[str] | None = None
        attempts: list[dict[str, Any]] = []
        rec["phase_b"] = {"system": pb.PROPOSAL_SYSTEM, "tool": tool["name"],
                          "scope": scope, "temperature": TEMPERATURE, "attempts": attempts}

        proposal: OptimizationProposal | None = None
        vres: ValidationResult | None = None

        for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
            messages = pb.render_proposal_messages(bundle, diag_raw, errors)
            a: dict[str, Any] = {"attempt": attempt}
            attempts.append(a)
            try:
                resp = await self.llm.call_tool(pb.PROPOSAL_SYSTEM, messages, tool, TEMPERATURE)
            except Exception as e:
                a["error"] = f"{type(e).__name__}: {e}"
                log.error("phase B attempt %d call failed: %s", attempt, e)
                errors = [f"the previous response could not be parsed: {e}"]
                continue

            a["raw_response"] = resp.tool_input
            a["usage"] = resp.usage

            # Re-validate the payload shape locally; never trust that the API
            # enforced the schema.
            schema_errors = self.validator.validate_tool_input(resp.tool_input, scope)
            if schema_errors:
                a["errors"] = schema_errors
                errors = schema_errors
                log.warning("phase B attempt %d: schema errors %s", attempt, schema_errors)
                continue

            try:
                proposal = OptimizationProposal.from_tool_input(resp.tool_input)
            except Exception as e:
                a["errors"] = [f"malformed proposal: {e}"]
                errors = a["errors"]
                continue

            if proposal.abstain:
                a["abstained"] = True
                return proposal, None

            vres = self.validator.validate(proposal, current, self.validator.bounds)
            a["validation"] = {"ok": vres.ok, "errors": vres.errors}
            if vres.ok:
                a["candidate_diff"] = [
                    {"path": c.path, "old": c.old, "new": c.new} for c in proposal.proposed_changes]
                log.info("phase B attempt %d validated: %s", attempt, a["candidate_diff"])
                return proposal, vres

            log.warning("phase B attempt %d rejected: %s", attempt, vres.error_text)
            errors = vres.errors

        # Exhausted attempts. If the model kept reaching for things outside the
        # whitelist, that is a signal worth a human's attention -- it may be right
        # that the fix lies outside the tunable set.
        #
        # Detect this from the RAW RESPONSES rather than from error text: a
        # forbidden path is caught by whichever layer sees it first (the tool
        # schema's enum, usually) and each layer words its rejection differently.
        # Matching on message strings would silently stop working the moment the
        # order of the checks changed.
        if self._reached_outside_whitelist(attempts):
            self._queue_for_human(rec, errors or [])
        rec["decision"] = f"rejected by validator after {MAX_PROPOSAL_ATTEMPTS} attempts"
        return proposal, vres

    def _reached_outside_whitelist(self, attempts: list[dict[str, Any]]) -> bool:
        allowed = set(self.validator.bounds.get("params", {}).keys())
        for a in attempts:
            raw = a.get("raw_response")
            if not isinstance(raw, dict):
                continue
            for ch in raw.get("proposed_changes") or []:
                if isinstance(ch, dict) and ch.get("path") not in allowed:
                    return True
        return False

    # -- sinks ---------------------------------------------------------------

    def _queue_for_human(self, rec: dict[str, Any], errors: list[str]) -> None:
        """Never auto-apply anything beyond the whitelist. Park it for a person.

        Worth reading these: the model repeatedly asking for a parameter it is not
        allowed to touch is weak evidence that the real fix lives outside the
        tunable set. That is a design question, not a tuning one.
        """
        entry = {
            "ts_ms": _now_ms(),
            "reason": "proposal reached for parameters outside the tunable whitelist",
            "requested_paths": sorted({
                ch.get("path") for a in rec.get("phase_b", {}).get("attempts", [])
                if isinstance(a.get("raw_response"), dict)
                for ch in (a["raw_response"].get("proposed_changes") or [])
                if isinstance(ch, dict)
            } - set(self.validator.bounds.get("params", {}).keys())),
            "errors": errors,
            "trigger": rec.get("trigger"),
            "phase_b_attempts": rec.get("phase_b", {}).get("attempts"),
        }
        _append_jsonl(self.human_review_path, entry)
        rec["queued_for_human_review"] = True
        log.warning("proposal queued for human review: %s", errors)

    def _finish(self, rec: dict[str, Any], outcome: str) -> dict[str, Any]:
        rec["outcome"] = outcome
        rec["cycle_finished_ms"] = _now_ms()
        _append_jsonl(self.audit_path, rec)
        log.info("optimizer cycle finished: %s", outcome)
        return rec


# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _bundle_summary(b: dict[str, Any]) -> dict[str, Any]:
    """A compact echo of the inputs, so an audit line is readable without
    re-deriving the whole bundle."""
    return {
        "kpi": b.get("kpi_report"),
        "baseline": b.get("baseline_kpis"),
        "regime": b.get("market_regime"),
        "economics": (b.get("economics") or {}).get("trade_economics"),
        "sizing": (b.get("economics") or {}).get("sizing_constraint"),
        "n_recent_trades": len(b.get("recent_trades") or []),
        "config_history_versions": [h["version"] for h in (b.get("config_history") or [])],
    }


def _now_ms() -> int:
    return int(time.time() * 1000)
