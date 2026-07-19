"""End-to-end drill for the self-optimization loop. No market, no API key, no cost.

Proves the whole Module 4 chain against REAL candle data using a MockLLMClient:

    T1 fires -> bundle built -> proposal validated -> sandbox runs -> v2 deployed
             -> shadow tracking opens -> shadow KPIs come in bad -> auto-rollback to v1

and the rejection paths that matter more than the happy path:

    out-of-bounds proposal   -> rejected, retried, abstained
    malformed tool payload   -> rejected by the local schema check
    variance diagnosis       -> cycle stops, cooldown doubles, nothing deploys

Runs entirely in a temp directory: it copies the candle table and the config docs,
so the drill can never touch the real config.json or the live database.

    python -m scripts.optimizer_drill
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import ConfigLoader                    # noqa: E402
from src.common.db import DB                                        # noqa: E402
from src.common.event_bus import EventBus                           # noqa: E402
from src.common.models import ClosedTrade                           # noqa: E402
from src.common.paths import BOUNDS_PATH, CONFIG_PATH, DB_PATH, SCHEMA_PATH  # noqa: E402
from src.evaluation.baseline import generate_baseline, save_baseline  # noqa: E402
from src.evaluation.kpi_tracker import KPITracker                   # noqa: E402
from src.evaluation.trigger_monitor import TRIGGERS, TriggerMonitor  # noqa: E402
from src.optimizer.backtest_sandbox import BacktestSandbox          # noqa: E402
from src.optimizer.burned import BurnedRegistry                     # noqa: E402
from src.optimizer.cycle import OptimizerCycle                      # noqa: E402
from src.optimizer.deployer import Deployer                         # noqa: E402
from src.optimizer.llm_client import MockLLMClient                  # noqa: E402
from src.optimizer.log_packager import LogPackager                  # noqa: E402
from src.optimizer.proposal_validator import ProposalValidator      # noqa: E402

MS_HOUR = 3_600_000
INITIAL_EQUITY = 10_000.0


def banner(s: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {s}")
    print("=" * 78)


def step(s: str) -> None:
    print(f"\n--- {s}")


class Drill:
    _open: list["Drill"] = []

    @classmethod
    def close_all(cls) -> None:
        """Release every sqlite handle before the temp dir is torn down; Windows
        refuses to unlink an open file."""
        for d in cls._open:
            try:
                d.db.close()
            except Exception:
                pass
        cls._open.clear()

    def __init__(self, workdir: Path) -> None:
        Drill._open.append(self)
        self.work = workdir
        self.cfg_dir = workdir / "config"
        self.logs = workdir / "logs"
        self.cfg_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg_dir / "versions").mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)

        # isolate: real config.json and loopquant.db must be untouchable
        shutil.copy(CONFIG_PATH, self.cfg_dir / "config.json")
        shutil.copy(SCHEMA_PATH, self.cfg_dir / "config.schema.json")
        shutil.copy(BOUNDS_PATH, self.cfg_dir / "bounds.json")

        self.db_path = workdir / "drill.db"
        self.db = DB(self.db_path)
        self.loader = ConfigLoader(self.cfg_dir / "config.json",
                                   self.cfg_dir / "config.schema.json",
                                   self.cfg_dir / "bounds.json")
        self.cfg = self.loader.load()
        self.bus = EventBus()
        self.audit = self.logs / "optimizer_audit.jsonl"
        self.human = self.logs / "human_review_queue.jsonl"

    def copy_candles(self) -> int:
        src = DB(DB_PATH)
        try:
            candles = src.get_candles(self.cfg["symbol"], self.cfg["timeframe"])
        finally:
            src.close()
        if not candles:
            raise SystemExit("no candles in the main DB. Run: python -m scripts.seed_data --days 30")
        self.db.upsert_candles(candles)
        self.now = candles[-1].ts_open_ms + 60_000
        return len(candles)

    def make_baseline(self):
        b = generate_baseline(self.cfg, self.db, now_ms=self.now)
        save_baseline(b, self.cfg_dir / "baseline.json")
        return b

    # -- synthetic live history -------------------------------------------

    def inject_trades(self, n_wins: int, n_stopouts: int, version: int = 1,
                      start_ms: int | None = None, pnl_pct: float = -0.2,
                      prefix: str = "d") -> None:
        """Fabricate live history: `n_wins` ordinary trades, then a run of
        `n_stopouts` stop-outs so T1's condition is met at the most recent trade."""
        t0 = start_ms if start_ms is not None else self.now - 30 * MS_HOUR
        i = 0
        for _ in range(n_wins):
            self._trade(f"{prefix}-w{i}", t0 + i * 60_000, "tp", 0.15, version)
            i += 1
        for _ in range(n_stopouts):
            self._trade(f"{prefix}-s{i}", t0 + i * 60_000, "sl", pnl_pct, version)
            i += 1

    def _trade(self, tid: str, ts: int, reason: str, pnl_pct: float, version: int) -> None:
        entry, qty = 60_000.0, 0.01
        notional = entry * qty
        self.db.insert_closed_trade(ClosedTrade(
            trade_id=tid, symbol=self.cfg["symbol"], side="BUY",
            entry_ts_ms=ts - 60_000, exit_ts_ms=ts, entry_px=entry,
            exit_px=entry * (1 + pnl_pct / 100), qty=qty,
            pnl_quote=notional * pnl_pct / 100, pnl_pct=pnl_pct,
            exit_reason=reason, slippage_bps=1.0, config_version=version,
            atr_at_entry=60.0))

    # -- wiring ------------------------------------------------------------

    def build(self, baseline, mock: MockLLMClient):
        self.db.insert_config_version(1, self.now - 48 * MS_HOUR,
                                      json.dumps(self.cfg), "initial", None)
        self.kpi = KPITracker(self.db, self.bus, INITIAL_EQUITY, self.loader.current, baseline)
        self.monitor = TriggerMonitor(baseline, TRIGGERS, self.db, self.loader.current,
                                      last_deploy_ms=self.now - 48 * MS_HOUR)
        self.burned = BurnedRegistry(self.cfg_dir / "burned.json", self.loader.bounds)
        self.deployer = Deployer(self.cfg_dir / "config.json", self.cfg_dir / "versions",
                                 self.cfg_dir / "baseline.json", self.db, self.bus,
                                 self.loader, self.burned)
        self.cycle = OptimizerCycle(
            packager=LogPackager(self.db, self.loader.current, self.loader.bounds),
            llm=mock,
            validator=ProposalValidator(self.loader.schema, self.loader.bounds, self.burned),
            sandbox=BacktestSandbox(self.db_path, ROOT),
            deployer=self.deployer, monitor=self.monitor, loader=self.loader,
            audit_path=self.audit, human_review_path=self.human, stress_windows=[],
        )


# -- canned LLM payloads ------------------------------------------------------

DIAG_STRATEGY = {
    "diagnosis": ("Stops are being hit by ordinary noise: atr_mult_sl of 1.5 puts the stop "
                  "roughly 0.15% from entry while the round-trip fee alone is 0.20%. Widening "
                  "the stop is the only in-bounds lever that changes the geometry."),
    "failure_class": "strategy",
    "evidence": [
        {"claim": "three consecutive stop-outs", "supporting_fields": ["kpi_report.consecutive_stopouts"]},
        {"claim": "breakeven win rate is unreachable",
         "supporting_fields": ["economics.trade_economics.breakeven_win_rate"]},
    ],
    "breakeven_analysis": {"rr_ratio": 1.667, "required_win_rate": 0.875, "actual_win_rate": 0.15},
}

DIAG_VARIANCE = {
    "diagnosis": "Three stop-outs in a row is unremarkable at a 15% win rate; p ~ 0.61.",
    "failure_class": "variance",
    "evidence": [{"claim": "sample too small", "supporting_fields": ["kpi_report.window_trades"]}],
}

PROP_VALID = {
    "proposed_changes": [
        {"path": "risk.atr_mult_sl", "old": 1.5, "new": 1.875,
         "rationale": "+25% (the per-cycle cap) widens the stop to ~0.19% of price, "
                      "moving breakeven from 87.5% toward 80%."},
        {"path": "risk.atr_mult_tp", "old": 2.5, "new": 3.125,
         "rationale": "holds R:R at 1.667 so the rr_floor invariant stays satisfied."},
    ],
    "expected_impact": {"kpi": "consecutive_stopouts", "direction": "down",
                        "mechanism": "a wider stop is outside the 1m noise band"},
    "simulation_plan": "Compare 30d expectancy and stop-out rate vs the incumbent.",
    "abstain": False,
}

PROP_OUT_OF_BOUNDS = {
    "proposed_changes": [
        {"path": "risk.atr_mult_sl", "old": 1.5, "new": 8.0,
         "rationale": "a much wider stop would stop getting hit"},
    ],
    "expected_impact": {"kpi": "consecutive_stopouts", "direction": "down", "mechanism": "x"},
    "simulation_plan": "x",
    "abstain": False,
}

PROP_ABSTAIN = {
    "proposed_changes": [],
    "simulation_plan": ("No in-bounds change fixes fee-dominated economics at 1m. "
                        "This needs a longer timeframe or a lower fee tier."),
    "abstain": True,
}

PROP_FORBIDDEN = {
    "proposed_changes": [
        {"path": "risk.max_daily_loss_pct", "old": 3.0, "new": 20.0,
         "rationale": "the daily loss halt keeps stopping us"},
    ],
    "simulation_plan": "x",
    "abstain": False,
}


async def scenario_happy_path_then_rollback(work: Path) -> bool:
    banner("SCENARIO 1: T1 fires -> deploy v2 -> shadow fails -> auto-rollback")
    d = Drill(work / "s1")
    print(f"copied {d.copy_candles()} real candles")

    step("generate baseline from the CURRENT config over 30d of real BTC data")
    b = d.make_baseline()
    print(f"    baseline v{b.config_version}: {b.n_trades} trades, "
          f"expectancy={b.expectancy:.4f}%, sharpe={b.sharpe:.2f}, maxDD={b.max_drawdown_pct:.2f}%")

    step("inject live history: 20 trades, ending in 3 consecutive stop-outs")
    d.inject_trades(n_wins=17, n_stopouts=3)

    mock = MockLLMClient()
    mock.queue("submit_diagnosis", DIAG_STRATEGY)
    mock.queue("submit_proposal", PROP_VALID)
    d.build(b, mock)

    step("compute KPIs and evaluate triggers")
    r = d.kpi.compute(d.now)
    print(f"    KPI: n={r.window_trades} stopouts={r.consecutive_stopouts} "
          f"expectancy={r.expectancy:.4f}% dev={r.bt_deviation_pct}")
    ev = d.monitor.evaluate(r)
    assert ev is not None and ev.trigger_id == "T1", f"expected T1, got {ev}"
    print(f"    TRIGGER {ev.trigger_id} fired: {ev.condition_text}")

    step("run the optimizer cycle (mock LLM, real validator, real sandbox subprocess)")
    rec = await d.cycle.run(ev)
    print(f"    outcome: {rec['outcome']}")
    print(f"    decision: {rec.get('decision')}")
    if rec.get("sandbox_verdict"):
        v = rec["sandbox_verdict"]
        print(f"    sandbox passed={v['passed']}")
        if v["candidate"]:
            print(f"      candidate : trades={v['candidate']['n_trades']} "
                  f"expectancy={v['candidate']['expectancy']:.4f}% "
                  f"maxDD={v['candidate']['max_drawdown_pct']:.2f}%")
            print(f"      incumbent : trades={v['incumbent']['n_trades']} "
                  f"expectancy={v['incumbent']['expectancy']:.4f}% "
                  f"maxDD={v['incumbent']['max_drawdown_pct']:.2f}%")
        for reason in v["reasons"]:
            print(f"      reject: {reason}")

    if rec["outcome"] != "deployed":
        print("\n    >> The sandbox REFUSED the deploy, on real market data. This is not a\n"
              "       drill artifact -- it is the gate doing its job. Widening the stop made\n"
              "       expectancy WORSE, and the candidate did not earn its deployment.\n"
              "       (`python -m scripts.analyze_config_space` shows that ZERO legal\n"
              "        single-cycle moves beat the incumbent on this dataset.)\n"
              "\n"
              "       The shadow/rollback machinery still needs exercising, so the deploy is\n"
              "       forced below with a SYNTHETIC passing verdict. Everything after this\n"
              "       point -- Deployer, shadow window, burn, rollback -- is the real code.")
        from src.common.models import BacktestResult, SandboxVerdict
        cand = BacktestResult(n_trades=40, expectancy=0.05, sharpe=1.0, max_drawdown_pct=2.0,
                              win_rate=0.5, profit_factor=1.2, total_return_pct=2.0,
                              pnl_pct_std=0.15)
        verdict = SandboxVerdict(True, [], cand, cand)
        d.deployer.deploy(_apply(PROP_VALID, d.loader.current()), verdict, now_ms=d.now)
        d.monitor.note_deploy(d.now)

    live = d.loader.current()
    print(f"\n    config.json is now v{live['version']}: "
          f"atr_mult_sl={live['risk']['atr_mult_sl']} atr_mult_tp={live['risk']['atr_mult_tp']}")
    assert live["version"] == 2, "expected v2 to be live"
    assert d.deployer.shadow_version == 2, "shadow window should be open on v2"
    print(f"    shadow window OPEN on v2: expecting ~{d.deployer.shadow_expected:.4f}%/trade")

    step("feed 15 BAD trades under v2 (worse than the sandbox promised)")
    d.inject_trades(n_wins=0, n_stopouts=15, version=2, start_ms=d.now + MS_HOUR,
                    pnl_pct=-0.9, prefix="shadow")
    r2 = d.kpi.compute(d.now + 20 * MS_HOUR)
    rolled = d.deployer.check_shadow(r2)
    print(f"    rollback triggered: {rolled}")
    assert rolled, "shadow window should have rejected v2"

    live = d.loader.current()
    print(f"    config.json is now v{live['version']}: "
          f"atr_mult_sl={live['risk']['atr_mult_sl']} (v1's value restored)")
    assert live["risk"]["atr_mult_sl"] == 1.5, "v1 parameters should be restored"
    assert len(d.burned) == 1, "the bad parameter set should be burned"
    print(f"    burned registry now holds {len(d.burned)} fingerprint(s)")

    step("re-proposing the burned config is now refused")
    from src.common.models import OptimizationProposal, ProposedChange
    v = ProposalValidator(d.loader.schema, d.loader.bounds, d.burned)
    again = OptimizationProposal(
        proposed_changes=[ProposedChange("risk.atr_mult_sl", 1.5, 1.875, "retry"),
                          ProposedChange("risk.atr_mult_tp", 2.5, 3.125, "retry")],
        expected_impact=None, simulation_plan="p", abstain=False)
    res = v.validate(again, d.loader.current(), d.loader.bounds)
    print(f"    validator: ok={res.ok} -> {res.errors[0][:90] if res.errors else ''}")
    assert not res.ok and any("burned" in e for e in res.errors)

    print(f"\n    audit log: {d.audit}")
    for line in d.audit.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        print(f"      [{rec['outcome']}] trigger={rec['trigger']['id']} "
              f"bundle={rec.get('bundle_hash')} decision={str(rec.get('decision'))[:60]}")
    d.db.close()
    return True


def _apply(prop: dict, cfg: dict) -> dict:
    import copy
    from src.common.config_loader import set_path
    out = copy.deepcopy(cfg)
    for c in prop["proposed_changes"]:
        set_path(out, c["path"], c["new"])
    return out


async def scenario_variance_offramp(work: Path) -> bool:
    banner("SCENARIO 2: variance diagnosis -> no action, cooldown doubles")
    d = Drill(work / "s2")
    d.copy_candles()
    b = d.make_baseline()
    d.inject_trades(n_wins=17, n_stopouts=3)

    mock = MockLLMClient()
    mock.queue("submit_diagnosis", DIAG_VARIANCE)
    d.build(b, mock)

    ev = d.monitor.evaluate(d.kpi.compute(d.now))
    assert ev is not None and ev.trigger_id == "T1"
    before_trades, before_hours = d.monitor.min_trades_required, d.monitor.min_hours_required

    rec = await d.cycle.run(ev)
    print(f"    outcome: {rec['outcome']}")
    print(f"    decision: {rec.get('decision')}")
    assert rec["outcome"] == "no_action_variance"
    assert d.loader.current()["version"] == 1, "nothing should have deployed"
    print(f"    config still v{d.loader.current()['version']} -- nothing deployed")
    print(f"    cooldown: {before_trades} trades/{before_hours:.0f}h -> "
          f"{d.monitor.min_trades_required} trades/{d.monitor.min_hours_required:.0f}h")
    assert d.monitor.min_trades_required == before_trades * 2
    assert len(mock.calls) == 1, "phase B must not run after a variance verdict"
    print("    phase B was never called -- the cycle stopped at diagnosis")
    d.db.close()
    return True


async def scenario_rejections(work: Path) -> bool:
    banner("SCENARIO 3: out-of-bounds proposal -> retry -> abstain (nothing deploys)")
    d = Drill(work / "s3")
    d.copy_candles()
    b = d.make_baseline()
    d.inject_trades(n_wins=17, n_stopouts=3)

    mock = MockLLMClient()
    mock.queue("submit_diagnosis", DIAG_STRATEGY)
    mock.queue("submit_proposal", PROP_OUT_OF_BOUNDS)   # attempt 1: 1.5 -> 8.0
    mock.queue("submit_proposal", PROP_ABSTAIN)         # attempt 2: model backs off
    d.build(b, mock)

    ev = d.monitor.evaluate(d.kpi.compute(d.now))
    rec = await d.cycle.run(ev)
    attempts = rec["phase_b"]["attempts"]
    print(f"    attempt 1 errors: {attempts[0].get('errors') or attempts[0].get('validation')}")
    print(f"    attempt 2: abstained={attempts[1].get('abstained')}")
    print(f"    outcome: {rec['outcome']} / {rec.get('decision')}")
    assert rec["outcome"] == "no_action_abstained"
    assert d.loader.current()["version"] == 1
    print(f"    config still v{d.loader.current()['version']} -- nothing deployed")

    banner("SCENARIO 4: proposal reaches for a NON-TUNABLE backstop -> human review queue")
    d2 = Drill(work / "s4")
    d2.copy_candles()
    b2 = d2.make_baseline()
    d2.inject_trades(n_wins=17, n_stopouts=3)
    m2 = MockLLMClient()
    m2.queue("submit_diagnosis", DIAG_STRATEGY)
    m2.queue("submit_proposal", PROP_FORBIDDEN)   # risk.max_daily_loss_pct 3 -> 20
    m2.queue("submit_proposal", PROP_FORBIDDEN)   # and again on the retry
    d2.build(b2, m2)

    ev2 = d2.monitor.evaluate(d2.kpi.compute(d2.now))
    rec2 = await d2.cycle.run(ev2)
    print(f"    outcome: {rec2['outcome']}")
    print(f"    attempt 1 errors: {rec2['phase_b']['attempts'][0].get('errors')}")
    assert rec2["outcome"] == "rejected_validation"
    assert d2.loader.current()["risk"]["max_daily_loss_pct"] == 3.0, "backstop must be untouched"
    print(f"    risk.max_daily_loss_pct is still {d2.loader.current()['risk']['max_daily_loss_pct']}"
          " -- the LLM cannot reach it")
    assert rec2.get("queued_for_human_review"), "should have been queued for a human"
    q = json.loads(d2.human.read_text(encoding="utf-8").splitlines()[0])
    print(f"    human review queue: {q['reason']}")
    d.db.close()
    d2.db.close()
    return True


async def scenario_malformed(work: Path) -> bool:
    banner("SCENARIO 5: malformed tool payload -> rejected by the local schema check")
    d = Drill(work / "s5")
    d.copy_candles()
    b = d.make_baseline()
    d.inject_trades(n_wins=17, n_stopouts=3)

    mock = MockLLMClient()
    mock.queue("submit_diagnosis", DIAG_STRATEGY)
    mock.queue("submit_proposal", {"garbage": True})              # no required keys
    mock.queue("submit_proposal", {"proposed_changes": "not a list",
                                   "simulation_plan": "x", "abstain": False})
    d.build(b, mock)

    ev = d.monitor.evaluate(d.kpi.compute(d.now))
    rec = await d.cycle.run(ev)
    for a in rec["phase_b"]["attempts"]:
        print(f"    attempt {a['attempt']}: {str(a.get('errors'))[:110]}")
    print(f"    outcome: {rec['outcome']}")
    assert rec["outcome"] == "rejected_validation"
    assert d.loader.current()["version"] == 1
    print(f"    config still v{d.loader.current()['version']} -- nothing deployed")
    d.db.close()
    return True


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)-8s %(name)-30s %(message)s")

    # ignore_cleanup_errors: sqlite keeps a handle open briefly on Windows, and a
    # cleanup race must not mask a real scenario failure.
    with tempfile.TemporaryDirectory(prefix="lq-drill-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        results = []
        for fn in (scenario_happy_path_then_rollback, scenario_variance_offramp,
                   scenario_rejections, scenario_malformed):
            try:
                results.append(await fn(work))
            except AssertionError as e:
                print(f"\n    SCENARIO FAILED: {e}")
                results.append(False)
        Drill.close_all()

    banner("DRILL COMPLETE -- all scenarios passed" if all(results) else "DRILL FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
