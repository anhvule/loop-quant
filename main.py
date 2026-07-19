"""Loop Quant entrypoint: wires the event bus and starts every module.

    set BINANCE_TESTNET_KEY=...      (Windows: $env:BINANCE_TESTNET_KEY="...")
    set BINANCE_TESTNET_SECRET=...
    set ANTHROPIC_API_KEY=...        (optional; without it the optimizer is disabled)
    python main.py

Testnet only. Keys come from the environment, but the HOST is hardcoded in
ExchangeAdapter -- a production key pasted into these variables still cannot reach
a production endpoint.

Control flow per minute:
    candle closes -> indicators update -> indicator_update event
        -> SignalEngine.compute
        -> RiskManager.approve_entry / signal exit
        -> OrderManager
    trade closes -> trade_closed event
        -> KPITracker -> kpi_report event
        -> Deployer.check_shadow  (may roll back)
        -> TriggerMonitor.evaluate -> underperformance -> OptimizerCycle
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import paths                                             # noqa: E402
from src.common.config_loader import ConfigLoader                        # noqa: E402
from src.common.db import DB                                            # noqa: E402
from src.common.event_bus import EventBus                               # noqa: E402
from src.common.logging_setup import setup_logging                      # noqa: E402
from src.common.models import IndicatorSnapshot, KPIReport, MarketState  # noqa: E402
from src.evaluation.baseline import ensure_baseline                     # noqa: E402
from src.evaluation.economics import sizing_constraint, trade_economics  # noqa: E402
from src.evaluation.kpi_tracker import KPITracker                       # noqa: E402
from src.evaluation.trigger_monitor import TRIGGERS, TriggerMonitor     # noqa: E402
from src.execution.exchange_adapter import ExchangeAdapter              # noqa: E402
from src.execution.order_manager import OrderManager                    # noqa: E402
from src.execution.risk_manager import RiskManager                      # noqa: E402
from src.execution.signal_engine import SignalEngine                    # noqa: E402
from src.ingestion.feed_handler import FeedHandler                      # noqa: E402
from src.ingestion.indicator_engine import IndicatorEngine              # noqa: E402
from src.optimizer.backtest_sandbox import (                            # noqa: E402
    BacktestSandbox, compute_stress_windows, load_stress_windows, save_stress_windows,
)
from src.optimizer.burned import BurnedRegistry                         # noqa: E402
from src.optimizer.cycle import OptimizerCycle                          # noqa: E402
from src.optimizer.deployer import Deployer                             # noqa: E402
from src.optimizer.llm_client import AnthropicLLMClient                 # noqa: E402
from src.optimizer.log_packager import LogPackager                      # noqa: E402
from src.optimizer.proposal_validator import ProposalValidator          # noqa: E402

log = logging.getLogger("loopquant")

EQUITY_REFRESH_SEC = 55


class Engine:
    def __init__(self) -> None:
        paths.ensure_dirs()
        self.loader = ConfigLoader(paths.CONFIG_PATH, paths.SCHEMA_PATH, paths.BOUNDS_PATH)
        self.cfg = self.loader.load()
        self.symbol = self.cfg["symbol"]
        self.tf = self.cfg["timeframe"]

        self.db = DB(paths.DB_PATH)
        self.bus = EventBus()
        self.market = MarketState(symbol=self.symbol)
        self.stop_event = asyncio.Event()
        self.equity = 0.0
        self._last_equity_ms = 0
        self._tasks: list[asyncio.Task] = []

        self.adapter = ExchangeAdapter(
            api_key=os.environ.get("BINANCE_TESTNET_KEY", ""),
            api_secret=os.environ.get("BINANCE_TESTNET_SECRET", ""),
        )

    # -- startup -----------------------------------------------------------

    async def start(self) -> None:
        st, rk = self.cfg["strategy"], self.cfg["risk"]
        await self.adapter.start()
        await self.adapter.sync_time()
        filters = await self.adapter.load_filters(self.symbol)

        self.indicators = IndicatorEngine(
            self.symbol, self.tf,
            rsi_period=int(st["rsi"]["period"]), macd_fast=int(st["macd"]["fast"]),
            macd_slow=int(st["macd"]["slow"]), macd_signal=int(st["macd"]["signal"]),
            atr_period=int(rk["atr_period"]), db=self.db, persist=True,
        )
        self.signal = SignalEngine(self.loader.current)
        self.rm = RiskManager(self.loader.current, filters, initial_equity=0.0)
        self.om = OrderManager(self.symbol, self.adapter, self.bus, self.db,
                               self.market, self.rm, self.loader.current)
        self.feed = FeedHandler(self.symbol, self.tf, self.adapter, self.bus, self.db,
                                self.market, self.indicators)

        # Record v1 once so the anti-thrash clock has an origin. INSERT OR REPLACE
        # would reset it on every restart, so only write if it is absent.
        if not any(int(r["version"]) == int(self.cfg["version"])
                   for r in self.db.get_config_versions(50)):
            import json
            self.db.insert_config_version(int(self.cfg["version"]), _now_ms(),
                                          json.dumps(self.cfg), "initial", None)

        baseline = ensure_baseline(self.loader.current(), self.db, paths.BASELINE_PATH)
        self.kpi = KPITracker(self.db, self.bus, initial_equity=0.0,
                              cfg_provider=self.loader.current, baseline=baseline)
        self.monitor = TriggerMonitor(baseline, TRIGGERS, self.db, self.loader.current,
                                      last_deploy_ms=self.db.last_deploy_ms() or _now_ms())

        self._setup_optimizer(baseline)
        self._check_economics()

        await self.adapter.get_account()      # fail fast on bad keys
        await self._refresh_equity(force=True)
        self.kpi.initial_equity = self.equity
        self.rm.equity = self.equity
        self.rm._day_start_equity = self.equity
        log.info("starting equity: %.2f quote", self.equity)

        await self.om.reconcile()
        await self.feed.start()
        if self.om.position.is_open and not self.om.position.protective_order_ids:
            await self.om.protect_adopted(float(self.indicators.snapshot().atr or 0.0))

        self.bus.subscribe("indicator_update", self.on_indicator_update)
        self.bus.subscribe("kpi_report", self.on_kpi_report)
        self.bus.subscribe("state_changed", self.on_state_changed)
        self._tasks.append(asyncio.create_task(self._kill_file_loop(), name="killswitch"))
        log.info("engine ready: %s %s, active triggers %s",
                 self.symbol, self.tf, self.monitor.active_trigger_ids)

    def _setup_optimizer(self, baseline) -> None:
        self.optimizer: OptimizerCycle | None = None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set: the self-optimization loop is DISABLED. "
                        "Triggers will still fire and be logged.")
            return

        bounds = self.loader.bounds
        burned = BurnedRegistry(paths.BURNED_PATH, bounds)
        windows = load_stress_windows(paths.STRESS_WINDOWS_PATH)
        if not windows:
            windows = compute_stress_windows(self.db, self.symbol, self.tf)
            if windows:
                save_stress_windows(windows, paths.STRESS_WINDOWS_PATH)

        deployer = Deployer(paths.CONFIG_PATH, paths.VERSIONS_DIR, paths.BASELINE_PATH,
                            self.db, self.bus, self.loader, burned,
                            on_deploy=self._on_deploy)
        self.deployer = deployer
        self.optimizer = OptimizerCycle(
            packager=LogPackager(self.db, self.loader.current, bounds),
            llm=AnthropicLLMClient(),
            validator=ProposalValidator(self.loader.schema, bounds, burned),
            sandbox=BacktestSandbox(paths.DB_PATH, ROOT),
            deployer=deployer, monitor=self.monitor, loader=self.loader,
            audit_path=paths.OPTIMIZER_AUDIT, human_review_path=paths.HUMAN_REVIEW_QUEUE,
            stress_windows=windows,
        )
        log.info("self-optimization ENABLED (stress windows: %s)",
                 [w["label"] for w in windows])

    def _on_deploy(self, version: int) -> None:
        from src.evaluation.baseline import load_baseline
        b = load_baseline(paths.BASELINE_PATH)
        self.kpi.set_baseline(b)
        self.monitor.set_baseline(b)
        # An open position deliberately keeps the stops it was opened with -- they
        # are already resting on the exchange as an OCO and were derived from the
        # ATR at entry. Re-deriving them under new parameters would move a stop the
        # trader never agreed to.
        if self.om.position.is_open:
            log.info("config v%d is live; the open position keeps its original stops", version)

    def _check_economics(self) -> None:
        """Refuse to start quietly losing money. See evaluation/economics.py."""
        snap = self.indicators.snapshot()
        candles = self.db.get_last_candles(self.symbol, self.tf, 1440)
        if not candles:
            return
        import statistics
        from src.backtest.backtester import Backtester
        price = statistics.fmean([c.close for c in candles])
        tmp = IndicatorEngine(self.symbol, self.tf, persist=False)
        tmp.warm_up(candles)
        atr = float(tmp.snapshot().atr or 0.0)
        if atr <= 0:
            return

        econ = trade_economics(self.loader.current(), atr, price)
        sizing = sizing_constraint(self.loader.current(), atr, price)
        log.info("economics @ ATR %.4f%% of price: stop=%.3f%% target=%.3f%% fees=%.2f%% -> %s",
                 econ.atr_pct_of_price, econ.stop_pct, econ.target_pct,
                 econ.fee_pct_round_trip, econ.note)
        if not econ.viable:
            log.critical(
                "CONFIG IS NOT ECONOMICALLY VIABLE AT CURRENT VOLATILITY: %s. "
                "No weight or threshold tuning can fix this -- it needs wider stops, "
                "a longer timeframe, or lower fees.", econ.note)
        if not sizing["risk_dial_is_live"]:
            log.warning(
                "risk.risk_per_trade_pct is INERT: the notional cap (%.1f%% of equity) binds "
                "instead, so effective risk is ~%.3f%%/trade, not the configured %.2f%%.",
                self.cfg["risk"]["max_position_pct_equity"],
                sizing["effective_risk_per_trade_pct"], sizing["configured_risk_per_trade_pct"])

    # -- event handlers ----------------------------------------------------

    async def on_indicator_update(self, snap: IndicatorSnapshot) -> None:
        await self._refresh_equity()
        now = _now_ms()
        sig = self.signal.compute(snap, self.market)

        if self.om.position.is_open:
            if sig.action == "EXIT":
                log.info("signal exit at score %.3f", sig.score)
                await self.om.exit_on_signal()
            return

        if self.om.position.state != "FLAT":
            return   # an entry or exit is mid-flight

        d = self.rm.approve_entry(sig, self.market, self.equity, now, float(snap.atr or 0.0))
        if not d.approved:
            if sig.action == "ENTER":
                log.info("entry declined: %s", d.reason)
            return
        log.info("ENTRY approved: qty=%.8f score=%.3f (sizing bound by %s)",
                 d.qty, sig.score, d.binding_constraint)
        await self.om.enter_long(d.qty, self.market, float(snap.atr or 0.0))

    async def on_kpi_report(self, r: KPIReport) -> None:
        if self.optimizer is None:
            ev = self.monitor.evaluate(r)
            if ev:
                log.warning("trigger %s fired but the optimizer is disabled: %s",
                            ev.trigger_id, ev.condition_text)
            return

        if self.deployer.check_shadow(r):
            return   # a rollback just happened; let the new config settle

        ev = self.monitor.evaluate(r)
        if self.monitor.thrash_halt:
            self.rm.halt("optimization thrash cap reached; human restart required")
            await self.om.flatten("thrash_halt")
            return
        if ev is None:
            return

        if ev.halt_trading:
            self.rm.halt(f"trigger {ev.trigger_id}: {ev.condition_text}")
            await self.om.flatten(f"trigger_{ev.trigger_id}")

        # Detached so a slow LLM round trip cannot stall the market loop.
        self._tasks.append(asyncio.create_task(self.optimizer.run(ev),
                                               name=f"optimize-{ev.trigger_id}"))

    async def on_state_changed(self, state: str) -> None:
        if state == "HALTED" and self.om.position.is_open:
            log.error("feed HALTED with an open position; flattening")
            await self.om.flatten("feed_halted")

    # -- background loops --------------------------------------------------

    async def _refresh_equity(self, force: bool = False) -> None:
        now = _now_ms()
        if not force and now - self._last_equity_ms < EQUITY_REFRESH_SEC * 1000:
            return
        try:
            px = self.market.last_price or 0.0
            if px <= 0:
                c = self.db.get_last_candles(self.symbol, self.tf, 1)
                px = c[0].close if c else 0.0
            if px > 0:
                self.equity = await self.adapter.get_equity_quote(self.symbol, px)
                self.rm.mark_equity(self.equity, now)
                self._last_equity_ms = now
        except Exception as e:
            log.warning("equity refresh failed: %s", e)

    async def _kill_file_loop(self) -> None:
        """`touch logs/KILL` -> flatten and exit. The backstop that works even if
        every other layer is confused."""
        while not self.stop_event.is_set():
            await asyncio.sleep(1.0)
            if paths.KILL_FILE.exists():
                log.critical("KILL file detected at %s; flattening and shutting down",
                             paths.KILL_FILE)
                self.rm.halt("kill file")
                with contextlib.suppress(Exception):
                    await self.om.flatten("kill_file")
                self.stop_event.set()
                return

    # -- shutdown ----------------------------------------------------------

    async def stop(self) -> None:
        log.info("shutting down")
        self.stop_event.set()
        with contextlib.suppress(Exception):
            await self.feed.stop()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await self.adapter.close()
        self.db.close()


def _now_ms() -> int:
    return int(time.time() * 1000)


async def amain() -> int:
    setup_logging(paths.ENGINE_LOG)
    if paths.KILL_FILE.exists():
        log.critical("refusing to start: KILL file present at %s. Remove it first.",
                     paths.KILL_FILE)
        return 1

    engine = Engine()
    try:
        await engine.start()
        await engine.stop_event.wait()
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("fatal error")
        return 1
    finally:
        await engine.stop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(130)
