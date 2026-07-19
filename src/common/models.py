"""Core dataclasses shared across all Loop Quant modules.

Every cross-module payload is one of these types. Modules never pass raw dicts
on the event bus -- the only dicts in the system are the config document and the
LLM proposal bundle (both of which are schema-validated at their boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Enumerations (Literal aliases keep them JSON-native and mypy-checkable)
# ---------------------------------------------------------------------------

SystemState = Literal["WARMUP", "READY", "DEGRADED", "HALTED"]
PositionState = Literal["FLAT", "PENDING_ENTRY", "OPEN", "PENDING_EXIT"]
ExitReason = Literal["tp", "sl", "signal", "flatten", "manual"]
Action = Literal["ENTER", "EXIT", "HOLD"]
Side = Literal["BUY", "SELL"]
FailureClass = Literal["strategy", "execution", "variance", "mixed"]
TriggerScope = Literal["strategy", "execution"]


# ---------------------------------------------------------------------------
# Module 1 -- market data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade from the exchange feed."""
    ts_ms: int
    symbol: str
    price: float
    qty: float
    is_buyer_maker: bool
    agg_trade_id: int


@dataclass(frozen=True, slots=True)
class Candle:
    """An OHLCV bar. `quote_volume` is sum(price*qty) over the bar's raw trades,
    which makes it the exact numerator for a raw-trade VWAP (never typical price).
    """
    ts_open_ms: int
    symbol: str
    tf: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    n_trades: int

    @property
    def ts_close_ms(self) -> int:
        return self.ts_open_ms + tf_to_ms(self.tf) - 1


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    ts_ms: int
    symbol: str
    best_bid: float
    best_ask: float
    bid_qty_top5: float
    ask_qty_top5: float

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return 0.0 if m <= 0 else (self.best_ask - self.best_bid) / m * 1e4


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Indicator state at a candle close. `close` is carried for convenience and
    is deliberately NOT persisted (the `indicators` table stores only the DDL cols).
    """
    ts_ms: int
    symbol: str
    tf: str
    vwap: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    atr: float | None
    close: float = 0.0

    @property
    def ready(self) -> bool:
        """True once every indicator the SignalEngine needs has a value."""
        return None not in (self.vwap, self.rsi, self.macd_hist, self.atr) and (self.atr or 0) > 0


@dataclass(slots=True)
class MarketState:
    """Mutable live view of the market. Module 2 reads this; only Module 1 writes it."""
    symbol: str
    last_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_qty_top5: float = 0.0
    ask_qty_top5: float = 0.0
    last_tick_ms: int = 0
    state: SystemState = "WARMUP"

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        return self.last_price

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0 or self.best_bid <= 0 or self.best_ask <= 0:
            return 0.0
        return (self.best_ask - self.best_bid) / m * 1e4

    @property
    def book_imbalance(self) -> float:
        """(bid-ask)/(bid+ask) over top 5 levels. +1 = all bids, -1 = all asks."""
        tot = self.bid_qty_top5 + self.ask_qty_top5
        return 0.0 if tot <= 0 else (self.bid_qty_top5 - self.ask_qty_top5) / tot


# ---------------------------------------------------------------------------
# Module 2 -- signals, risk, orders
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Signal:
    ts_ms: int
    score: float
    components: dict[str, float]
    action: Action


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    qty: float = 0.0
    reason: str = ""
    stop_px: float = 0.0
    tp_px: float = 0.0
    # Which sizing rule actually decided the size. When this is "notional", the
    # `risk_per_trade_pct` dial is inert -- worth surfacing, because the optimizer
    # is allowed to tune that dial and would otherwise be turning a dead knob.
    binding_constraint: Literal["risk", "notional", "none"] = "none"


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    client_order_id: str
    ts_ms: int
    symbol: str
    side: Side
    price: float
    qty: float
    fee: float = 0.0
    fee_asset: str = ""
    intended_px: float = 0.0

    @property
    def slippage_bps(self) -> float:
        """Signed against the trader: positive = filled worse than intended."""
        if self.intended_px <= 0:
            return 0.0
        sign = 1.0 if self.side == "BUY" else -1.0
        return (self.price - self.intended_px) / self.intended_px * 1e4 * sign


@dataclass(slots=True)
class Position:
    symbol: str
    state: PositionState = "FLAT"
    side: Side | None = None
    qty: float = 0.0
    entry_px: float = 0.0
    entry_ts_ms: int = 0
    stop_px: float = 0.0
    tp_px: float = 0.0
    atr_at_entry: float = 0.0
    config_version: int = 0
    trade_id: str = ""
    entry_slippage_bps: float = 0.0
    protective_order_ids: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.state in ("OPEN", "PENDING_EXIT")


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    trade_id: str
    symbol: str
    side: Side
    entry_ts_ms: int
    exit_ts_ms: int
    entry_px: float
    exit_px: float
    qty: float
    pnl_quote: float
    pnl_pct: float
    exit_reason: ExitReason
    slippage_bps: float
    config_version: int
    atr_at_entry: float

    @property
    def is_win(self) -> bool:
        return self.pnl_quote > 0


# ---------------------------------------------------------------------------
# Module 3 -- evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KPIReport:
    ts_ms: int
    window_trades: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    sharpe: float | None
    max_drawdown_pct: float | None
    avg_slippage_bps: float | None
    consecutive_stopouts: int
    bt_deviation_pct: float | None
    equity: float
    config_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestBaseline:
    """KPIs the current config produced in simulation. The yardstick live results
    are measured against."""
    generated_ms: int
    config_version: int
    window_days: int
    n_trades: int
    expectancy: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Trigger:
    id: str
    description: str
    scope: TriggerScope
    min_trades: int
    halt_on_fire: bool = False


@dataclass(frozen=True, slots=True)
class UnderperformanceEvent:
    trigger_id: str
    scope: TriggerScope
    condition_text: str
    kpi_report: KPIReport
    baseline: BacktestBaseline | None
    fired_at_ms: int
    halt_trading: bool = False


# ---------------------------------------------------------------------------
# Module 4 -- optimizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Evidence:
    claim: str
    supporting_fields: list[str]


@dataclass(frozen=True, slots=True)
class BreakevenAnalysis:
    rr_ratio: float
    required_win_rate: float
    actual_win_rate: float


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Phase A output."""
    diagnosis: str
    failure_class: FailureClass
    evidence: list[Evidence]
    breakeven_analysis: BreakevenAnalysis | None = None

    @staticmethod
    def from_tool_input(d: dict[str, Any]) -> "Diagnosis":
        ba = d.get("breakeven_analysis")
        return Diagnosis(
            diagnosis=d["diagnosis"],
            failure_class=d["failure_class"],
            evidence=[Evidence(e["claim"], list(e.get("supporting_fields", [])))
                      for e in d.get("evidence", [])],
            breakeven_analysis=BreakevenAnalysis(
                float(ba["rr_ratio"]), float(ba["required_win_rate"]), float(ba["actual_win_rate"])
            ) if ba else None,
        )


@dataclass(frozen=True, slots=True)
class ProposedChange:
    path: str
    old: Any
    new: Any
    rationale: str


@dataclass(frozen=True, slots=True)
class ExpectedImpact:
    kpi: str
    direction: Literal["up", "down"]
    mechanism: str


@dataclass(frozen=True, slots=True)
class OptimizationProposal:
    """Phase B output. Pure DATA -- never code, never executed."""
    proposed_changes: list[ProposedChange]
    expected_impact: ExpectedImpact | None
    simulation_plan: str
    abstain: bool = False

    @staticmethod
    def from_tool_input(d: dict[str, Any]) -> "OptimizationProposal":
        ei = d.get("expected_impact")
        return OptimizationProposal(
            proposed_changes=[
                ProposedChange(c["path"], c.get("old"), c["new"], c.get("rationale", ""))
                for c in d.get("proposed_changes", [])
            ],
            expected_impact=ExpectedImpact(ei["kpi"], ei["direction"], ei.get("mechanism", ""))
            if ei else None,
            simulation_plan=d.get("simulation_plan", ""),
            abstain=bool(d.get("abstain", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_changes": [asdict(c) for c in self.proposed_changes],
            "expected_impact": asdict(self.expected_impact) if self.expected_impact else None,
            "simulation_plan": self.simulation_plan,
            "abstain": self.abstain,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    candidate_config: dict[str, Any] | None = None

    @property
    def error_text(self) -> str:
        return "; ".join(self.errors)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    n_trades: int
    expectancy: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_return_pct: float
    # Per-trade dispersion. Crosses the subprocess boundary because the Deployer's
    # shadow window needs it to compute a standard error -- without it there is no
    # principled threshold for "worse than the sandbox promised".
    pnl_pct_std: float = 0.0
    pnl_pct_series: list[float] = field(default_factory=list)
    trade_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("pnl_pct_series", None)   # the series itself is too big to ship
        return d


@dataclass(frozen=True, slots=True)
class SandboxVerdict:
    passed: bool
    reasons: list[str]
    candidate: BacktestResult | None
    incumbent: BacktestResult | None
    stress: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "incumbent": self.incumbent.to_dict() if self.incumbent else None,
            "stress": self.stress,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_TF_MS = {"1m": 60_000, "5m": 300_000, "1d": 86_400_000}


def tf_to_ms(tf: str) -> int:
    try:
        return _TF_MS[tf]
    except KeyError:
        raise ValueError(f"unsupported timeframe: {tf!r}") from None
