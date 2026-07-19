# Loop Quant — Self-Optimizing Algorithmic Trading Engine

A paper-trading research system that trades VWAP/RSI/MACD signals on Binance Spot
**Testnet**, measures its own performance, and — when KPIs degrade past defined
thresholds — packages its failure logs into an LLM optimization prompt that proposes
bounded changes to `config.json`, validates them in a backtest sandbox, and deploys
them with automatic rollback.

Built to the blueprint in `../LOOP_QUANT_BLUEPRINT.md`.

> **Not investment advice. Testnet only.** No live-trading code path exists in v1:
> `ExchangeAdapter` refuses to construct against a production Binance host. This is
> a research harness for studying a self-optimizing loop, not a money-making system —
> see the findings below.

---

## ⚠️ Read this first: three findings from the build

The machinery works and is well tested. The **strategy defaults the blueprint
specifies do not**, and the system proves it on real data. These are your decisions
to make, not mine.

### 1. The shipped config cannot be profitable at 1m. It loses 5.9%/month on real BTC.

Fees are charged on *notional*; stops are measured in *ATR*. On a 1m BTC bar, ATR is
~0.05–0.15% of price, so a `1.5 × ATR` stop is ~15bps — while a 10bps taker fee each
way is **20bps round trip**. The fee is larger than the entire stop distance.

| ATR (% of price) | stop | target | fee r/t | breakeven win rate |
|---|---|---|---|---|
| 0.05% | 0.075% | 0.125% | 0.20% | **impossible** (a winner doesn't cover fees) |
| 0.10% | 0.150% | 0.250% | 0.20% | **87.5%** |
| 0.15% | 0.225% | 0.375% | 0.20% | 70.8% |
| 0.50% | 0.750% | 1.250% | 0.20% | 47.5% |

Backtest over 30 days of **real** BTCUSDT 1m data: **289 trades, 15.2% win rate,
−0.21% expectancy per trade, −5.9% total.** It needs 87.5% and gets 15.2%.

Fixes, in rough order of effectiveness — all outside the optimizer's reach:
- **Longer timeframe** (15m/1h), where ATR% is large enough that fees are noise.
- **Lower fees** (BNB discount, VIP tier, maker-only entries).
- **Wider stops**: `atr_mult_sl 3.0 / atr_mult_tp 6.0` — the bounds maximum — takes
  breakeven from 87.5% to 55.6%. Better; still likely not enough.
- **Reconsider the signal.** `s_vwap` is momentum (above VWAP = bullish) while
  `s_rsi` is mean-reversion (oversold = bullish). They fight each other, and the
  15% win rate against a 1.67 R:R suggests a negative edge before fees ever apply.

### 2. `risk_per_trade_pct` is a dead dial — and the optimizer is allowed to tune it.

Risk-based sizing binds only when `stop_distance/price ≥ risk_pct/max_pos_pct` — i.e.
a **5% stop**. At 1m BTC the stop is ~0.15%, so `max_position_pct_equity` (10%) always
binds instead. Effective risk is **~0.015% of equity per trade, not the 0.5% configured**.

Empirically confirmed: `risk_per_trade_pct` at 0.375, 0.4, 0.6, and 0.625 all produce
**byte-identical** backtests (289 trades, −0.2105%, 5.90% DD). Run
`python -m scripts.analyze_config_space` — it reports every inert dial.

On spot there is no fix: risking 0.5% across a 0.2% stop requires a 250% position.
Either accept ~0.015% risk/trade, or move to a timeframe with wider stops.

### 3. Zero of 28–56 legal single-cycle moves pass the sandbox gate.

The optimizer will trigger, diagnose, propose, be rejected, and hit its 4-cycles-per-24h
thrash cap — then halt for a human. **That is the safety framework working correctly.**
It fails safe rather than deploying garbage. But it means the loop cannot rescue this
configuration; the fix is outside the tunable set.

The designed escape hatch works: Phase B is explicitly told to `abstain` when no
in-bounds change helps, and the bundle hands the model the deterministic breakeven
arithmetic so it can say so.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

.venv/Scripts/python -m pytest tests/ -q            # 181 tests
.venv/Scripts/python -m scripts.seed_data --days 30 # real BTCUSDT 1m history
.venv/Scripts/python -m scripts.analyze_config_space
.venv/Scripts/python -m scripts.optimizer_drill     # full Module 4 loop, no API key
```

To run the engine (needs Binance **testnet** keys from https://testnet.binance.vision):

```powershell
$env:BINANCE_TESTNET_KEY="..."
$env:BINANCE_TESTNET_SECRET="..."
$env:ANTHROPIC_API_KEY="..."    # optional; without it the optimizer is disabled
.venv/Scripts/python main.py
```

**Kill switch:** `New-Item logs/KILL` → flattens any open position and exits.

---

## Architecture

```
main.py ── EventBus ──┬─ Module 1  ingestion/   feed → candles → indicators
                      ├─ Module 2  execution/   signal → risk → orders
                      ├─ Module 3  evaluation/  KPIs → triggers
                      └─ Module 4  optimizer/   bundle → LLM → validate → sandbox → deploy
```

Per minute: `candle_closed` → `indicator_update` → SignalEngine → RiskManager → OrderManager.
Per trade: `trade_closed` → KPITracker → `kpi_report` → shadow check → TriggerMonitor → OptimizerCycle.

### Module 4 flow

```
UnderperformanceEvent
  → LogPackager        facts only: KPIs, baseline, config history + what each produced,
                       last 50 trades, market regime, deterministic breakeven economics
  → Phase A            diagnose. failure_class == "variance" → STOP, double the cooldown
  → Phase B            propose ≤3 bounded changes, or abstain
  → ProposalValidator  7 checks + stale-`old` + no-op + burned  [1 retry with the errors]
  → BacktestSandbox    subprocess, 30d + 2 stress windows, must EARN the deploy
  → Deployer           atomic swap, 15-trade shadow window, auto-rollback + burn
  → optimizer_audit.jsonl
```

### Safety framework (defense in depth)

| Layer | Mechanism |
|---|---|
| Structural | Forced tool-use JSON. The LLM emits **data, never code**. `path` is an enum built from `bounds.json` — a forbidden parameter is *unrepresentable*, not merely rejected. Execution-scoped triggers get a filtered enum, so strategy weights are unreachable while diagnosing a fills problem. |
| Bounds | Absolute min/max, 25%/cycle delta cap, ≤3 params, whitelist. Re-enforced on **every** config load, so even a hand-corrupted `config.json` cannot start the engine. |
| Behavioural | Deterministic sandbox (byte-identical reruns, enforced by test) + stress windows. Candidates must beat the incumbent, not merely differ. |
| Temporal | ≥20 trades **and** ≥6h between cycles; one cycle in flight; 4/24h then hard halt; 15-trade shadow window with auto-rollback; burned configs rejected for 7 days. |
| Absolute | `max_daily_loss_pct` kill switch, `logs/KILL` sentinel, testnet-only host guard. **None of these are in `bounds.json`** — the optimizer cannot reach them at all. |
| Audit | Every cycle appended to `optimizer_audit.jsonl`, including the cycles where nothing happened. |
| Human gate | Anything beyond the whitelist → `logs/human_review_queue.jsonl`, never auto-applied. |

---

## Deliberate deviations from the blueprint

Each of these is a place the spec was wrong or ambiguous, not a shortcut.

| # | Blueprint | What was built | Why |
|---|---|---|---|
| 1 | Sandbox: `candidate expectancy ≥ incumbent × 1.05` | `candidate ≥ incumbent + 0.05×abs(incumbent)` | **The spec has a sign bug.** Multiplying a *negative* expectancy by 1.05 *lowers* the bar: −0.10 × 1.05 = −0.105, so a still-losing candidate "passes". The optimizer fires precisely when expectancy is negative, so this is the common case, not an edge case. |
| 2 | Shadow: roll back below `sandbox expectancy − 1.5σ` | σ = **standard error** of the 15-trade mean (σ/√n) | With σ = per-trade stdev the band is so wide rollback would never fire. The quantity being compared is a mean of 15, so its dispersion is the standard error. |
| 3 | `CandleAggregator.on_trade → Candle \| None` | `→ list[Candle]` | Closing a multi-minute gap must emit the real bar **plus** one flat bar per skipped minute. A single-value return cannot express that without a hidden queue. |
| 4 | Stream `<symbol>@trade` | `<symbol>@aggTrade` | The schema's dedup key is `agg_trade_id`, which only the aggTrade stream carries (field `a`). Also less bandwidth for identical OHLCV. |
| 5 | `approve_entry(sig, mkt, equity)` | `+ now_ms, atr` | Sizing and cooldown depend on both. Sourcing them from ambient state would break the backtest/live parity this module exists to guarantee. |
| 6 | Sandbox "subprocess with no network access" | subprocess = crash/state isolation | Not an OS-level network jail; the backtester simply makes no network calls by construction. Overclaiming would be worse than saying it plainly. |
| 7 | VWAP "from raw trades" | from each bar's `quote_volume` | `quote_volume` **is** Σ(price×qty) over the bar's raw trades — identical math, not typical price. Accumulating per-bar is what makes live == backtest exactly. |

## Bugs caught by tests during the build

- **RSI on a flat series returned 100.** The naive `avg_loss == 0 → RSI 100` branch treated
  a dead-flat market (0 gains *and* 0 losses — genuinely 0/0) as maximally overbought, which
  the SignalEngine reads as a full-strength bearish signal. Gap-filled zero-volume bars
  produce exactly this series, so it was a live path. Now resolves to 50 (neutral).
- **Score could exceed 1.0** by float error in weight normalization, breaking the scale
  `entry_threshold` is specified against. Now clamped.
- **Human-review queue never fired.** It matched on the validator's error wording, but the
  tool-schema enum caught forbidden paths first and worded it differently. Now detected from
  the raw response, independent of which layer rejects.

## Notes

- **`rsi_band_ordering` is currently unreachable.** Bounds cap oversold at 40 and floor
  overbought at 60, so the minimum legal gap (20) already exceeds the invariant's 10. It is a
  backstop for a future bounds edit, and is tested where it can actually fire.
- **`scripts/seed_data.py` is the only code that touches `api.binance.com`**, with no API key
  and no secret — `_sign` raises without a secret, so every order-placing endpoint is
  unreachable from that process. Downloading public candles is not trading; the testnet guard
  exists to keep *orders* off production, and still does.
- **Not implemented:** live-paper smoke test against testnet (needs keys I don't have), and
  the `5m` timeframe path (schema allows it; only `1m` is exercised).

## Layout

```
config/     config.json (LLM-writable) · bounds.json (NEVER LLM-writable) · schema · versions/
src/common/     models · event_bus · config_loader · db · paths · logging_setup
src/ingestion/  feed_handler · candle_aggregator · indicator_engine
src/execution/  signal_engine · risk_manager · order_manager · exchange_adapter
src/evaluation/ kpi_tracker · trigger_monitor · metrics · baseline · economics
src/optimizer/  log_packager · prompt_builder · llm_client · proposal_validator
                backtest_sandbox · deployer · burned · cycle
src/backtest/   backtester  (shared by sandbox + baseline; deterministic)
scripts/        seed_data · optimizer_drill · analyze_config_space
tests/          181 tests
```
