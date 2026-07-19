# Plan: SPY September–October 2026 outlook (probabilistic, not point-forecast)

## Context

The existing `src/forecast/` pipeline (GBM + ARIMA, daily bars) is honest but weak at
2–3 month horizons: drift ≈ 0, the cone just widens with √t, and the output collapses to
"today's price ± a lot." For a Sep/Oct 2026 view we change the *question* from
"what price?" to "what is the probability distribution of prices at specific dates?" —
and we **validate** that the distributions are calibrated before trusting them.

Reframed deliverable: for month-end September (≈ trading day 51) and month-end October
(≈ trading day 73), a report with:
- P(SPY above today), P(> +5%), P(< −5%), P(< −10%), expected value, full percentile table
- date-mapped output (calendar dates, not "day 51")
- seasonality context (how Sep/Oct have historically behaved) — reported, never baked in
- a calibration score proving (or disproving) the bands mean what they claim

> NOT INVESTMENT ADVICE. All outputs remain statistical baselines with a disclaimer.

## Why the current model underperforms at this horizon, and the fixes

| Weakness | Fix | Phase |
|---|---|---|
| Drift from 500d window is noise-dominated | Longer history (10y) + shrinkage toward long-run mean | 1 |
| Constant σ from trailing window ignores vol regimes | EWMA (RiskMetrics λ=0.94) vol + optional ^VIX market-implied anchor | 2 |
| Normal shocks miss fat tails / vol clustering | Block bootstrap of real historical returns (blocks ≈ 10d) | 3 |
| "day N" output is unreadable | Trading-day → calendar-date mapping; Sep/Oct summary tables | 4 |
| No evidence bands are trustworthy | Walk-forward calibration backtest (the honesty gate) | 5 |

## Phases

### Phase 1 — `src/forecast/longrange.py`: drift done properly
- Fetch ~10 years of SPY daily bars (reuse `fetch_yfinance`, bigger `days`).
- Drift estimator: `mu_blend = w * mu_recent(500d) + (1-w) * mu_long(10y)`, default w=0.3.
  Report both components so the user sees how little the recent window matters.

### Phase 2 — volatility term structure
- `ewma_sigma(returns, lam=0.94)` — numpy-only, no new deps.
- Optional `--vix-anchor`: fetch `^VIX` close via yfinance; forward daily σ ≈ VIX/100/√252;
  blend 50/50 with EWMA. Degrades gracefully (EWMA-only) if ^VIX fetch fails.

### Phase 3 — `src/forecast/bootstrap.py`: block-bootstrap Monte Carlo
- Sample contiguous blocks (default 10 trading days) of *actual* historical daily returns,
  stitch to horizon length, n_paths=20k, seeded rng (deterministic like `gbm.py`).
- Preserves fat tails, skew, and vol clustering that the normal-shock GBM cannot.
- Runs alongside GBM; report both so disagreement is visible.

### Phase 4 — `src/forecast/outlook.py` + `scripts/outlook.py`
- Trading-day calendar: map horizon days to dates (weekday grid is fine; NYSE holidays
  hardcoded for 2026 H2 — Labor Day Sep 7, Thanksgiving Nov 26).
- Probability tables at requested dates from the terminal distributions
  (GBM already returns `terminal`; bootstrap will too).
- Seasonality context: month-of-year return stats from the 10y history (mean, median,
  hit rate for September and October) — printed as *context*, never mixed into the sim.
- CLI: `python -m scripts.outlook --symbol SPY --dates 2026-09-30,2026-10-30`
  (flags: `--vix-anchor`, `--paths`, `--seed`, `--no-chart`).
- Chart: history + both cones + vertical markers at the two target dates.

### Phase 5 — `scripts/validate_outlook.py`: the honesty gate
- Walk-forward: for each month-end over the past ~8 years, fit on data available *then*,
  predict the 51/73-day-ahead distribution, record where the realized price landed.
- Report coverage (90% band should contain ~90% of outcomes; 50% band ~50%) and CRPS-style
  sharpness comparison: GBM vs bootstrap vs VIX-anchored.
- **Decision rule:** whichever variant is best-calibrated becomes the headline in the
  Sep/Oct report; if all are badly calibrated, the report must say so.

### Tests
- `test_forecast_bootstrap.py` — determinism (same seed ⇒ identical), band ordering,
  block sampling stays within history, horizon length.
- `test_forecast_longrange.py` — shrinkage math, EWMA against a hand-computed series.
- `test_forecast_outlook.py` — date mapping (weekends/holidays skipped), probability
  table sums/monotonicity.

## What this will NOT do
- No macro/event modeling (FOMC, CPI, elections). Those dominate 2–3 month outcomes and
  are invisible to price-history models. The report lists this limitation explicitly.
- No buy/sell recommendation — probabilities and ranges only.

## Dependencies
None new. EWMA and bootstrap are numpy-only; ^VIX comes through the existing yfinance dep.

## Verification
1. `pytest tests/ -q` — all existing 199 + new tests pass.
2. `python -m scripts.validate_outlook --symbol SPY` — calibration report; check 90% ≈ 90%.
3. `python -m scripts.outlook --symbol SPY --dates 2026-09-30,2026-10-30 --vix-anchor` —
   final Sep/Oct probability report + chart.
