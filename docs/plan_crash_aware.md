# Plan: crash-aware upgrade to the SPY outlook (drawdown, tails, stress scenarios)

## Context

The Sep–Dec 2026 outlook reports **terminal-date** probability bands, validated at 90%
coverage. But for crash risk it has real gaps: no intra-path drawdown probabilities, thin
(normal) tails in the headline GBM, no tail-specific calibration, no historical stress
scenarios, and season-blind volatility. The user is specifically worried about an October
crash. We do not build "a crash will happen" in as fact (unknowable; October's crash fame
is event-driven — its 20y mean here is +1.1%); instead we make the tool able to *quantify*
crash risk honestly and *simulate the user's scenario* explicitly.

> NOT INVESTMENT ADVICE. Odds and scenarios, not predictions; no positioning guidance.

## Phase A — intra-path drawdown risk (the biggest gap)

New `src/forecast/drawdown.py`:
- `touch_prob_gbm(s0, mu, sigma, h, level)` — closed-form first-passage probability
  under GBM (reflection principle): P(path touches `level` at any time ≤ h).
- `path_risk(prices_matrix, s0)` — from the existing `bootstrap_prices` matrix:
  P(touch −5/−10/−15/−20% at any time), plus the max-drawdown distribution
  (median / p90 / p99 max drawdown).
- Wire into `outlook.py` per target date and the console/CSV report as a new
  "path risk (any time before date)" block, GBM analytic vs bootstrap side by side.

## Phase B — fat-tail shocks for the parametric model

- Add Student-t innovations to `simulate_gbm` (`shock_dist="t"`, df fitted from excess
  kurtosis of the return history, floored at 4, variance-normalized so sigma is preserved).
- Outlook gains `--tails {normal,t}`; default decided by Phase D calibration.

## Phase C — named stress scenarios ("what if you're right")

New `src/forecast/scenarios.py`:
- From the full history, extract the worst Aug→Dec windows (auto-ranked; will surface
  2008, plus others in-sample e.g. 2000, 2018, 2020; 1987 hardcoded from its known
  monthly returns since our data starts ~1997).
- Replay each window's daily return sequence from today's price, report the resulting
  Oct/Nov/Dec price path and trough. Pure arithmetic, clearly labelled "scenario, not
  probability".
- Also a **crisis-conditioned bootstrap**: sample blocks only from high-vol regimes
  (top-quartile EWMA vol days) → a "stressed cone" printed next to the base cone.

## Phase D — tail-specific calibration (honesty gate for crash numbers)

Extend `scripts/validate_outlook.py`:
- Report lower/upper tail exceedance: share of outcomes below p5 (target 5%) and above
  p95, per method and horizon — normal-GBM vs t-GBM vs bootstrap.
- Drawdown calibration: predicted P(touch −10%) vs realized touch frequency across the
  344 walk-forward month-ends.
- The best tail-calibrated variant becomes the default `--tails` / headline for path risk.

## Phase E — seasonal volatility (the October question, done honestly)

- `monthly_vol_multipliers(history)` — per-calendar-month realized vol vs overall
  (expect Sep/Oct > 1). Since the sim already maps day → calendar date, scale sigma
  per simulated day; flag `--seasonal-vol`.
- Only becomes default if Phase D shows it *improves* tail calibration; otherwise it
  stays an option and the report says it didn't help.

## Smaller fixes
- `--vix-anchor` fetches `^VIX3M` (falls back to ^VIX) for multi-month horizons.
- Report block lists known Q4-2026 FOMC meeting dates as context (manual constant,
  clearly labelled unverified).

## Tests
- `test_forecast_drawdown.py` — analytic touch prob vs Monte Carlo agreement (±1%),
  monotonicity in level/horizon; max-drawdown distribution sanity.
- `test_forecast_scenarios.py` — replay arithmetic exact on a synthetic window;
  crisis-bootstrap uses only high-vol blocks.
- t-shock determinism + variance normalization test in `test_forecast_gbm.py`.

## Verification
1. Full pytest suite green.
2. `python -m scripts.validate_outlook --symbol SPY --horizons 51,73,116` — tail
   exceedance ≈ 5%/5% for the chosen default; drawdown calibration reported.
3. `python -m scripts.outlook --symbol SPY --dates 2026-10-30,2026-12-31 --vix-anchor`
   — now includes path-risk block + stress scenarios table.
