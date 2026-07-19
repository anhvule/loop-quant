# Plan: high-beta basket outlook (ASTS, IREN, NBIS, TE, CRWV) to year-end 2026

## Context

The user wants a forecasting strategy for five high-beta names through Dec 2026. Today's
validated findings dictate the design:

1. **Risk is predictable, return is not.** SPY's cone is calibrated (90.0% coverage,
   28y walk-forward). Single-name drift from recent momentum is noise (ASTS's raw
   estimate annualizes to ~+50%/yr — untrustworthy).
2. **These names co-move** (ASTS–IREN r=0.32, each ~0.3–0.4 to SPY, ~2–4x beta).
   Five independent cones would fake diversification; a crash hits them together.
3. **Calibration must be re-earned per asset class.** SPY's validation does not
   transfer to 6–8%/day-vol names; the plan validates on the names themselves.

Strategy in one line: **simulate the market factor with the validated SPY machinery,
propagate it into each name via beta, bootstrap the residuals JOINTLY to preserve
co-crashes, and anchor every name's drift to beta x market drift (not its own momentum).**

> NOT INVESTMENT ADVICE. Output is calibrated odds + scenarios, never buy/sell calls.

## Ticker checks (data realities)

| Ticker | Note |
|---|---|
| ASTS | ~1,700 bars. Fine. |
| IREN | ~1,170 bars. Fine. |
| NBIS | Nebius — relisted Oct 2024, ~440 bars. Beta estimable; validation impossible. |
| TE | **Ambiguous ticker** (resolved earlier to a $5.84 instrument; NOT TE Connectivity=TEL). Proceed with yfinance's TE; flag in output. |
| CRWV | CoreWeave — IPO Mar 2025, ~330 bars. Thin: joint window is capped by it. |

## Phase 1 — `src/forecast/beta.py`: joint factor engine

- `align_returns(symbol->closes)` -> matrix of daily log returns on common dates
  (window capped by youngest listing; report the window length honestly).
- `beta_stats(name_rets, spy_rets)`: EWMA-weighted beta, idio vol, r². Betas also
  computed on each name's FULL overlap with SPY (longer window than the joint one).
- `joint_bootstrap(ret_matrix, horizon, ...)`: block bootstrap sampling the SAME
  date-blocks for all columns -> (n_paths, horizon, n_assets) price paths. Preserves
  the correlation matrix, fat tails, and co-crash days by construction.
- **Drift discipline**: recenter SPY's column to its validated blended mu; recenter
  each name to `beta_i * mu_spy` (idio drift = 0 by default; `--idio-drift` to opt in).
  Report raw-momentum drift next to the anchored drift so the user sees what was refused.

## Phase 2 — `scripts/beta_outlook.py`: the basket report

Per name, monthly (Aug..Dec, cumulative and marginal):
- P(up), median move, dip-touch probabilities (-10/-20/-30%), max-drawdown distribution.
- Year-end cone (median, 90% range, P(above today)) with the beta-anchored drift.
- **Conditional table** — the point of the factor model: median outcome per name GIVEN
  SPY ends -10% / flat / +10% (read off the joint sim; shows beta cutting both ways).

Basket level (equal-weight portfolio of the 5):
- Same tables for the basket — correlated sim means drawdowns do NOT diversify away;
  quantify how much worse the basket is than a naive independent assumption.
- SPY-crash stress: apply the Phase-C scenario replays to the SPY column and read the
  implied basket damage through the joint structure.

Chart: per-name cones (small multiples) + basket cone + conditional-on-SPY panel.
CSV: `data/beta_outlook_<name>.csv` + `data/beta_outlook_basket.csv`.

## Phase 3 — validation (the honesty gate, adapted)

- Walk-forward on the names with history (ASTS, IREN, TE): at each month-end, fit
  betas/vols on data-so-far, project 51d/73d bands two ways — (a) naive single-name
  bootstrap, (b) beta-anchored joint model — and score coverage + lower-tail exceedance.
- Report which wins per name. NBIS/CRWV: too short to validate — the report must SAY
  their bands are unvalidated extrapolation.
- Success bar: cov90 within ~0.85–0.95 and lower-tail ~0.05 for the winner.

## Phase 4 — tests

- `test_forecast_beta.py`: alignment drops non-common dates; beta of a synthetic
  2x-SPY series ~= 2; joint bootstrap preserves cross-correlation (r of sampled cols
  ~= r of input cols); drift recentering hits the target mean; determinism by seed.

## Explicit non-goals / honesty rails

- No timing calls, no per-name "price targets" — medians are labelled as drift lines.
- No catalyst modeling (launches, BTC price, AI contracts) — stated limitation; these
  dominate idio risk for exactly these five names.
- TE stays flagged as unconfirmed identity until the user verifies the ticker.

## Verification

1. `pytest tests/ -q` all green.
2. `python -m scripts.beta_outlook --validate` -> calibration table per name.
3. `python -m scripts.beta_outlook --symbols ASTS,IREN,NBIS,TE,CRWV --dates 2026-12-31`
   -> full basket report + chart; sanity-check betas (expect ~1.5-4x) and that the
   conditional-on-SPY table shows symmetric beta amplification.
