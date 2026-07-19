# Plan: high-beta screener — criteria, scoring, and frontend adaptation

## Context

The user invests in high-beta names by preference and wants criteria for picking "good"
ones, surfaced in the web UI. Session evidence: direction is unpredictable; what IS
measurable is verifiability, survivability, redundancy, liquidity, and payoff shape.
So the screener grades tickers on those axes with explicit thresholds, and any
composite ranking uses USER-ADJUSTABLE weights — the tool measures, the user decides
what "good" means. Banner on everything: none of these criteria predict returns.

> NOT INVESTMENT ADVICE. A scorecard of measurable properties, not a buy list.

## The seven criteria (each shows its number, its grade, and its threshold)

| # | Criterion | Metric | Grades (thresholds visible in UI) |
|---|---|---|---|
| 1 | **True high-beta?** | beta, ±se, r² vs SPY | beta>=2 & r2>=0.10 = "market amplifier"; beta>=2 & r2<0.10 = "idio lottery — beta unreliable"; beta<2 = "not high-beta" |
| 2 | **Verifiable?** | shared bars; stored calibration verdict (cov90 + CI) | >=700 bars & calibration consistent = pass; >=700 unvalidated = warn ("run validation"); <700 = fail ("extrapolation only") |
| 3 | **Survivable?** | % of 126d windows losing >=50%; worst 126d; max DD ever | <5% halved = mild; 5–20% = rough; >20% = severe ("halving is routine") |
| 4 | **Adds a new bet?** | avg pairwise corr vs the OTHER submitted tickers | <0.30 = diversifier; 0.30–0.45 = partial; >0.45 = duplicate of the cluster |
| 5 | **Liquid enough?** | median daily dollar volume (needs volume from chart API) | >$50M = fine; $5–50M = caution; <$5M = thin (execution risk) |
| 6 | **Vol regime now** | current EWMA vol percentile vs own history | <70th = normal; 70–85th = elevated; >85th = "entering during a storm" |
| 7 | **Payoff shape** (descriptive only) | % windows >=2x vs % <=1/2; best/worst 126d precedent | no grade — displayed as "X% doubled / Y% halved", explicitly non-predictive |

Composite: `sum(weight_i * normalized_i)` with sliders defaulting to equal weights,
labelled "**your weighted preference — not a forecast**". Re-sorts live client-side.

## Phase 1 — `src/forecast/screener.py`

- `ticker_metrics(closes, highs, lows, volumes, mkt_closes)` -> dict of criteria 1–3, 5–7
  (reuses `beta_stats`, `ewma_vol_path`, `realized_multiples`; nothing new mathematically).
- `cohort_redundancy(list of return series)` -> per-name avg corr + pairwise matrix
  (criterion 4 is only defined relative to the submitted set — stated in output).
- `grade(metrics)` -> per-criterion {value, grade, threshold_text}; pure function, testable.
- Calibration verdict read from `data/beta_validation_status.json` (existing machinery).

## Phase 2 — `scripts/screen.py`

`python -m scripts.screen --symbols WYFI,OKLO,IREN,...` or `--csv portfolio.csv
[--min-beta 2.0]` -> console scorecard table + `data/screen.csv`. Serves as the
source-of-truth implementation the web copy is pinned against.

## Phase 3 — web API

- Extend `web/api/_engine.fetch_prices` to also return **volume** (chart JSON already
  carries it; we currently discard it). All three call sites updated.
- `web/api/_screener.py` — vendored metrics/grades (numpy+stdlib), parity-tested.
- `web/api/screen.py` — `GET /api/screen?symbols=A,B,C` (cap ~20 symbols): per-name
  metrics + grades + redundancy matrix + SPY fetched once. Failures per-symbol, not fatal.
- `serve.py` routes `/api/screen` too.

## Phase 4 — frontend

- Mode toggle: "Single ticker" | "High-beta screener" (screener pre-fills the high-beta
  names from the user's portfolio as chips; editable ticker list).
- Scorecard table: one row per ticker; criteria as coloured grade pills with the value
  and threshold in a tooltip/subtext; payoff shape shown as plain numbers, no colour.
- Weight sliders (0–3) per criterion -> live composite column + re-sort. Header on the
  composite: "your weighted preference — not a forecast".
- Warnings surface inline (e.g. "WYFI: 237 bars — cannot be verified").
- Existing single-ticker view untouched; screener links each ticker to it.

## Phase 5 — tests

- `test_forecast_screener.py`: each grade boundary (both sides), redundancy on a
  synthetic 2x-clone (corr ~1 -> duplicate) vs independent series (-> diversifier),
  dollar-volume math, vol percentile, JSON-safety.
- `test_web_screener.py`: parity per metric vs `src/forecast/screener.py`; payload
  contract; per-symbol failure isolation; volume plumbed through `fetch_prices`.

## Honesty rails (non-negotiable)

- Banner on the screener: "These criteria measure verifiability, survivability,
  redundancy, liquidity and payoff shape. **None of them predicts returns** — direction
  failed every test we ran."
- Composite column is impossible to render without its "your weighted preference" label.
- Criterion 7 carries "descriptive, not predictive" inline.
- Unverifiable names show their fail state prominently, not as a footnote.

## Verification

1. pytest green (incl. parity).
2. `python -m scripts.screen --csv portfolio.csv --min-beta 2.0` -> scorecard for the
   27 high-beta names; sanity-check against the numbers already computed this session
   (WYFI unverifiable, BMNR severe, OKLO/SMR duplicates, MRVL/AAOI diversifier-ish).
3. Browser: screener mode with those tickers; move sliders -> re-rank updates; single
   ticker links still work.
