# Stock Risk Outlook — deployable web app

A ticker-in, probability-out web front end for the beta-anchored forecasting engine.
Enter any ticker; get calibrated **ranges, drawdown odds and reality checks** — never a
price target and never a recommendation.

> **NOT INVESTMENT ADVICE.** The app models *risk* from price history only. Direction is
> not predictable; medians are labelled as assumptions throughout.

## Run it locally (no Node, no build step, no deploy)

```powershell
cd web
..\.venv\Scripts\python serve.py --open
```

Opens <http://127.0.0.1:8000>. `serve.py` serves the static files and answers
`/api/predict` with the *same* engine the serverless function uses, so local behaviour
matches a deployment exactly. Flags: `--port 9000`, `--host 0.0.0.0`, `--open`.

Each forecast takes well under a second and prints a one-line summary in the console.

## Deploy to Vercel (optional)

```bash
cd web
npx vercel --prod
```

Zero configuration — Vercel detects `api/predict.py` as a Python serverless function
(deps from `requirements.txt`: numpy only) and serves the static files at root. Or import
the repo in the dashboard with **Root Directory** set to `web`.

> Untested from Vercel's network: Yahoo Finance sometimes rate-limits datacenter IP
> ranges. The engine already retries across `query1`/`query2` and surfaces a clear
> "rate-limited, try again" message, but running locally avoids the question entirely.

## Layout

```
web/
  serve.py         local runner (static files + /api/predict)
  api/analyse.py   GET /api/analyse?symbols=NVDA,COIN  <- what the UI calls
  api/_analyse.py  fetches each ticker once, then feeds both engines + the forecast
  api/predict.py   GET /api/predict?symbol=ASTS&months=6   (still live, no longer used by the UI)
  api/screen.py    GET /api/screen?symbols=OKLO,IREN       (   "    "  )
  api/classify.py  GET /api/classify?symbols=OKLO,IREN     (   "    "  )
  api/_engine.py   numpy-only engine: price fetch + the validated forecasting math
  api/_screener.py vendored screener, parity-tested against src/forecast/screener.py
  api/_classify.py vendored classifier, parity-tested against src/forecast/classify.py
  index.html       UI shell (honesty banner, search, results)
  app.js           renders the JSON payload; draws the cone chart as inline SVG
  styles.css       light/dark styling, no external assets
  requirements.txt numpy
  vercel.json      function memory/duration + security headers
```

No build step, no framework, no external JS/CSS/font requests.

## Tests

Three suites cover this app, all offline (no network):

| file | covers |
|---|---|
| `tests/test_web_engine.py` | forecasting engine; **pins it to the validated pipeline** (`log_returns`, `ewma_vol_path`, `vol_half_life`, `blended_drift`, `beta_stats`, `reality_check` asserted equal to `src/forecast/*`) |
| `tests/test_web_waves.py` | Elliott/Fibonacci parity + chart payload |
| `tests/test_web_screener.py` | screener parity — identical thresholds, grades and composites vs the CLI |
| `tests/test_web_classify.py` | classifier parity — identical thresholds, gate grades and verdicts vs the CLI, with and without fundamentals |
| `tests/test_web_analyse.py` | the unified analysis — including the property that motivated it: each ticker is fetched **exactly once** |
| `tests/test_web_handlers.py` | the HTTP layer itself: query parsing, status codes, truncation reporting, static routing |

```powershell
..\.venv\Scripts\python -m pytest ..\tests -q -k "web"
```

## One analysis

There is a single input and a single **Analyse** button. What you type decides the
depth:

- **A list of tickers** → market-regime strip, then one cohort table carrying *both*
  readings per name: the seven-criterion risk **scorecard** (weighted live by the
  sliders) and the ten-gate **verdict**.
- **A single ticker** → everything above, plus the full forecast: hero metrics, the
  range stat-tiles, the cone chart with a hover crosshair, and wave structure.

The two readings sit side by side and are deliberately **never merged into one score**.
The scorecard measures risk and refuses to rank on returns; the verdict additionally
asks whether a name was historically paid for its beta. Where they disagree — a name can
top the scorecard and still be `reckless` — that disagreement is the finding.

Everything is served by one request to `/api/analyse`, which fetches each ticker's
history **once** and feeds both engines from the same arrays. Calling the older
per-view endpoints would refetch the same prices per view, and Yahoo rate-limits.

## UI conventions

Styling targets an institutional research terminal: tabular figures everywhere, a
single blue accent, and status colour (green / amber / red) **reserved** for grades and
verdicts so colour never decorates. Both themes are *selected* — dark mode has its own
colour steps validated against the dark surface, not an inverted light palette — and
every text role clears WCAG AA on both (verified in-browser: 4.85–9.49 dark,
5.7–7.35 light). No colour carries meaning alone: every pill and badge ships its own
text label. There are no webfonts, icon fonts, or external requests of any kind.

## API — `GET /api/screen?symbols=OKLO,IREN,MRVL`

Up to `MAX_SYMBOLS` tickers (200 locally; extras are returned in `truncated`, never
silently dropped). Response:

| field | meaning |
|---|---|
| `banner` | the standing rail: none of these criteria predicts returns |
| `cards[]` | per ticker: `symbol`, `name`, `spot`, `criteria[]`, `composite_equal`, `warnings[]` |
| `cards[].criteria[]` | `key`, `title`, `value`, `display`, `grade` (pass/warn/fail/info), `threshold`, `note` |
| `failures[]` | tickers that could not be fetched or scored, with the reason |
| `truncated[]`, `max_symbols` | tickers beyond the cap that were **not** screened |

The seven criteria: **beta** (true amplifier vs idiosyncratic lottery), **verifiable**
(enough history + stored calibration), **survivable** (share of 126-day windows that
halved), **redundancy** (avg correlation to the *submitted cohort*), **liquidity**
(median daily $ volume), **vol_regime** (current vol percentile vs its own history), and
**payoff** — which is deliberately ungraded (`info`), because describing which tail has
been fat is not a claim about which comes next.

`composite_equal` is an equal-weighted score; the UI recomputes it live from the weight
sliders. **The weights are a value judgement, so the ranking is a preference ordering,
never a forecast.**

CLI equivalent (the source of truth the web copy is parity-tested against):

```bash
python -m scripts.screen --csv portfolio.csv --min-beta 2.0 --weights survivable=3
```

Note: `--min-beta` filters *before* redundancy is computed, so "adds a new bet" is
measured against the names actually shown.

## API — `GET /api/classify?symbols=OKLO,IREN,MRVL&rf=0.04`

Up to `MAX_SYMBOLS` tickers (extras land in `truncated`). `rf` is the flat annual risk-free rate
used by the Treynor gate, clamped to 0–20%. Response:

| field | meaning |
|---|---|
| `banner` | the standing rail: classification, not prediction |
| `regime`, `risk_on` | the market's position vs its 200-day SMA, computed once per run |
| `quartile_mode` | `quartile` (ranked) or `absolute` (too few eligible names to rank) |
| `n_eligible` | names that cleared the two hard gates and were actually ranked |
| `results[]` | per ticker: `symbol`, `name`, `spot`, `verdict`, `gates[]`, `metrics`, `reasons[]` |
| `results[].gates[]` | `key`, `title`, `value`, `display`, `grade`, `threshold`, `note` |
| `failures[]`, `truncated[]` | tickers not classified, with reasons — never silently dropped |

The ten gates. **adv** (median daily $ volume > $10M) and **beta** (252d beta ≥ 1.5 with
the 63d window confirming) are *hard eligibility* — failing either yields `excluded_*`,
which is a statement of scope, not a criticism. **regime** is an overlay: below the
200d SMA nothing is `investable`, the best is `stand_aside`. Then:

| gate | asks | can condemn? |
|---|---|---|
| `treynor` | top quartile of return per unit of beta | required for `reckless` |
| `idio` | is the market, not its own story, driving it | yes |
| `survivable` | how often 126-day windows halved, over *all* history | yes |
| `dilution` | trailing-year share-count growth | yes |
| `runway` | quarters of cash at the current burn | yes |
| `own_trend` | is the name above its own 200d SMA | **no** — advisory |
| `revenue` | does an operating business exist | **no** — advisory |

`investable` needs every gate PASS (INFO is exempt — unverifiable is not bad news).
`reckless` needs a failed **treynor** *and* at least **two** corroborating structural
red flags — any single gate can be wrong about a single name, so the harshest label
never turns on one reading. One flag, or a name that was paid for its beta, stays
`mixed`. The flags that fired are returned in `metrics.flags`, so the label always
ships with its evidence. The two advisory gates can withhold a blessing but never
condemn, because both shade into predicting returns.

The fundamentals gates recognise their own blind spots: **dilution** caps at WARN for
names public under ~2 years (a share count cannot be read across an IPO) *and* for
names with TTM revenue over $10M (a lender issuing equity to grow its book is capital
formation, not a survival treadmill — the SOFI case); **runway** caps at WARN on the
same revenue bar (burning cash on growth capex is a financing choice, not a countdown).

**Fundamentals are CLI-only.** The web engine fetches prices from Yahoo's chart endpoint;
share counts and statements need authentication it does not carry, so `dilution`,
`runway` and `revenue` always read INFO here and the payload sets
`fundamentals_available: false`. Verdicts stay parity-correct because INFO never blocks.

Because the web runs on the handful of tickers you submit, the eligible universe is
almost always below the 8-name floor and the Treynor gate falls back to its absolute rule
(excess return above zero). `quartile_mode` reports which rule applied. A true
top-quartile rank against the whole index is CLI-only:

```bash
python -m scripts.classify                 # the whole S&P 500
python -m scripts.classify --symbols NVDA,MRVL,COIN --rf 0.045
```

## API — `GET /api/predict?symbol=IREN&months=6`

Returns JSON containing:

| field | meaning |
|---|---|
| `beta`, `beta_se`, `r2` | market sensitivity, its uncertainty, and share of variance explained |
| `anchored_mu_daily` / `raw_mu_daily` | the drift used vs the momentum drift that was **refused** |
| `months[]` | per month: P(up), median move, p5/p50/p95, cumulative dip odds |
| `reality_check[]` | each percentile vs what the stock actually did, plus the correction applied |
| `anchored_range` | the headline p5 / median / p95 after the reality check |
| `precedent` | worst/best/median realised move over the same horizon length |
| `alpha_sensitivity` | how P(up) decays if the stock does not earn its beta |
| `warnings[]` | thin history, low r², extreme volatility |

Errors return `{"error": "..."}` with HTTP 400 (bad/unknown/thin ticker) or 500.

Responses are cached at the edge for 15 minutes (`s-maxage=900`); forecasts only change
when a new daily bar arrives.

## Method (and its limits)

* Drift is anchored to `beta × market drift`. A stock's own recent momentum is treated as
  noise and discarded — projecting it forward is how forecasts become fantasy.
* Returns are volatility-standardised and rescaled to *today's* volatility, with a decaying
  term structure so a current panic regime is not projected forever.
* Blocks are resampled **jointly with SPY**, so correlation, fat tails and co-crash days
  survive into the simulation. Recent regimes are weighted more heavily than distant ones.
* Every percentile is checked against the stock's own realised moves. Unprecedented upside
  is trimmed; a downside gloomier than history is kept (history bounds how bad things have
  been, not how bad they can get).
* Beta's own standard error is propagated into the bands.

**Known limits, stated up front:** the market-model calibration (90% bands covering ~90% of
outcomes) was established on SPY over 28 years. Individual stocks with under ~700 shared
bars cannot be validated and are flagged in the UI. Nothing here models earnings, launches,
contracts, crypto prices or macro events — which is most of what actually moves high-beta
names. A price-history model can never represent a company failing outright, so the true
downside is worse than any percentile shown.

## Known difference vs the CLI

Both use a 10-year Yahoo window for per-name history (identical bar counts), but
`scripts/beta_outlook.py` pulls a slightly longer SPY series for the market drift
(~12.7y vs 10y here), so the anchored drift differs by ~0.004%/day. Over six months that
is well under 1% on the median and does not affect any range or drawdown figure — but it
is why the two can print marginally different medians for the same ticker.

## If you publish this publicly

It is a research tool, not a financial service. Keep the disclaimers intact, don't present
outputs as recommendations, and be aware that jurisdictions differ on what counts as
investment advice.
