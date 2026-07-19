# Plan: Survivability, Own-Trend, and Fundamentals Gates for the Classifier

## Context — why this change

The five-gate classifier (`src/forecast/classify.py`) grades everything on
**trailing-one-year, name-vs-market price** measures. Live validation exposed the blind
spot: `SPCE` and `CRWV` both come out `borderline` with near-identical gate patterns,
yet they are opposite cases. SPCE's defining property is *chronic collapse* — 21% of all
126-day windows in its history halved, worst window −85%, max drawdown ~−100% (the
serial-diluter signature). CRWV is a 16-month-old IPO with 7% halvings and a two-sided
payoff. No one-year price window can see this difference; the whole-history
survivability measure in `src/forecast/screener.py::c_survivable` sees it immediately
(screener composites: CRWV 0.75 vs SPCE 0.33).

Price history is a *proxy* for the underlying disease (persistent halvings are what
dilution looks like on a chart). This plan adds both the proxy and the disease itself:

**Price-based (part 1):**
1. **Gate 6 — Survivable?** (verdict-affecting): the screener's halving-frequency math
   over the name's full history.
2. **Gate 7 — Own trend** (advisory): name's close vs its *own* 200d SMA, mirroring the
   market regime gate. Momentum-flavoured, and the project philosophy refuses return
   prediction — so it may **block `investable` but never cause `reckless`**.

**Fundamentals-based (part 2):**
3. **Gate 8 — Dilution** (verdict-affecting): trailing-12m share-count growth. The
   direct measurement of the serial-diluter treadmill.
4. **Gate 9 — Runway** (verdict-affecting): quarters of cash left at the current burn
   rate. A cash-starved name's next move is dilution or death; a binary-event profile
   plus a short runway is the full biotech-trap signature.
5. **Gate 10 — Real revenue?** (advisory): does an operating business exist, or is the
   price all story? Advisory only — judging revenue *quality* is forecasting, which
   this project refuses.

Fundamentals are **CLI-only in v1** (see the web section for why) and every
fundamentals gate must degrade to INFO when data is missing — Yahoo fundamentals are
patchy (ADRs, fresh IPOs, trusts), and a gate that FAILs on absent data would condemn
names for Yahoo's coverage gaps.

## Files to change

| file | change |
|---|---|
| `src/forecast/classify.py` | five new gate functions (pure), verdict logic, `classify_universe` wiring |
| `src/forecast/fundamentals.py` | **new** — yfinance fetch + JSON cache + pure derivation of the three metrics |
| `scripts/classify.py` | fundamentals fetch loop + `--no-fundamentals`, table columns, CSV columns |
| `web/api/_classify.py` | vendored copy of all five gates + verdict logic; fundamentals arrive as INFO placeholders |
| `web/app.js` | `CGATES` list gains the new entries |
| `web/README.md` | gate list five → ten in the `/api/classify` section; note fundamentals are CLI-only |
| `tests/test_forecast_classify.py` | new gate + verdict tests |
| `tests/test_forecast_fundamentals.py` | **new** — pure derivations + cache behaviour, downloader injected |
| `tests/test_web_classify.py` | parity for new gates and verdicts |

No changes to `screener.py`, `serve.py`, `web/api/classify.py` (payload just grows),
handlers tests, or `universe.py`.

---

# Part 1 — price-based gates

## src/forecast/classify.py

### New constants

```python
SURV_WINDOW = 126              # matches screener.WINDOW -- the ~6-month horizon
T_HALVED_MILD = 0.05           # same numbers as screener.T_HALVED_MILD/_SEVERE;
T_HALVED_SEVERE = 0.20         #   redeclared here so this module stays self-describing
OWN_SMA_WINDOW = 200           # gate 7, mirrors SMA_WINDOW
```

(Do **not** import the thresholds from `screener` — the two modules deliberately state
their own cutoffs, and the web copy must be numpy+stdlib-standalone anyway. A comment
noting they intentionally match `screener.py` is enough.)

### Gate 6 — `g_survivable(closes) -> tuple[Criterion, dict]`

Same math as `screener.c_survivable` (`src/forecast/screener.py:114-138`), reimplemented
locally (keys/titles differ, and classify's copy must not drift if the screener's
display changes):

```python
px = np.asarray(closes, dtype=float)
# px.size <= SURV_WINDOW + 1  ->  Criterion grade WARN, "insufficient history to judge
#   survivability", stats {} (short history must NOT pass silently -- WARN blocks
#   investable, which is correct for young listings)
fwd = px[SURV_WINDOW:] / px[:-SURV_WINDOW]
halved = float(np.mean(fwd <= 0.5))
worst = float(fwd.min() - 1.0)
max_dd = float(np.max(1.0 - px / np.maximum.accumulate(px)))
grade = PASS if halved < T_HALVED_MILD else WARN if halved < T_HALVED_SEVERE else FAIL
```

- key `"survivable"`, title `"Survivable?"`
- display: `f"{halved:.0%} of {SURV_WINDOW}d windows halved | worst {worst:+.0%} | max DD {-max_dd:.0%}"`
- threshold: `f"pass < {T_HALVED_MILD:.0%}, fail >= {T_HALVED_SEVERE:.0%} of {SURV_WINDOW}d windows halved"`
- notes: FAIL `"halving is routine -- chronic-collapse / serial-diluter profile"`,
  WARN `"occasional halvings"`, PASS `"halvings are rare"`
- stats dict for metrics: `{"halved_126d": halved, "worst_window": worst, "max_dd": max_dd}`

Input is `d["closes"]` — the same full-history (unaligned) close array gate 1 already
uses. More history = better estimate; alignment with SPY is irrelevant here.

### Gate 7 — `g_own_trend(closes) -> tuple[Criterion, bool]`

Structurally identical to `g_regime` but for the name itself:

```python
# px.size < OWN_SMA_WINDOW -> INFO, above=True ("too little history to judge -- not held
#   against the name"), mirroring g_regime's permissive short-history behaviour
sma = float(np.mean(px[-OWN_SMA_WINDOW:]))
above = float(px[-1]) > sma
grade = PASS if above else WARN        # WARN, not FAIL: advisory by design
```

- key `"own_trend"`, title `"Own trend"`
- display: `f"close {last:,.2f} vs its {OWN_SMA_WINDOW}d SMA {sma:,.2f} ({gap:+.1%})"`
- threshold: `f"advisory: close > own {OWN_SMA_WINDOW}d SMA blesses; below only blocks 'investable'"`
- notes: `"in its own uptrend"` / `"below its own {OWN_SMA_WINDOW}d SMA -- blessing withheld"`
- Store the gap into metrics as `"own_sma_gap"` so the CSV writer needs no special-casing.

---

# Part 2 — fundamentals gates

## src/forecast/fundamentals.py (new module)

Follows the `universe.py` pattern exactly: pure derivation functions testable without a
network, a lazy-imported yfinance downloader, a JSON cache in `DATA_DIR`, and
provenance strings so a run is never silently built on stale numbers.

### What is fetched (per symbol, via yfinance — lazy import)

```python
t = yf.Ticker(symbol)
shares = t.get_shares_full(start=<2y ago>)     # pandas Series: date -> share count
bs = t.quarterly_balance_sheet                 # rows incl. "Cash And Cash Equivalents"
                                               #   (+ "Other Short Term Investments" when present)
cf = t.quarterly_cashflow                      # rows incl. "Free Cash Flow"
                                               #   (fallback: "Operating Cash Flow")
inc = t.quarterly_income_stmt                  # row "Total Revenue"
```

Each of the four pulls is independently try/except-wrapped: one missing statement must
not discard the others. The raw pulls are reduced **immediately** to a small plain dict
(the cache never stores DataFrames):

```python
{"shares_now": float|None, "shares_1y_ago": float|None,
 "cash": float|None,                    # latest quarter: cash + short-term investments
 "fcf_quarters": [float, ...],         # up to 4 most recent quarterly FCF values
 "revenue_ttm": float|None,            # sum of up to 4 quarters, None if none reported
 "n_revenue_quarters": int,
 "as_of": "YYYY-MM-DD"}
```

`shares_1y_ago`: the share count at the date closest to 365 days before the latest
observation (tolerate ±45 days; else None). Guard: yfinance share series contain
occasional zero/garbage rows — drop non-positive values before use.

### Pure derivations (unit-test these, no network)

```python
def dilution_yoy(shares_now, shares_1y_ago) -> float | None
    # (now / ago) - 1; None if either is None or ago <= 0

def runway_quarters(cash, fcf_quarters) -> float | None
    # burn = mean of NEGATIVE quarterly FCF values only.
    # FCF >= 0 on average -> float("inf") (self-funding).
    # None if cash is None or fcf_quarters is empty.
    # else cash / abs(burn)

def revenue_ttm(...)   # already reduced at fetch; passthrough
```

### Cache

`DATA_DIR / "fundamentals_cache.json"`: `{symbol: reduced_dict}` with `as_of` per
symbol, TTL **7 days** (fundamentals move quarterly; 7 keeps staleness bounded without
hammering Yahoo). API mirrors `universe.py`:

```python
def fetch_fundamentals(symbols, cache_path=..., max_age_days=7, refresh=False,
                       downloader=None) -> tuple[dict[str, dict], list[str]]
    # returns ({symbol: reduced_dict}, failed_symbols)
    # per-symbol failure -> keep going; failed symbols simply get no fundamentals
    # (their gates grade INFO downstream). Never raise for one bad ticker.
```

A corrupt cache file is treated as absent (same rule as `universe._read_cache`).

## Gates 8–10 in src/forecast/classify.py (pure — they take numbers, never fetch)

### New constants

```python
T_DILUTION_OK = 0.03           # <= 3%/yr share growth: neutral or buying back
T_DILUTION_HEAVY = 0.15        # > 15%/yr: the treadmill
T_RUNWAY_OK = 8.0              # quarters of cash at current burn
T_RUNWAY_CRITICAL = 4.0        # < 4 quarters: forced raise ahead
T_REVENUE_REAL = 10e6          # TTM revenue below this = pre-revenue story
```

### Gate 8 — `g_dilution(dilution: float | None) -> Criterion`

- `None` → INFO, `"share-count history unavailable -- dilution unverified"`. INFO does
  **not** block `investable` (see verdict rationale below) but the note lands in
  `reasons` so an unverified pass is visible.
- PASS `dilution <= T_DILUTION_OK` — note `"share count flat or shrinking"`
- WARN `<= T_DILUTION_HEAVY` — note `"moderate issuance -- watch the treadmill"`
- FAIL otherwise — note `"heavy dilution -- holders are paying for the story"`
- key `"dilution"`, title `"Diluting holders?"`,
  display `f"share count {dilution:+.1%} YoY"`, value = dilution.

### Gate 9 — `g_runway(quarters: float | None) -> Criterion`

- `None` → INFO `"cash-flow statements unavailable -- runway unverified"`
- `inf` → PASS, display `"self-funding (FCF >= 0)"`
- PASS `quarters >= T_RUNWAY_OK`; WARN `>= T_RUNWAY_CRITICAL`;
  FAIL below — note `"under a year of cash -- forced dilution or worse ahead"`
- key `"runway"`, title `"Cash runway"`, display `f"~{quarters:.1f} quarters of cash at current burn"`
  (cap the printed number at 99 for display sanity), value = quarters (JSON: emit
  `None`→null and `inf` as the float cap 99.0 — `json.dumps(float('inf'))` produces
  invalid JSON, so clamp **before** it reaches any payload; parity tests must cover the
  self-funding case).

### Gate 10 — `g_revenue(rev_ttm: float | None, n_quarters: int) -> Criterion`

Advisory (never FAIL — same rationale as the trend gate: whether $8M of revenue is
"real" is a judgement about the future):

- `None` or `n_quarters == 0` → INFO `"income statements unavailable"`
- WARN `rev_ttm < T_REVENUE_REAL` — note `"pre-revenue story -- the price is all narrative"`
- PASS otherwise — display `f"TTM revenue ${rev_ttm/1e6:,.0f}M over {n_quarters}q"`

### Metrics

Add to the metrics dict when available: `dilution_yoy`, `runway_quarters`
(inf clamped to 99.0), `revenue_ttm`.

---

# Verdict logic — replace `_verdict`

```python
def _verdict(c_beta, c_treynor, c_idio, c_surv, c_trend,
             c_dilution, c_runway, c_revenue, risk_on):
    opinion = (c_beta, c_treynor, c_idio, c_surv, c_trend, c_dilution, c_runway, c_revenue)
    # INFO = unverifiable, exempt from the all-PASS requirement (fundamentals coverage
    # gaps and short histories must not make 'investable' unreachable), but the INFO
    # notes still surface in `reasons`.
    investable = all(c.grade in (PASS, INFO) for c in opinion)
    # Reckless = not paid for the beta AND at least one structural red flag.
    # c_trend and c_revenue are deliberately absent: advisory gates never condemn.
    red_flags = (c_idio.grade, c_surv.grade, c_dilution.grade, c_runway.grade)
    reckless = c_treynor.grade == FAIL and FAIL in red_flags
    base = INVESTABLE if investable else RECKLESS if reckless else BORDERLINE
    reasons = [c.note for c in opinion if c.note]
    if base == INVESTABLE and not risk_on:
        return STAND_ASIDE, [...same stand-aside reason as now...] + reasons
    return base, reasons
```

Docstring update: 'reckless' = uncompensated beta **plus** a structural red flag —
binary-event idio, chronic-collapse history, heavy dilution, or a critical runway. One
red flag without the Treynor fail stays borderline; a paid name is never reckless.

**Note the INFO exemption is a deliberate asymmetry against gate 1:** missing *volume*
excludes a name (you cannot clear a liquidity bar that cannot be measured), while
missing *fundamentals* does not block `investable`. Liquidity is measurable for
anything actually tradable, so its absence is itself a red flag; fundamentals coverage
is a property of the data vendor, not the name. Record this in both modules' comments.

## classify_universe wiring

Signature: `classify_universe(per_name, mkt_closes, rf=..., min_adv=..., min_beta=...,
market=...)` — unchanged. Fundamentals ride in as an **optional key** on each name:
`per_name[sym]["fundamentals"] = reduced_dict | None` (absent → all three gates INFO).
This keeps the module pure and the callers (CLI passes real data, web passes nothing)
identical in shape.

In the staged loop (after `c_beta` passes):

```python
c_surv, sstats = g_survivable(d["closes"])
c_trend, _ = g_own_trend(d["closes"])
f = d.get("fundamentals") or {}
c_dil = g_dilution(dilution_from(f))        # pure helpers reading the reduced dict
c_run = g_runway(runway_from(f))
c_rev = g_revenue(f.get("revenue_ttm"), f.get("n_revenue_quarters", 0))
```

(`dilution_from` / `runway_from` live in `fundamentals.py` as the pure derivations
above; classify.py may import them **only if** the web copy re-vendors them — simpler:
have the CLI pre-derive and store `{"dilution_yoy": ..., "runway_quarters": ...,
"revenue_ttm": ..., "n_revenue_quarters": ...}` in the fundamentals dict, so classify.py
and the web copy just read plain numbers. **Choose this.**)

Gates list per result:
`[c_adv, c_beta, regime_c, c_treynor, c_idio, c_surv, c_trend, c_dil, c_run, c_rev]`.
Excluded rows unchanged. Existing sort (verdict rank, then Treynor) unchanged.

## scripts/classify.py

- After the price fetch loop, when `--no-fundamentals` is not set:
  `funda, f_failed = fetch_fundamentals(names)` with a progress line (`fetching
  fundamentals for N names (cached 7d) ...`); report `f_failed` count the same way
  price failures are reported. Attach the derived plain-number dict onto
  `per_name[n]["fundamentals"]`.
- New flag `--no-fundamentals` (fundamentals default ON; the flag exists because 500
  yfinance statement pulls on a full S&P sweep are slow on a cold cache — document in
  the flag help: "price-only run; fundamentals gates grade INFO").
- `--refresh-fundamentals` flag mapping to `refresh=True`.
- `_GATES` gains `("survivable", "survivable"), ("own_trend", "trend"),
  ("dilution", "dilution"), ("runway", "runway"), ("revenue", "revenue")`.
  (Console width check: 10 gate columns × 11 chars ≈ 130 chars — acceptable; if it
  wraps badly, shorten the header labels, not the gates.)
- `_CSV_COLS` gains `halved_126d, max_dd, own_sma_gap, dilution_yoy, runway_quarters,
  revenue_ttm` (before `regime`). All flow through the existing `g()` numeric helper.

## Web (v1 scope: fundamentals shown as unverified, not computed)

The web engine fetches prices from Yahoo's public **chart** endpoint only. Statements
and share counts come from the quoteSummary/fundamentals endpoints, which need the
crumb/cookie dance yfinance implements — re-vendoring that into the serverless copy is
real, brittle work. Therefore:

- `web/api/_classify.py` vendors all five gate functions and the new `_verdict`
  verbatim (dict style). Its `classify()` top-level passes **no fundamentals**, so
  gates 8–10 grade INFO with the "unavailable" notes. Verdicts remain parity-correct
  by construction (INFO is investable-exempt on both sides).
- `gate_order` becomes the full ten-key list.
- `web/app.js` `CGATES` gains the five new entries (INFO renders as the grey `n/a`
  pill already).
- `web/README.md`: document that fundamentals gates are computed by the CLI only and
  always read "unverified" on the site.

Parity stays testable because the gate functions themselves are pure: the parity tests
inject synthetic fundamentals dicts into `classify_universe` on both sides.

---

# Tests

## tests/test_forecast_classify.py

Reuse the existing fixtures (`_mkt`, `_name_data`). Note `_name_data` builds ~300-bar
histories: 300 bars → 174 forward windows for survivability (fine) and exactly 300 for
the own-SMA (fine). New tests:

1. `g_survivable` boundaries: flat series → PASS 0% halved; engineered decay
   `100 * exp(linspace(0, -4, 600))` → FAIL (reuse the screener test's construction);
   mild-decay series landing in (5%, 20%) → WARN; 100 bars → WARN "insufficient" with
   empty stats.
2. `g_own_trend`: rising → PASS; falling → WARN (**assert it is WARN, not FAIL** — the
   anti-momentum rail); 150 bars → INFO with `above=True`.
3. `g_dilution`: None → INFO; 0.02 → PASS; 0.03 → PASS (boundary inclusive); 0.10 →
   WARN; 0.151 → FAIL; negative (buyback) → PASS.
4. `g_runway`: None → INFO; `inf` → PASS with "self-funding"; 9.0 → PASS; 5.0 → WARN;
   3.9 → FAIL; value clamped to 99.0 in the Criterion for `inf`.
5. `g_revenue`: None/0 quarters → INFO; 5e6 → WARN "pre-revenue"; 50e6 → PASS;
   **never FAIL** (assert over the whole input grid).
6. Verdict interactions:
   - treynor FAIL + survivable FAIL → `reckless` (the SPCE case).
   - treynor FAIL + dilution FAIL (others PASS/INFO) → `reckless`.
   - treynor FAIL + runway FAIL → `reckless`.
   - treynor FAIL + everything else PASS/INFO → `borderline` (no red flag).
   - treynor PASS + survivable FAIL → `borderline`, NOT reckless (paid names are never
     reckless — assert the asymmetry).
   - all PASS except own_trend WARN → `borderline` (trend blocks the blessing).
   - all PASS except revenue WARN → `borderline`.
   - all PASS with fundamentals gates INFO (no data) → `investable` (the INFO
     exemption), and the "unverified" notes appear in `reasons`.
   - all PASS including fundamentals → `investable`; risk-off overlay → `stand_aside`.
7. End-to-end `_universe` fixture: attach synthetic fundamentals to a couple of names
   (one clean, one heavy-diluter) and assert bucket movement; GOOD names still reach
   `investable` (if a GOOD name dips below its own SMA by fixture luck, raise its `mu`,
   don't touch the gate). Existing bucket/quartile/sort tests stay green with the
   10-gate list.
8. Metrics: `halved_126d`, `max_dd`, `own_sma_gap` finite for eligible names;
   fundamentals metrics present when supplied, absent (not null-crash) when not.
   Whole payload `json.dumps`-safe **including a self-funding (inf-runway) name**.

## tests/test_forecast_fundamentals.py (new)

All offline; downloader injected.

1. `dilution_yoy`: plain case, None propagation, `ago <= 0` → None.
2. `runway_quarters`: mixed-sign FCF quarters (burn = mean of negatives only);
   all-positive → inf; empty list → None; cash None → None.
3. Reduction from fake yfinance frames: build small pandas objects mimicking
   `get_shares_full` / `quarterly_balance_sheet` / `quarterly_cashflow` /
   `quarterly_income_stmt`; assert the reduced dict, including: zero/garbage share rows
   dropped, `shares_1y_ago` tolerance window (a series with nothing within ±45d of one
   year ago → None), missing cash-flow statement → `fcf_quarters` empty while revenue
   still reduces.
4. Cache: fetch writes; fresh cache short-circuits (downloader that raises
   AssertionError, as in `test_forecast_universe.py`); TTL expiry refetches; corrupt
   file treated as absent; per-symbol failure lands in `failed`, other symbols
   unaffected.

## tests/test_web_classify.py

- Threshold parity list gains `SURV_WINDOW, T_HALVED_MILD, T_HALVED_SEVERE,
  OWN_SMA_WINDOW, T_DILUTION_OK, T_DILUTION_HEAVY, T_RUNWAY_OK, T_RUNWAY_CRITICAL,
  T_REVENUE_REAL`.
- Per-gate parity across the same input grids as the src tests (including None/INFO
  branches and the inf-runway clamp).
- Whole-universe parity with synthetic fundamentals injected on both sides, and a
  second run with **no** fundamentals asserting identical verdicts + INFO grades.

---

# Acceptance criteria (live, after tests are green)

```powershell
.\.venv\Scripts\python -m scripts.classify --symbols SPCE,CRWV
```

- SPCE → `reckless` (treynor FAIL + survivable FAIL ~21% halved; expect the dilution
  gate to corroborate with a FAIL/WARN — report its actual YoY figure).
- CRWV → `borderline` (survivable WARN ~7%) — NOT reckless.
- Both rows show dilution/runway/revenue with real numbers (INFO acceptable per name if
  Yahoo lacks a statement — but if *all* fundamentals come back INFO for both names,
  treat the fetch path as broken and investigate before declaring done).

```powershell
.\.venv\Scripts\python -m scripts.classify --symbols NVDA,MRVL,SMCI,PLTR,COIN,AMD
```

- COIN must remain `reckless` (its idio-FAIL route is untouched).
- NVDA expected `investable` (rare halvings, above own SMA, buybacks → dilution PASS,
  self-funding → runway PASS). If it drops to `borderline`, inspect which gate fired
  and report — do not tune thresholds to force the old verdict.
- `--no-fundamentals` run completes with gates 8–10 INFO and identical
  investable/borderline membership except where fundamentals were the deciding gate.
- Full suite: `.\.venv\Scripts\python -m pytest -q` — all green.
- Web check: start the server (`.venv\Scripts\python web\serve.py`), classify
  `SPCE,CRWV` in the "Investable vs reckless" tab: ten pill columns render,
  fundamentals show the grey n/a pill with "unavailable in the web app"-style notes,
  price-gate verdicts match the CLI.

# Guardrails for the implementer

- The trend and revenue gates must never produce FAIL and never appear in the reckless
  condition — advisory rails, kept deliberately weaker because they shade into return
  prediction, which this project refuses.
- Fundamentals gates INFO on missing data, and INFO never blocks `investable` — the
  opposite of gate 1's missing-volume rule; both asymmetries are deliberate and must be
  commented at the definition site.
- classify.py stays pure: no yfinance import, no cache I/O — plain numbers in, gates
  out. All fetching lives in `fundamentals.py`; all wiring in `scripts/classify.py`.
- The web copy must not attempt fundamentals fetching. INFO placeholders only.
- Clamp `inf` runway before any payload; `json.dumps` on every payload is already a
  test — keep it passing with a self-funding name present.
- Do not import screener thresholds or call `c_survivable` directly; restate constants
  locally with a comment that they match.
- Exit code 255 from PowerShell when piping CLI output is a stderr artifact of yfinance
  progress lines, not a failure — check `$LASTEXITCODE` on a clean run or the presence
  of `wrote ...classify.csv`.
- yfinance statement row labels drift between versions ("Free Cash Flow" vs "FreeCashFlow");
  match row names case-/space-insensitively and fall back to Operating Cash Flow when
  FCF is absent. The reduction tests pin the matching logic.
