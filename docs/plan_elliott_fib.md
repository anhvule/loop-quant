# Plan: mechanical Elliott Wave stage + Fibonacci levels for high-beta names

## Context and honesty rails

The user wants: which Elliott stage a stock is in, and its Fibonacci bottom/top levels.
Elliott Wave is subjective in the wild and has no demonstrated predictive validity;
Fibonacci levels are mechanical but weakly evidenced. We therefore build:

1. a **deterministic** implementation (ZigZag pivots -> rule-based counts -> fib levels)
   so the same chart always yields the same answer and the method is *testable*;
2. **rival counts shown with scores** -- ambiguity is displayed, never hidden behind one
   confident label;
3. a **validation phase** measuring, walk-forward, whether fib levels and wave stages
   beat placebo. Its verdict is stamped on every output (like the calibration status in
   beta_outlook). If the answer is "decorative, not predictive", the tool says so.

> NOT INVESTMENT ADVICE. This adds a *description* of price structure, not a validated
> forecast. The calibrated risk machinery (beta_outlook) remains the headline tool.

## Phase 1 -- swing detection: `src/forecast/waves.py`

- `zigzag(closes, highs, lows, threshold)` -> alternating pivot list [(idx, price, hi/lo)].
  Reversal threshold is ATR-scaled (k x ATR%, default ~3) with a %-floor, so a 6%/day
  name doesn't produce a pivot every week and a calm name still produces some.
- Threshold sensitivity is part of the output: pivots recomputed at 0.75x/1x/1.5x
  threshold; pivots that survive all three are "robust", others "fragile".

## Phase 2 -- mechanical wave counting

- Candidate 5-3 counts over the last N pivots, enforcing the three HARD rules:
  W2 retraces < 100% of W1; W3 never the shortest of 1/3/5; W4 does not overlap W1
  (for impulses). Corrections labeled A-B-C.
- Score each candidate by soft guidelines: fib proportions between waves (W2 ~0.5-0.618
  of W1, W3 ~1.618x W1, W4 ~0.382 of W3), alternation, channel fit.
- `wave_state()` returns the TOP 3 counts with scores and the implied current stage
  ("in W3 of impulse up", "in wave C of correction", ...). If scores are within 15%
  of each other, the stage is reported as "AMBIGUOUS between X and Y" -- by design.

## Phase 3 -- Fibonacci levels

- From the last robust swing (and from the best count's wave structure):
  * retracements: 23.6 / 38.2 / 50 / 61.8 / 78.6% -> the "fib bottom" ladder below price
    (or above, in a downtrend);
  * extensions: 127.2 / 161.8 / 261.8% -> the "fib top" ladder / targets.
- Confluence detection: levels from different swings within 2% of each other are
  merged and flagged "confluent" (the only fib levels with even anecdotal strength).
- Each level annotated with distance from spot and whether it coincides with a prior
  pivot (real support) or floats in empty space.

## Phase 4 -- the honesty gate: does any of this predict?

`scripts/validate_waves.py`, walk-forward over SPY + ASTS/IREN/TE (+700-bar names):
- **Fib test**: at each month-end, compute levels from data-so-far; measure whether
  forward price paths reverse (>=1 ATR bounce) within +/-1.5% of a fib level more often
  than at PLACEBO levels (random levels drawn uniformly over the same range, 200 draws).
  Report hit-rate vs placebo with binomial CIs (reuse `src/forecast/calib.py`).
- **Stage test**: bucket forward 21/63-day returns by reported stage (W3 vs W5 vs
  correction...). If stage carries signal, buckets differ beyond noise; test with the
  same CI machinery. Expect: no significant difference -- and print whatever we find.
- Verdict persisted to `data/waves_validation.json`; every waves output stamps it,
  e.g. "fib levels: not distinguishable from placebo (n=...)" or the reverse.

## Phase 5 -- outputs

- `scripts/waves.py --symbol ASTS`: console summary (stage + rival counts + fib ladders
  + robustness + validation stamp) and `data/waves_<SYM>.png` chart: price, pivots,
  labeled best count, fib levels drawn as horizontal lines with confluence highlighted.
- Optional later: a card in the web app (only after Phase 4's verdict exists, so the
  stamp is never missing).

## Tests

- zigzag: alternation invariant, threshold monotonicity (higher threshold -> fewer
  pivots), determinism, ATR scaling.
- wave rules: synthetic textbook impulse labeled correctly; rule violations rejected
  (W4 overlapping W1 kills a count); ambiguity reported when two counts score close.
- fib: level arithmetic exact; confluence merging; downtrend mirroring.
- validation: placebo machinery on synthetic data returns ~equal hit rates (no
  false positives by construction).

## Verification

1. pytest green.
2. `python -m scripts.validate_waves` -> verdicts with CIs for SPY/ASTS/IREN/TE.
3. `python -m scripts.waves --symbol ASTS` -> chart + stage + fib ladder, stamped with
   the validation verdict; sanity-check the count against the visible swings.
