# Plan: realistic worst/best case for high-beta names (fix the $12-$573 problem)

## Context

The vol-standardized model fixed calibration (TE 0.74 -> 0.90) but produced an ASTS
year-end 90% range of $12-$573 that the user reasonably challenges. Diagnosis:

1. **Vol persistence overdone**: current EWMA vol (8.1%/day, 90th pct of ASTS's own
   history) is held constant for the whole horizon. Measured vol half-life ~162
   trading days -- persistent, but not infinite; a term structure should decay it
   toward the long-run level (5.6-6%/day).
2. **Stale-regime blocks**: the upper tail resamples ASTS's 2024 13.77x micro-cap
   mania. At ~30x that market cap, those blocks overstate today's upside. The
   sampler should prefer recent history over ancient regimes.
3. **"Worst/best case" needs anchors, not just percentiles**: the report should place
   statistical bands next to the name's own realized extremes (worst actual 126d
   stretch -58%; 6% of windows >4x) so the user can see what's precedent vs tail.

> Honesty rail: we can NARROW the range only where the data justifies it. If
> validation shows the narrower cone breaks coverage, the wide cone stands.

## Phase 1 -- vol term structure (src/forecast/beta.py)

- `vol_half_life(rets)`: fit from 21-day autocorrelation of log EWMA vol.
- Extend `joint_bootstrap_prices(..., vol_schedule=...)`: per-day sigma multiplier
  `sigma_t = long_run + (current - long_run) * exp(-t/hl)` per asset, replacing the
  constant-current-vol assumption. Default ON for the report (`--flat-vol` to disable).

## Phase 2 -- recency-weighted block sampling (src/forecast/beta.py)

- `joint_bootstrap_prices(..., recency_half_life_days=500)`: sample block START
  indices with exponential recency weights instead of uniform. The 2024 mania still
  appears (it should -- it happened) but at a weight reflecting its age, not parity
  with last month. Weight floor so early history never fully vanishes.

## Phase 3 -- "realistic bounds" section in scripts/beta_outlook.py

Per name, print a bounds ladder with explicit provenance:
```
            statistical      historical precedent
worst 1%    $X               (worst actual 126d: -58% -> $24.3)
worst 5%    $X               ...
median      $X
best 5%     $X               (6% of 126d windows gained >4x -- at 1/30th today's cap)
best 1%     $X
```
plus one line each: "catastrophic scenario (financing/launch failure): approaches $0,
not modelable from prices" and "mania scenario: requires 2024-style repricing at
today's scale -- no precedent at this cap."

## Phase 4 -- re-validate (the gate)

- `--validate` grows a `beta+volstd+ts` method (term structure + recency). Keep it
  only if 51/73d coverage stays consistent with 0.90 (CIs) AND the 116d cone narrows.
- Report ASTS/IREN/TE before/after ranges in the summary.

## Tests

- half-life fit on synthetic AR(1) log-vol recovers the input half-life (+/-30%).
- vol_schedule: day-1 sigma ~= current, day-inf ~= long-run; monotone decay.
- recency weights: newer blocks sampled more often (frequency ratio matches weights);
  determinism preserved; correlation preservation still holds.
- validation smoke: new method runs and prints CIs.

## Verification

1. pytest green.
2. `--validate --vpaths 3000` on ASTS,IREN,TE: new method coverage consistent w/ 0.90.
3. Report: ASTS range narrows from $12-$573 toward something defensible; document
   the before/after and WHY in the output.
