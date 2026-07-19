"""Mechanical Elliott Wave staging and Fibonacci levels.

READ THIS FIRST. Elliott Wave as practised is subjective: two analysts routinely produce
opposite counts from one chart, and there is no credible evidence of predictive power.
Fibonacci levels are mechanical but only weakly evidenced -- at best self-fulfilling
focal points. This module therefore:

  * is fully DETERMINISTIC (ATR-scaled ZigZag -> rule-checked counts), so the same
    prices always give the same answer and the method can actually be tested;
  * reports RIVAL counts with scores and marks them AMBIGUOUS when close, instead of
    projecting false confidence;
  * marks pivots "robust" only if they survive several detection thresholds;
  * is designed to be judged by `scripts/validate_waves.py`, which tests the levels
    against random placebo levels. Whatever that says is stamped on every output.

Treat the output as a DESCRIPTION of price structure, never as a forecast. The
calibrated risk tooling (beta_outlook / outlook) remains the predictive machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ZigZag threshold = clamp(K_ATR * ATR%, FLOOR, CEIL). A 6%/day name needs a much wider
# swing filter than SPY or every wiggle becomes a "wave".
K_ATR = 3.0
THRESHOLD_FLOOR = 0.03
THRESHOLD_CEIL = 0.25
ROBUSTNESS_SCALES = (0.75, 1.0, 1.5)
AMBIGUITY_MARGIN = 0.15          # top-2 scores within 15% -> report as ambiguous

RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSIONS = (1.272, 1.618, 2.618)
CONFLUENCE_TOL = 0.02            # levels within 2% merge into one zone


@dataclass(frozen=True, slots=True)
class Pivot:
    index: int
    price: float
    kind: str            # "H" or "L"
    confirmed: bool = True
    robust: bool = False


@dataclass(frozen=True, slots=True)
class WaveCount:
    kind: str            # impulse_up | impulse_down | correction_up | correction_down
    stage: str           # human label of the wave currently in progress
    score: float
    complete: bool
    points: list[Pivot]
    ratios: dict = field(default_factory=dict)

    @property
    def rising(self) -> bool:
        """Does this structure TRAVEL upward? The `_up`/`_down` suffix describes the
        direction the structure moves, NOT the trend it is correcting -- so
        `correction_down` is a falling A-B-C, not 'a correction of a downtrend'."""
        return self.kind.endswith("_up")

    @property
    def label(self) -> str:
        """Unambiguous human description, e.g. 'correction FALLING (A-B-C down)'."""
        arrow = "RISING" if self.rising else "FALLING"
        if self.kind.startswith("impulse"):
            return f"impulse {arrow} (5-wave {'up' if self.rising else 'down'})"
        return f"correction {arrow} (A-B-C {'up' if self.rising else 'down'})"


@dataclass(frozen=True, slots=True)
class Level:
    label: str
    price: float
    kind: str            # "retracement" | "extension"
    source: str
    near_pivot: bool = False
    confluent: bool = False


# ---------------------------------------------------------------------------
# swing detection
# ---------------------------------------------------------------------------

def atr_pct(highs, lows, closes, period: int = 14) -> float:
    """ATR as a fraction of price (Wilder smoothing), used to scale the swing filter."""
    h, l, c = np.asarray(highs, float), np.asarray(lows, float), np.asarray(closes, float)
    if c.size < 2:
        return 0.02
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
    if tr.size == 0:
        return 0.02
    a = float(np.mean(tr[:period])) if tr.size >= period else float(np.mean(tr))
    for x in tr[period:]:
        a = (a * (period - 1) + float(x)) / period
    px = float(c[-1])
    return float(np.clip(a / px if px > 0 else 0.02, 1e-4, 0.5))


def zigzag(prices, threshold: float) -> list[Pivot]:
    """Alternating swing pivots. A pivot is confirmed only once price reverses by
    `threshold` from the running extreme, so pivots never repaint once emitted.

    The final element is the *provisional* extreme (confirmed=False): the swing still
    in progress. Any honest reading of "what wave are we in" depends on it."""
    p = np.asarray(prices, dtype=float)
    n = p.size
    if n < 3 or threshold <= 0:
        return []
    out: list[Pivot] = []
    hi_i = lo_i = 0
    hi_p = lo_p = float(p[0])
    trend = 0
    for i in range(1, n):
        v = float(p[i])
        if trend >= 0 and v > hi_p:
            hi_i, hi_p = i, v
        if trend <= 0 and v < lo_p:
            lo_i, lo_p = i, v
        if trend >= 0 and v <= hi_p * (1 - threshold):
            out.append(Pivot(hi_i, hi_p, "H"))
            trend, lo_i, lo_p = -1, i, v
        elif trend <= 0 and v >= lo_p * (1 + threshold):
            out.append(Pivot(lo_i, lo_p, "L"))
            trend, hi_i, hi_p = 1, i, v
    if trend == 1:
        out.append(Pivot(hi_i, hi_p, "H", confirmed=False))
    elif trend == -1:
        out.append(Pivot(lo_i, lo_p, "L", confirmed=False))
    return out


def robust_pivots(prices, base_threshold: float,
                  scales=ROBUSTNESS_SCALES) -> list[Pivot]:
    """Pivots at the base threshold, flagged robust when they also appear (within a
    few bars) at every other threshold. Fragile pivots are where wave counts flip."""
    base = zigzag(prices, base_threshold)
    if not base:
        return []
    others = [zigzag(prices, base_threshold * s) for s in scales if s != 1.0]
    tol = max(2, len(np.asarray(prices)) // 200)
    out = []
    for pv in base:
        ok = all(any(abs(o.index - pv.index) <= tol and o.kind == pv.kind for o in oth)
                 for oth in others) if others else True
        out.append(Pivot(pv.index, pv.price, pv.kind, pv.confirmed, ok))
    return out


# ---------------------------------------------------------------------------
# wave counting
# ---------------------------------------------------------------------------

def _closeness(x: float, ideal: float, width: float = 0.35) -> float:
    """1.0 when x == ideal, decaying smoothly. Used to score fib proportions."""
    if not math.isfinite(x) or ideal <= 0:
        return 0.0
    return float(math.exp(-((x / ideal - 1.0) ** 2) / (2 * width * width)))


def _impulse(points: list[Pivot], up: bool) -> tuple[bool, list[str], dict, float]:
    """Check the three HARD Elliott rules and score the soft fib guidelines.

    Hard rules (a count violating any of these is not an impulse, full stop):
      1. wave 2 never retraces more than 100% of wave 1
      2. wave 3 is never the shortest of waves 1, 3, 5
      3. wave 4 never enters wave 1's price territory
    """
    p = [q.price for q in points]
    s = 1.0 if up else -1.0
    w1 = s * (p[1] - p[0])
    w2 = s * (p[1] - p[2])
    w3 = s * (p[3] - p[2]) if len(p) > 3 else float("nan")
    w4 = s * (p[3] - p[4]) if len(p) > 4 else float("nan")
    w5 = s * (p[5] - p[4]) if len(p) > 5 else float("nan")

    bad: list[str] = []
    if w1 <= 0:
        bad.append("wave 1 has no extent")
    if w2 >= w1:
        bad.append("rule 1: wave 2 retraced 100%+ of wave 1")
    if len(p) > 3 and w3 <= 0:
        bad.append("wave 3 has no extent")
    if len(p) > 5 and math.isfinite(w3) and math.isfinite(w5):
        if w3 < w1 and w3 < w5:
            bad.append("rule 2: wave 3 is the shortest of 1/3/5")
    if len(p) > 4:
        overlap = (p[4] <= p[1]) if up else (p[4] >= p[1])
        if overlap:
            bad.append("rule 3: wave 4 overlaps wave 1")

    ratios, scores = {}, []
    if w1 > 0:
        r2 = w2 / w1
        ratios["w2_retrace"] = r2
        scores.append(max(_closeness(r2, 0.5), _closeness(r2, 0.618)))
        if math.isfinite(w3):
            e3 = w3 / w1
            ratios["w3_x_w1"] = e3
            scores.append(max(_closeness(e3, 1.618), _closeness(e3, 2.618)))
        if math.isfinite(w4) and math.isfinite(w3) and w3 > 0:
            r4 = w4 / w3
            ratios["w4_retrace"] = r4
            scores.append(max(_closeness(r4, 0.382), _closeness(r4, 0.5)))
        if math.isfinite(w5):
            e5 = w5 / w1
            ratios["w5_x_w1"] = e5
            scores.append(max(_closeness(e5, 1.0), _closeness(e5, 0.618)))
    return (not bad), bad, ratios, (float(np.mean(scores)) if scores else 0.0)


def _correction(points: list[Pivot], up: bool) -> tuple[bool, list[str], dict, float]:
    """A-B-C. Only a light rule: B must not fully retrace A (that would be a new trend)."""
    p = [q.price for q in points]
    s = 1.0 if up else -1.0
    a = s * (p[1] - p[0])
    b = s * (p[1] - p[2])
    c = s * (p[3] - p[2]) if len(p) > 3 else float("nan")
    bad = []
    if a <= 0:
        bad.append("wave A has no extent")
    if b >= a:
        bad.append("wave B retraced 100%+ of wave A")
    ratios, scores = {}, []
    if a > 0:
        rb = b / a
        ratios["b_retrace"] = rb
        scores.append(max(_closeness(rb, 0.5), _closeness(rb, 0.618), _closeness(rb, 0.786)))
        if math.isfinite(c):
            rc = c / a
            ratios["c_x_a"] = rc
            scores.append(max(_closeness(rc, 1.0), _closeness(rc, 1.618)))
    return (not bad), bad, ratios, (float(np.mean(scores)) if scores else 0.0)


_IMPULSE_STAGE = {3: "wave 3 (in progress)", 4: "wave 4 pullback (in progress)",
                  5: "wave 5 (in progress)", 6: "impulse complete -> correction expected"}
_CORRECTION_STAGE = {3: "wave C (in progress)", 4: "correction complete -> new impulse expected"}
# NOTE: "-> ... expected" describes what Elliott theory says comes NEXT structurally.
# It is not a validated directional forecast -- the walk-forward stage test found no
# usable signal in forward returns. Treat it as a label, not a prediction.


def count_waves(pivots: list[Pivot], max_lookback: int = 8) -> list[WaveCount]:
    """Enumerate rule-valid structures ending at the most recent pivot; score and rank.

    Partial structures are included on purpose -- the useful question is "which wave are
    we in *now*", which is by definition an incomplete count."""
    if len(pivots) < 3:
        return []
    tail = pivots[-max_lookback:]
    out: list[WaveCount] = []
    for start in range(len(tail) - 2):
        pts = tail[start:]
        n = len(pts)
        up = pts[0].kind == "L"                    # impulse up starts from a low
        if 3 <= n <= 6:
            ok, bad, ratios, score = _impulse(pts, up)
            if ok:
                out.append(WaveCount(
                    kind="impulse_up" if up else "impulse_down",
                    stage=_IMPULSE_STAGE.get(n, f"wave {n} (in progress)"),
                    score=score, complete=(n == 6), points=list(pts), ratios=ratios))
        if 3 <= n <= 4:
            ok, bad, ratios, score = _correction(pts, up)
            if ok:
                out.append(WaveCount(
                    kind="correction_up" if up else "correction_down",
                    stage=_CORRECTION_STAGE.get(n, "correction (in progress)"),
                    # a correction reading is a weaker structural claim than a full
                    # impulse; nudge it down so ties break toward the stricter fit
                    score=score * 0.95, complete=(n == 4),
                    points=list(pts), ratios=ratios))
    out.sort(key=lambda w: w.score, reverse=True)
    return out[:3]


def is_ambiguous(counts: list[WaveCount], margin: float = AMBIGUITY_MARGIN) -> bool:
    """True when the top two readings are too close to call -- which is common, and
    the single most important thing an Elliott tool can be honest about."""
    if len(counts) < 2 or counts[0].score <= 0:
        return len(counts) < 1
    return (counts[0].score - counts[1].score) / counts[0].score < margin


# ---------------------------------------------------------------------------
# Fibonacci levels
# ---------------------------------------------------------------------------

def fib_levels(low: float, high: float, direction: str, source: str) -> list[Level]:
    """Retracements inside [low, high] plus extensions beyond the swing's end.

    direction "up": the swing ran low -> high, so retracements are potential SUPPORT
    below and extensions are targets above. "down" mirrors it."""
    if high <= low:
        return []
    span = high - low
    out: list[Level] = []
    for r in RETRACEMENTS:
        px = high - span * r if direction == "up" else low + span * r
        out.append(Level(f"{r * 100:.1f}% retrace", float(px), "retracement", source))
    for e in EXTENSIONS:
        px = high + span * (e - 1.0) if direction == "up" else low - span * (e - 1.0)
        if px > 0:
            out.append(Level(f"{e * 100:.1f}% extension", float(px), "extension", source))
    return out


def mark_near_pivots(levels: list[Level], pivots: list[Pivot],
                     tol: float = CONFLUENCE_TOL) -> list[Level]:
    """Flag levels that sit on a historical turning point. A fib level backed by a real
    prior pivot is at least *plausible* support; one floating in empty space is a line."""
    out = []
    for lv in levels:
        near = any(abs(p.price - lv.price) / max(lv.price, 1e-9) <= tol for p in pivots)
        out.append(Level(lv.label, lv.price, lv.kind, lv.source, near, lv.confluent))
    return out


def find_confluence(levels: list[Level], tol: float = CONFLUENCE_TOL) -> list[Level]:
    """Mark levels that cluster with a level from a *different* swing."""
    out = []
    for lv in levels:
        hit = any(other.source != lv.source
                  and abs(other.price - lv.price) / max(lv.price, 1e-9) <= tol
                  for other in levels)
        out.append(Level(lv.label, lv.price, lv.kind, lv.source, lv.near_pivot, hit))
    return out


def ladder(levels: list[Level], spot: float):
    """Split into levels below spot (the 'fib bottom' ladder) and above (the 'fib top')."""
    below = sorted([l for l in levels if l.price < spot], key=lambda l: -l.price)
    above = sorted([l for l in levels if l.price >= spot], key=lambda l: l.price)
    return below, above


def swings_used(pivots: list[Pivot], counts: list[WaveCount]) -> list[tuple]:
    """The (low, high, direction, source) swings that `analyse` derives levels from.

    Exposed so validation can build a STRICTER placebo: random ratios of these same
    swings. Beating uniform-random lines only shows that levels near real price
    structure matter; beating random ratios of the same swing is what would actually
    implicate the Fibonacci numbers themselves."""
    out = []
    if len(pivots) >= 2:
        a, b = pivots[-2], pivots[-1]
        out.append((min(a.price, b.price), max(a.price, b.price),
                    "up" if b.kind == "H" else "down", "last swing"))
    if len(pivots) >= 3:
        a, b = pivots[-3], pivots[-1]
        out.append((min(a.price, b.price), max(a.price, b.price),
                    "up" if b.price >= a.price else "down", "prior swing"))
    if counts:
        pts = counts[0].points
        out.append((min(p.price for p in pts), max(p.price for p in pts),
                    "up" if counts[0].kind.endswith("up") else "down", "best count"))
    return out


def random_ratio_levels(swings: list[tuple], rng, n_retr: int = len(RETRACEMENTS),
                        n_ext: int = len(EXTENSIONS)) -> np.ndarray:
    """Placebo levels: the SAME swings, same COUNT, same ratio SPAN -- but random ratios.

    The span is matched to the Fibonacci range deliberately (retracements over
    [0.236, 0.786], extensions over [1.272, 2.618]). A wider placebo span would spread
    levels further from the swing's middle, and since turning points cluster inside the
    recent range, fib would win on concentration alone rather than on its actual values.
    Matching the span isolates the one question worth asking: do the specific Fibonacci
    numbers matter, or would any levels drawn from the same swing do just as well?"""
    lo_r, hi_r = min(RETRACEMENTS), max(RETRACEMENTS)
    lo_e, hi_e = min(EXTENSIONS), max(EXTENSIONS)
    out = []
    for lo, hi, direction, _ in swings:
        span = hi - lo
        if span <= 0:
            continue
        for r in rng.uniform(lo_r, hi_r, n_retr):
            out.append(hi - span * r if direction == "up" else lo + span * r)
        for e in rng.uniform(lo_e, hi_e, n_ext):
            px = hi + span * (e - 1.0) if direction == "up" else lo - span * (e - 1.0)
            if px > 0:
                out.append(px)
    return np.asarray(out, dtype=float)


def analyse(closes, highs, lows, k_atr: float = K_ATR) -> dict:
    """Full mechanical read: pivots, rival counts, fib ladders. No forecast."""
    c = np.asarray(closes, dtype=float)
    if c.size < 60:
        raise ValueError("need at least 60 bars for a swing analysis")
    a_pct = atr_pct(highs, lows, closes)
    threshold = float(np.clip(k_atr * a_pct, THRESHOLD_FLOOR, THRESHOLD_CEIL))
    pivots = robust_pivots(c, threshold)
    counts = count_waves(pivots)
    spot = float(c[-1])

    swings = swings_used(pivots, counts)
    levels: list[Level] = []
    for lo, hi, direction, source in swings:
        levels += fib_levels(lo, hi, direction, source)
    levels = find_confluence(mark_near_pivots(levels, pivots))
    below, above = ladder(levels, spot)
    return {
        "spot": spot,
        "atr_pct": a_pct,
        "threshold": threshold,
        "pivots": pivots,
        "counts": counts,
        "ambiguous": is_ambiguous(counts),
        "swings": swings,
        "levels": levels,
        "fib_below": below,
        "fib_above": above,
    }
