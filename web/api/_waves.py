"""Mechanical Elliott / Fibonacci for the web API (numpy + stdlib only).

Vendored from `src/forecast/waves.py`; `tests/test_web_waves.py` pins the two copies
together so they cannot drift apart silently.

The honesty rails travel with the maths:
  * deterministic ZigZag -> rule-checked counts (reproducible, therefore testable)
  * rival counts returned with scores, and an explicit AMBIGUOUS flag
  * pivots marked robust only if they survive several detection thresholds
  * levels marked "confluent" / "at a prior pivot" vs "floats in empty space"
  * the walk-forward placebo verdict is attached so the UI can stamp it
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

K_ATR = 3.0
THRESHOLD_FLOOR = 0.03
THRESHOLD_CEIL = 0.25
ROBUSTNESS_SCALES = (0.75, 1.0, 1.5)
AMBIGUITY_MARGIN = 0.15
RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSIONS = (1.272, 1.618, 2.618)
CONFLUENCE_TOL = 0.02

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERDICT_PATHS = (
    os.path.join(_HERE, "waves_validation.json"),                       # bundled copy
    os.path.join(_HERE, "..", "..", "data", "waves_validation.json"),   # repo copy
)


# ---------------------------------------------------------------------------
# swings
# ---------------------------------------------------------------------------

def atr_pct(highs, lows, closes, period: int = 14) -> float:
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


def zigzag(prices, threshold: float) -> list[dict]:
    p = np.asarray(prices, dtype=float)
    n = p.size
    if n < 3 or threshold <= 0:
        return []
    out: list[dict] = []
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
            out.append({"index": hi_i, "price": hi_p, "kind": "H", "confirmed": True})
            trend, lo_i, lo_p = -1, i, v
        elif trend <= 0 and v >= lo_p * (1 + threshold):
            out.append({"index": lo_i, "price": lo_p, "kind": "L", "confirmed": True})
            trend, hi_i, hi_p = 1, i, v
    if trend == 1:
        out.append({"index": hi_i, "price": hi_p, "kind": "H", "confirmed": False})
    elif trend == -1:
        out.append({"index": lo_i, "price": lo_p, "kind": "L", "confirmed": False})
    return out


def robust_pivots(prices, base_threshold: float, scales=ROBUSTNESS_SCALES) -> list[dict]:
    base = zigzag(prices, base_threshold)
    if not base:
        return []
    others = [zigzag(prices, base_threshold * s) for s in scales if s != 1.0]
    tol = max(2, len(np.asarray(prices)) // 200)
    for pv in base:
        pv["robust"] = all(
            any(abs(o["index"] - pv["index"]) <= tol and o["kind"] == pv["kind"] for o in oth)
            for oth in others) if others else True
    return base


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------

def _closeness(x: float, ideal: float, width: float = 0.35) -> float:
    if not math.isfinite(x) or ideal <= 0:
        return 0.0
    return float(math.exp(-((x / ideal - 1.0) ** 2) / (2 * width * width)))


def _impulse(points, up):
    p = [q["price"] for q in points]
    s = 1.0 if up else -1.0
    w1 = s * (p[1] - p[0])
    w2 = s * (p[1] - p[2])
    w3 = s * (p[3] - p[2]) if len(p) > 3 else float("nan")
    w4 = s * (p[3] - p[4]) if len(p) > 4 else float("nan")
    w5 = s * (p[5] - p[4]) if len(p) > 5 else float("nan")
    bad = []
    if w1 <= 0:
        bad.append("wave 1 has no extent")
    if w2 >= w1:
        bad.append("rule 1: wave 2 retraced 100%+ of wave 1")
    if len(p) > 3 and w3 <= 0:
        bad.append("wave 3 has no extent")
    if len(p) > 5 and math.isfinite(w3) and math.isfinite(w5) and w3 < w1 and w3 < w5:
        bad.append("rule 2: wave 3 is the shortest of 1/3/5")
    if len(p) > 4 and ((p[4] <= p[1]) if up else (p[4] >= p[1])):
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
    return (not bad), ratios, (float(np.mean(scores)) if scores else 0.0)


def _correction(points, up):
    p = [q["price"] for q in points]
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
    return (not bad), ratios, (float(np.mean(scores)) if scores else 0.0)


_IMPULSE_STAGE = {3: "wave 3 (in progress)", 4: "wave 4 pullback (in progress)",
                  5: "wave 5 (in progress)", 6: "impulse complete -> correction expected"}
_CORRECTION_STAGE = {3: "wave C (in progress)",
                     4: "correction complete -> new impulse expected"}


def direction_label(kind: str) -> str:
    """Unambiguous description of the structure's travel direction.

    The `_up`/`_down` suffix says which way the structure MOVES, not which trend it is
    correcting -- 'correction_down' is a FALLING A-B-C. Rendering the arrow avoids the
    obvious misreading."""
    rising = kind.endswith("_up")
    arrow = "RISING" if rising else "FALLING"
    if kind.startswith("impulse"):
        return f"impulse {arrow} (5-wave {'up' if rising else 'down'})"
    return f"correction {arrow} (A-B-C {'up' if rising else 'down'})"


def count_waves(pivots: list[dict], max_lookback: int = 8) -> list[dict]:
    if len(pivots) < 3:
        return []
    tail = pivots[-max_lookback:]
    out = []
    for start in range(len(tail) - 2):
        pts = tail[start:]
        n = len(pts)
        up = pts[0]["kind"] == "L"
        if 3 <= n <= 6:
            ok, ratios, score = _impulse(pts, up)
            if ok:
                out.append({"kind": "impulse_up" if up else "impulse_down",
                            "stage": _IMPULSE_STAGE.get(n, f"wave {n} (in progress)"),
                            "score": score, "complete": n == 6,
                            "points": list(pts), "ratios": ratios})
        if 3 <= n <= 4:
            ok, ratios, score = _correction(pts, up)
            if ok:
                out.append({"kind": "correction_up" if up else "correction_down",
                            "stage": _CORRECTION_STAGE.get(n, "correction (in progress)"),
                            "score": score * 0.95, "complete": n == 4,
                            "points": list(pts), "ratios": ratios})
    out.sort(key=lambda w: w["score"], reverse=True)
    return out[:3]


def is_ambiguous(counts: list[dict], margin: float = AMBIGUITY_MARGIN) -> bool:
    if len(counts) < 2 or counts[0]["score"] <= 0:
        return len(counts) < 1
    return (counts[0]["score"] - counts[1]["score"]) / counts[0]["score"] < margin


# ---------------------------------------------------------------------------
# fibonacci
# ---------------------------------------------------------------------------

def fib_levels(low, high, direction, source) -> list[dict]:
    if high <= low:
        return []
    span = high - low
    out = []
    for r in RETRACEMENTS:
        px = high - span * r if direction == "up" else low + span * r
        out.append({"label": f"{r * 100:.1f}% retrace", "price": float(px),
                    "kind": "retracement", "source": source})
    for e in EXTENSIONS:
        px = high + span * (e - 1.0) if direction == "up" else low - span * (e - 1.0)
        if px > 0:
            out.append({"label": f"{e * 100:.1f}% extension", "price": float(px),
                        "kind": "extension", "source": source})
    return out


def swings_used(pivots: list[dict], counts: list[dict]) -> list[tuple]:
    out = []
    if len(pivots) >= 2:
        a, b = pivots[-2], pivots[-1]
        out.append((min(a["price"], b["price"]), max(a["price"], b["price"]),
                    "up" if b["kind"] == "H" else "down", "last swing"))
    if len(pivots) >= 3:
        a, b = pivots[-3], pivots[-1]
        out.append((min(a["price"], b["price"]), max(a["price"], b["price"]),
                    "up" if b["price"] >= a["price"] else "down", "prior swing"))
    if counts:
        pts = counts[0]["points"]
        out.append((min(p["price"] for p in pts), max(p["price"] for p in pts),
                    "up" if counts[0]["kind"].endswith("up") else "down", "best count"))
    return out


def annotate(levels: list[dict], pivots: list[dict], tol: float = CONFLUENCE_TOL):
    for lv in levels:
        lv["near_pivot"] = any(
            abs(p["price"] - lv["price"]) / max(lv["price"], 1e-9) <= tol for p in pivots)
        lv["confluent"] = any(
            o["source"] != lv["source"]
            and abs(o["price"] - lv["price"]) / max(lv["price"], 1e-9) <= tol
            for o in levels)
    return levels


def load_verdict(symbol: str) -> str:
    for path in _VERDICT_PATHS:
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            continue
        v = data.get(symbol.upper())
        if not v:
            return (f"Not yet validated for {symbol.upper()}. Until it is, treat these "
                    f"levels as decoration.")
        if str(v.get("verdict", "")).startswith("insufficient"):
            return f"Insufficient history to validate {symbol.upper()} (n={v.get('n', 0)})."
        return (f"Walk-forward test ({v['when']}, n={v['n']}): {v['verdict']}. "
                f"Skill ratio {v.get('skill_ratio')} vs matched random ratios "
                f"(1.0 = no better than chance); fib closer in "
                f"{(v.get('win_rate') or 0) * 100:.0f}% of windows.")
    return "Not yet validated -- run scripts/validate_waves.py. Treat as decoration."


def wave_labels(kind: str, n_points: int) -> list[str]:
    """Point labels: the origin, then the wave that ENDS at each subsequent pivot."""
    if kind.startswith("impulse"):
        return ["0"] + ["1", "2", "3", "4", "5"][:max(0, n_points - 1)]
    return ["start"] + ["A", "B", "C"][:max(0, n_points - 1)]


def _chart_data(closes, counts, levels, spot, pad: int = 60, max_points: int = 240) -> dict:
    """Compact, JSON-safe drawing data: a downsampled price window zoomed to the
    labelled structure, the wave points for the best + rival counts, and only the
    levels worth drawing. Downsampled so the payload stays small."""
    c = np.asarray(closes, dtype=float)
    if counts:
        first = min(p["index"] for p in counts[0]["points"])
        lo = max(0, first - pad)
    else:
        lo = max(0, c.size - 300)
    seg = c[lo:]
    if seg.size < 2:
        return {"available": False}
    step = max(1, seg.size // max_points)
    series = [{"i": int(lo + k), "p": round(float(seg[k]), 4)}
              for k in range(0, seg.size, step)]
    if series[-1]["i"] != c.size - 1:
        series.append({"i": int(c.size - 1), "p": round(float(c[-1]), 4)})

    def pack(cnt):
        if not cnt:
            return None
        labs = wave_labels(cnt["kind"], len(cnt["points"]))
        return {
            "label": direction_label(cnt["kind"]), "stage": cnt["stage"],
            "complete": cnt["complete"],
            "points": [{"i": int(p["index"]), "p": round(float(p["price"]), 4),
                        "l": labs[j] if j < len(labs) else ""}
                       for j, p in enumerate(cnt["points"])],
        }

    vis_lo, vis_hi = float(seg.min()) * 0.85, float(seg.max()) * 1.15
    strong = [{"price": round(float(l["price"]), 4), "label": l["label"]}
              for l in levels
              if (l.get("confluent") or l.get("near_pivot")) and vis_lo <= l["price"] <= vis_hi]
    return {
        "available": True, "from": int(lo), "to": int(c.size - 1),
        "series": series, "spot": round(float(spot), 4),
        "best": pack(counts[0] if counts else None),
        "rival": pack(counts[1] if len(counts) > 1 else None),
        "levels": strong[:10],
    }


def analyse(closes, highs, lows, symbol: str = "", k_atr: float = K_ATR) -> dict:
    c = np.asarray(closes, dtype=float)
    if c.size < 60:
        return {"available": False,
                "reason": "needs at least 60 daily bars for a swing analysis"}
    a_pct = atr_pct(highs, lows, closes)
    threshold = float(np.clip(k_atr * a_pct, THRESHOLD_FLOOR, THRESHOLD_CEIL))
    pivots = robust_pivots(c, threshold)
    counts = count_waves(pivots)
    spot = float(c[-1])
    swings = swings_used(pivots, counts)
    levels = []
    for lo, hi, direction, source in swings:
        levels += fib_levels(lo, hi, direction, source)
    levels = annotate(levels, pivots)
    below = sorted([l for l in levels if l["price"] < spot], key=lambda l: -l["price"])[:6]
    above = sorted([l for l in levels if l["price"] >= spot], key=lambda l: l["price"])[:6]
    return {
        "available": True,
        "atr_pct": a_pct,
        "threshold": threshold,
        "n_pivots": len(pivots),
        "n_robust": sum(1 for p in pivots if p.get("robust")),
        "recent_pivots": pivots[-6:],
        "counts": [dict({k: v for k, v in c0.items() if k != "points"},
                        label=direction_label(c0["kind"]),
                        rising=c0["kind"].endswith("_up")) for c0 in counts],
        "ambiguous": is_ambiguous(counts),
        "fib_below": below,
        "fib_above": above,
        "chart": _chart_data(c, counts, levels, spot),
        "verdict": load_verdict(symbol),
    }
