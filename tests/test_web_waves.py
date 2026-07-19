"""Parity between the web waves engine (web/api/_waves.py) and src/forecast/waves.py.

The web copy is hand-vendored. Without these tests the two implementations can drift and
the site would quietly show numbers the validation never covered.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))

import _waves as WW  # noqa: E402

from src.forecast import waves as W  # noqa: E402


def _series(seed=3, n=900):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.03, n)))


def _ohlc(px):
    return px, px * 1.02, px * 0.98


def test_atr_pct_parity():
    px, hi, lo = _ohlc(_series())
    assert WW.atr_pct(hi, lo, px) == pytest.approx(W.atr_pct(hi, lo, px))


def test_zigzag_parity():
    px = _series()
    a = WW.zigzag(px, 0.08)
    b = W.zigzag(px, 0.08)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["index"] == y.index
        assert x["price"] == pytest.approx(y.price)
        assert x["kind"] == y.kind
        assert x["confirmed"] == y.confirmed


def test_robust_pivots_parity():
    px = _series(seed=5)
    a = WW.robust_pivots(px, 0.08)
    b = W.robust_pivots(px, 0.08)
    assert [x["robust"] for x in a] == [y.robust for y in b]


def test_count_waves_parity():
    px = _series(seed=7)
    a = WW.count_waves(WW.robust_pivots(px, 0.08))
    b = W.count_waves(W.robust_pivots(px, 0.08))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["kind"] == y.kind
        assert x["stage"] == y.stage
        assert x["score"] == pytest.approx(y.score)


def test_fib_levels_parity():
    a = WW.fib_levels(100.0, 250.0, "up", "s")
    b = W.fib_levels(100.0, 250.0, "up", "s")
    assert [x["label"] for x in a] == [y.label for y in b]
    for x, y in zip(a, b):
        assert x["price"] == pytest.approx(y.price)


def test_swings_used_parity():
    px = _series(seed=11)
    wp = WW.robust_pivots(px, 0.08)
    sp = W.robust_pivots(px, 0.08)
    a = WW.swings_used(wp, WW.count_waves(wp))
    b = W.swings_used(sp, W.count_waves(sp))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == pytest.approx(y[0])
        assert x[1] == pytest.approx(y[1])
        assert x[2] == y[2] and x[3] == y[3]


def test_ambiguity_parity():
    px = _series(seed=13)
    wp, sp = WW.robust_pivots(px, 0.08), W.robust_pivots(px, 0.08)
    assert WW.is_ambiguous(WW.count_waves(wp)) == W.is_ambiguous(W.count_waves(sp))


# ---- web-specific behaviour ----

def test_analyse_payload_is_json_serialisable():
    px, hi, lo = _ohlc(_series())
    out = WW.analyse(px, hi, lo, symbol="TEST")
    assert out["available"] is True
    json.dumps(out)                       # must survive the API boundary
    for key in ("atr_pct", "threshold", "n_pivots", "n_robust", "counts", "ambiguous",
                "fib_below", "fib_above", "verdict"):
        assert key in out


def test_analyse_ladders_are_split_around_spot():
    px, hi, lo = _ohlc(_series(seed=17))
    out = WW.analyse(px, hi, lo, symbol="TEST")
    spot = float(px[-1])
    assert all(l["price"] < spot for l in out["fib_below"])
    assert all(l["price"] >= spot for l in out["fib_above"])
    assert [l["price"] for l in out["fib_below"]] == sorted(
        [l["price"] for l in out["fib_below"]], reverse=True)


def test_analyse_degrades_gracefully_on_short_history():
    px = np.linspace(10, 12, 30)
    out = WW.analyse(px, px * 1.01, px * 0.99, symbol="TEST")
    assert out["available"] is False
    assert "60" in out["reason"]


def test_direction_label_parity_and_semantics():
    """Web labels must match the source module and state travel direction plainly."""
    from src.forecast.waves import WaveCount

    for kind in ("impulse_up", "impulse_down", "correction_up", "correction_down"):
        web = WW.direction_label(kind)
        ref = WaveCount(kind, "s", 0.5, False, []).label
        assert web == ref, (kind, web, ref)
    assert "FALLING" in WW.direction_label("correction_down")
    assert "RISING" in WW.direction_label("correction_up")


def test_analyse_counts_carry_direction_fields():
    px, hi, lo = _ohlc(_series(seed=23))
    out = WW.analyse(px, hi, lo, symbol="TEST")
    for c in out["counts"]:
        assert "label" in c and "rising" in c
        assert c["rising"] == c["kind"].endswith("_up")
        assert ("RISING" if c["rising"] else "FALLING") in c["label"]


def test_wave_labels_origin_then_ending_waves():
    assert WW.wave_labels("impulse_up", 6) == ["0", "1", "2", "3", "4", "5"]
    assert WW.wave_labels("impulse_down", 4) == ["0", "1", "2", "3"]
    assert WW.wave_labels("correction_down", 4) == ["start", "A", "B", "C"]
    assert WW.wave_labels("correction_up", 3) == ["start", "A", "B"]


def test_chart_payload_is_compact_and_consistent():
    px, hi, lo = _ohlc(_series(seed=31, n=1200))
    ch = WW.analyse(px, hi, lo, symbol="TEST")["chart"]
    assert ch["available"] is True
    json.dumps(ch)
    # downsampled, not the whole history
    assert 2 <= len(ch["series"]) <= 245
    xs = [s["i"] for s in ch["series"]]
    assert xs == sorted(xs)
    assert xs[-1] == ch["to"] == px.size - 1          # window ends at the latest bar
    assert all(s["p"] > 0 for s in ch["series"])
    if ch["best"]:
        assert len(ch["best"]["points"]) >= 3
        assert all("l" in p and "i" in p and "p" in p for p in ch["best"]["points"])
        # every labelled point sits inside the drawn window
        assert all(ch["from"] <= p["i"] <= ch["to"] for p in ch["best"]["points"])


def test_chart_only_carries_meaningful_levels():
    px, hi, lo = _ohlc(_series(seed=37, n=900))
    out = WW.analyse(px, hi, lo, symbol="TEST")
    strong_prices = {round(l["price"], 4) for l in out["fib_below"] + out["fib_above"]
                     if l.get("confluent") or l.get("near_pivot")}
    for l in out["chart"]["levels"]:
        assert l["price"] > 0
    assert len(out["chart"]["levels"]) <= 10
    # nothing in the chart's level list should be a weak/unmarked level from the ladders
    weak = {round(l["price"], 4) for l in out["fib_below"] + out["fib_above"]
            if not (l.get("confluent") or l.get("near_pivot"))}
    assert not ({l["price"] for l in out["chart"]["levels"]} & (weak - strong_prices))


def test_chart_degrades_when_no_structure():
    px = np.linspace(10, 12, 40)
    out = WW.analyse(px, px * 1.01, px * 0.99, symbol="TEST")
    assert out["available"] is False          # too short overall; no chart key needed


def test_verdict_is_always_a_string_even_when_unknown():
    v = WW.load_verdict("NOSUCHTICKER")
    assert isinstance(v, str) and v
    assert "decoration" in v or "validate" in v.lower()
