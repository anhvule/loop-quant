"""Mechanical Elliott/Fibonacci: pivot detection, hard rules, fib arithmetic, ambiguity."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast.waves import (
    Level,
    Pivot,
    analyse,
    atr_pct,
    count_waves,
    fib_levels,
    find_confluence,
    is_ambiguous,
    ladder,
    mark_near_pivots,
    robust_pivots,
    zigzag,
)


def _leg(start, end, n):
    return list(np.linspace(start, end, n, endpoint=False))


def _impulse_series(up=True):
    """A textbook 5-wave impulse with clean fib proportions (W2 .5, W3 1.618, W4 .382)."""
    pts = [100, 120, 110, 142.4, 130.0, 150] if up else [150, 130, 140, 107.6, 120, 100]
    s = []
    for a, b in zip(pts, pts[1:]):
        s += _leg(a, b, 25)
    s.append(pts[-1])
    return np.asarray(s, float)


# ---- zigzag ----

def test_zigzag_alternates_and_is_deterministic():
    p = _impulse_series()
    a = zigzag(p, 0.05)
    b = zigzag(p, 0.05)
    assert [(x.index, x.kind) for x in a] == [(x.index, x.kind) for x in b]
    kinds = [x.kind for x in a]
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)), kinds


def test_zigzag_last_pivot_is_provisional():
    p = _impulse_series()
    pv = zigzag(p, 0.05)
    assert pv and pv[-1].confirmed is False
    assert all(x.confirmed for x in pv[:-1])


def test_zigzag_threshold_monotonicity():
    p = _impulse_series()
    assert len(zigzag(p, 0.03)) >= len(zigzag(p, 0.12))


def test_zigzag_finds_the_real_turning_points():
    p = _impulse_series()
    pv = [x for x in zigzag(p, 0.05) if x.confirmed]
    prices = sorted(round(x.price, 1) for x in pv)
    # the interior turning points of the constructed impulse
    for want in (110.0, 120.0, 130.0):
        assert any(abs(pr - want) < 1.5 for pr in prices), (want, prices)


def test_zigzag_handles_degenerate_input():
    assert zigzag([1.0, 1.0], 0.05) == []
    assert zigzag(np.ones(50), 0.05) == []          # flat series -> no swings


def test_robust_pivots_flag_survivors():
    p = _impulse_series()
    pv = robust_pivots(p, 0.05)
    assert pv
    assert any(x.robust for x in pv)


def test_atr_pct_scales_with_volatility():
    n = 300
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    a_calm = atr_pct(calm * 1.005, calm * 0.995, calm)
    a_wild = atr_pct(wild * 1.05, wild * 0.95, wild)
    assert a_wild > a_calm


# ---- wave rules ----

def test_textbook_impulse_is_recognised():
    p = _impulse_series(up=True)
    counts = count_waves(robust_pivots(p, 0.05))
    assert counts
    assert any(c.kind == "impulse_up" for c in counts)
    assert counts[0].score > 0.3


def test_rule3_wave4_overlap_rejects_impulse():
    """W4 dipping into W1 territory must invalidate an impulse reading."""
    good = [Pivot(0, 100, "L"), Pivot(1, 120, "H"), Pivot(2, 110, "L"),
            Pivot(3, 140, "H"), Pivot(4, 125, "L"), Pivot(5, 150, "H")]
    bad = list(good)
    bad[4] = Pivot(4, 115, "L")            # below the wave-1 high of 120 -> overlap
    assert any(c.kind == "impulse_up" for c in count_waves(good))
    got = [c for c in count_waves(bad) if c.kind == "impulse_up" and len(c.points) == 6]
    assert not got


def test_rule1_wave2_full_retrace_rejects_impulse():
    bad = [Pivot(0, 100, "L"), Pivot(1, 120, "H"), Pivot(2, 99, "L"),
           Pivot(3, 140, "H"), Pivot(4, 130, "L"), Pivot(5, 150, "H")]
    got = [c for c in count_waves(bad) if c.kind == "impulse_up" and len(c.points) == 6]
    assert not got


def test_rule2_shortest_wave3_rejects_impulse():
    # W1 = 30, W3 = 5, W5 = 30 -> wave 3 is the shortest, which is illegal
    bad = [Pivot(0, 100, "L"), Pivot(1, 130, "H"), Pivot(2, 120, "L"),
           Pivot(3, 125, "H"), Pivot(4, 122, "L"), Pivot(5, 152, "H")]
    got = [c for c in count_waves(bad) if c.kind == "impulse_up" and len(c.points) == 6]
    assert not got


def test_partial_counts_report_wave_in_progress():
    partial = [Pivot(0, 100, "L"), Pivot(1, 120, "H"), Pivot(2, 110, "L"),
               Pivot(3, 142, "H")]
    counts = count_waves(partial)
    assert counts
    assert any("wave 4" in c.stage or "wave 3" in c.stage or "wave C" in c.stage
               for c in counts)


def test_direction_suffix_means_travel_direction_not_corrected_trend():
    """Pins the semantics a user reasonably misread: `correction_down` is a FALLING
    A-B-C, not 'a correction of a downtrend' (which would rise)."""
    from src.forecast.waves import WaveCount

    falling = WaveCount("correction_down", "wave C (in progress)", 0.5, False, [])
    rising = WaveCount("correction_up", "wave C (in progress)", 0.5, False, [])
    assert falling.rising is False and "FALLING" in falling.label
    assert rising.rising is True and "RISING" in rising.label
    assert "A-B-C down" in falling.label and "A-B-C up" in rising.label

    imp = WaveCount("impulse_down", "wave 3 (in progress)", 0.5, False, [])
    assert imp.rising is False and "impulse FALLING" in imp.label


def test_correction_down_is_built_from_a_falling_abc():
    """A structure starting at a HIGH and ending lower must be labelled falling."""
    pts = [Pivot(0, 133.0, "H"), Pivot(1, 65.0, "L"),
           Pivot(2, 89.0, "H"), Pivot(3, 55.0, "L")]
    counts = count_waves(pts)
    corr = [c for c in counts if c.kind.startswith("correction")]
    assert corr, [c.kind for c in counts]
    assert corr[0].kind == "correction_down"
    assert corr[0].rising is False
    assert pts[-1].price < pts[0].price          # price genuinely fell


def test_ambiguity_detection():
    close = [type("C", (), {"score": 0.80})(), type("C", (), {"score": 0.76})()]
    clear = [type("C", (), {"score": 0.90})(), type("C", (), {"score": 0.40})()]
    assert is_ambiguous(close)
    assert not is_ambiguous(clear)


# ---- fibonacci ----

def test_fib_retracement_arithmetic_up():
    lv = {l.label: l.price for l in fib_levels(100.0, 200.0, "up", "s")}
    assert lv["50.0% retrace"] == pytest.approx(150.0)
    assert lv["61.8% retrace"] == pytest.approx(138.2)
    assert lv["161.8% extension"] == pytest.approx(261.8)


def test_fib_mirrors_for_downtrend():
    lv = {l.label: l.price for l in fib_levels(100.0, 200.0, "down", "s")}
    assert lv["50.0% retrace"] == pytest.approx(150.0)
    assert lv["61.8% retrace"] == pytest.approx(161.8)
    assert lv["161.8% extension"] == pytest.approx(38.2)


def test_fib_rejects_degenerate_swing():
    assert fib_levels(100.0, 100.0, "up", "s") == []


def test_confluence_only_across_different_swings():
    a = Level("x", 100.0, "retracement", "swing A")
    b = Level("y", 100.5, "retracement", "swing B")     # 0.5% away, different source
    c = Level("z", 100.2, "retracement", "swing A")     # same source -> not confluence
    out = {(l.label, l.confluent) for l in find_confluence([a, b, c])}
    assert ("x", True) in out and ("y", True) in out
    assert ("z", True) in out or ("z", False) in out    # z is confluent with b too


def test_mark_near_pivots():
    lv = [Level("a", 100.0, "retracement", "s"), Level("b", 500.0, "retracement", "s")]
    piv = [Pivot(0, 100.4, "L")]
    out = {l.label: l.near_pivot for l in mark_near_pivots(lv, piv)}
    assert out["a"] is True and out["b"] is False


def test_ladder_splits_and_orders():
    lv = [Level(str(p), float(p), "retracement", "s") for p in (80, 90, 110, 130)]
    below, above = ladder(lv, spot=100.0)
    assert [l.price for l in below] == [90.0, 80.0]      # nearest support first
    assert [l.price for l in above] == [110.0, 130.0]    # nearest resistance first


# ---- end to end ----

def test_analyse_contract():
    p = _impulse_series()
    p = np.concatenate([p, p * 1.02])                    # enough bars
    an = analyse(p, p * 1.01, p * 0.99)
    for key in ("spot", "atr_pct", "threshold", "pivots", "counts", "ambiguous",
                "levels", "fib_below", "fib_above"):
        assert key in an
    assert all(l.price < an["spot"] for l in an["fib_below"])
    assert all(l.price >= an["spot"] for l in an["fib_above"])
    assert 0 < an["threshold"] <= 0.25


def test_analyse_requires_enough_bars():
    with pytest.raises(ValueError):
        analyse(np.ones(10), np.ones(10), np.ones(10))


# ---- the placebo control itself ----

def test_random_ratio_levels_match_fib_count_and_span():
    """The strict control must be matched on count AND ratio span, or fib wins on
    geometry (concentration near the swing middle) rather than on its values."""
    from src.forecast.waves import EXTENSIONS, RETRACEMENTS, random_ratio_levels

    swings = [(100.0, 200.0, "up", "s")]
    rng = np.random.default_rng(0)
    rl = random_ratio_levels(swings, rng)
    fl = [l.price for l in fib_levels(100.0, 200.0, "up", "s")]
    assert rl.size == len(fl)                        # same number of levels

    # every placebo retracement lies inside the fib retracement span
    lo_r, hi_r = min(RETRACEMENTS), max(RETRACEMENTS)
    retr = rl[rl >= 200.0 - 100.0 * hi_r - 1e-9]
    retr = retr[retr <= 200.0 - 100.0 * lo_r + 1e-9]
    assert retr.size == len(RETRACEMENTS)

    ext = rl[rl > 200.0]
    lo_e, hi_e = min(EXTENSIONS), max(EXTENSIONS)
    assert ext.size == len(EXTENSIONS)
    assert np.all(ext >= 200.0 + 100.0 * (lo_e - 1) - 1e-9)
    assert np.all(ext <= 200.0 + 100.0 * (hi_e - 1) + 1e-9)


def test_random_ratio_levels_are_deterministic_per_seed():
    from src.forecast.waves import random_ratio_levels
    sw = [(50.0, 90.0, "down", "s")]
    a = random_ratio_levels(sw, np.random.default_rng(3))
    b = random_ratio_levels(sw, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_swings_used_reports_sources():
    from src.forecast.waves import swings_used
    piv = [Pivot(0, 100, "L"), Pivot(1, 120, "H"), Pivot(2, 110, "L"), Pivot(3, 140, "H")]
    sw = swings_used(piv, [])
    assert [s[3] for s in sw] == ["last swing", "prior swing"]
    for lo, hi, direction, _ in sw:
        assert lo < hi and direction in ("up", "down")
