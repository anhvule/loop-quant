"""Calibration helpers: Wilson intervals, overlap adjustment, verdicts."""

from __future__ import annotations

import math

import pytest

from src.forecast.calib import effective_n, rate, wilson


def test_wilson_contains_point_estimate():
    lo, hi = wilson(90, 100)
    assert lo < 0.90 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_widens_as_n_shrinks():
    wide = wilson(37, 40)
    tight = wilson(370, 400)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0]) * 2


def test_wilson_handles_zero_and_full():
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson(20, 20)
    # upper bound is 1 in exact arithmetic; float lands a few ulps below
    assert 0 < lo < 1 and hi == pytest.approx(1.0)


def test_wilson_empty():
    lo, hi = wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_effective_n_discounts_overlap():
    # monthly forecasts (21d apart) looking 73d ahead overlap heavily
    assert effective_n(100, 73) == pytest.approx(100 * 21 / 73)
    # a horizon shorter than the step is fully independent
    assert effective_n(100, 10) == 100.0
    assert effective_n(100, 0) == 100.0


def test_rate_overlap_widens_interval():
    plain = rate(37, 40)
    overlapped = rate(37, 40, horizon_days=73)
    assert overlapped.n_eff < plain.n_eff
    assert (overlapped.hi - overlapped.lo) > (plain.hi - plain.lo)
    assert overlapped.p == pytest.approx(37 / 40)      # observed rate unchanged


def test_verdicts():
    # 329 trials at 0.90 -> tight and consistent
    strong = rate(296, 329)
    assert strong.covers(0.90)
    # small sample -> consistent but explicitly weak
    weak = rate(37, 40, horizon_days=73)
    assert "weak" in weak.verdict(0.90) or weak.covers(0.90)
    # clearly broken coverage
    broken = rate(45, 61)
    assert not broken.covers(0.90)
    assert "RULED OUT" in broken.verdict(0.90)


def test_rate_no_data():
    r = rate(0, 0)
    assert math.isnan(r.p)
    assert r.verdict() == "no data"
