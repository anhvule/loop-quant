"""Realistic-bounds helpers: realized-window precedent and the bounds ladder."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from src.forecast.bounds import (
    MIN_WINDOWS_FOR_CHECK,
    anchored_range,
    bounds_ladder,
    reality_check,
    realized_multiples,
    realized_windows,
    unmodelable_notes,
)


def _series(mults, s0=10.0):
    """Build a price path from successive per-step multipliers."""
    px = [s0]
    for m in mults:
        px.append(px[-1] * m)
    return px


def test_realized_windows_finds_best_and_worst():
    # 40 flat steps, then a 4x run, then a halving
    px = _series([1.0] * 40 + [1.05] * 30 + [0.97] * 30)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(px))]
    prec = realized_windows(px, dates, horizon=20)
    assert prec is not None
    assert prec.n_windows == len(px) - 20
    assert prec.best_mult > 1.5
    assert prec.worst_mult < 1.0
    assert 0.0 <= prec.share_ge_2x <= 1.0
    assert "->" in prec.best_date and "->" in prec.worst_date


def test_realized_windows_none_when_too_short():
    px = _series([1.0] * 5)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(px))]
    assert realized_windows(px, dates, horizon=20) is None


def test_realized_windows_shares_are_consistent():
    px = _series([1.02] * 200)          # steady compounding -> many big windows
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(px))]
    prec = realized_windows(px, dates, horizon=40)
    assert prec.share_ge_2x >= prec.share_ge_4x        # 4x is a subset of 2x
    assert prec.share_le_half == 0.0                    # never fell
    assert prec.median_mult > 1.0


def test_bounds_ladder_is_ordered_and_labeled():
    rng = np.random.default_rng(0)
    terminal = 100.0 * np.exp(rng.normal(0.0, 0.5, 20000))
    rows = bounds_ladder(terminal, s0=100.0, prec=None)
    labels = [r[0].strip() for r in rows]
    assert labels == ["worst  1%", "worst  5%", "worst 25%", "median",
                      "best  25%", "best   5%", "best   1%"]
    # prices ascend down the ladder
    prices = [float(r[1].split("(")[0].strip().lstrip("$").replace(",", "")) for r in rows]
    assert prices == sorted(prices)


def test_bounds_ladder_includes_precedent_notes():
    px = _series([1.01] * 120)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(px))]
    prec = realized_windows(px, dates, horizon=30)
    rng = np.random.default_rng(1)
    rows = bounds_ladder(100.0 * np.exp(rng.normal(0, 0.3, 5000)), 100.0, prec)
    notes = " ".join(r[2] for r in rows)
    assert "worst actual" in notes and "best actual" in notes


def test_unmodelable_notes_mention_zero_and_regime():
    notes = unmodelable_notes("ASTS")
    joined = " ".join(notes)
    assert "ASTS" in joined
    assert "$0" in joined or "bankruptcy" in joined
    assert "market cap" in joined


# ---------------------------------------------------------------------------
# automatic reality check
# ---------------------------------------------------------------------------

H = 20


def _tame_history(n=600, s0=100.0, seed=0):
    """A history whose 20-day moves stay within roughly +/-30%."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, n)
    return list(s0 * np.exp(np.cumsum(r)))


def _terminal(mult_lo, mult_hi, s0=100.0, n=20000, seed=1):
    """A simulated terminal distribution spanning [lo, hi] multiples (log-uniform)."""
    rng = np.random.default_rng(seed)
    lo, hi = math.log(mult_lo), math.log(mult_hi)
    return s0 * np.exp(rng.uniform(lo, hi, n))


def test_realized_multiples_none_when_short():
    assert realized_multiples([1.0, 2.0, 3.0], horizon=20) is None


def test_upside_beyond_precedent_is_trimmed():
    """Model says the 95th percentile is 5x; history never did more than ~1.3x."""
    hist = _tame_history()
    s0 = hist[-1]
    term = _terminal(0.9, 6.0, s0=s0)
    checks = {c.q: c for c in reality_check(term, s0, hist, H, tol=1.5)}
    hi = checks[95.0]
    assert hi.ratio > 1.5
    assert "HIGH" in hi.flag and "trimmed" in hi.flag
    # Anchored sits at or below the model and at or above precedent: it is pulled toward
    # the realized figure, but the monotonicity pass may hold it up to a lower percentile
    # when the model is broadly above precedent (which is itself the signal).
    assert hi.anchored < hi.model
    assert hi.anchored >= hi.realized


def test_downside_more_severe_than_precedent_is_KEPT():
    """The safety asymmetry: a gloomier-than-history downside must NOT be softened."""
    hist = _tame_history()
    s0 = hist[-1]
    term = _terminal(0.05, 1.2, s0=s0)                   # model allows a -95% crash
    lo = {c.q: c for c in reality_check(term, s0, hist, H, tol=1.5)}[5.0]
    assert lo.ratio < 1 / 1.5
    assert "more severe" in lo.flag and "kept" in lo.flag
    assert lo.anchored == pytest.approx(lo.model)        # unchanged, NOT raised


def test_downside_too_mild_is_tightened():
    """A downside milder than precedent gets pulled down to what actually happened."""
    hist = _tame_history()
    s0 = hist[-1]
    term = _terminal(0.98, 1.02, s0=s0)                  # model barely moves
    lo = {c.q: c for c in reality_check(term, s0, hist, H, tol=1.05)}[5.0]
    assert "MILD" in lo.flag and "tightened" in lo.flag
    assert lo.anchored == pytest.approx(lo.realized)
    assert lo.anchored < lo.model


def test_supported_percentiles_are_unflagged():
    hist = _tame_history()
    s0 = hist[-1]
    mult = realized_multiples(hist, H)
    # build a terminal distribution that mirrors the realized one
    term = s0 * np.asarray(mult, dtype=float)
    for c in reality_check(term, s0, hist, H, tol=1.5):
        assert c.flag == ""
        assert c.anchored == pytest.approx(c.model)


def test_thin_precedent_is_unchecked_not_silently_trusted():
    short = _tame_history(n=MIN_WINDOWS_FOR_CHECK + H - 20)
    s0 = short[-1]
    checks = reality_check(_terminal(0.1, 10.0, s0=s0), s0, short, H)
    assert all(c.realized is None for c in checks)
    assert all("thin precedent" in c.flag for c in checks)
    assert all(c.anchored == pytest.approx(c.model) for c in checks)


def test_anchored_range_returns_p5_median_p95():
    hist = _tame_history()
    s0 = hist[-1]
    checks = reality_check(_terminal(0.5, 3.0, s0=s0), s0, hist, H)
    lo, mid, hi = anchored_range(checks)
    # non-strict: when the model sits broadly above precedent the upper ladder can
    # compress onto a single anchored level
    assert lo <= mid <= hi


def test_anchored_percentiles_stay_monotonic():
    """Regression: trimming an upper percentile to precedent must not drop it below a
    lower percentile that was left alone -- a quantile ladder has to be ordered."""
    hist = _tame_history()
    s0 = hist[-1]
    # wide model: p95 gets trimmed hard while the median is left near its model value
    checks = reality_check(_terminal(0.5, 3.0, s0=s0), s0, hist, H, tol=1.5)
    anchored = [c.anchored for c in sorted(checks, key=lambda c: c.q)]
    assert anchored == sorted(anchored)
    assert any("ordered" in c.flag for c in checks)      # correction is disclosed


def test_tolerance_controls_sensitivity():
    hist = _tame_history()
    s0 = hist[-1]
    term = _terminal(0.9, 2.5, s0=s0)
    strict = {c.q: c for c in reality_check(term, s0, hist, H, tol=1.1)}[95.0]
    loose = {c.q: c for c in reality_check(term, s0, hist, H, tol=10.0)}[95.0]
    assert strict.flag != "" and loose.flag == ""
