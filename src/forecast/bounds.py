"""Realistic worst/best case: statistical percentiles next to historical precedent.

A percentile alone invites two opposite errors: treating a 1-in-100 tail as "the worst
case" (it isn't -- see `unmodelable_notes`), or dismissing a wide band as unrealistic
when the stock has actually done that move before.

So every bound is printed with its provenance:
  * what the SIMULATION says (percentile of the terminal distribution)
  * what the STOCK ITSELF has actually done over the same horizon length
  * what neither can capture (financing failure -> ~0; a re-rating with no precedent
    at the current market cap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class Precedent:
    """What actually happened to this name over `horizon`-day windows."""
    horizon: int
    n_windows: int
    worst_mult: float          # lowest realized end/start multiple
    best_mult: float           # highest realized end/start multiple
    worst_date: str
    best_date: str
    share_ge_2x: float
    share_ge_4x: float
    share_le_half: float
    median_mult: float


def realized_windows(closes, dates, horizon: int) -> Precedent | None:
    """Every overlapping `horizon`-day forward move the name has actually delivered."""
    px = np.asarray(closes, dtype=float)
    if px.size <= horizon + 1:
        return None
    fwd = px[horizon:] / px[:-horizon]
    i_w, i_b = int(np.argmin(fwd)), int(np.argmax(fwd))
    fmt = lambda i: f"{dates[i]}->{dates[i + horizon]}"
    return Precedent(
        horizon=horizon, n_windows=int(fwd.size),
        worst_mult=float(fwd.min()), best_mult=float(fwd.max()),
        worst_date=fmt(i_w), best_date=fmt(i_b),
        share_ge_2x=float(np.mean(fwd >= 2.0)), share_ge_4x=float(np.mean(fwd >= 4.0)),
        share_le_half=float(np.mean(fwd <= 0.5)), median_mult=float(np.median(fwd)),
    )


def bounds_ladder(terminal, s0: float, prec: Precedent | None) -> list[tuple[str, str, str]]:
    """Rows of (label, simulated price, precedent note) for the report."""
    t = np.asarray(terminal, dtype=float)
    rows: list[tuple[str, str, str]] = []

    def px(q):
        v = float(np.percentile(t, q))
        return f"${v:>9,.2f} ({v / s0 - 1:+6.0%})"

    worst_note = ("" if prec is None else
                  f"worst actual {prec.horizon}d: {prec.worst_mult - 1:+.0%} "
                  f"-> ${s0 * prec.worst_mult:,.2f}  ({prec.worst_date})")
    best_note = ("" if prec is None else
                 f"best actual {prec.horizon}d: {prec.best_mult - 1:+.0%} "
                 f"-> ${s0 * prec.best_mult:,.2f}  ({prec.best_date})")
    freq_lo = ("" if prec is None else
               f"{prec.share_le_half:.0%} of windows lost >=50%")
    freq_hi = ("" if prec is None else
               f"{prec.share_ge_2x:.0%} doubled, {prec.share_ge_4x:.0%} quadrupled")

    rows.append(("worst  1%", px(1), worst_note))
    rows.append(("worst  5%", px(5), freq_lo))
    rows.append(("worst 25%", px(25), ""))
    rows.append(("median   ", px(50),
                 "" if prec is None else f"median actual: {prec.median_mult - 1:+.0%}"))
    rows.append(("best  25%", px(75), ""))
    rows.append(("best   5%", px(95), freq_hi))
    rows.append(("best   1%", px(99), best_note))
    return rows


# ---------------------------------------------------------------------------
# Automatic reality check
#
# Compares every model percentile against what the name has ACTUALLY delivered at
# that same frequency, and corrects the model where it is not supported by evidence.
#
# The rule is deliberately ASYMMETRIC, because the two errors are not equally safe:
#
#   UPSIDE   -- if the model's best case exceeds anything the name has done at that
#               frequency, trim it to the realized figure. Big upside requires a
#               re-rating, and "no precedent" is real evidence against it.
#   DOWNSIDE -- never make the downside LESS severe than the model says. History is
#               a lower bound on how bad things can get, not an upper bound: the
#               bankruptcy that has not happened yet is absent from every dataset.
#               So a model gloomier than precedent is kept as-is and merely noted.
# ---------------------------------------------------------------------------

MIN_WINDOWS_FOR_CHECK = 150
DEFAULT_TOLERANCE = 1.5
CHECK_QS = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


@dataclass(frozen=True, slots=True)
class Check:
    q: float
    model: float               # price at this percentile from the simulation
    realized: float | None     # price implied by the name's own realized move at this pct
    ratio: float | None        # model / realized
    flag: str                  # "" when the model is supported by precedent
    anchored: float            # the number the report should feature


def realized_multiples(closes, horizon: int):
    px = np.asarray(closes, dtype=float)
    if px.size <= horizon + 1:
        return None
    return px[horizon:] / px[:-horizon]


def reality_check(terminal, s0: float, closes, horizon: int,
                  qs: tuple[float, ...] = CHECK_QS,
                  tol: float = DEFAULT_TOLERANCE) -> list[Check]:
    """Flag and (where justified) correct model percentiles against realized history."""
    t = np.asarray(terminal, dtype=float)
    mult = realized_multiples(closes, horizon)
    thin = mult is None or mult.size < MIN_WINDOWS_FOR_CHECK
    out: list[Check] = []
    for q in qs:
        m = float(np.percentile(t, q))
        if thin:
            out.append(Check(q, m, None, None, "thin precedent - unchecked", m))
            continue
        rp = float(s0 * np.percentile(mult, q))
        ratio = m / rp if rp > 0 else float("inf")
        flag, anchored = "", m
        if q >= 50.0:                       # upside: trim what has no precedent
            if ratio > tol:
                flag, anchored = f"MODEL {ratio:.1f}x HIGH -> trimmed", rp
        else:                               # downside: never soften it
            if ratio > tol:
                flag, anchored = f"MODEL {ratio:.1f}x MILD -> tightened", rp
            elif ratio < 1.0 / tol:
                flag = f"model {ratio:.1f}x more severe than precedent (kept)"
        out.append(Check(q, m, rp, ratio, flag, anchored))

    # Trimming an upper percentile to precedent can drop it below a lower one that was
    # left alone, producing a non-monotonic (and meaningless) quantile ladder. Enforce
    # a non-decreasing anchored sequence, noting wherever the correction bites.
    out.sort(key=lambda c: c.q)
    fixed: list[Check] = []
    running = -math.inf
    for c in out:
        a = max(c.anchored, running)
        running = a
        if a != c.anchored:
            note = "raised to keep percentiles ordered"
            c = replace(c, anchored=a, flag=f"{c.flag}; {note}" if c.flag else note)
        fixed.append(c)
    return fixed


def anchored_range(checks: list[Check]) -> tuple[float, float, float]:
    """(p5, median, p95) after the reality check -- the headline realistic range."""
    by_q = {c.q: c.anchored for c in checks}
    return by_q.get(5.0, float("nan")), by_q.get(50.0, float("nan")), by_q.get(95.0, float("nan"))


def unmodelable_notes(symbol: str) -> list[str]:
    """The two ends no price-history model can produce. Stated, not simulated."""
    return [
        f"BELOW the table: {symbol} going to ~$0 (financing failure, launch/technical "
        f"failure, severe dilution). A bootstrap of past returns can never generate a "
        f"bankruptcy the history does not contain -- so the true floor is lower than "
        f"any percentile shown.",
        f"ABOVE the table: extreme upside percentiles resample the name's own past "
        f"manias. If those happened at a far smaller market cap, the same multiple is "
        f"much harder now -- treat the top 1% as regime-dependent, not a forecast.",
    ]
