"""High-beta screening criteria: measurable properties, explicit thresholds.

READ THIS FIRST. Every criterion here measures something about RISK, LIQUIDITY,
VERIFIABILITY or REDUNDANCY. **None of them predicts returns.** Direction failed every
predictive test in this project (ARIMA selected a random walk; Elliott stage carried no
forward signal; single-name drift is noise). A stock scoring well here is not "going
up" -- it is *analysable, survivable, tradable, and not a duplicate of what you already
hold*. That is a different and more honest claim.

The composite score deliberately takes CALLER-SUPPLIED weights. What counts as "good"
is a value judgement (survivability vs lottery-shape vs diversification), so the tool
measures and the user decides. Default weights are equal, never tuned to flatter a name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from src.forecast.beta import beta_stats, ewma_vol_path
from src.forecast.longrange import log_returns

WINDOW = 126                 # ~6 months, the horizon used throughout the project
MIN_VERIFY_BARS = 700
PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"

# Every threshold is a constant so the UI can print it next to the grade -- a grade
# whose cutoff is hidden is just an opinion with a colour.
T_BETA = 2.0
T_R2 = 0.10
T_HALVED_MILD, T_HALVED_SEVERE = 0.05, 0.20
T_CORR_DIVERSIFIER, T_CORR_DUPLICATE = 0.30, 0.45
T_DOLLAR_VOL_OK, T_DOLLAR_VOL_THIN = 50e6, 5e6
T_VOL_PCTL_ELEVATED, T_VOL_PCTL_STORM = 0.70, 0.85


@dataclass(frozen=True, slots=True)
class Criterion:
    key: str
    title: str
    value: float | None
    display: str
    grade: str            # pass | warn | fail | info
    threshold: str        # human-readable cutoff, shown in the UI
    note: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "value": self.value,
                "display": self.display, "grade": self.grade,
                "threshold": self.threshold, "note": self.note}


@dataclass(frozen=True, slots=True)
class Scorecard:
    symbol: str
    criteria: list[Criterion]
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def by_key(self, key: str) -> Criterion | None:
        return next((c for c in self.criteria if c.key == key), None)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "error": self.error,
                "warnings": list(self.warnings),
                "criteria": [c.to_dict() for c in self.criteria]}


# ---------------------------------------------------------------------------
# individual criteria
# ---------------------------------------------------------------------------

def c_beta(name_rets, mkt_rets) -> tuple[Criterion, dict]:
    """1. Is it genuinely a market amplifier, or just individually wild?

    A high beta with a low r^2 is a warning, not a feature: the beta estimate itself is
    unreliable and the name is driven by its own story, so 'high beta' tells you little."""
    b = beta_stats("x", np.asarray(name_rets), np.asarray(mkt_rets))
    if b.beta < T_BETA:
        grade, note = FAIL, "below the high-beta cut"
    elif b.r2 < T_R2:
        grade, note = WARN, "idiosyncratic lottery -- the beta estimate is unreliable"
    else:
        grade, note = PASS, "market amplifier"
    return (Criterion("beta", "True high-beta?", float(b.beta),
                      f"beta {b.beta:.2f} +/-{b.se:.2f} | r2 {b.r2:.2f}",
                      grade, f"pass: beta >= {T_BETA} and r2 >= {T_R2}", note),
            {"beta": b.beta, "se": b.se, "r2": b.r2, "vol": b.vol})


def c_verifiable(n_bars: int, verdict: dict | None) -> Criterion:
    """2. Can any claim about this name actually be tested?"""
    if n_bars < MIN_VERIFY_BARS:
        return Criterion("verifiable", "Verifiable?", float(n_bars),
                         f"{n_bars} shared bars", FAIL,
                         f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent",
                         "too short to calibration-test -- any range is extrapolation")
    if not verdict:
        return Criterion("verifiable", "Verifiable?", float(n_bars),
                         f"{n_bars} bars | not yet tested", WARN,
                         f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent",
                         "run scripts.beta_outlook --validate")
    v = str(verdict.get("verdict", ""))
    ok = not v.startswith("RULED OUT")
    return Criterion("verifiable", "Verifiable?", float(n_bars),
                     f"{n_bars} bars | cov90 {verdict.get('cov90', float('nan')):.2f}",
                     PASS if ok else FAIL,
                     f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent",
                     v)


def c_survivable(closes, window: int = WINDOW) -> tuple[Criterion, dict]:
    """3. How often has holding this for ~6 months cost you half your money?"""
    px = np.asarray(closes, dtype=float)
    if px.size <= window + 1:
        return (Criterion("survivable", "Survivable?", None, "insufficient history",
                          WARN, f"mild: < {T_HALVED_MILD:.0%} of {window}d windows halved"),
                {})
    fwd = px[window:] / px[:-window]
    halved = float(np.mean(fwd <= 0.5))
    worst = float(fwd.min() - 1.0)
    run_max = np.maximum.accumulate(px)
    max_dd = float(np.max(1.0 - px / run_max))
    grade = (PASS if halved < T_HALVED_MILD else
             WARN if halved < T_HALVED_SEVERE else FAIL)
    note = ("halving is routine for this name" if grade == FAIL else
            "occasional halvings" if grade == WARN else "halvings are rare")
    return (Criterion("survivable", "Survivable?", halved,
                      f"{halved:.0%} of {window}d windows halved | worst {worst:+.0%} | "
                      f"max DD {-max_dd:.0%}",
                      grade,
                      f"pass < {T_HALVED_MILD:.0%}, warn < {T_HALVED_SEVERE:.0%} halved",
                      note),
            {"halved": halved, "worst_window": worst, "max_dd": max_dd,
             "doubled": float(np.mean(fwd >= 2.0)),
             "best_window": float(fwd.max() - 1.0)})


def c_redundancy(avg_corr: float | None, n_peers: int) -> Criterion:
    """4. Does it add a bet, or repeat one you already have?

    Defined ONLY relative to the submitted cohort -- a 'diversifier' against these
    tickers may be a duplicate against a different list."""
    if avg_corr is None or n_peers < 1:
        return Criterion("redundancy", "Adds a new bet?", None, "no peers to compare",
                         INFO, f"diversifier < {T_CORR_DIVERSIFIER}, duplicate > {T_CORR_DUPLICATE}")
    grade = (PASS if avg_corr < T_CORR_DIVERSIFIER else
             WARN if avg_corr < T_CORR_DUPLICATE else FAIL)
    note = ("largely a duplicate of the rest of this list" if grade == FAIL else
            "partly overlapping" if grade == WARN else "genuinely different bet")
    return Criterion("redundancy", "Adds a new bet?", float(avg_corr),
                     f"avg corr {avg_corr:.2f} vs {n_peers} peers", grade,
                     f"pass < {T_CORR_DIVERSIFIER}, fail > {T_CORR_DUPLICATE} "
                     f"(relative to the tickers you submitted)", note)


def c_liquidity(closes, volumes) -> tuple[Criterion, dict]:
    """5. Can a position be entered and exited without moving the price?"""
    px = np.asarray(closes, dtype=float)
    vol = np.asarray(volumes, dtype=float) if volumes is not None else None
    if vol is None or vol.size == 0 or not np.any(vol > 0):
        return (Criterion("liquidity", "Liquid enough?", None, "no volume data", INFO,
                          f"ok > ${T_DOLLAR_VOL_OK / 1e6:.0f}M/day, "
                          f"thin < ${T_DOLLAR_VOL_THIN / 1e6:.0f}M/day"), {})
    n = min(px.size, vol.size)
    dollar = float(np.median(px[-n:][-60:] * vol[-n:][-60:]))
    grade = (PASS if dollar > T_DOLLAR_VOL_OK else
             WARN if dollar > T_DOLLAR_VOL_THIN else FAIL)
    note = ("thin -- expect slippage and gaps" if grade == FAIL else
            "moderate depth" if grade == WARN else "deep")
    return (Criterion("liquidity", "Liquid enough?", dollar,
                      f"${dollar / 1e6:,.0f}M median daily $ volume", grade,
                      f"pass > ${T_DOLLAR_VOL_OK / 1e6:.0f}M, "
                      f"fail < ${T_DOLLAR_VOL_THIN / 1e6:.0f}M", note),
            {"dollar_volume": dollar})


def c_vol_regime(closes) -> tuple[Criterion, dict]:
    """6. Is today calm or turbulent BY THIS NAME'S OWN STANDARDS?

    Entering at a vol extreme means wider outcomes than the name's typical behaviour --
    relevant for sizing, and it says nothing about direction."""
    r = log_returns(np.asarray(closes, dtype=float))
    if r.size < 90:
        return (Criterion("vol_regime", "Vol regime now", None, "insufficient history",
                          INFO, f"elevated > {T_VOL_PCTL_ELEVATED:.0%} percentile"), {})
    path, cur = ewma_vol_path(r)
    pctl = float(np.mean(path <= cur))
    grade = (PASS if pctl < T_VOL_PCTL_ELEVATED else
             WARN if pctl < T_VOL_PCTL_STORM else FAIL)
    note = ("entering during a storm -- unusually wide outcomes" if grade == FAIL else
            "somewhat elevated" if grade == WARN else "normal for this name")
    return (Criterion("vol_regime", "Vol regime now", pctl,
                      f"{cur * 100:.1f}%/day | {pctl:.0%} percentile of its own history",
                      grade, f"pass < {T_VOL_PCTL_ELEVATED:.0%}, "
                             f"fail > {T_VOL_PCTL_STORM:.0%} percentile", note),
            {"vol_now": cur, "vol_pctl": pctl})


def c_payoff(stats: dict, window: int = WINDOW) -> Criterion:
    """7. Payoff SHAPE. Deliberately ungraded (INFO): describing how fat each tail has
    been is not a claim about which one comes next."""
    if not stats or "doubled" not in stats:
        return Criterion("payoff", "Payoff shape", None, "insufficient history", INFO,
                         "descriptive only -- not predictive")
    return Criterion("payoff", "Payoff shape", float(stats["doubled"]),
                     f"{stats['doubled']:.0%} of {window}d windows doubled | "
                     f"{stats['halved']:.0%} halved | best {stats['best_window']:+.0%}",
                     INFO, "descriptive only -- not predictive",
                     "historical shape; says nothing about which tail comes next")


# ---------------------------------------------------------------------------
# cohort + assembly
# ---------------------------------------------------------------------------

def cohort_redundancy(returns_by_symbol: dict[str, np.ndarray]) -> dict[str, float]:
    """Average pairwise correlation of each name against the others submitted."""
    syms = [s for s, r in returns_by_symbol.items() if r is not None and r.size > 30]
    if len(syms) < 2:
        return {}
    n = min(len(returns_by_symbol[s]) for s in syms)
    M = np.column_stack([returns_by_symbol[s][-n:] for s in syms])
    C = np.corrcoef(M.T)
    np.fill_diagonal(C, np.nan)
    return {s: float(np.nanmean(C[i])) for i, s in enumerate(syms)}


def build_scorecard(symbol: str, closes, highs, lows, volumes, mkt_rets,
                    n_shared_bars: int, avg_corr: float | None = None,
                    n_peers: int = 0, verdict: dict | None = None) -> Scorecard:
    """All seven criteria for one ticker."""
    try:
        name_rets = log_returns(np.asarray(closes, dtype=float))
        m = np.asarray(mkt_rets, dtype=float)
        k = min(name_rets.size, m.size)
        if k < 40:
            return Scorecard(symbol, [], [], "not enough overlapping history")
        # Both tails are taken from the same shared trading calendar, so element i of
        # each refers to the same session. A one-off halt in a single name could shift
        # a few rows; the effect on an EWMA beta over ~200 effective observations is
        # noise-level, and the web copy behaves identically (pinned by parity tests).
        cb, bstats = c_beta(name_rets[-k:], m[-k:])
        cs, sstats = c_survivable(closes)
        cl, _ = c_liquidity(closes, volumes)
        cv, _ = c_vol_regime(closes)
        crit = [cb, c_verifiable(n_shared_bars, verdict), cs,
                c_redundancy(avg_corr, n_peers), cl, cv, c_payoff(sstats)]
    except Exception as e:  # noqa: BLE001 -- one bad ticker must not kill the screen
        return Scorecard(symbol, [], [], f"could not score: {type(e).__name__}")

    warns = [c.note for c in crit if c.grade == FAIL and c.note]
    return Scorecard(symbol, crit, warns)


# Higher is "better" only in the sense of the criterion's own grade -- never returns.
_GRADE_POINTS = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


def composite(card: Scorecard, weights: dict[str, float] | None = None) -> float | None:
    """Weighted average of grade points. The WEIGHTS ARE THE USER'S VALUE JUDGEMENT --
    this is a preference ranking, not a forecast. INFO criteria are excluded because
    they are descriptive and carry no direction of 'better'."""
    graded = [c for c in card.criteria if c.grade in _GRADE_POINTS]
    if not graded:
        return None
    w = weights or {}
    num = sum(w.get(c.key, 1.0) * _GRADE_POINTS[c.grade] for c in graded)
    den = sum(w.get(c.key, 1.0) for c in graded)
    return float(num / den) if den > 0 else None


def load_verdicts(path) -> dict:
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return {}
