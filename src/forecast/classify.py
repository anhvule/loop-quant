"""Classify high-beta names as investable or reckless: a gate stack, not a forecast.

READ THIS FIRST, and read it against `screener.py`. That module deliberately refuses
to rank on returns, and nothing here changes its mind. This module answers a *different*
question and must not be confused with it:

    screener.py  -- "is this name analysable, survivable, tradable, non-duplicative?"
    classify.py  -- "of the names that are genuinely high-beta and genuinely liquid,
                     which ones have HISTORICALLY been paid for that beta, without
                     their story drowning out the market?"

Gate 4 (Treynor) is the honest exception in this project: it ranks REALIZED return per
unit of beta over the trailing year. That is a property of the past, not a claim about
the next year. A name in the top quartile is not "going up" -- it is one that has, so
far, compensated its holders for the market risk they carried. Momentum in that measure
failed every predictive test here, so treat the ranking as a description of what you
would have been paid, never as an expectation of what you will be.

The gate stack, in order:

  1. ADV        -- median daily dollar volume above a floor. HARD: fails -> excluded.
  2. Beta       -- 252d OLS beta above the cut, with the 63d window confirming.
                   HARD: fails -> excluded (not high-beta, so out of scope).
  3. Regime     -- market above its 200d SMA. An OVERLAY, computed once: risk-off
                   turns "investable" into "stand aside", never into "reckless".
                   The regime is a fact about the market, not about the name.
  4. Treynor    -- top quartile of the eligible universe (see the small-universe rule).
  5. Idio share -- 1 - r2, low or compressing. High AND expanding is the binary-event
                   profile (trial readouts, litigation) this gate exists to catch.

Gates 1 and 2 are eligibility: failing them is not a verdict about quality, so those
names are reported as `excluded_*` rather than "reckless". Only names that clear both
get an opinion attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.forecast.screener import FAIL, INFO, PASS, WARN, Criterion

MARKET_LABEL = "SPY"
TRADING_DAYS = 252

# Every threshold is a module constant so the UI can print it beside the grade --
# a grade whose cutoff is hidden is just an opinion with a colour (see screener.py).
T_ADV_DOLLAR = 10e6            # gate 1: median daily dollar volume floor
ADV_WINDOW = 60                # bars, matching screener.c_liquidity
T_BETA_MIN = 1.5               # gate 2: applied to the 252d beta
BETA_CONFIRM_FRAC = 0.75       # the 63d beta must reach this fraction of the cut
BETA_DIVERGE = 0.50            # |b63 - b252| / b252 above this -> unstable, WARN
SMA_WINDOW = 200               # gate 3
TREYNOR_TOP_FRAC = 0.25        # gate 4
MIN_UNIVERSE_FOR_QUARTILE = 8  # below this the quartile is meaningless -> absolute rule
T_IDIO_SHARE = 0.75            # gate 5: idio share at/above this counts as "high"
T_IDIO_COMPRESS = 0.05         # 5pp move in idio share counts as compressing/expanding
WIN_SHORT, WIN_LONG = 63, 252
MIN_BARS_CLASSIFY = 300        # 252 returns plus slack; below this -> an error row
RF_DEFAULT = 0.04              # flat annual risk-free; the caller adjusts as rates move

# Gate 6-7: whole-history price behaviour. The halving thresholds intentionally match
# screener.T_HALVED_MILD/_SEVERE, but are restated rather than imported so each module
# states its own cutoffs and the web copy stays numpy+stdlib standalone.
SURV_WINDOW = 126              # ~6 months, the horizon used throughout the project
T_HALVED_MILD = 0.05
T_HALVED_SEVERE = 0.20
OWN_SMA_WINDOW = 200           # gate 7, mirroring SMA_WINDOW for the name itself

# Gates 8-10: fundamentals. Every one of these degrades to INFO on missing data.
T_DILUTION_OK = 0.03           # <= 3%/yr share growth: flat or buying back
T_DILUTION_HEAVY = 0.15        # > 15%/yr: the treadmill
T_RUNWAY_OK = 8.0              # quarters of cash at the current burn
T_RUNWAY_CRITICAL = 4.0        # < 4 quarters: a forced raise is the next event
T_REVENUE_REAL = 10e6          # TTM revenue below this is a pre-revenue story
RUNWAY_DISPLAY_CAP = 99.0      # json.dumps(inf) emits invalid JSON -- clamp before payload
DILUTION_MIN_BARS = 504        # ~2y: below this the share count still reflects the IPO
# Any one gate can be wrong about any one name; the harshest label should not turn on a
# single reading. A genuinely broken name trips several at once.
RECKLESS_MIN_FLAGS = 2

INVESTABLE = "investable"
# "mixed" rather than "borderline": the name is not sitting near a threshold, it is a
# name whose gates DISAGREE -- some clean, some flagged, not enough to condemn. The old
# word implied a marginal call, which is not what this bucket means.
MIXED = "mixed"
RECKLESS = "reckless"
STAND_ASIDE = "stand_aside"
EXCLUDED_ILLIQUID = "excluded_illiquid"
EXCLUDED_LOW_BETA = "excluded_low_beta"
ERROR = "error"

# Display order: opinions first, then the names that never earned one.
VERDICT_ORDER = [INVESTABLE, MIXED, STAND_ASIDE, RECKLESS,
                 EXCLUDED_LOW_BETA, EXCLUDED_ILLIQUID, ERROR]

QUARTILE_RELATIVE = "quartile"
QUARTILE_ABSOLUTE = "absolute"

BANNER = ("CLASSIFICATION, NOT PREDICTION. The Treynor gate ranks realized return per "
          "unit of beta over the trailing year -- a historical property, not a forecast. "
          "'Investable' means the name cleared liquidity, was genuinely high-beta on "
          "both windows, was paid for that beta in the past, and is not dominated by its "
          "own story. It does not mean the name will rise.")


@dataclass(frozen=True, slots=True)
class Classification:
    symbol: str
    verdict: str
    gates: list[Criterion] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def by_key(self, key: str) -> Criterion | None:
        return next((g for g in self.gates if g.key == key), None)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "verdict": self.verdict,
                "gates": [g.to_dict() for g in self.gates],
                "metrics": dict(self.metrics), "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class UniverseResult:
    """Everything a caller needs to render a run, including how gate 4 was decided."""
    results: list[Classification]
    regime: Criterion
    risk_on: bool
    quartile_mode: str          # "quartile" (relative) or "absolute" (tiny universe)
    n_eligible: int             # names that cleared gates 1 and 2

    def to_dict(self) -> dict:
        return {"banner": BANNER, "regime": self.regime.to_dict(),
                "risk_on": self.risk_on, "quartile_mode": self.quartile_mode,
                "n_eligible": self.n_eligible,
                "verdict_order": list(VERDICT_ORDER),
                "results": [r.to_dict() for r in self.results]}


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def rolling_ols_beta(name_rets, mkt_rets, window: int) -> tuple[float, float]:
    """Plain OLS beta and r^2 over the trailing `window` daily log returns.

    Deliberately NOT `beta_stats` (EWMA, half-life ~138d): the two fixed windows are
    the point here. Comparing a 63d and a 252d fit is what exposes a beta that has
    quietly changed, which one exponentially-weighted number blends away. The pair of
    r^2 values also feeds the idiosyncratic gate for free.
    """
    y = np.asarray(name_rets, dtype=float)
    x = np.asarray(mkt_rets, dtype=float)
    if y.size != x.size:
        raise ValueError("return series length mismatch")
    if y.size < window:
        raise ValueError(f"need {window} observations for the beta window, have {y.size}")
    y, x = y[-window:], x[-window:]
    var_m = float(np.var(x, ddof=1))
    if var_m <= 0:
        raise ValueError("market variance is zero over the window")
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_m)
    r2 = 0.0 if float(np.std(y, ddof=1)) <= 0 else float(np.corrcoef(x, y)[0, 1] ** 2)
    return beta, r2


def treynor_ratio(name_rets, beta: float, rf_annual: float = RF_DEFAULT,
                  window: int = WIN_LONG) -> float:
    """Annualized excess log return per unit of beta over the trailing window.

    Only called for names that already cleared the beta gate, so the denominator is
    bounded away from zero and the ratio cannot explode on a near-zero beta."""
    r = np.asarray(name_rets, dtype=float)
    if r.size < window:
        raise ValueError(f"need {window} observations for Treynor, have {r.size}")
    if beta == 0.0:
        raise ValueError("beta is zero; Treynor undefined")
    return float((TRADING_DAYS * float(np.mean(r[-window:])) - rf_annual) / beta)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def g_liquidity(closes, volumes, min_adv: float = T_ADV_DOLLAR
                ) -> tuple[Criterion, float | None]:
    """1. Can a position be entered and exited without paying for the privilege?

    Median rather than mean dollar volume: one halt-and-resume day or index-rebalance
    print should not qualify a name that is otherwise untradeable. Missing volume data
    grades INFO and still fails the gate -- you cannot clear a liquidity bar that
    cannot be measured."""
    px = np.asarray(closes, dtype=float) if closes is not None else None
    vol = np.asarray(volumes, dtype=float) if volumes is not None else None
    threshold = f"pass: median daily $ volume > ${min_adv / 1e6:,.0f}M over {ADV_WINDOW}d"
    if px is None or px.size == 0 or vol is None or vol.size == 0 or not np.any(vol > 0):
        return (Criterion("adv", "Liquid enough?", None, "no volume data", INFO,
                          threshold, "liquidity cannot be verified -- excluded"), None)
    n = min(px.size, vol.size)
    adv = float(np.median(px[-n:][-ADV_WINDOW:] * vol[-n:][-ADV_WINDOW:]))
    grade = PASS if adv > min_adv else FAIL
    note = ("deep enough to trade" if grade == PASS else
            "illiquid -- slippage and gap risk dominate any edge")
    return (Criterion("adv", "Liquid enough?", adv,
                      f"${adv / 1e6:,.1f}M median daily $ volume", grade, threshold, note),
            adv)


def g_beta(name_rets, mkt_rets, min_beta: float = T_BETA_MIN) -> tuple[Criterion, dict]:
    """2. Is it high-beta on BOTH windows, or was it high-beta a year ago?

    The 252d fit is the structural claim; the 63d fit only has to confirm the name is
    still an amplifier today. A large gap between them grades WARN, not FAIL: the name
    stays eligible, but 'its beta is 2.1' is no longer a stable description of it."""
    b_long, r2_long = rolling_ols_beta(name_rets, mkt_rets, WIN_LONG)
    b_short, r2_short = rolling_ols_beta(name_rets, mkt_rets, WIN_SHORT)
    stats = {"beta_63": b_short, "beta_252": b_long,
             "r2_63": r2_short, "r2_252": r2_long}
    confirm = BETA_CONFIRM_FRAC * min_beta
    diverge = abs(b_short - b_long) / abs(b_long) if b_long != 0 else float("inf")
    if b_long < min_beta:
        grade, note = FAIL, f"252d beta below the {min_beta:g} cut -- not high-beta"
    elif b_short < confirm:
        grade, note = FAIL, "63d beta has collapsed -- the high-beta claim is stale"
    elif diverge > BETA_DIVERGE:
        grade, note = WARN, "beta is shifting between windows -- the estimate is unstable"
    else:
        grade, note = PASS, "market amplifier on both windows"
    return (Criterion("beta", "High beta, both windows?", b_long,
                      f"252d {b_long:.2f} | 63d {b_short:.2f} | r2 {r2_long:.2f}",
                      grade,
                      f"pass: 252d >= {min_beta:g} and 63d >= {confirm:.2f}; "
                      f"warn if the windows differ by > {BETA_DIVERGE:.0%}", note),
            stats)


def g_regime(mkt_closes, window: int = SMA_WINDOW,
             market: str = MARKET_LABEL) -> tuple[Criterion, bool]:
    """3. Does the market environment favour carrying amplified risk at all?

    Computed once per run from the market's own closes -- the regime is a fact about
    the market, not about any name. Below the SMA the gate does not condemn names; it
    withholds the 'investable' label (see `_verdict`)."""
    px = np.asarray(mkt_closes, dtype=float)
    threshold = f"risk-on: {market} close > its {window}d SMA"
    if px.size < window:
        return (Criterion("regime", "Market regime", None,
                          f"only {px.size} {market} bars (need {window})", INFO, threshold,
                          "too little market history to judge -- treated as risk-on"),
                True)
    sma = float(np.mean(px[-window:]))
    last = float(px[-1])
    gap = last / sma - 1.0 if sma > 0 else 0.0
    risk_on = last > sma
    return (Criterion("regime", "Market regime", gap,
                      f"{market} {last:,.2f} vs {window}d SMA {sma:,.2f} ({gap:+.1%})",
                      PASS if risk_on else FAIL, threshold,
                      "risk-on" if risk_on else
                      "risk-off -- high-beta longs stand aside regardless of name quality"),
            risk_on)


def g_treynor(treynor: float, pctl: float, n_eligible: int,
              top_frac: float = TREYNOR_TOP_FRAC) -> Criterion:
    """4. Was the holder actually paid for the beta they carried?

    Relative by design: 'good return per unit of beta' only means something against a
    peer set. Below `MIN_UNIVERSE_FOR_QUARTILE` names a quartile is arithmetic theatre,
    so the gate falls back to the absolute question (was excess return positive at all)
    and says so in the threshold text rather than pretending to rank."""
    if n_eligible < MIN_UNIVERSE_FOR_QUARTILE:
        ok = treynor > 0.0
        return Criterion("treynor", "Paid for its beta?", treynor,
                         f"Treynor {treynor:+.2f}/yr | universe of {n_eligible} "
                         f"too small to rank",
                         PASS if ok else FAIL,
                         f"fallback (< {MIN_UNIVERSE_FOR_QUARTILE} eligible names): "
                         f"pass: Treynor > 0",
                         "positive excess return per unit of beta" if ok else
                         "carried the beta and was not paid for it")
    ok = pctl >= 1.0 - top_frac
    return Criterion("treynor", "Paid for its beta?", treynor,
                     f"Treynor {treynor:+.2f}/yr | {pctl:.0%} percentile of "
                     f"{n_eligible} eligible",
                     PASS if ok else FAIL,
                     f"pass: top {top_frac:.0%} of the eligible universe",
                     "top-quartile return per unit of beta" if ok else
                     "middling or negative return per unit of beta")


def g_idio(r2_63: float, r2_252: float) -> Criterion:
    """5. Is this a market amplifier, or a coin-flip wearing a beta?

    Idiosyncratic share is 1 - r2. High *and rising* is the signature this gate exists
    to catch: a name whose own story (trial readout, verdict, single contract) is
    taking over from the market is one where beta tells you nothing about the outcome
    that matters. High but stable is a warning, not a disqualification -- plenty of
    real businesses are simply not very market-driven.

    THE TWO WINDOWS HAVE DIFFERENT JOBS. r^2 over 63 observations carries large
    sampling error, so the short window is not trusted to establish the LEVEL -- only
    the 252d fit does that. The short window's job is to corroborate the DIRECTION.
    A name therefore fails only when the reliable window says "high" and the pair says
    "and getting worse".

    An earlier version let either window declare the level, so a name at 67% over 252d
    reading 77% over 63d was condemned outright. That rule was written from a single
    observation (COIN) and SOFI promptly produced a near-identical signature that we did
    not believe. Such a name now grades WARN: the short window may raise a flag, it may
    not condemn on its own."""
    idio_long = 1.0 - float(r2_252)
    idio_short = 1.0 - float(r2_63)
    delta = idio_short - idio_long
    high_long = idio_long >= T_IDIO_SHARE
    high_short = idio_short >= T_IDIO_SHARE
    expanding = delta > T_IDIO_COMPRESS
    compressing = delta <= -T_IDIO_COMPRESS
    threshold = (f"fail: 252d idio share >= {T_IDIO_SHARE:.0%} AND rising by "
                 f">= {T_IDIO_COMPRESS:.0%}; the 63d window corroborates, it cannot "
                 f"condemn alone")
    if high_long and expanding:
        grade, note = FAIL, ("idiosyncratic share is high and expanding -- "
                             "binary-event profile")
    elif high_long and not compressing:
        grade, note = WARN, "high but stable idiosyncratic share -- own-story risk"
    elif high_short and expanding:
        grade, note = WARN, ("recent idiosyncratic share is high and rising, but the "
                             "252d window does not corroborate it yet")
    elif high_long:      # high on the long window, but compressing toward the market
        grade, note = PASS, "idiosyncratic share is compressing toward the market"
    else:
        grade, note = PASS, "market-driven, not story-driven"
    return Criterion("idio", "Market-driven?", idio_long,
                     f"idio share 252d {idio_long:.0%} | 63d {idio_short:.0%} "
                     f"({delta:+.0%})", grade, threshold, note)


def g_survivable(closes, window: int = SURV_WINDOW) -> tuple[Criterion, dict]:
    """6. How often has holding this for ~6 months cost half the position?

    The only gate that looks at the name's WHOLE history rather than the trailing year,
    and the one that separates a volatile business from a chronic collapse. SPCE and
    CRWV are indistinguishable on every one-year measure; over full history one halves
    in 21% of windows and the other in 7%. Persistent halvings are what serial dilution
    looks like on a chart -- gate 8 measures the same disease directly.

    Short history grades WARN, not PASS: a name too young to have survived anything has
    not demonstrated survival, and WARN correctly blocks `investable` without condemning."""
    px = np.asarray(closes, dtype=float)
    threshold = (f"pass < {T_HALVED_MILD:.0%}, fail >= {T_HALVED_SEVERE:.0%} of "
                 f"{window}d windows halved")
    if px.size <= window + 1:
        return (Criterion("survivable", "Survivable?", None,
                          f"only {px.size} bars (need > {window + 1})", WARN, threshold,
                          "too little history to judge survivability"), {})
    fwd = px[window:] / px[:-window]
    halved = float(np.mean(fwd <= 0.5))
    worst = float(fwd.min() - 1.0)
    max_dd = float(np.max(1.0 - px / np.maximum.accumulate(px)))
    grade = (PASS if halved < T_HALVED_MILD else
             WARN if halved < T_HALVED_SEVERE else FAIL)
    note = ("halving is routine -- chronic-collapse / serial-diluter profile"
            if grade == FAIL else
            "occasional halvings" if grade == WARN else "halvings are rare")
    return (Criterion("survivable", "Survivable?", halved,
                      f"{halved:.0%} of {window}d windows halved | worst {worst:+.0%} | "
                      f"max DD {-max_dd:.0%}", grade, threshold, note),
            {"halved_126d": halved, "worst_window": worst, "max_dd": max_dd})


def g_own_trend(closes, window: int = OWN_SMA_WINDOW) -> tuple[Criterion, bool]:
    """7. Is the name itself in an uptrend? ADVISORY ONLY.

    Deliberately the weakest gate in the stack. A trend filter is a momentum signal, and
    momentum failed every predictive test in this project -- so this one may WITHHOLD a
    blessing but never condemn. It grades WARN, never FAIL, and takes no part in the
    `reckless` decision. Treat it as "the market disagrees with you right now", not as
    evidence about what happens next."""
    px = np.asarray(closes, dtype=float)
    threshold = (f"advisory: close > its own {window}d SMA blesses; below only blocks "
                 f"'investable', never condemns")
    if px.size < window:
        return (Criterion("own_trend", "Own trend", None,
                          f"only {px.size} bars (need {window})", INFO, threshold,
                          "too little history to judge -- not held against the name"),
                True)
    sma = float(np.mean(px[-window:]))
    last = float(px[-1])
    gap = last / sma - 1.0 if sma > 0 else 0.0
    above = last > sma
    return (Criterion("own_trend", "Own trend", gap,
                      f"close {last:,.2f} vs its {window}d SMA {sma:,.2f} ({gap:+.1%})",
                      PASS if above else WARN, threshold,
                      "in its own uptrend" if above else
                      f"below its own {window}d SMA -- blessing withheld"),
            above)


def g_dilution(dilution: float | None, n_bars: int | None = None,
               rev_ttm: float | None = None) -> Criterion:
    """8. Is the company printing shares to stay alive?

    The disease gate 6 only sees the symptom of. A name issuing 30% more stock a year is
    transferring the upside from holders to the treasury; the chart shows it as a slow
    bleed that no amount of good news arrests.

    TWO CAPS, both at WARN, because in each case the number does not mean what the
    threshold assumes:

      * RECENT IPOs. A company public for under `DILUTION_MIN_BARS` has a share count
        dominated by the flotation and its lockup expiries, not by a funding treadmill.
        CRWV reads +27% YoY sixteen months after listing -- that is simply what going
        public looks like, and year one cannot tell the two apart.

      * A REAL REVENUE BASE. Issuing stock while booking billions in sales is capital
        formation; a lender must raise equity to grow its book at all. Issuing stock
        with no product is survival. Both show the same share-count growth, and this is
        the SAME judgement `g_runway` already makes about negative free cash flow --
        applying it there but not here was an inconsistency, not a decision.

    SPCE is untouched by either cap: years of history behind its +195%, and $1.3M of
    revenue is nowhere near the bar."""
    threshold = (f"pass <= {T_DILUTION_OK:.0%}/yr share growth, "
                 f"fail > {T_DILUTION_HEAVY:.0%}/yr")
    if dilution is None:
        return Criterion("dilution", "Diluting holders?", None,
                         "share-count history unavailable", INFO, threshold,
                         "dilution unverified -- not held against the name")
    if dilution <= T_DILUTION_OK:
        grade, note = PASS, "share count flat or shrinking"
    elif dilution <= T_DILUTION_HEAVY:
        grade, note = WARN, "moderate issuance -- watch the treadmill"
    else:
        grade, note = FAIL, "heavy dilution -- holders are paying for the story"
    young = n_bars is not None and n_bars < DILUTION_MIN_BARS
    funded = rev_ttm is not None and float(rev_ttm) >= T_REVENUE_REAL
    if grade == FAIL and young:
        grade = WARN
        note = ("share count still reflects the IPO, not a dilution trend -- "
                "too young to tell them apart")
    elif grade == FAIL and funded:
        grade = WARN
        note = ("issuing stock against a real revenue base -- capital formation, "
                "not a survival treadmill")
    return Criterion("dilution", "Diluting holders?", float(dilution),
                     f"share count {dilution:+.1%} YoY", grade, threshold, note)


def g_runway(quarters: float | None, rev_ttm: float | None = None,
             fcf_meaningful: bool = True) -> Criterion:
    """9. How long until the company must raise, sell itself, or stop?

    A short runway is not merely a risk, it is a schedule. Combined with a binary-event
    idiosyncratic profile it is the complete clinical-biotech trap: the next corporate
    event is either the readout or the raise, and the raise is gate 8's problem.

    A REAL REVENUE BASE CAPS THIS AT WARN. Burning cash while booking billions in sales
    is a financing decision about growth capex; burning cash with no product is a
    countdown. Both show negative free cash flow, and treating them alike would condemn
    every capital-intensive growth company on the same evidence as a dying one. The
    countdown reading is reserved for companies without the revenue to choose."""
    threshold = (f"pass >= {T_RUNWAY_OK:.0f} quarters of cash, "
                 f"fail < {T_RUNWAY_CRITICAL:.0f}")
    if not fcf_meaningful:
        # A lender books loan origination as an operating outflow, so a healthy growing
        # bank reads as though it is burning cash. Reporting no number is honest; a
        # number that measures the wrong thing is not.
        return Criterion("runway", "Cash runway", None,
                         "not meaningful for this sector", INFO, threshold,
                         "free cash flow does not measure burn for a lender -- "
                         "runway not assessed")
    if quarters is None:
        return Criterion("runway", "Cash runway", None,
                         "cash-flow statements unavailable", INFO, threshold,
                         "runway unverified -- not held against the name")
    if quarters == float("inf"):
        return Criterion("runway", "Cash runway", RUNWAY_DISPLAY_CAP,
                         "self-funding (free cash flow >= 0)", PASS, threshold,
                         "generates its own cash -- no runway clock")
    q = float(quarters)
    if q >= T_RUNWAY_OK:
        grade, note = PASS, "comfortably funded"
    elif q >= T_RUNWAY_CRITICAL:
        grade, note = WARN, "under two years of cash -- a raise is foreseeable"
    else:
        grade, note = FAIL, "under a year of cash -- forced dilution or worse ahead"
    funded = rev_ttm is not None and float(rev_ttm) >= T_REVENUE_REAL
    if funded and grade == FAIL:
        grade = WARN
        note = ("burning cash against a real revenue base -- a financing choice, "
                "not a countdown")
    return Criterion("runway", "Cash runway", min(q, RUNWAY_DISPLAY_CAP),
                     f"~{min(q, RUNWAY_DISPLAY_CAP):.1f} quarters of cash at current burn",
                     grade, threshold, note)


def g_revenue(rev_ttm: float | None, n_quarters: int = 0) -> Criterion:
    """10. Does an operating business exist, or is the price entirely narrative?

    ADVISORY ONLY, and never FAIL. Whether $8M of revenue is "real" is a judgement about
    the future, which this project refuses to make. The gate reports what is there and
    lets a pre-revenue name block its own blessing, nothing more."""
    threshold = (f"advisory: TTM revenue >= ${T_REVENUE_REAL / 1e6:,.0f}M blesses; "
                 f"below only blocks 'investable', never condemns")
    if rev_ttm is None or n_quarters <= 0:
        return Criterion("revenue", "Real revenue?", None,
                         "income statements unavailable", INFO, threshold,
                         "revenue unverified -- not held against the name")
    rev = float(rev_ttm)
    if rev < T_REVENUE_REAL:
        return Criterion("revenue", "Real revenue?", rev,
                         f"TTM revenue ${rev / 1e6:,.1f}M over {n_quarters}q", WARN,
                         threshold, "pre-revenue story -- the price is all narrative")
    return Criterion("revenue", "Real revenue?", rev,
                     f"TTM revenue ${rev / 1e6:,.0f}M over {n_quarters}q", PASS,
                     threshold, "an operating business underneath the beta")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def _verdict(c_beta: Criterion, c_treynor: Criterion, c_idio: Criterion,
             c_surv: Criterion, c_trend: Criterion, c_dilution: Criterion,
             c_runway: Criterion, c_revenue: Criterion,
             risk_on: bool) -> tuple[str, list[str]]:
    """Every gate can withhold a blessing; only some can condemn.

    `investable` requires a clean sweep, with one exemption: INFO means *unverifiable*,
    not *bad*. Yahoo's fundamentals coverage is patchy and young listings have no
    200-day history, so treating INFO as a blocker would make the label unreachable for
    reasons that say nothing about the company. The INFO notes still land in `reasons`,
    so an unverified pass is always visible as one.

    Note the deliberate asymmetry against gate 1, which EXCLUDES a name whose volume is
    unknown. Liquidity is measurable for anything genuinely tradable, so its absence is
    itself the finding; fundamentals coverage is a property of the data vendor.

    `reckless` needs an uncompensated beta AND at least `RECKLESS_MIN_FLAGS` structural
    red flags -- from a binary-event idio profile, a chronic-collapse history, heavy
    dilution, or a critical runway. Requiring TWO is deliberate: any single gate can be
    wrong about a single name (a threshold sits one point away, a vendor number is
    stale), and the harshest label in the vocabulary should not turn on one reading.
    Corroboration is cheap here because a genuinely broken name lights up several at
    once -- SPCE carries three.

    A name that WAS paid for its beta is never reckless, whatever its history. The two
    advisory gates (trend, revenue) are absent from the red-flag set entirely: both
    shade into predicting returns, so neither may condemn.

    The flags that fired are RETURNED, not just counted, so the label always ships with
    its evidence -- "reckless" on three counts reads differently from a marginal two.
    """
    opinion = (c_beta, c_treynor, c_idio, c_surv, c_trend,
               c_dilution, c_runway, c_revenue)
    investable = all(c.grade in (PASS, INFO) for c in opinion)
    flags = [c.key for c in (c_idio, c_surv, c_dilution, c_runway) if c.grade == FAIL]
    reckless = c_treynor.grade == FAIL and len(flags) >= RECKLESS_MIN_FLAGS
    base = INVESTABLE if investable else RECKLESS if reckless else MIXED
    reasons = [c.note for c in opinion if c.note]
    if base == INVESTABLE and not risk_on:
        return STAND_ASIDE, ["would be investable, but the market is below its "
                             f"{SMA_WINDOW}d SMA"] + reasons, flags
    return base, reasons, flags


def classify_universe(per_name: dict[str, dict], mkt_closes,
                      rf: float = RF_DEFAULT, min_adv: float = T_ADV_DOLLAR,
                      min_beta: float = T_BETA_MIN,
                      market: str = MARKET_LABEL) -> UniverseResult:
    """Run the gate stack over a whole universe.

    `per_name[sym]` supplies pre-aligned arrays -- {"closes", "volumes", "name_rets",
    "mkt_rets"} -- so this module never touches the network or a database and stays
    directly testable. The caller owns alignment; both return series must sit on the
    same trading calendar.

    `per_name[sym]["fundamentals"]` is OPTIONAL: a dict of already-derived plain numbers
    ({"dilution_yoy", "runway_quarters", "revenue_ttm", "n_revenue_quarters"}) as
    produced by `src/forecast/fundamentals.py`. Absent or empty, gates 8-10 grade INFO.
    Deriving upstream keeps this module free of any fundamentals import, so the vendored
    web copy reads the same plain numbers without re-implementing the reduction.

    Gate 4 is universe-relative, so it runs only after every name has been staged:
    the quartile is measured across names that cleared gates 1 and 2, never across the
    raw submission (otherwise illiquid junk would dilute the ranking of real candidates).
    """
    regime_c, risk_on = g_regime(mkt_closes, market=market)
    staged: dict[str, dict] = {}
    out: list[Classification] = []

    for sym, d in per_name.items():
        try:
            c_adv, adv = g_liquidity(d.get("closes"), d.get("volumes"), min_adv)
            if c_adv.grade != PASS:
                out.append(Classification(sym, EXCLUDED_ILLIQUID, [c_adv, regime_c],
                                          {"adv_dollar": adv}, [c_adv.note]))
                continue
            c_beta, bstats = g_beta(d["name_rets"], d["mkt_rets"], min_beta)
            metrics = {"adv_dollar": adv, **bstats}
            if c_beta.grade == FAIL:
                out.append(Classification(sym, EXCLUDED_LOW_BETA,
                                          [c_adv, c_beta, regime_c], metrics,
                                          [c_beta.note]))
                continue
            metrics["treynor"] = treynor_ratio(d["name_rets"], bstats["beta_252"], rf)

            c_surv, sstats = g_survivable(d["closes"])
            metrics.update(sstats)
            c_trend, _above = g_own_trend(d["closes"])
            metrics["own_sma_gap"] = c_trend.value

            f = d.get("fundamentals") or {}
            # Both gates are told enough to recognise their own blind spots: a share
            # count is uninterpretable across an IPO, and negative free cash flow means
            # something different when there is revenue behind it.
            c_dil = g_dilution(f.get("dilution_yoy"), n_bars=int(np.size(d["closes"])),
                               rev_ttm=f.get("revenue_ttm"))
            c_run = g_runway(f.get("runway_quarters"), rev_ttm=f.get("revenue_ttm"),
                             fcf_meaningful=f.get("fcf_meaningful", True))
            c_rev = g_revenue(f.get("revenue_ttm"), f.get("n_revenue_quarters", 0))
            for key, crit in (("dilution_yoy", c_dil), ("runway_quarters", c_run),
                              ("revenue_ttm", c_rev)):
                if crit.value is not None:
                    metrics[key] = crit.value   # already inf-clamped by the gate

            staged[sym] = {"c_adv": c_adv, "c_beta": c_beta, "c_surv": c_surv,
                           "c_trend": c_trend, "c_dil": c_dil, "c_run": c_run,
                           "c_rev": c_rev, "metrics": metrics}
        except Exception as e:  # noqa: BLE001 -- one bad ticker must not kill the run
            out.append(Classification(sym, ERROR, [], {},
                                      [f"could not classify: {type(e).__name__}: {e}"]))

    n_eligible = len(staged)
    mode = (QUARTILE_RELATIVE if n_eligible >= MIN_UNIVERSE_FOR_QUARTILE
            else QUARTILE_ABSOLUTE)
    tvals = np.asarray([v["metrics"]["treynor"] for v in staged.values()], dtype=float)

    for sym, v in staged.items():
        m = dict(v["metrics"])
        # Percentile rank: the share of the eligible universe at or below this name.
        m["treynor_pctl"] = float(np.mean(tvals <= m["treynor"]))
        c_treynor = g_treynor(m["treynor"], m["treynor_pctl"], n_eligible)
        c_idio = g_idio(m["r2_63"], m["r2_252"])
        verdict, reasons, flags = _verdict(
            v["c_beta"], c_treynor, c_idio, v["c_surv"], v["c_trend"],
            v["c_dil"], v["c_run"], v["c_rev"], risk_on)
        m["flags"] = flags
        out.append(Classification(sym, verdict,
                                  [v["c_adv"], v["c_beta"], regime_c, c_treynor, c_idio,
                                   v["c_surv"], v["c_trend"], v["c_dil"], v["c_run"],
                                   v["c_rev"]],
                                  m, reasons))

    rank = {v: i for i, v in enumerate(VERDICT_ORDER)}
    out.sort(key=lambda c: (rank.get(c.verdict, len(VERDICT_ORDER)),
                            -(c.metrics.get("treynor") or 0.0), c.symbol))
    return UniverseResult(out, regime_c, risk_on, mode, n_eligible)
