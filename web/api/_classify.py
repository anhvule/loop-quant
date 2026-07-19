"""High-beta investable/reckless classifier for the web API (numpy + stdlib only).

Vendored from `src/forecast/classify.py`; `tests/test_web_classify.py` pins the two
together so the site cannot label a ticker differently from the CLI.

CLASSIFICATION, NOT PREDICTION. The Treynor gate ranks realized return per unit of beta
over the trailing year -- a property of the past, not a forecast. See the source module
for the full reasoning behind each gate.

One deliberate difference from the CLI, and it is reported rather than hidden: the web
runs on the handful of tickers a visitor submits, so the eligible universe is almost
always below `MIN_UNIVERSE_FOR_QUARTILE` and the Treynor gate falls back to its absolute
rule. `quartile_mode` in the payload says which rule was applied. A true top-quartile
rank against the S&P 500 is a CLI-only capability.
"""

from __future__ import annotations

import numpy as np

import _engine as E

PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"

T_ADV_DOLLAR = 10e6
ADV_WINDOW = 60
T_BETA_MIN = 1.5
BETA_CONFIRM_FRAC = 0.75
BETA_DIVERGE = 0.50
SMA_WINDOW = 200
TREYNOR_TOP_FRAC = 0.25
MIN_UNIVERSE_FOR_QUARTILE = 8
T_IDIO_SHARE = 0.75
T_IDIO_COMPRESS = 0.05
WIN_SHORT, WIN_LONG = 63, 252
MIN_BARS_CLASSIFY = 300
RF_DEFAULT = 0.04
TRADING_DAYS = 252
MARKET_LABEL = "SPY"

SURV_WINDOW = 126
T_HALVED_MILD = 0.05
T_HALVED_SEVERE = 0.20
OWN_SMA_WINDOW = 200

T_DILUTION_OK = 0.03
T_DILUTION_HEAVY = 0.15
T_RUNWAY_OK = 8.0
T_RUNWAY_CRITICAL = 4.0
T_REVENUE_REAL = 10e6
RUNWAY_DISPLAY_CAP = 99.0
DILUTION_MIN_BARS = 504
RECKLESS_MIN_FLAGS = 2

INVESTABLE = "investable"
MIXED = "mixed"
RECKLESS = "reckless"
STAND_ASIDE = "stand_aside"
EXCLUDED_ILLIQUID = "excluded_illiquid"
EXCLUDED_LOW_BETA = "excluded_low_beta"
ERROR = "error"

VERDICT_ORDER = [INVESTABLE, MIXED, STAND_ASIDE, RECKLESS,
                 EXCLUDED_LOW_BETA, EXCLUDED_ILLIQUID, ERROR]

QUARTILE_RELATIVE = "quartile"
QUARTILE_ABSOLUTE = "absolute"

BANNER = ("CLASSIFICATION, NOT PREDICTION. The Treynor gate ranks realized return per "
          "unit of beta over the trailing year -- a historical property, not a forecast. "
          "'Investable' means the name cleared liquidity, was genuinely high-beta on "
          "both windows, was paid for that beta in the past, and is not dominated by its "
          "own story. It does not mean the name will rise.")

VERDICT_TEXT = {
    INVESTABLE: "cleared every gate",
    MIXED: "cleared the hard gates, but they disagree on the rest",
    RECKLESS: "not paid for its beta AND increasingly driven by its own story",
    STAND_ASIDE: "would clear every gate, but the market is below its 200d SMA",
    EXCLUDED_LOW_BETA: "not high-beta -- outside the scope of this screen",
    EXCLUDED_ILLIQUID: "too illiquid to trade without paying for it",
    ERROR: "could not be classified",
}


def _crit(key, title, value, display, grade, threshold, note=""):
    return {"key": key, "title": title, "value": value, "display": display,
            "grade": grade, "threshold": threshold, "note": note}


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def rolling_ols_beta(name_rets, mkt_rets, window):
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


def treynor_ratio(name_rets, beta, rf_annual=RF_DEFAULT, window=WIN_LONG):
    r = np.asarray(name_rets, dtype=float)
    if r.size < window:
        raise ValueError(f"need {window} observations for Treynor, have {r.size}")
    if beta == 0.0:
        raise ValueError("beta is zero; Treynor undefined")
    return float((TRADING_DAYS * float(np.mean(r[-window:])) - rf_annual) / beta)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def g_liquidity(closes, volumes, min_adv=T_ADV_DOLLAR):
    px = np.asarray(closes, dtype=float) if closes is not None else None
    vol = np.asarray(volumes, dtype=float) if volumes is not None else None
    threshold = f"pass: median daily $ volume > ${min_adv / 1e6:,.0f}M over {ADV_WINDOW}d"
    if px is None or px.size == 0 or vol is None or vol.size == 0 or not np.any(vol > 0):
        return (_crit("adv", "Liquid enough?", None, "no volume data", INFO,
                      threshold, "liquidity cannot be verified -- excluded"), None)
    n = min(px.size, vol.size)
    adv = float(np.median(px[-n:][-ADV_WINDOW:] * vol[-n:][-ADV_WINDOW:]))
    grade = PASS if adv > min_adv else FAIL
    note = ("deep enough to trade" if grade == PASS else
            "illiquid -- slippage and gap risk dominate any edge")
    return (_crit("adv", "Liquid enough?", adv,
                  f"${adv / 1e6:,.1f}M median daily $ volume", grade, threshold, note),
            adv)


def g_beta(name_rets, mkt_rets, min_beta=T_BETA_MIN):
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
    return (_crit("beta", "High beta, both windows?", b_long,
                  f"252d {b_long:.2f} | 63d {b_short:.2f} | r2 {r2_long:.2f}", grade,
                  f"pass: 252d >= {min_beta:g} and 63d >= {confirm:.2f}; "
                  f"warn if the windows differ by > {BETA_DIVERGE:.0%}", note),
            stats)


def g_regime(mkt_closes, window=SMA_WINDOW, market=MARKET_LABEL):
    px = np.asarray(mkt_closes, dtype=float)
    threshold = f"risk-on: {market} close > its {window}d SMA"
    if px.size < window:
        return (_crit("regime", "Market regime", None,
                      f"only {px.size} {market} bars (need {window})", INFO, threshold,
                      "too little market history to judge -- treated as risk-on"), True)
    sma = float(np.mean(px[-window:]))
    last = float(px[-1])
    gap = last / sma - 1.0 if sma > 0 else 0.0
    risk_on = last > sma
    return (_crit("regime", "Market regime", gap,
                  f"{market} {last:,.2f} vs {window}d SMA {sma:,.2f} ({gap:+.1%})",
                  PASS if risk_on else FAIL, threshold,
                  "risk-on" if risk_on else
                  "risk-off -- high-beta longs stand aside regardless of name quality"),
            risk_on)


def g_treynor(treynor, pctl, n_eligible, top_frac=TREYNOR_TOP_FRAC):
    if n_eligible < MIN_UNIVERSE_FOR_QUARTILE:
        ok = treynor > 0.0
        return _crit("treynor", "Paid for its beta?", treynor,
                     f"Treynor {treynor:+.2f}/yr | universe of {n_eligible} "
                     f"too small to rank",
                     PASS if ok else FAIL,
                     f"fallback (< {MIN_UNIVERSE_FOR_QUARTILE} eligible names): "
                     f"pass: Treynor > 0",
                     "positive excess return per unit of beta" if ok else
                     "carried the beta and was not paid for it")
    ok = pctl >= 1.0 - top_frac
    return _crit("treynor", "Paid for its beta?", treynor,
                 f"Treynor {treynor:+.2f}/yr | {pctl:.0%} percentile of "
                 f"{n_eligible} eligible",
                 PASS if ok else FAIL,
                 f"pass: top {top_frac:.0%} of the eligible universe",
                 "top-quartile return per unit of beta" if ok else
                 "middling or negative return per unit of beta")


def g_idio(r2_63, r2_252):
    """Level from the 252d fit (the reliable estimator); direction from the pair. The
    63d window has too few observations to declare the level on its own."""
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
    elif high_long:
        grade, note = PASS, "idiosyncratic share is compressing toward the market"
    else:
        grade, note = PASS, "market-driven, not story-driven"
    return _crit("idio", "Market-driven?", idio_long,
                 f"idio share 252d {idio_long:.0%} | 63d {idio_short:.0%} "
                 f"({delta:+.0%})", grade, threshold, note)


def g_survivable(closes, window=SURV_WINDOW):
    px = np.asarray(closes, dtype=float)
    threshold = (f"pass < {T_HALVED_MILD:.0%}, fail >= {T_HALVED_SEVERE:.0%} of "
                 f"{window}d windows halved")
    if px.size <= window + 1:
        return (_crit("survivable", "Survivable?", None,
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
    return (_crit("survivable", "Survivable?", halved,
                  f"{halved:.0%} of {window}d windows halved | worst {worst:+.0%} | "
                  f"max DD {-max_dd:.0%}", grade, threshold, note),
            {"halved_126d": halved, "worst_window": worst, "max_dd": max_dd})


def g_own_trend(closes, window=OWN_SMA_WINDOW):
    px = np.asarray(closes, dtype=float)
    threshold = (f"advisory: close > its own {window}d SMA blesses; below only blocks "
                 f"'investable', never condemns")
    if px.size < window:
        return (_crit("own_trend", "Own trend", None,
                      f"only {px.size} bars (need {window})", INFO, threshold,
                      "too little history to judge -- not held against the name"), True)
    sma = float(np.mean(px[-window:]))
    last = float(px[-1])
    gap = last / sma - 1.0 if sma > 0 else 0.0
    above = last > sma
    return (_crit("own_trend", "Own trend", gap,
                  f"close {last:,.2f} vs its {window}d SMA {sma:,.2f} ({gap:+.1%})",
                  PASS if above else WARN, threshold,
                  "in its own uptrend" if above else
                  f"below its own {window}d SMA -- blessing withheld"), above)


def g_dilution(dilution, n_bars=None, rev_ttm=None):
    """Two WARN caps: a share count cannot be read across an IPO, and issuance against
    a real revenue base is capital formation rather than a survival treadmill -- the
    same judgement g_runway already makes about negative free cash flow."""
    threshold = (f"pass <= {T_DILUTION_OK:.0%}/yr share growth, "
                 f"fail > {T_DILUTION_HEAVY:.0%}/yr")
    if dilution is None:
        return _crit("dilution", "Diluting holders?", None,
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
    return _crit("dilution", "Diluting holders?", float(dilution),
                 f"share count {dilution:+.1%} YoY", grade, threshold, note)


def g_runway(quarters, rev_ttm=None, fcf_meaningful=True):
    threshold = (f"pass >= {T_RUNWAY_OK:.0f} quarters of cash, "
                 f"fail < {T_RUNWAY_CRITICAL:.0f}")
    if not fcf_meaningful:
        # Loan origination is an operating outflow, so a growing lender reads as though
        # it is burning cash. No number beats a wrong number.
        return _crit("runway", "Cash runway", None,
                     "not meaningful for this sector", INFO, threshold,
                     "free cash flow does not measure burn for a lender -- "
                     "runway not assessed")
    if quarters is None:
        return _crit("runway", "Cash runway", None,
                     "cash-flow statements unavailable", INFO, threshold,
                     "runway unverified -- not held against the name")
    if quarters == float("inf"):
        return _crit("runway", "Cash runway", RUNWAY_DISPLAY_CAP,
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
    return _crit("runway", "Cash runway", min(q, RUNWAY_DISPLAY_CAP),
                 f"~{min(q, RUNWAY_DISPLAY_CAP):.1f} quarters of cash at current burn",
                 grade, threshold, note)


def g_revenue(rev_ttm, n_quarters=0):
    threshold = (f"advisory: TTM revenue >= ${T_REVENUE_REAL / 1e6:,.0f}M blesses; "
                 f"below only blocks 'investable', never condemns")
    if rev_ttm is None or n_quarters <= 0:
        return _crit("revenue", "Real revenue?", None,
                     "income statements unavailable", INFO, threshold,
                     "revenue unverified -- not held against the name")
    rev = float(rev_ttm)
    if rev < T_REVENUE_REAL:
        return _crit("revenue", "Real revenue?", rev,
                     f"TTM revenue ${rev / 1e6:,.1f}M over {n_quarters}q", WARN,
                     threshold, "pre-revenue story -- the price is all narrative")
    return _crit("revenue", "Real revenue?", rev,
                 f"TTM revenue ${rev / 1e6:,.0f}M over {n_quarters}q", PASS,
                 threshold, "an operating business underneath the beta")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def _verdict(c_beta, c_treynor, c_idio, c_surv, c_trend, c_dilution, c_runway,
             c_revenue, risk_on):
    """INFO is exempt from the all-PASS requirement: unverifiable is not bad news.
    Deliberately the opposite of gate 1, which excludes a name whose volume is unknown --
    liquidity is measurable for anything tradable, fundamentals coverage is the vendor's
    property. The advisory gates (trend, revenue) never appear in the red-flag set.

    `reckless` needs RECKLESS_MIN_FLAGS corroborating flags: any single gate can be
    wrong about a single name, and the harshest label should not turn on one reading.
    The flags are returned so the label always ships with its evidence."""
    opinion = (c_beta, c_treynor, c_idio, c_surv, c_trend,
               c_dilution, c_runway, c_revenue)
    investable = all(c["grade"] in (PASS, INFO) for c in opinion)
    flags = [c["key"] for c in (c_idio, c_surv, c_dilution, c_runway)
             if c["grade"] == FAIL]
    reckless = c_treynor["grade"] == FAIL and len(flags) >= RECKLESS_MIN_FLAGS
    base = INVESTABLE if investable else RECKLESS if reckless else MIXED
    reasons = [c["note"] for c in opinion if c["note"]]
    if base == INVESTABLE and not risk_on:
        return STAND_ASIDE, ["would be investable, but the market is below its "
                             f"{SMA_WINDOW}d SMA"] + reasons, flags
    return base, reasons, flags


def classify_universe(per_name, mkt_closes, rf=RF_DEFAULT, min_adv=T_ADV_DOLLAR,
                      min_beta=T_BETA_MIN, market=MARKET_LABEL) -> dict:
    regime_c, risk_on = g_regime(mkt_closes, market=market)
    staged, out = {}, []

    for sym, d in per_name.items():
        try:
            c_adv, adv = g_liquidity(d.get("closes"), d.get("volumes"), min_adv)
            if c_adv["grade"] != PASS:
                out.append({"symbol": sym, "verdict": EXCLUDED_ILLIQUID,
                            "gates": [c_adv, regime_c], "metrics": {"adv_dollar": adv},
                            "reasons": [c_adv["note"]]})
                continue
            c_beta, bstats = g_beta(d["name_rets"], d["mkt_rets"], min_beta)
            metrics = dict({"adv_dollar": adv}, **bstats)
            if c_beta["grade"] == FAIL:
                out.append({"symbol": sym, "verdict": EXCLUDED_LOW_BETA,
                            "gates": [c_adv, c_beta, regime_c], "metrics": metrics,
                            "reasons": [c_beta["note"]]})
                continue
            metrics["treynor"] = treynor_ratio(d["name_rets"], bstats["beta_252"], rf)

            c_surv, sstats = g_survivable(d["closes"])
            metrics.update(sstats)
            c_trend, _above = g_own_trend(d["closes"])
            metrics["own_sma_gap"] = c_trend["value"]

            f = d.get("fundamentals") or {}
            c_dil = g_dilution(f.get("dilution_yoy"), n_bars=int(np.size(d["closes"])),
                               rev_ttm=f.get("revenue_ttm"))
            c_run = g_runway(f.get("runway_quarters"), rev_ttm=f.get("revenue_ttm"),
                             fcf_meaningful=f.get("fcf_meaningful", True))
            c_rev = g_revenue(f.get("revenue_ttm"), f.get("n_revenue_quarters", 0))
            for key, crit in (("dilution_yoy", c_dil), ("runway_quarters", c_run),
                              ("revenue_ttm", c_rev)):
                if crit["value"] is not None:
                    metrics[key] = crit["value"]

            staged[sym] = {"c_adv": c_adv, "c_beta": c_beta, "c_surv": c_surv,
                           "c_trend": c_trend, "c_dil": c_dil, "c_run": c_run,
                           "c_rev": c_rev, "metrics": metrics}
        except Exception as e:  # noqa: BLE001 -- one bad ticker must not kill the run
            out.append({"symbol": sym, "verdict": ERROR, "gates": [], "metrics": {},
                        "reasons": [f"could not classify: {type(e).__name__}: {e}"]})

    n_eligible = len(staged)
    mode = (QUARTILE_RELATIVE if n_eligible >= MIN_UNIVERSE_FOR_QUARTILE
            else QUARTILE_ABSOLUTE)
    tvals = np.asarray([v["metrics"]["treynor"] for v in staged.values()], dtype=float)

    for sym, v in staged.items():
        m = dict(v["metrics"])
        m["treynor_pctl"] = float(np.mean(tvals <= m["treynor"]))
        c_treynor = g_treynor(m["treynor"], m["treynor_pctl"], n_eligible)
        c_idio = g_idio(m["r2_63"], m["r2_252"])
        verdict, reasons, flags = _verdict(
            v["c_beta"], c_treynor, c_idio, v["c_surv"], v["c_trend"],
            v["c_dil"], v["c_run"], v["c_rev"], risk_on)
        m["flags"] = flags
        out.append({"symbol": sym, "verdict": verdict,
                    "gates": [v["c_adv"], v["c_beta"], regime_c, c_treynor, c_idio,
                              v["c_surv"], v["c_trend"], v["c_dil"], v["c_run"],
                              v["c_rev"]],
                    "metrics": m, "reasons": reasons})

    rank = {v: i for i, v in enumerate(VERDICT_ORDER)}
    out.sort(key=lambda c: (rank.get(c["verdict"], len(VERDICT_ORDER)),
                            -(c["metrics"].get("treynor") or 0.0), c["symbol"]))
    return {"banner": BANNER, "regime": regime_c, "risk_on": risk_on,
            "quartile_mode": mode, "n_eligible": n_eligible,
            "verdict_order": list(VERDICT_ORDER), "verdict_text": dict(VERDICT_TEXT),
            "results": out}


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def classify(symbols: list, max_symbols: int = 20, rf: float = RF_DEFAULT) -> dict:
    """Fetch, gate and label a submitted cohort. Per-symbol failures are isolated.

    Over-long lists are truncated, but the drop is REPORTED (`truncated`) rather than
    silent -- a screen that quietly ignores half your tickers is worse than an error."""
    requested = [s.strip().upper() for s in symbols if s.strip()]
    syms = requested[:max_symbols]
    dropped = requested[max_symbols:]
    if not syms:
        raise E.DataError("no symbols supplied.")

    m_dates, m_px, _, _, _, _ = E.fetch_prices(E.MARKET, E.HISTORY_RANGE)
    mkt = dict(zip(m_dates, m_px))

    per_name, meta, failures = {}, {}, []
    for s in syms:
        try:
            d, px, _hi, _lo, vol, info = E.fetch_prices(s, E.HISTORY_RANGE)
        except E.DataError as err:
            failures.append({"symbol": s, "reason": str(err)})
            continue
        common = [(x, p) for x, p in zip(d, px) if x in mkt]
        if len(common) < MIN_BARS_CLASSIFY:
            failures.append({"symbol": s, "reason":
                             f"only {len(common)} bars shared with {E.MARKET}; "
                             f"need {MIN_BARS_CLASSIFY} for a 252d beta."})
            continue
        aligned = np.asarray([p for _, p in common], dtype=float)
        mkt_aligned = np.asarray([mkt[x] for x, _ in common], dtype=float)
        per_name[s] = {"closes": px, "volumes": vol,
                       "name_rets": E.log_returns(aligned),
                       "mkt_rets": E.log_returns(mkt_aligned)}
        meta[s] = {"name": info.get("name", s), "spot": float(px[-1]),
                   "shared": len(common)}

    if not per_name:
        raise E.DataError("none of those tickers had enough history to classify.")

    body = classify_universe(per_name, m_px, rf=rf, market=E.MARKET)
    for r in body["results"]:
        r.update(meta.get(r["symbol"], {}))
    body.update({"failures": failures, "n_requested": len(syms),
                 "n_classified": len(body["results"]), "truncated": dropped,
                 "max_symbols": max_symbols, "rf": rf,
                 "fundamentals_available": False,
                 "gate_order": ["adv", "beta", "regime", "treynor", "idio",
                                "survivable", "own_trend", "dilution", "runway",
                                "revenue"]})
    return body
