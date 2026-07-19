"""The scored checks behind the company report: DCF, 30 pass/fail tests, snowflake.

Pure module -- no network, no pandas. It consumes the plain dict `company.reduce_raw`
produces and returns plain dicts, so it is trivially unit-testable and the web layer
can serialise the result untouched.

Three rules the whole file obeys:

  1. TRI-STATE, ALWAYS. A check is "pass", "fail", or "info". `info` means the data
     to judge it is absent -- NOT that the company failed. Condemning a name for the
     vendor's coverage gap would be worse than admitting we do not know.

  2. NO CHECK INVENTS A NUMBER. Every threshold is a published constant declared at
     the top of its section, and every `display` string states the actual value the
     check saw, so a reader can audit the verdict without reading this file.

  3. THE DCF IS DELIBERATELY SIMPLE AND SAYS SO. Its assumptions ride along in the
     returned dict and are rendered in the UI. A two-stage FCF model with a flat
     discount rate is a sanity check on price, never a price target.
"""

from __future__ import annotations

# ---- DCF assumptions (echoed to the UI; change these and the report says so) ----
DISCOUNT_RATE = 0.09        # flat, not CAPM -- simple and stated, rather than precise
TERMINAL_GROWTH = 0.025     # roughly long-run nominal GDP
STAGE1_YEARS = 5            # analyst-ish growth
STAGE2_YEARS = 5            # linear fade to terminal
GROWTH_CLAMP = (-0.20, 0.30)
DEFAULT_GROWTH = 0.04       # when nothing better exists
FCF_OUTLIER_RATIO = 0.60    # below this share of its own average, a year is a cycle
                            # (a heavy capex build), not the new run rate

# ---- value ----
FAIR_VALUE_MARGIN = 0.80    # "significantly" below fair value
MARKET_PE = 25.0            # broad US market, order-of-magnitude
PEG_MAX = 1.0
PB_MAX = 3.0
ANALYST_UPSIDE = 0.10
MIN_ANALYSTS = 3

# ---- future ----
GROWTH_OK = 0.04            # beats a rough risk-free hurdle
EPS_GROWTH_HIGH = 0.15
REV_GROWTH_HIGH = 0.10

# ---- past ----
ROE_MIN = 0.15
MIN_TREND_YEARS = 3         # years needed before "improving/accelerating" means anything
MAX_SERIES_YEARS = 4        # annual bars charted

# ---- health ----
DEBT_EQUITY_MAX = 0.40
DEBT_COVERAGE_MIN = 0.20    # operating cash flow as a share of total debt
INTEREST_COVER_MIN = 5.0

# ---- dividend ----
YIELD_NOTABLE = 0.015
YIELD_HIGH = 0.04
DIVIDEND_HISTORY_YEARS = 10
DPS_CUT_TOLERANCE = 0.20    # a >20% year-on-year drop counts as a cut
PAYOUT_MAX = 0.90

AXES = (("value", "Value"), ("future", "Future"), ("past", "Past"),
        ("health", "Health"), ("dividend", "Dividend"))
CHECKS_PER_AXIS = 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chk(key: str, title: str, grade: str, display: str, threshold: str) -> dict:
    return {"key": key, "title": title, "grade": grade,
            "display": display, "threshold": threshold}


def _at(series, i: int):
    """Element `i` of a list that may be short or absent, as a float or None."""
    if not series or i >= len(series) or i < 0:
        return None
    v = series[i]
    return None if v is None else float(v)


def _grade(cond, display: str, threshold: str, key: str, title: str,
           unknown: str = "no data") -> dict:
    """pass/fail from a boolean, or info when the condition could not be evaluated."""
    if cond is None:
        return _chk(key, title, "info", unknown, threshold)
    return _chk(key, title, "pass" if cond else "fail", display, threshold)


def _pct(v, dp: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{dp}f}%"


def _x(v, dp: int = 2) -> str:
    return "—" if v is None else f"{v:.{dp}f}×"


def _money(v) -> str:
    if v is None:
        return "—"
    a = abs(v)
    for cut, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= cut:
            return f"{v / cut:,.2f}{suf}"
    return f"{v:,.2f}"


def _ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def _cagr(series) -> float | None:
    """Compound growth from the OLDEST to the NEWEST reading of a newest-first list.
    Requires every reading positive: a CAGR across a sign change is meaningless."""
    vals = [v for v in (series or []) if v is not None]
    if len(vals) < MIN_TREND_YEARS or any(v <= 0 for v in vals):
        return None
    newest, oldest = vals[0], vals[-1]
    years = len(vals) - 1
    return (newest / oldest) ** (1.0 / years) - 1.0


def _score(checks: list[dict]) -> tuple[int, int]:
    """(passes, checks that could be evaluated)."""
    return (sum(1 for c in checks if c["grade"] == "pass"),
            sum(1 for c in checks if c["grade"] != "info"))


def _section(checks: list[dict], **extra) -> dict:
    passes, evaluable = _score(checks)
    return {"score": passes, "n_evaluable": evaluable, "max": CHECKS_PER_AXIS,
            "checks": checks, **extra}


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

def growth_estimate(d: dict) -> tuple[float, str]:
    """(growth, where it came from). Analyst estimates first, history second, a
    conservative constant last -- and always clamped, because a 300% analyst number
    compounded for five years produces a fair value with no contact with reality."""
    for key, label in (("eps_growth_1y", "analyst EPS estimate"),
                       ("rev_growth_1y", "analyst revenue estimate")):
        g = d.get(key)
        if g is not None:
            return max(GROWTH_CLAMP[0], min(GROWTH_CLAMP[1], float(g))), label
    hist = _cagr(d.get("revenue"))
    if hist is not None:
        return (max(GROWTH_CLAMP[0], min(GROWTH_CLAMP[1], hist)),
                "historical revenue CAGR")
    return DEFAULT_GROWTH, "default (no estimate available)"


def fcf_base(series) -> tuple[float | None, str]:
    """(free cash flow to grow from, how it was chosen).

    The latest year, EXCEPT when it is far below its own multi-year average. A
    company part-way through a heavy capex build reports a collapsed free cash flow
    that is a phase, not a run rate -- compounding it for ten years produces a fair
    value that is confidently, uselessly wrong (Amazon in 2025: $7.7bn against a
    $32.9bn prior year). Averaging the cycle is cruder but survives it.
    """
    vals = [v for v in (series or []) if v is not None]
    if not vals:
        return None, ""
    latest = vals[0]
    mean = sum(vals) / len(vals)
    if latest > 0 and (len(vals) < MIN_TREND_YEARS or mean <= 0
                       or latest >= mean * FCF_OUTLIER_RATIO):
        return latest, "latest reported year"
    if mean > 0:
        return mean, f"{len(vals)}-year average (the latest year is well below it)"
    positive = [v for v in vals if v > 0]
    if positive:
        pmean = sum(positive) / len(positive)
        return pmean, f"average of the {len(positive)} profitable years"
    return None, ""


def dcf_fair_value(d: dict) -> dict | None:
    """Two-stage discounted free cash flow, per share. None when not meaningful.

    Returns the assumptions alongside the number so the UI can print them: a fair
    value whose inputs are hidden is indistinguishable from a price target.
    """
    base, basis = fcf_base(d.get("fcf"))
    if base is None or base <= 0:
        return None

    shares = d.get("shares_outstanding")
    if shares is None or shares <= 0:
        return None

    # Statements in one currency and a price in another (ADRs, LSE pence) would give a
    # fair value wrong by the exchange rate. Refuse rather than mislead.
    fin_ccy, px_ccy = d.get("financial_currency"), d.get("currency")
    if fin_ccy and px_ccy and fin_ccy.upper() != px_ccy.upper():
        return None

    g, source = growth_estimate(d)
    r = DISCOUNT_RATE
    pv = 0.0
    fcf = base
    for t in range(1, STAGE1_YEARS + 1):
        fcf *= (1.0 + g)
        pv += fcf / (1.0 + r) ** t
    for j in range(1, STAGE2_YEARS + 1):
        gt = g + (TERMINAL_GROWTH - g) * (j / STAGE2_YEARS)
        fcf *= (1.0 + gt)
        pv += fcf / (1.0 + r) ** (STAGE1_YEARS + j)
    terminal = fcf * (1.0 + TERMINAL_GROWTH) / (r - TERMINAL_GROWTH)
    pv += terminal / (1.0 + r) ** (STAGE1_YEARS + STAGE2_YEARS)

    return {"fair_value": pv / shares, "growth_used": g, "growth_source": source,
            "discount_rate": r, "terminal_growth": TERMINAL_GROWTH,
            "fcf_base": base, "fcf_basis": basis,
            "years": STAGE1_YEARS + STAGE2_YEARS}


# ---------------------------------------------------------------------------
# VALUE
# ---------------------------------------------------------------------------

def value_section(d: dict) -> dict:
    price = d.get("price")
    dcf = dcf_fair_value(d)
    fv = dcf["fair_value"] if dcf else None
    checks = []

    disc = _ratio(price, fv)
    # Spelled out as "X% above/below" rather than a bare signed percentage: a lone
    # "98.0%" beside two prices reads as "trading at 98% of fair value", which is
    # the opposite of what a 98% premium means.
    gap = "" if disc is None else (
        f" — {_pct(disc - 1.0)} above" if disc >= 1.0 else f" — {_pct(1.0 - disc)} below")
    checks.append(_grade(
        None if disc is None else disc < 1.0,
        f"{_money(price)} vs fair value {_money(fv)}{gap}",
        "price below the DCF fair-value estimate",
        "below_fair_value", "Below fair value",
        "no usable free cash flow or share count for a DCF"))
    checks.append(_grade(
        None if disc is None else disc < FAIR_VALUE_MARGIN,
        f"trading at {_pct(disc, 0) if disc else '—'} of fair value",
        f"price more than {(1 - FAIR_VALUE_MARGIN) * 100:.0f}% below fair value",
        "significantly_below_fair_value", "Significantly below fair value",
        "no DCF available"))

    pe, eps = d.get("trailing_pe"), d.get("trailing_eps")
    if pe is not None and pe > 0:
        pe_ok, pe_disp = pe < MARKET_PE, f"P/E {pe:.1f}"
    elif eps is not None and eps < 0:
        pe_ok, pe_disp = False, "loss-making (no meaningful P/E)"
    else:
        pe_ok, pe_disp = None, "no earnings data"
    checks.append(_grade(pe_ok, pe_disp,
                         f"P/E below {MARKET_PE:.0f} (broad US market)",
                         "pe_vs_market", "P/E vs market", pe_disp))

    peg, fpe, eg = d.get("peg"), d.get("forward_pe"), d.get("eps_growth_1y")
    if peg is None and fpe is not None and eg is not None and eg > 0:
        peg = fpe / (eg * 100.0)
    checks.append(_grade(
        None if peg is None or peg <= 0 else peg < PEG_MAX,
        f"PEG {peg:.2f}" if peg else "—",
        f"PEG below {PEG_MAX:.0f} (P/E justified by growth)",
        "peg", "Growth-adjusted P/E",
        "no forecast growth to price against"))

    pb = d.get("price_to_book")
    checks.append(_grade(
        None if pb is None or pb <= 0 else pb < PB_MAX,
        f"P/B {pb:.2f}" if pb else "—",
        f"price-to-book below {PB_MAX:.0f}", "pb", "Price to book"))

    tm, n = d.get("target_mean"), d.get("n_analysts")
    upside = _ratio(tm, price)
    enough = n is not None and n >= MIN_ANALYSTS
    checks.append(_grade(
        None if upside is None or not enough else upside - 1.0 > ANALYST_UPSIDE,
        f"analyst target {_money(tm)} — {_pct(upside - 1.0) if upside else '—'} "
        f"from {_money(price)} ({int(n) if n else 0} analysts)",
        f"mean analyst target more than {ANALYST_UPSIDE * 100:.0f}% above price, "
        f"from at least {MIN_ANALYSTS} analysts",
        "analyst_upside", "Analyst target",
        f"only {int(n) if n else 0} analysts cover it" if upside is not None
        else "no analyst coverage"))

    return _section(checks, dcf=dcf, stats={
        "pe": pe if (pe is not None and pe > 0) else None,
        "forward_pe": fpe, "pb": pb, "peg": peg, "price": price,
        "fair_value": fv, "target_mean": tm, "target_high": d.get("target_high"),
        "target_low": d.get("target_low"), "n_analysts": n})


# ---------------------------------------------------------------------------
# FUTURE
# ---------------------------------------------------------------------------

def future_section(d: dict) -> dict:
    eg, rg = d.get("eps_growth_1y"), d.get("rev_growth_1y")
    now, nxt = d.get("eps_now"), d.get("eps_next")
    checks = [
        _grade(None if eg is None else eg > GROWTH_OK, f"forecast {_pct(eg)}",
               f"forecast earnings growth above {GROWTH_OK * 100:.0f}%",
               "earnings_growth_positive", "Earnings growth",
               "no analyst earnings forecast"),
        _grade(None if eg is None else eg > EPS_GROWTH_HIGH, f"forecast {_pct(eg)}",
               f"forecast earnings growth above {EPS_GROWTH_HIGH * 100:.0f}%",
               "earnings_growth_high", "High earnings growth",
               "no analyst earnings forecast"),
        _grade(None if rg is None else rg > GROWTH_OK, f"forecast {_pct(rg)}",
               f"forecast revenue growth above {GROWTH_OK * 100:.0f}%",
               "revenue_growth_positive", "Revenue growth",
               "no analyst revenue forecast"),
        _grade(None if rg is None else rg > REV_GROWTH_HIGH, f"forecast {_pct(rg)}",
               f"forecast revenue growth above {REV_GROWTH_HIGH * 100:.0f}%",
               "revenue_growth_high", "High revenue growth",
               "no analyst revenue forecast"),
        _grade(None if (now is None or nxt is None) else nxt > now,
               f"EPS {now:.2f} → {nxt:.2f}" if (now is not None and nxt is not None)
               else "—",
               "next year's EPS estimate above this year's",
               "eps_improving", "EPS improving", "no EPS estimates"),
        _grade(None if nxt is None else nxt > 0, f"forecast EPS {nxt:.2f}"
               if nxt is not None else "—",
               "forecast to be profitable next year",
               "becoming_profitable", "Profitable ahead", "no EPS estimate"),
    ]
    return _section(checks, stats={
        "eps_growth_1y": eg, "rev_growth_1y": rg, "eps_now": now, "eps_next": nxt,
        "rev_now": d.get("rev_now"), "rev_next": d.get("rev_next")})


# ---------------------------------------------------------------------------
# PAST
# ---------------------------------------------------------------------------

def past_section(d: dict) -> dict:
    ni, rev = d.get("net_income") or [], d.get("revenue") or []
    eq, ta = d.get("equity") or [], d.get("total_assets") or []
    ebit, cl = d.get("ebit") or [], d.get("current_liabilities") or []
    ni0, ni1 = _at(ni, 0), _at(ni, 1)
    rev0, rev1 = _at(rev, 0), _at(rev, 1)

    checks = [_grade(None if ni0 is None else ni0 > 0,
                     f"net income {_money(ni0)}",
                     "profitable over the last reported year",
                     "profitable", "Profitable", "no income statement")]

    if ni0 is None or ni1 is None:
        checks.append(_grade(None, "—", "earnings higher than the year before",
                             "earnings_grew", "Earnings grew", "need two years"))
    elif ni1 <= 0 < ni0:
        checks.append(_chk("earnings_grew", "Earnings grew", "pass",
                           f"became profitable ({_money(ni1)} → {_money(ni0)})",
                           "earnings higher than the year before"))
    else:
        checks.append(_grade(ni0 > ni1, f"{_money(ni1)} → {_money(ni0)}",
                             "earnings higher than the year before",
                             "earnings_grew", "Earnings grew"))

    yoy = _ratio(ni0, ni1) if (ni0 is not None and ni1 is not None and ni1 > 0) else None
    yoy = None if yoy is None else yoy - 1.0
    cagr = _cagr(ni)
    checks.append(_grade(
        None if (yoy is None or cagr is None) else yoy > cagr,
        f"last year {_pct(yoy)} vs {len(ni) - 1}-year average {_pct(cagr)}",
        "earnings growing faster than their own multi-year average",
        "growth_accelerating", "Growth accelerating",
        "need three profitable years to compare"))

    checks.append(_grade(
        None if (rev0 is None or rev1 is None) else rev0 > rev1,
        f"{_money(rev1)} → {_money(rev0)}", "revenue higher than the year before",
        "revenue_grew", "Revenue grew", "need two years of revenue"))

    roe = _ratio(ni0, _at(eq, 0))
    if roe is None:
        roe = d.get("roe_info")
    checks.append(_grade(None if roe is None else roe > ROE_MIN, f"ROE {_pct(roe)}",
                         f"return on equity above {ROE_MIN * 100:.0f}%",
                         "roe", "Return on equity", "no equity or income data"))

    def roce_at(i):
        cap = None
        a, c = _at(ta, i), _at(cl, i)
        if a is not None and c is not None:
            cap = a - c
        return _ratio(_at(ebit, i), cap)

    roce_now = roce_at(0)
    oldest = min(len(ebit), len(ta), len(cl)) - 1
    roce_old = roce_at(oldest) if oldest >= MIN_TREND_YEARS - 1 else None
    checks.append(_grade(
        None if (roce_now is None or roce_old is None) else roce_now > roce_old,
        f"ROCE {_pct(roce_old)} → {_pct(roce_now)} over {oldest} years",
        "return on capital employed higher than it was three years ago",
        "roce_improving", "Returns improving",
        "need three years of balance sheet and operating income"))

    return _section(checks, series={
        "years": (d.get("fiscal_years") or [])[:MAX_SERIES_YEARS],
        "revenue": rev[:MAX_SERIES_YEARS], "net_income": ni[:MAX_SERIES_YEARS],
    }, stats={"roe": roe, "roce": roce_now, "roa": _ratio(ni0, _at(ta, 0)),
              "net_income": ni0, "revenue": rev0})


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

def health_section(d: dict) -> dict:
    ca, cl = _at(d.get("current_assets"), 0), _at(d.get("current_liabilities"), 0)
    tl = _at(d.get("total_liabilities"), 0)
    debt_series = d.get("total_debt") or []
    eq_series = d.get("equity") or []
    debt, eq = _at(debt_series, 0), _at(eq_series, 0)
    ocf, ebit = _at(d.get("ocf"), 0), _at(d.get("ebit"), 0)
    interest = _at(d.get("interest_expense"), 0)

    # No debt row on a balance sheet that otherwise parsed means debt-free, not unknown.
    debt_free = (not debt_series) and eq is not None and eq > 0
    if debt_free:
        debt = 0.0

    lt_liab = None if (tl is None or cl is None) else tl - cl
    checks = [
        _grade(None if (ca is None or cl is None) else ca > cl,
               f"current assets {_money(ca)} vs {_money(cl)} due within a year",
               "short-term assets cover short-term liabilities",
               "short_term_liabilities", "Short-term liabilities",
               "no current assets/liabilities"),
        _grade(None if (ca is None or lt_liab is None) else ca > lt_liab,
               f"current assets {_money(ca)} vs long-term liabilities {_money(lt_liab)}",
               "short-term assets cover long-term liabilities",
               "long_term_liabilities", "Long-term liabilities",
               "no liabilities breakdown"),
    ]

    de = _ratio(debt, eq) if (eq is not None and eq > 0) else None
    if debt_free:
        checks.append(_chk("debt_level", "Debt level", "pass", "no debt on the books",
                           f"total debt below {DEBT_EQUITY_MAX * 100:.0f}% of equity"))
    else:
        checks.append(_grade(None if de is None else de < DEBT_EQUITY_MAX,
                             f"debt/equity {_pct(de)}",
                             f"total debt below {DEBT_EQUITY_MAX * 100:.0f}% of equity",
                             "debt_level", "Debt level",
                             "no debt or equity figure"))

    oldest = min(len(debt_series), len(eq_series)) - 1
    de_old = None
    if oldest >= MIN_TREND_YEARS - 1:
        e_old = _at(eq_series, oldest)
        de_old = _ratio(_at(debt_series, oldest), e_old) if (e_old and e_old > 0) else None
    if debt_free:
        checks.append(_chk("debt_reduction", "Debt reduced", "pass",
                           "debt-free", "debt/equity lower than three years ago"))
    else:
        checks.append(_grade(
            None if (de is None or de_old is None) else de < de_old,
            f"debt/equity {_pct(de_old)} → {_pct(de)} over {oldest} years",
            "debt/equity lower than three years ago",
            "debt_reduction", "Debt reduced",
            "need three years of debt and equity"))

    cov = _ratio(ocf, debt) if (debt and debt > 0) else None
    if debt_free:
        checks.append(_chk("debt_coverage", "Debt covered by cash flow", "pass",
                           "no debt to cover",
                           f"operating cash flow above "
                           f"{DEBT_COVERAGE_MIN * 100:.0f}% of total debt"))
    else:
        checks.append(_grade(
            None if cov is None else cov > DEBT_COVERAGE_MIN,
            f"operating cash flow {_money(ocf)} = {_pct(cov)} of debt {_money(debt)}",
            f"operating cash flow above {DEBT_COVERAGE_MIN * 100:.0f}% of total debt",
            "debt_coverage", "Debt covered by cash flow",
            "no operating cash flow or debt figure"))

    icov_thr = f"EBIT at least {INTEREST_COVER_MIN:.0f}× interest expense"
    if interest is None or interest == 0:
        icov = None
        # No interest line on a debt-free balance sheet is the good case, not a gap.
        # On a leveraged one it is missing data, and inventing a pass would be a lie.
        checks.append(_chk("interest_coverage", "Interest covered", "pass",
                           "no material interest expense", icov_thr) if debt_free
                      else _chk("interest_coverage", "Interest covered", "info",
                                "no interest expense reported", icov_thr))
    else:
        icov = _ratio(ebit, abs(interest))
        checks.append(_grade(
            None if icov is None else icov > INTEREST_COVER_MIN,
            f"EBIT covers interest {_x(icov, 1)}", icov_thr,
            "interest_coverage", "Interest covered", "no operating income"))

    return _section(checks, stats={
        "debt_to_equity": de, "total_debt": debt, "equity": eq,
        "cash": _at(d.get("cash"), 0), "current_ratio": _ratio(ca, cl),
        "interest_coverage": icov, "debt_free": debt_free})


# ---------------------------------------------------------------------------
# DIVIDEND
# ---------------------------------------------------------------------------

def _effective_yield(d: dict) -> float | None:
    """Yield as a fraction. Derived from the trailing rate and price where possible;
    `dividendYield` is used last because it flipped from fraction (0.005) to percent
    (0.51) form across yfinance versions and cannot be told apart for small yields."""
    rate, price = d.get("trailing_div_rate"), d.get("price")
    if rate is not None and price:
        return rate / price
    dps = d.get("dps_by_year") or {}
    if dps and price:
        return dps[max(dps)] / price
    raw = d.get("dividend_yield_info")
    if raw is None:
        return None
    return raw / 100.0 if raw > 1.0 else raw


def dividend_section(d: dict) -> dict:
    dps = d.get("dps_by_year") or {}
    y = _effective_yield(d)
    pays = bool(dps) or bool(y and y > 0)

    if not pays:
        checks = [_chk(k, t, "fail", "does not pay a dividend", thr) for k, t, thr in (
            ("yield_notable", "Notable yield", f"yield above {YIELD_NOTABLE * 100:.1f}%"),
            ("yield_high", "High yield", f"yield above {YIELD_HIGH * 100:.0f}%"),
            ("stable", "Stable dividend",
             f"paid every year for {DIVIDEND_HISTORY_YEARS} years without a cut"),
            ("growing", "Growing dividend",
             f"dividend per share higher than {DIVIDEND_HISTORY_YEARS} years ago"),
            ("earnings_coverage", "Covered by earnings",
             f"payout ratio below {PAYOUT_MAX * 100:.0f}%"),
            ("cash_flow_coverage", "Covered by cash flow",
             f"dividends below {PAYOUT_MAX * 100:.0f}% of free cash flow"),
        )]
        return _section(checks, pays_dividend=False,
                        series={"years": [], "dps": []},
                        stats={"yield": None, "payout_ratio": None})

    years = sorted(dps)
    vals = [dps[y_] for y_ in years]

    checks = [
        _grade(None if y is None else y > YIELD_NOTABLE, f"yield {_pct(y, 2)}",
               f"yield above {YIELD_NOTABLE * 100:.1f}%",
               "yield_notable", "Notable yield", "no yield figure"),
        _grade(None if y is None else y > YIELD_HIGH, f"yield {_pct(y, 2)}",
               f"yield above {YIELD_HIGH * 100:.0f}%",
               "yield_high", "High yield", "no yield figure"),
    ]

    recent = years[-DIVIDEND_HISTORY_YEARS:]
    if len(recent) < DIVIDEND_HISTORY_YEARS:
        checks.append(_chk("stable", "Stable dividend", "fail",
                           f"only {len(recent)} years of payments",
                           f"paid every year for {DIVIDEND_HISTORY_YEARS} years "
                           f"without a cut"))
    else:
        gaps = [int(recent[i + 1]) - int(recent[i]) != 1 for i in range(len(recent) - 1)]
        cuts = [dps[recent[i + 1]] < dps[recent[i]] * (1 - DPS_CUT_TOLERANCE)
                for i in range(len(recent) - 1)]
        why = ("a year was skipped" if any(gaps)
               else f"cut by more than {DPS_CUT_TOLERANCE * 100:.0f}%" if any(cuts)
               else f"{len(recent)} unbroken years")
        checks.append(_grade(not (any(gaps) or any(cuts)), why,
                             f"paid every year for {DIVIDEND_HISTORY_YEARS} years "
                             f"without a cut", "stable", "Stable dividend"))

    if len(years) >= 5:
        base_year = years[-DIVIDEND_HISTORY_YEARS] if len(years) >= DIVIDEND_HISTORY_YEARS \
            else years[0]
        checks.append(_grade(
            dps[years[-1]] > dps[base_year],
            f"{dps[base_year]:.2f} ({base_year}) → {dps[years[-1]]:.2f} ({years[-1]})",
            f"dividend per share higher than {DIVIDEND_HISTORY_YEARS} years ago",
            "growing", "Growing dividend"))
    else:
        checks.append(_grade(None, "—",
                             f"dividend per share higher than "
                             f"{DIVIDEND_HISTORY_YEARS} years ago",
                             "growing", "Growing dividend",
                             f"only {len(years)} years of history"))

    paid = _at(d.get("dividends_paid"), 0)
    payout = d.get("payout_ratio_info")
    if payout is None:
        payout = _ratio(abs(paid) if paid is not None else None,
                        _at(d.get("net_income"), 0))
        if payout is not None and payout < 0:
            payout = None
    checks.append(_grade(None if payout is None else payout < PAYOUT_MAX,
                         f"payout ratio {_pct(payout)}",
                         f"payout ratio below {PAYOUT_MAX * 100:.0f}% of earnings",
                         "earnings_coverage", "Covered by earnings",
                         "no payout ratio or earnings"))

    fcf = _at(d.get("fcf"), 0)
    if paid is None or fcf is None:
        checks.append(_grade(None, "—",
                             f"dividends below {PAYOUT_MAX * 100:.0f}% of free cash flow",
                             "cash_flow_coverage", "Covered by cash flow",
                             "no dividends-paid or free-cash-flow figure"))
    elif fcf <= 0:
        checks.append(_chk("cash_flow_coverage", "Covered by cash flow", "fail",
                           f"free cash flow is {_money(fcf)}",
                           f"dividends below {PAYOUT_MAX * 100:.0f}% of free cash flow"))
    else:
        cov = abs(paid) / fcf
        checks.append(_grade(cov < PAYOUT_MAX,
                             f"dividends {_money(abs(paid))} = {_pct(cov)} of "
                             f"free cash flow",
                             f"dividends below {PAYOUT_MAX * 100:.0f}% of free cash flow",
                             "cash_flow_coverage", "Covered by cash flow"))

    return _section(checks, pays_dividend=True,
                    series={"years": years, "dps": vals},
                    stats={"yield": y, "payout_ratio": payout,
                           "dps_latest": vals[-1] if vals else None})


# ---------------------------------------------------------------------------
# management + ownership (informational -- no axis, matching Simply Wall St)
# ---------------------------------------------------------------------------

def management_section(d: dict) -> dict:
    officers = d.get("officers") or []
    ceo = next((o for o in officers
                if "ceo" in (o.get("title") or "").lower()
                or "chief executive" in (o.get("title") or "").lower()), None)
    return {"officers": officers, "ceo": ceo, "n_officers": len(officers)}


def ownership_section(d: dict) -> dict:
    buys, sells = d.get("insider_buys_12m"), d.get("insider_sells_12m")
    net = d.get("insider_net_shares_12m")
    return {"insider_pct": d.get("insider_pct"),
            "institution_pct": d.get("institution_pct"),
            "top_institutions": d.get("top_institutions") or [],
            "insider_buys_12m": buys, "insider_sells_12m": sells,
            "insider_net_shares_12m": net,
            "insider_bias": (None if buys is None or sells is None
                             else "buying" if net and net > 0
                             else "selling" if net and net < 0 else "flat")}


# ---------------------------------------------------------------------------
# snowflake
# ---------------------------------------------------------------------------

_AXIS_PHRASE = {
    "value": ("looks cheap on the checks here", "looks expensive on the checks here"),
    "future": ("has strong forecast growth", "has weak forecast growth"),
    "past": ("has a strong earnings record", "has a weak earnings record"),
    "health": ("has a strong balance sheet", "has a strained balance sheet"),
    "dividend": ("is an established dividend payer", "pays little or nothing"),
}


def snowflake(sections: dict) -> dict:
    axes = [{"key": k, "label": label,
             "score": int(sections.get(k, {}).get("score", 0)),
             "max": CHECKS_PER_AXIS}
            for k, label in AXES]
    total = sum(a["score"] for a in axes)
    best = max(axes, key=lambda a: a["score"])
    worst = min(axes, key=lambda a: a["score"])
    if best["score"] <= 1:
        summary = "Fails or cannot evaluate almost every check — treat with caution."
    elif worst["score"] >= 4:
        summary = "Scores well across every axis."
    else:
        summary = (_AXIS_PHRASE[best["key"]][0].capitalize() + " ("
                   f"{best['score']}/{CHECKS_PER_AXIS}), but "
                   f"{_AXIS_PHRASE[worst['key']][1]} ({worst['score']}/"
                   f"{CHECKS_PER_AXIS}).")
    return {"axes": axes, "total": total, "max": CHECKS_PER_AXIS * len(AXES),
            "summary": summary}
