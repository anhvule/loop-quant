"""One analysis pass: risk scorecard + investable/reckless verdict + optional forecast.

This exists because the three views were really one question asked three ways, and
running them separately made the site fetch every ticker's ten-year history from Yahoo
once per view. Yahoo rate-limits, so a twenty-name list was three times as likely to come
back empty-handed for no analytical gain.

The fix is to fetch once and feed both engines from the same arrays. That is possible
only because the scoring functions are pure: `_screener.build_scorecard` and
`_classify.classify_universe` take numbers, not tickers, so this module owns all the I/O
and neither of them repeats it.

The two engines answer deliberately different questions and are NOT merged into a single
score:

    scorecard  -- is this analysable, survivable, tradable, non-duplicative?
                  Weighted by the user; explicitly refuses to rank on returns.
    verdict    -- of the genuinely liquid, genuinely high-beta names, which were paid
                  for that beta without their own story drowning out the market?

A name can be a fine scorecard and a poor verdict, and that disagreement is information.
Collapsing them into one number would destroy it.
"""

from __future__ import annotations

import time

import numpy as np

import _classify as CL
import _engine as E
import _screener as SC

MIN_SCREEN_BARS = 60          # below this even the scorecard is guesswork
INSUFFICIENT = "insufficient_history"

# Yahoo throttles bursts, and each miss costs up to 30s (two hosts x 15s timeout). A
# 20-name cohort could therefore stall for minutes with the caller none the wiser.
# Once the budget is spent, remaining symbols are REPORTED as skipped rather than
# fetched -- partial results now beat complete results after an unbounded wait.
FETCH_BUDGET_S = 75.0

# The classifier's order, with a slot for names too young for a 252d beta. They are not
# errors and not judgements -- they simply have not existed long enough to be gated.
VERDICT_ORDER = ([v for v in CL.VERDICT_ORDER if v != CL.ERROR]
                 + [INSUFFICIENT, CL.ERROR])
VERDICT_TEXT = dict(CL.VERDICT_TEXT)
VERDICT_TEXT[INSUFFICIENT] = "too little shared history to gate"

BANNER = (
    "NOT INVESTMENT ADVICE. Two independent readings, deliberately not merged into one "
    "number. The SCORECARD measures risk, liquidity and verifiability and predicts "
    "nothing. The VERDICT additionally asks whether a name was historically paid for its "
    "beta -- a property of the past year, not a forecast. Where they disagree, the "
    "disagreement is the finding."
)


def _fetch_cohort(syms: list) -> tuple[dict, list, np.ndarray]:
    """Fetch the market and every name ONCE. Returns (data, failures, market closes).

    Bounded by FETCH_BUDGET_S: a cohort that outruns the budget stops fetching and the
    untouched symbols come back as failures, so the response time stays predictable
    even when the upstream is crawling.
    """
    started = time.monotonic()
    m_dates, m_px, _, _, _, _ = E.fetch_prices(E.MARKET, E.HISTORY_RANGE)
    mkt = dict(zip(m_dates, m_px))

    data, failures = {}, []
    for i, s in enumerate(syms):
        spent = time.monotonic() - started
        if spent > FETCH_BUDGET_S and data:
            for rest in syms[i:]:
                failures.append({"symbol": rest, "reason":
                                 f"skipped after {spent:.0f}s — the price source was "
                                 f"too slow to finish this cohort. Try fewer tickers."})
            break
        try:
            d, px, _hi, _lo, vol, info = E.fetch_prices(s, E.HISTORY_RANGE)
        except E.DataError as err:
            failures.append({"symbol": s, "reason": str(err)})
            continue
        common = [(x, p) for x, p in zip(d, px) if x in mkt]
        if len(common) < MIN_SCREEN_BARS:
            failures.append({"symbol": s, "reason":
                             f"only {len(common)} bars shared with {E.MARKET}; "
                             f"need {MIN_SCREEN_BARS} to score anything."})
            continue
        aligned = np.asarray([p for _, p in common], dtype=float)
        mkt_aligned = np.asarray([mkt[x] for x, _ in common], dtype=float)
        data[s] = {"px": px, "vol": vol, "info": info, "shared": len(common),
                   "name_rets": E.log_returns(aligned),
                   "mkt_rets": E.log_returns(mkt_aligned)}
    return data, failures, m_px


def analyse(symbols: list, max_symbols: int = 20, rf: float = CL.RF_DEFAULT,
            months: int = 6) -> dict:
    """Score, gate and (for a lone ticker) forecast a submitted cohort.

    Over-long lists are truncated, but the drop is REPORTED (`truncated`) rather than
    silent -- an analysis that quietly ignores half your tickers is worse than an error.
    """
    requested = [s.strip().upper() for s in symbols if s.strip()]
    syms = list(dict.fromkeys(requested))[:max_symbols]
    dropped = list(dict.fromkeys(requested))[max_symbols:]
    if not syms:
        raise E.DataError("enter at least one ticker.")

    data, failures, m_px = _fetch_cohort(syms)
    if not data:
        raise E.DataError("none of those tickers had usable history.")

    # --- scorecard: every name that cleared the minimum bar count ---
    corr = SC.cohort_redundancy({s: v["name_rets"] for s, v in data.items()})
    verdicts = SC.load_verdicts()
    cards = {}
    for s, v in data.items():
        card = SC.build_scorecard(s, v["px"], v["vol"], v["mkt_rets"], v["shared"],
                                  avg_corr=corr.get(s), n_peers=max(0, len(corr) - 1),
                                  verdict=verdicts.get(s))
        card["composite_equal"] = SC.composite(card["criteria"])
        cards[s] = card

    # --- verdict: only names with enough shared history for a 252d beta ---
    per_name = {s: {"closes": v["px"], "volumes": v["vol"],
                    "name_rets": v["name_rets"], "mkt_rets": v["mkt_rets"]}
                for s, v in data.items() if v["shared"] >= CL.MIN_BARS_CLASSIFY}
    cl = CL.classify_universe(per_name, m_px, rf=rf)     # empty input is handled
    gated = {r["symbol"]: r for r in cl["results"]}

    rows = []
    for s, v in data.items():
        row = dict(gated.get(s) or {
            "symbol": s, "verdict": INSUFFICIENT, "gates": [], "metrics": {},
            "reasons": [f"only {v['shared']} bars shared with {E.MARKET}; "
                        f"{CL.MIN_BARS_CLASSIFY} are needed for a 252d beta"]})
        card = cards[s]
        row.update({"name": v["info"].get("name", s), "spot": float(v["px"][-1]),
                    "shared": v["shared"], "criteria": card["criteria"],
                    "composite_equal": card["composite_equal"],
                    "scorecard_error": card["error"]})
        rows.append(row)

    rank = {v: i for i, v in enumerate(VERDICT_ORDER)}
    rows.sort(key=lambda r: (rank.get(r["verdict"], len(VERDICT_ORDER)),
                             -(r["composite_equal"] or 0.0), r["symbol"]))

    # --- forecast: only for a lone ticker, where a cone is readable and worth the wait ---
    forecast = None
    if len(data) == 1:
        only = next(iter(data))
        try:
            forecast = E.predict(only, months=months)
        except Exception as err:  # noqa: BLE001 -- the cohort view must still render
            failures.append({"symbol": only,
                             "reason": f"forecast unavailable ({type(err).__name__})"})

    return {
        "banner": BANNER,
        "regime": cl["regime"], "risk_on": cl["risk_on"],
        "quartile_mode": cl["quartile_mode"], "n_eligible": cl["n_eligible"],
        "fundamentals_available": False,
        "verdict_order": VERDICT_ORDER, "verdict_text": VERDICT_TEXT,
        "rows": rows, "forecast": forecast,
        "failures": failures, "truncated": dropped,
        "n_requested": len(syms), "n_analysed": len(rows),
        "max_symbols": max_symbols, "rf": rf, "months": months,
        "criteria_order": ["beta", "verifiable", "survivable", "redundancy",
                           "liquidity", "vol_regime", "payoff"],
        "gate_order": ["adv", "beta", "regime", "treynor", "idio", "survivable",
                       "own_trend", "dilution", "runway", "revenue"],
    }
