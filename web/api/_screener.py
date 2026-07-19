"""High-beta screening criteria for the web API (numpy + stdlib only).

Vendored from `src/forecast/screener.py`; `tests/test_web_screener.py` pins the two
together so the web copy cannot drift from the CLI's numbers.

Every criterion measures RISK, LIQUIDITY, VERIFIABILITY or REDUNDANCY. **None predicts
returns.** The composite uses caller-supplied weights because "good" is a value
judgement, not something the data can settle.
"""

from __future__ import annotations

import json
import os

import numpy as np

import _engine as E

WINDOW = 126
MIN_VERIFY_BARS = 700
PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"

T_BETA = 2.0
T_R2 = 0.10
T_HALVED_MILD, T_HALVED_SEVERE = 0.05, 0.20
T_CORR_DIVERSIFIER, T_CORR_DUPLICATE = 0.30, 0.45
T_DOLLAR_VOL_OK, T_DOLLAR_VOL_THIN = 50e6, 5e6
T_VOL_PCTL_ELEVATED, T_VOL_PCTL_STORM = 0.70, 0.85

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERDICT_PATHS = (
    os.path.join(_HERE, "beta_validation_status.json"),
    os.path.join(_HERE, "..", "..", "data", "beta_validation_status.json"),
)

BANNER = ("These criteria measure verifiability, survivability, redundancy, liquidity "
          "and payoff shape. NONE of them predicts returns — direction failed every "
          "predictive test in this project. A good score means analysable and "
          "survivable, not 'going up'.")


def _crit(key, title, value, display, grade, threshold, note=""):
    return {"key": key, "title": title, "value": value, "display": display,
            "grade": grade, "threshold": threshold, "note": note}


def load_verdicts() -> dict:
    for p in _VERDICT_PATHS:
        try:
            return json.loads(open(p, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            continue
    return {}


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

def c_beta(name_rets, mkt_rets):
    b = E.beta_stats(np.asarray(name_rets), np.asarray(mkt_rets))
    if b["beta"] < T_BETA:
        grade, note = FAIL, "below the high-beta cut"
    elif b["r2"] < T_R2:
        grade, note = WARN, "idiosyncratic lottery - the beta estimate is unreliable"
    else:
        grade, note = PASS, "market amplifier"
    return _crit("beta", "True high-beta?", float(b["beta"]),
                 f"beta {b['beta']:.2f} +/-{b['se']:.2f} | r2 {b['r2']:.2f}", grade,
                 f"pass: beta >= {T_BETA} and r2 >= {T_R2}", note), b


def c_verifiable(n_bars, verdict):
    if n_bars < MIN_VERIFY_BARS:
        return _crit("verifiable", "Verifiable?", float(n_bars),
                     f"{n_bars} shared bars", FAIL,
                     f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent",
                     "too short to calibration-test - any range is extrapolation")
    if not verdict:
        return _crit("verifiable", "Verifiable?", float(n_bars),
                     f"{n_bars} bars | not yet tested", WARN,
                     f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent",
                     "no stored validation for this ticker")
    v = str(verdict.get("verdict", ""))
    ok = not v.startswith("RULED OUT")
    return _crit("verifiable", "Verifiable?", float(n_bars),
                 f"{n_bars} bars | cov90 {verdict.get('cov90', float('nan')):.2f}",
                 PASS if ok else FAIL,
                 f"pass: >= {MIN_VERIFY_BARS} bars and calibration consistent", v)


def c_survivable(closes, window=WINDOW):
    px = np.asarray(closes, dtype=float)
    if px.size <= window + 1:
        return _crit("survivable", "Survivable?", None, "insufficient history", WARN,
                     f"pass < {T_HALVED_MILD:.0%} of {window}d windows halved"), {}
    fwd = px[window:] / px[:-window]
    halved = float(np.mean(fwd <= 0.5))
    worst = float(fwd.min() - 1.0)
    max_dd = float(np.max(1.0 - px / np.maximum.accumulate(px)))
    grade = PASS if halved < T_HALVED_MILD else WARN if halved < T_HALVED_SEVERE else FAIL
    note = ("halving is routine for this name" if grade == FAIL else
            "occasional halvings" if grade == WARN else "halvings are rare")
    return (_crit("survivable", "Survivable?", halved,
                  f"{halved:.0%} of {window}d windows halved | worst {worst:+.0%} | "
                  f"max DD {-max_dd:.0%}", grade,
                  f"pass < {T_HALVED_MILD:.0%}, warn < {T_HALVED_SEVERE:.0%} halved", note),
            {"halved": halved, "worst_window": worst, "max_dd": max_dd,
             "doubled": float(np.mean(fwd >= 2.0)), "best_window": float(fwd.max() - 1.0)})


def c_redundancy(avg_corr, n_peers):
    if avg_corr is None or n_peers < 1:
        return _crit("redundancy", "Adds a new bet?", None, "no peers to compare", INFO,
                     f"diversifier < {T_CORR_DIVERSIFIER}, duplicate > {T_CORR_DUPLICATE}")
    grade = (PASS if avg_corr < T_CORR_DIVERSIFIER else
             WARN if avg_corr < T_CORR_DUPLICATE else FAIL)
    note = ("largely a duplicate of the rest of this list" if grade == FAIL else
            "partly overlapping" if grade == WARN else "genuinely different bet")
    return _crit("redundancy", "Adds a new bet?", float(avg_corr),
                 f"avg corr {avg_corr:.2f} vs {n_peers} peers", grade,
                 f"pass < {T_CORR_DIVERSIFIER}, fail > {T_CORR_DUPLICATE} "
                 f"(relative to the tickers you submitted)", note)


def c_liquidity(closes, volumes):
    px = np.asarray(closes, dtype=float)
    vol = np.asarray(volumes, dtype=float) if volumes is not None else None
    if vol is None or vol.size == 0 or not np.any(vol > 0):
        return _crit("liquidity", "Liquid enough?", None, "no volume data", INFO,
                     f"pass > ${T_DOLLAR_VOL_OK / 1e6:.0f}M/day, "
                     f"fail < ${T_DOLLAR_VOL_THIN / 1e6:.0f}M/day"), {}
    n = min(px.size, vol.size)
    dollar = float(np.median(px[-n:][-60:] * vol[-n:][-60:]))
    grade = PASS if dollar > T_DOLLAR_VOL_OK else WARN if dollar > T_DOLLAR_VOL_THIN else FAIL
    note = ("thin - expect slippage and gaps" if grade == FAIL else
            "moderate depth" if grade == WARN else "deep")
    return (_crit("liquidity", "Liquid enough?", dollar,
                  f"${dollar / 1e6:,.0f}M median daily $ volume", grade,
                  f"pass > ${T_DOLLAR_VOL_OK / 1e6:.0f}M, "
                  f"fail < ${T_DOLLAR_VOL_THIN / 1e6:.0f}M", note),
            {"dollar_volume": dollar})


def c_vol_regime(closes):
    r = E.log_returns(np.asarray(closes, dtype=float))
    if r.size < 90:
        return _crit("vol_regime", "Vol regime now", None, "insufficient history", INFO,
                     f"pass < {T_VOL_PCTL_ELEVATED:.0%} percentile"), {}
    path, cur = E.ewma_vol_path(r)
    pctl = float(np.mean(path <= cur))
    grade = (PASS if pctl < T_VOL_PCTL_ELEVATED else
             WARN if pctl < T_VOL_PCTL_STORM else FAIL)
    note = ("entering during a storm - unusually wide outcomes" if grade == FAIL else
            "somewhat elevated" if grade == WARN else "normal for this name")
    return (_crit("vol_regime", "Vol regime now", pctl,
                  f"{cur * 100:.1f}%/day | {pctl:.0%} percentile of its own history",
                  grade, f"pass < {T_VOL_PCTL_ELEVATED:.0%}, "
                         f"fail > {T_VOL_PCTL_STORM:.0%} percentile", note),
            {"vol_now": cur, "vol_pctl": pctl})


def c_payoff(stats, window=WINDOW):
    if not stats or "doubled" not in stats:
        return _crit("payoff", "Payoff shape", None, "insufficient history", INFO,
                     "descriptive only - not predictive")
    return _crit("payoff", "Payoff shape", float(stats["doubled"]),
                 f"{stats['doubled']:.0%} of {window}d windows doubled | "
                 f"{stats['halved']:.0%} halved | best {stats['best_window']:+.0%}",
                 INFO, "descriptive only - not predictive",
                 "historical shape; says nothing about which tail comes next")


def cohort_redundancy(returns_by_symbol: dict) -> dict:
    syms = [s for s, r in returns_by_symbol.items() if r is not None and r.size > 30]
    if len(syms) < 2:
        return {}
    n = min(returns_by_symbol[s].size for s in syms)
    M = np.column_stack([returns_by_symbol[s][-n:] for s in syms])
    C = np.corrcoef(M.T)
    np.fill_diagonal(C, np.nan)
    return {s: float(np.nanmean(C[i])) for i, s in enumerate(syms)}


_GRADE_POINTS = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


def composite(criteria: list, weights: dict | None = None):
    graded = [c for c in criteria if c["grade"] in _GRADE_POINTS]
    if not graded:
        return None
    w = weights or {}
    num = sum(w.get(c["key"], 1.0) * _GRADE_POINTS[c["grade"]] for c in graded)
    den = sum(w.get(c["key"], 1.0) for c in graded)
    return float(num / den) if den > 0 else None


def build_scorecard(symbol, closes, volumes, mkt_rets, n_shared_bars,
                    avg_corr=None, n_peers=0, verdict=None) -> dict:
    try:
        name_rets = E.log_returns(np.asarray(closes, dtype=float))
        m = np.asarray(mkt_rets, dtype=float)
        k = min(name_rets.size, m.size)
        if k < 40:
            return {"symbol": symbol, "error": "not enough overlapping history",
                    "criteria": [], "warnings": []}
        # Same tail-alignment assumption as src/forecast/screener.py -- see the note
        # there. Kept byte-identical in behaviour so the parity tests stay meaningful.
        cb, _ = c_beta(name_rets[-k:], m[-k:])
        cs, sstats = c_survivable(closes)
        cl, _ = c_liquidity(closes, volumes)
        cv, _ = c_vol_regime(closes)
        crit = [cb, c_verifiable(n_shared_bars, verdict), cs,
                c_redundancy(avg_corr, n_peers), cl, cv, c_payoff(sstats)]
    except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the screen
        return {"symbol": symbol, "error": f"could not score: {type(e).__name__}",
                "criteria": [], "warnings": []}
    return {"symbol": symbol, "error": "", "criteria": crit,
            "warnings": [c["note"] for c in crit if c["grade"] == FAIL and c["note"]]}


def screen(symbols: list[str], max_symbols: int = 20) -> dict:
    """Fetch, score and rank a cohort. Per-symbol failures are isolated.

    Over-long lists are truncated, but the drop is REPORTED (`truncated`) rather than
    silent -- a screen that quietly ignores half your tickers is worse than an error."""
    requested = [s.strip().upper() for s in symbols if s.strip()]
    syms = requested[:max_symbols]
    dropped = requested[max_symbols:]
    if not syms:
        raise E.DataError("no symbols supplied.")
    m_dates, m_px, _, _, _, _ = E.fetch_prices(E.MARKET, E.HISTORY_RANGE)
    mkt = dict(zip(m_dates, m_px))

    data, failures = {}, []
    for s in syms:
        try:
            d, px, _hi, _lo, vol, info = E.fetch_prices(s, E.HISTORY_RANGE)
        except E.DataError as err:
            failures.append({"symbol": s, "reason": str(err)})
            continue
        common = [(x, p) for x, p in zip(d, px) if x in mkt]
        if len(common) < 60:
            failures.append({"symbol": s, "reason": "insufficient overlap with SPY"})
            continue
        data[s] = {"px": px, "vol": vol, "info": info,
                   "shared": len(common),
                   "aligned": np.asarray([p for _, p in common], dtype=float),
                   "mkt_aligned": np.asarray([mkt[x] for x, _ in common], dtype=float)}

    rets = {s: E.log_returns(v["aligned"]) for s, v in data.items()}
    corr = cohort_redundancy(rets)
    verdicts = load_verdicts()

    cards = []
    for s, v in data.items():
        card = build_scorecard(
            s, v["px"], v["vol"], E.log_returns(v["mkt_aligned"]), v["shared"],
            avg_corr=corr.get(s), n_peers=max(0, len(corr) - 1),
            verdict=verdicts.get(s))
        card["name"] = v["info"].get("name", s)
        card["spot"] = float(v["px"][-1])
        card["composite_equal"] = composite(card["criteria"])
        cards.append(card)

    return {"banner": BANNER, "cards": cards, "failures": failures,
            "n_requested": len(syms), "n_scored": len(cards),
            "truncated": dropped, "max_symbols": max_symbols,
            "criteria_order": ["beta", "verifiable", "survivable", "redundancy",
                               "liquidity", "vol_regime", "payoff"]}
