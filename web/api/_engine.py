"""Self-contained forecasting engine for the web API (numpy + stdlib only).

This is a deliberately dependency-light port of the validated pipeline in
`src/forecast/` (beta.py, bounds.py, drawdown.py). It keeps every honesty rail:

  * drift is ANCHORED to beta x market drift -- a name's own momentum is refused
  * returns are VOLATILITY-STANDARDIZED and rescaled to today's vol, with a decaying
    term structure so a current panic-vol regime is not projected forever
  * blocks are sampled JOINTLY with the market (correlation and co-crashes survive)
    and weighted toward recent regimes
  * every percentile is REALITY-CHECKED against the name's own realized moves, with
    an asymmetric rule: unprecedented upside is trimmed, a gloomier-than-history
    downside is kept
  * calibration status is reported, never assumed

No pandas / yfinance / statsmodels: prices come from Yahoo's public chart JSON via
urllib so the function stays inside Vercel's size and cold-start budget.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import _waves

MARKET = "SPY"
# Two hosts: Yahoo intermittently throttles one while the other answers. A genuine 404
# (unknown ticker) short-circuits; transport errors fall through to the next host.
CHART_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
CHART_URL = "https://{host}/v8/finance/chart/{sym}?range={rng}&interval=1d"
UA = {"User-Agent": "Mozilla/5.0 (compatible; loop-quant-outlook/1.0)"}
HISTORY_RANGE = "10y"    # match the CLI's window so web and terminal agree

VOL_LAMBDA = 0.94
BETA_LAMBDA = 0.995
RECENCY_HALF_LIFE = 500.0
DEFAULT_BLOCK = 10
MIN_BARS = 120
MIN_VALIDATE_BARS = 700
MIN_WINDOWS_FOR_CHECK = 150
REALITY_TOL = 1.5
CHECK_QS = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
DRAWDOWN_LEVELS = (-0.10, -0.20, -0.30, -0.50)

# NYSE full-day closures 2026-2027 (used only to map calendar dates -> trading days).
HOLIDAYS = frozenset({
    dt.date(2026, 1, 1), dt.date(2026, 1, 19), dt.date(2026, 2, 16), dt.date(2026, 4, 3),
    dt.date(2026, 5, 25), dt.date(2026, 6, 19), dt.date(2026, 7, 3), dt.date(2026, 9, 7),
    dt.date(2026, 11, 26), dt.date(2026, 12, 25),
    dt.date(2027, 1, 1), dt.date(2027, 1, 18), dt.date(2027, 2, 15), dt.date(2027, 3, 26),
    dt.date(2027, 5, 31), dt.date(2027, 6, 18), dt.date(2027, 7, 5), dt.date(2027, 9, 6),
    dt.date(2027, 11, 25), dt.date(2027, 12, 24),
})


class DataError(RuntimeError):
    """Ticker unusable: not found, empty, or too little history to model."""


# ---------------------------------------------------------------------------
# price data
# ---------------------------------------------------------------------------

def _fetch_chart(symbol: str, rng: str) -> dict:
    """GET the chart JSON, trying each Yahoo host before giving up."""
    quoted = urllib.parse.quote(symbol.upper(), safe="")
    last = None
    for host in CHART_HOSTS:
        url = CHART_URL.format(host=host, sym=quoted, rng=rng)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:            # ticker really does not exist -- do not retry
                raise DataError(f"'{symbol}' not found on Yahoo Finance (HTTP 404).") from None
            last = f"HTTP {e.code}"
        except Exception as e:           # noqa: BLE001 -- timeouts, DNS, JSON, etc.
            last = type(e).__name__
    raise DataError(
        f"could not reach the price data source for '{symbol}' ({last}). "
        f"It may be rate-limiting requests -- wait a moment and try again."
    )


def fetch_prices(symbol: str, rng: str = HISTORY_RANGE):
    """(dates, closes, highs, lows, volumes, info). Highs/lows feed the ATR-scaled swing
    filter (falling back to closes when Yahoo omits them); volumes feed the screener's
    liquidity criterion."""
    payload = _fetch_chart(symbol, rng)
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise DataError(f"'{symbol}' not recognised by Yahoo Finance.")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise DataError(f"no data returned for '{symbol}'.")

    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    hi_raw = quote.get("high") or []
    lo_raw = quote.get("low") or []

    vol_raw = quote.get("volume") or []
    rows = []
    for i, (t, c) in enumerate(zip(ts, closes)):
        if not c or c <= 0:
            continue
        h = hi_raw[i] if i < len(hi_raw) and hi_raw[i] else c
        l = lo_raw[i] if i < len(lo_raw) and lo_raw[i] else c
        v = vol_raw[i] if i < len(vol_raw) and vol_raw[i] else 0.0
        rows.append((t, float(c), float(max(h, c)), float(min(l, c)), float(v)))
    if len(rows) < MIN_BARS:
        raise DataError(
            f"'{symbol}' has only {len(rows)} usable daily bars (need >= {MIN_BARS}). "
            f"Too little history to model responsibly."
        )
    dates = [dt.datetime.fromtimestamp(r[0], dt.timezone.utc).date() for r in rows]
    px = np.asarray([r[1] for r in rows], dtype=float)
    highs = np.asarray([r[2] for r in rows], dtype=float)
    lows = np.asarray([r[3] for r in rows], dtype=float)
    volumes = np.asarray([r[4] for r in rows], dtype=float)
    meta = result.get("meta") or {}
    info = {
        "name": meta.get("longName") or meta.get("shortName") or symbol.upper(),
        "currency": meta.get("currency") or "",
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName") or "",
    }
    return dates, px, highs, lows, volumes, info


def align(d1, p1, d2, p2):
    """Restrict two date/price series to their shared dates."""
    m2 = dict(zip(d2, p2))
    common = [(d, v, m2[d]) for d, v in zip(d1, p1) if d in m2]
    if len(common) < 60:
        raise DataError("insufficient overlap with the market series.")
    return ([c[0] for c in common],
            np.asarray([c[1] for c in common]),
            np.asarray([c[2] for c in common]))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def log_returns(px: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=float)
    return np.diff(np.log(px[px > 0]))


def ewma_vol_path(r: np.ndarray, lam: float = VOL_LAMBDA, seed_window: int = 20):
    n = r.size
    if n == 0:
        return np.empty(0), 0.0
    vol = np.empty(n)
    var = float(np.var(r[:min(seed_window, n)])) or float(np.var(r)) or 1e-8
    for i in range(n):
        vol[i] = math.sqrt(max(var, 1e-12))
        var = lam * var + (1.0 - lam) * float(r[i]) ** 2
    return vol, math.sqrt(max(var, 1e-12))


def vol_half_life(vol: np.ndarray, lag: int = 21) -> float:
    if vol.size < 60:
        return 400.0
    lv = np.log(np.maximum(vol[30:], 1e-12))
    if lv.size <= lag + 10:
        return 400.0
    a = float(np.corrcoef(lv[:-lag], lv[lag:])[0, 1])
    if not (0.0 < a < 1.0):
        return 400.0
    return float(np.clip(-lag * math.log(2.0) / math.log(a), 5.0, 400.0))


def blended_drift(px: np.ndarray, recent: int = 500, w_recent: float = 0.30):
    r = log_returns(px)
    if r.size == 0:
        return 0.0, 0.0, 0.0
    mu_long = float(r.mean())
    mu_recent = float(r[-recent:].mean()) if r.size > recent else mu_long
    return w_recent * mu_recent + (1 - w_recent) * mu_long, mu_recent, mu_long


def beta_stats(name_r: np.ndarray, mkt_r: np.ndarray, lam: float = BETA_LAMBDA) -> dict:
    n = name_r.size
    w = lam ** np.arange(n - 1, -1, -1)
    w = w / w.sum()
    mx, my = float(w @ mkt_r), float(w @ name_r)
    var_m = float(w @ (mkt_r - mx) ** 2)
    if var_m <= 0:
        raise DataError("market variance is zero over the shared window.")
    beta = float(w @ ((mkt_r - mx) * (name_r - my))) / var_m
    resid = name_r - ((my - beta * mx) + beta * mkt_r)
    rv = float(w @ (resid - float(w @ resid)) ** 2)
    var_y = float(w @ (name_r - my) ** 2)
    n_eff = 1.0 / float(np.sum(w ** 2))
    idio = math.sqrt(max(rv, 0.0))
    return {
        "beta": beta,
        "se": idio / (math.sqrt(n_eff) * math.sqrt(var_m)),
        "r2": 0.0 if var_y <= 0 else max(0.0, 1.0 - rv / var_y),
        "idio_vol": idio,
        "vol": float(np.std(name_r)),
        "raw_mu": float(np.mean(name_r)),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def simulate(mkt_r, name_r, s0_mkt, s0_name, horizon, target_mu_mkt, target_mu_name,
             n_paths=8000, seed=7, block=DEFAULT_BLOCK, beta_se=0.0):
    """Joint, vol-standardized, term-structured, recency-weighted block bootstrap.

    Returns (market_prices, name_prices), each (n_paths, horizon)."""
    R = np.column_stack([mkt_r, name_r])
    T = R.shape[0]
    if T < block + 1:
        raise DataError("history too short to bootstrap.")

    Z = np.empty_like(R)
    cur = np.empty(2)
    longrun = np.empty(2)
    hl = np.empty(2)
    for k in range(2):
        vol, cur[k] = ewma_vol_path(R[:, k])
        Z[:, k] = R[:, k] / np.maximum(vol, 1e-12)
        longrun[k] = float(np.median(vol))
        hl[k] = vol_half_life(vol)

    t = np.arange(1, horizon + 1)[:, None]
    sched = longrun[None, :] + (cur[None, :] - longrun[None, :]) * np.power(0.5, t / hl[None, :])

    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    n_starts = T - block + 1
    ages = np.arange(n_starts - 1, -1, -1, dtype=float)
    w = np.power(0.5, ages / RECENCY_HALF_LIFE)
    w = w + 0.02 * w.max()
    w /= w.sum()
    starts = rng.choice(n_starts, size=(n_paths, n_blocks), p=w)
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]
           ).reshape(n_paths, n_blocks * block)[:, :horizon]

    sampled = Z[idx] * sched[None, :, :]
    base_mu = Z.mean(axis=0) * sched.mean(axis=0)
    target = np.asarray([target_mu_mkt, target_mu_name])
    sampled = sampled + (target - base_mu)[None, None, :]

    if beta_se > 0:                       # propagate beta estimation uncertainty
        extra = rng.normal(0.0, beta_se, n_paths) * target_mu_mkt
        sampled[:, :, 1] += extra[:, None]

    paths = np.exp(np.cumsum(sampled, axis=1))
    return s0_mkt * paths[:, :, 0], s0_name * paths[:, :, 1]


# ---------------------------------------------------------------------------
# reality check (asymmetric, monotonic)
# ---------------------------------------------------------------------------

def realized_multiples(px: np.ndarray, horizon: int):
    if px.size <= horizon + 1:
        return None
    return px[horizon:] / px[:-horizon]


def reality_check(terminal, s0, px, horizon, tol=REALITY_TOL):
    mult = realized_multiples(px, horizon)
    thin = mult is None or mult.size < MIN_WINDOWS_FOR_CHECK
    rows = []
    for q in CHECK_QS:
        m = float(np.percentile(terminal, q))
        if thin:
            rows.append({"q": q, "model": m, "realized": None, "ratio": None,
                         "flag": "thin precedent - unchecked", "anchored": m})
            continue
        rp = float(s0 * np.percentile(mult, q))
        ratio = m / rp if rp > 0 else float("inf")
        flag, anchored = "", m
        if q >= 50.0:
            if ratio > tol:
                flag, anchored = f"model {ratio:.1f}x above precedent - trimmed", rp
        else:
            if ratio > tol:
                flag, anchored = f"model {ratio:.1f}x milder than precedent - tightened", rp
            elif ratio < 1.0 / tol:
                flag = f"model {ratio:.1f}x more severe than precedent - kept"
        rows.append({"q": q, "model": m, "realized": rp, "ratio": ratio,
                     "flag": flag, "anchored": anchored})

    running = -math.inf                    # percentiles must stay ordered
    for row in rows:
        if row["anchored"] < running:
            row["anchored"] = running
            note = "raised to keep percentiles ordered"
            row["flag"] = f'{row["flag"]}; {note}' if row["flag"] else note
        running = row["anchored"]
    return rows


def precedent(px: np.ndarray, dates, horizon: int):
    mult = realized_multiples(px, horizon)
    if mult is None:
        return None
    i_w, i_b = int(np.argmin(mult)), int(np.argmax(mult))
    return {
        "n_windows": int(mult.size),
        "worst_pct": float(mult.min() - 1) * 100,
        "best_pct": float(mult.max() - 1) * 100,
        "median_pct": float(np.median(mult) - 1) * 100,
        "worst_window": f"{dates[i_w]} to {dates[i_w + horizon]}",
        "best_window": f"{dates[i_b]} to {dates[i_b + horizon]}",
        "share_le_half": float(np.mean(mult <= 0.5)),
        "share_ge_2x": float(np.mean(mult >= 2.0)),
    }


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def trading_days_between(start: dt.date, end: dt.date) -> int:
    n, d = 0, start
    while d < end:
        d += dt.timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


def month_milestones(anchor: dt.date, months: int = 6):
    """(label, trading-day index) for each upcoming month end."""
    out = []
    y, m = anchor.year, anchor.month
    for _ in range(months):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        last = (dt.date(y + (m == 12), (m % 12) + 1, 1) - dt.timedelta(days=1))
        n = trading_days_between(anchor, last)
        if n > 0:
            out.append((last.strftime("%b %Y"), n, last.isoformat()))
    return out


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def predict(symbol: str, months: int = 6, n_paths: int = 8000, seed: int = 7) -> dict:
    symbol = symbol.strip().upper()
    if not symbol or len(symbol) > 12:
        raise DataError("please enter a valid ticker symbol.")

    n_dates, n_px, n_hi, n_lo, _n_vol, info = fetch_prices(symbol, HISTORY_RANGE)
    m_dates, m_px, _, _, _, _ = fetch_prices(MARKET, HISTORY_RANGE)
    dates, name_px, mkt_px = align(n_dates, n_px, m_dates, m_px)

    name_r, mkt_r = log_returns(name_px), log_returns(mkt_px)
    b = beta_stats(name_r, mkt_r)
    mu_mkt, _, _ = blended_drift(m_px)          # full market history
    anchored_mu = b["beta"] * mu_mkt

    anchor = dates[-1]
    miles = month_milestones(anchor, months)
    if not miles:
        raise DataError("no future month milestones to project.")
    horizon = miles[-1][1]
    s0 = float(name_px[-1])

    _, paths = simulate(mkt_r, name_r, float(mkt_px[-1]), s0, horizon,
                        mu_mkt, anchored_mu, n_paths=n_paths, seed=seed,
                        beta_se=b["se"])

    months_out = []
    prev = 0
    for label, day, iso in miles:
        col = paths[:, day - 1]
        ref = np.full(paths.shape[0], s0) if prev == 0 else paths[:, prev - 1]
        mret = col / ref - 1.0
        cum_low = paths[:, :day].min(axis=1) / s0 - 1.0
        months_out.append({
            "label": label, "date": iso, "trading_day": int(day),
            "p_up": float(np.mean(mret > 0)),
            "median_ret": float(np.median(mret)),
            "p5": float(np.percentile(col, 5)),
            "p50": float(np.percentile(col, 50)),
            "p95": float(np.percentile(col, 95)),
            "dips": {f"{int(abs(l * 100))}": float(np.mean(cum_low <= l))
                     for l in DRAWDOWN_LEVELS},
            "median_worst_dip": float(-np.median(cum_low)),
        })
        prev = day

    terminal = paths[:, -1]
    checks = reality_check(terminal, s0, name_px, horizon)
    anc = {c["q"]: c["anchored"] for c in checks}
    lr = np.log(terminal / s0)
    alpha_rows = [{"label": lbl, "p_up": float(np.mean(lr + al * horizon > 0))}
                  for lbl, al in (("as modeled", 0.0), ("-12%/yr", -0.0005),
                                  ("-22%/yr", -0.0010), ("-31%/yr", -0.0015))]

    bars = len(dates)
    warnings = []
    if bars < MIN_VALIDATE_BARS:
        warnings.append(
            f"Only {bars} bars of shared history (< {MIN_VALIDATE_BARS}). This ticker "
            f"CANNOT be calibration-tested here -- its ranges are extrapolation, not "
            f"validated output.")
    if b["r2"] < 0.25:
        warnings.append(
            f"The market explains only {b['r2'] * 100:.0f}% of this name's moves, so "
            f"{100 - b['r2'] * 100:.0f}% is idiosyncratic -- driven by company-specific "
            f"events this model cannot see.")
    if b["vol"] > 0.05:
        warnings.append(
            f"Very high volatility ({b['vol'] * 100:.1f}%/day). Expect wide ranges and "
            f"large drawdowns as normal behaviour, not as the bad case.")

    return {
        "symbol": symbol,
        "info": info,
        "spot": s0,
        "as_of": anchor.isoformat(),
        "bars": bars,
        "horizon_days": int(horizon),
        "beta": b["beta"], "beta_se": b["se"], "r2": b["r2"],
        "vol_daily": b["vol"], "idio_vol_daily": b["idio_vol"],
        "market_mu_daily": mu_mkt,
        "anchored_mu_daily": anchored_mu,
        "raw_mu_daily": b["raw_mu"],
        "months": months_out,
        "terminal": {
            "median": float(np.median(terminal)),
            "p5": float(np.percentile(terminal, 5)),
            "p95": float(np.percentile(terminal, 95)),
            "p_up": float(np.mean(terminal > s0)),
        },
        "reality_check": checks,
        "anchored_range": {"p5": anc.get(5.0), "median": anc.get(50.0), "p95": anc.get(95.0)},
        "precedent": precedent(name_px, dates, horizon),
        "alpha_sensitivity": alpha_rows,
        # Descriptive only -- structure, never a forecast. Carries its own walk-forward
        # verdict so the UI can never show levels without their track record.
        "waves": _waves.analyse(n_px, n_hi, n_lo, symbol=symbol),
        "warnings": warnings,
        "disclaimer": ("NOT INVESTMENT ADVICE. Calibrated odds from price history only. "
                       "No earnings, launches, contracts, crypto prices or macro events "
                       "are modeled. Direction is not predictable; treat ranges and "
                       "drawdown odds as the output, never the median as a target."),
    }
