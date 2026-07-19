"""Tests for the web API engine (web/api/_engine.py).

Two jobs:
  1. Exercise the engine offline -- no network. `predict()` is driven through a stubbed
     fetch so the whole payload contract is covered.
  2. PARITY: the web engine is a hand-vendored copy of `src/forecast/*`. Without a test
     pinning them together the two copies drift silently, which is exactly how a
     "validated" number stops being the validated number.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))

import _engine as W  # noqa: E402

from src.forecast import beta as B  # noqa: E402
from src.forecast import bounds as BD  # noqa: E402
from src.forecast import longrange as LR  # noqa: E402


# ---------------------------------------------------------------------------
# parity with the validated pipeline
# ---------------------------------------------------------------------------

def _series(n=800, seed=3, vol=0.02, drift=0.0004):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def test_log_returns_parity():
    px = _series()
    assert np.allclose(W.log_returns(px), LR.log_returns(px))


def test_ewma_vol_path_parity():
    r = W.log_returns(_series())
    wv, wc = W.ewma_vol_path(r)
    bv, bc = B.ewma_vol_path(r)
    assert np.allclose(wv, bv)
    assert wc == pytest.approx(bc)


def test_vol_half_life_parity():
    vol, _ = W.ewma_vol_path(W.log_returns(_series(1500, seed=9)))
    assert W.vol_half_life(vol) == pytest.approx(B.vol_half_life_from_path(vol))


def test_blended_drift_parity():
    px = _series()
    assert W.blended_drift(px)[0] == pytest.approx(LR.blended_drift(px)[0])


def test_beta_stats_parity():
    mkt = W.log_returns(_series(900, seed=1, vol=0.01))
    name = 2.5 * mkt + np.random.default_rng(4).normal(0, 0.02, mkt.size)
    w = W.beta_stats(name, mkt)
    b = B.beta_stats("X", name, mkt)
    assert w["beta"] == pytest.approx(b.beta)
    assert w["se"] == pytest.approx(b.se)
    assert w["r2"] == pytest.approx(b.r2)
    assert w["idio_vol"] == pytest.approx(b.idio_vol)


def test_reality_check_parity():
    px = _series(1200, seed=5)
    rng = np.random.default_rng(0)
    terminal = px[-1] * np.exp(rng.normal(0.2, 0.6, 5000))
    web = W.reality_check(terminal, px[-1], px, horizon=60)
    ref = BD.reality_check(terminal, px[-1], list(px), 60)
    assert [c["q"] for c in web] == [c.q for c in ref]
    for a, b in zip(web, ref):
        assert a["model"] == pytest.approx(b.model)
        assert a["anchored"] == pytest.approx(b.anchored)
        assert bool(a["flag"]) == bool(b.flag)


# ---------------------------------------------------------------------------
# engine behaviour
# ---------------------------------------------------------------------------

def test_reality_check_keeps_severe_downside_and_trims_upside():
    px = _series(1200, seed=7, vol=0.01)          # tame history
    s0 = px[-1]
    rng = np.random.default_rng(1)
    terminal = s0 * np.exp(rng.normal(0.5, 1.2, 8000))   # wild model both ways
    rows = {c["q"]: c for c in W.reality_check(terminal, s0, px, horizon=60)}
    assert "trimmed" in rows[95.0]["flag"]
    assert rows[95.0]["anchored"] < rows[95.0]["model"]
    low = rows[5.0]
    if low["ratio"] < 1 / W.REALITY_TOL:
        assert "kept" in low["flag"]
        assert low["anchored"] == pytest.approx(low["model"])


def test_reality_check_percentiles_stay_ordered():
    px = _series(1200, seed=11, vol=0.008)
    s0 = px[-1]
    rng = np.random.default_rng(2)
    rows = W.reality_check(s0 * np.exp(rng.normal(0.4, 1.0, 6000)), s0, px, 60)
    vals = [r["anchored"] for r in rows]
    assert vals == sorted(vals)


def test_simulate_is_deterministic_and_targets_drift():
    mkt = W.log_returns(_series(900, seed=2, vol=0.01))
    name = 2.0 * mkt + np.random.default_rng(6).normal(0, 0.02, mkt.size)
    args = dict(s0_mkt=100.0, s0_name=50.0, horizon=60,
                target_mu_mkt=0.0005, target_mu_name=0.001, n_paths=4000, seed=7)
    _, a = W.simulate(mkt, name, **args)
    _, b = W.simulate(mkt, name, **args)
    assert np.array_equal(a, b)
    realized = float(np.mean(np.log(a[:, -1] / 50.0)) / 60)
    assert realized == pytest.approx(0.001, abs=5e-4)


def test_simulate_rejects_short_history():
    r = np.zeros(5)
    with pytest.raises(W.DataError):
        W.simulate(r, r, 100.0, 100.0, 30, 0.0, 0.0)


def test_month_milestones_are_trading_days_and_increasing():
    ms = W.month_milestones(dt.date(2026, 7, 17), months=6)
    assert len(ms) == 6
    days = [d for _, d, _ in ms]
    assert days == sorted(days) and days[0] > 0
    assert W.is_trading_day(dt.date(2026, 9, 8))          # normal Tuesday
    assert not W.is_trading_day(dt.date(2026, 9, 7))      # Labor Day
    assert not W.is_trading_day(dt.date(2026, 7, 18))     # Saturday


def test_precedent_reports_extremes():
    px = _series(900, seed=13)
    dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(px.size)]
    p = W.precedent(px, dates, 60)
    assert p["n_windows"] == px.size - 60
    assert p["worst_pct"] <= p["median_pct"] <= p["best_pct"]
    assert 0.0 <= p["share_le_half"] <= 1.0


# ---------------------------------------------------------------------------
# predict() end to end, offline
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_fetch(monkeypatch):
    """Serve deterministic synthetic history instead of calling Yahoo."""
    def make(symbol, n=1200, vol=0.03, seed=21):
        px = _series(n, seed=seed, vol=vol)
        end = dt.date(2026, 7, 17)
        dates, d = [], end
        while len(dates) < n:
            if W.is_trading_day(d):
                dates.append(d)
            d -= dt.timedelta(days=1)
        return (list(reversed(dates)), px, px * 1.02, px * 0.98,
                np.full(n, 1_000_000.0),
                {"name": f"{symbol} Test Co", "currency": "USD", "exchange": "TEST"})

    def fake(symbol, rng=W.HISTORY_RANGE):
        if symbol == W.MARKET:
            return make("SPY", 1200, vol=0.009, seed=2)
        # 6%/day mirrors a real high-beta name (ASTS ~6%) and clears the 5% warning bar
        return make(symbol, 1200, vol=0.06, seed=21)

    monkeypatch.setattr(W, "fetch_prices", fake)
    return fake


def test_predict_payload_contract(stub_fetch):
    d = W.predict("TEST", months=6, n_paths=2000)
    for key in ("symbol", "info", "spot", "as_of", "bars", "horizon_days", "beta", "beta_se",
                "r2", "months", "terminal", "reality_check", "anchored_range", "precedent",
                "alpha_sensitivity", "warnings", "disclaimer"):
        assert key in d, f"missing {key}"
    assert len(d["months"]) == 6
    assert d["horizon_days"] > 0
    a = d["anchored_range"]
    assert a["p5"] <= a["median"] <= a["p95"]
    assert "NOT INVESTMENT ADVICE" in d["disclaimer"]


def test_predict_months_are_ordered_with_growing_dip_odds(stub_fetch):
    d = W.predict("TEST", months=6, n_paths=3000)
    days = [m["trading_day"] for m in d["months"]]
    assert days == sorted(days)
    dips = [m["dips"]["20"] for m in d["months"]]
    assert dips == sorted(dips)          # cumulative risk can only accumulate
    for m in d["months"]:
        assert m["p5"] <= m["p50"] <= m["p95"]
        assert 0.0 <= m["p_up"] <= 1.0


def test_predict_alpha_sensitivity_decays(stub_fetch):
    d = W.predict("TEST", months=6, n_paths=3000)
    ups = [a["p_up"] for a in d["alpha_sensitivity"]]
    assert ups == sorted(ups, reverse=True)   # more drag -> lower P(up)


def test_predict_warns_on_high_volatility(stub_fetch):
    d = W.predict("TEST", months=6, n_paths=1500)
    assert any("volatility" in w for w in d["warnings"])


def test_predict_flags_thin_history(monkeypatch):
    """Few enough bars that BOTH guards fire: no calibration test, and too few
    historical windows for the reality check to have anything to compare against."""
    def thin(symbol, rng=W.HISTORY_RANGE):
        n = 200 if symbol != W.MARKET else 1200
        px = _series(n, seed=5, vol=0.03)
        end = dt.date(2026, 7, 17)
        dates, d = [], end
        while len(dates) < n:
            if W.is_trading_day(d):
                dates.append(d)
            d -= dt.timedelta(days=1)
        return (list(reversed(dates)), px, px * 1.02, px * 0.98,
                np.full(n, 1_000_000.0),
                {"name": "Thin Co", "currency": "USD", "exchange": "T"})

    monkeypatch.setattr(W, "fetch_prices", thin)
    d = W.predict("THIN", months=6, n_paths=1500)
    assert any("CANNOT be calibration-tested" in w for w in d["warnings"])
    assert all("thin precedent" in c["flag"] for c in d["reality_check"])


def test_predict_rejects_bad_symbols():
    for bad in ("", "   ", "WAYTOOLONGTICKER"):
        with pytest.raises(W.DataError):
            W.predict(bad)
