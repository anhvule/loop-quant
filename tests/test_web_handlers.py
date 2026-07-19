"""HTTP layer tests for the web app.

The engines are well covered; the *handlers* (query parsing, status codes, JSON shape,
truncation reporting) were only ever verified by hand in a browser. A regression there --
a mis-parsed parameter, a 200 that should be a 400 -- would slip straight through the
rest of the suite.

Runs against `web/serve.py`'s handler with the engines stubbed, so no network is used.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "api"))
sys.path.insert(0, os.path.join(ROOT, "web"))

import _engine as E  # noqa: E402
import serve as SV  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(monkeypatch):
    """Serve the real handler with stubbed engines."""
    def fake_predict(symbol, months=6):
        if symbol.upper() == "BAD":
            raise E.DataError("'BAD' not found on Yahoo Finance (HTTP 404).")
        if symbol.upper() == "BOOM":
            raise RuntimeError("engine exploded")
        return {"symbol": symbol.upper(), "months_requested": months,
                "spot": 1.0, "waves": {"available": False}}

    def fake_screen(symbols, max_symbols=20):
        if not symbols:
            raise E.DataError("enter at least one ticker.")
        kept, dropped = symbols[:max_symbols], symbols[max_symbols:]
        return {"banner": "b", "cards": [{"symbol": s} for s in kept],
                "failures": [], "n_requested": len(kept), "n_scored": len(kept),
                "truncated": dropped, "max_symbols": max_symbols}

    def fake_classify(symbols, max_symbols=20, rf=0.04):
        if not symbols:
            raise E.DataError("enter at least one ticker.")
        if symbols[0].upper() == "BOOM":
            raise RuntimeError("classifier exploded")
        kept, dropped = symbols[:max_symbols], symbols[max_symbols:]
        return {"banner": "b", "rf": rf, "risk_on": True, "quartile_mode": "absolute",
                "n_eligible": len(kept), "regime": {"display": "d"},
                "results": [{"symbol": s, "verdict": "mixed"} for s in kept],
                "failures": [], "n_requested": len(kept), "n_classified": len(kept),
                "truncated": dropped, "max_symbols": max_symbols}

    def fake_analyse(symbols, max_symbols=20, rf=0.04, months=6):
        if not symbols:
            raise E.DataError("enter at least one ticker.")
        if symbols[0].upper() == "BOOM":
            raise RuntimeError("analysis exploded")
        kept, dropped = symbols[:max_symbols], symbols[max_symbols:]
        return {"banner": "b", "rf": rf, "months": months, "risk_on": True,
                "quartile_mode": "absolute", "n_eligible": len(kept),
                "regime": {"display": "d"}, "verdict_order": ["mixed"],
                "rows": [{"symbol": s, "verdict": "mixed"} for s in kept],
                "forecast": {"symbol": kept[0]} if len(kept) == 1 else None,
                "failures": [], "n_requested": len(kept), "n_analysed": len(kept),
                "truncated": dropped, "max_symbols": max_symbols}

    monkeypatch.setattr(SV, "predict", fake_predict)
    monkeypatch.setattr(SV, "screen", fake_screen)
    monkeypatch.setattr(SV, "classify", fake_classify)
    monkeypatch.setattr(SV, "analyse", fake_analyse)

    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), SV.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---- /api/predict ----

def test_predict_ok(server):
    code, body = get(f"{server}/api/predict?symbol=asts&months=9")
    assert code == 200
    assert body["symbol"] == "ASTS"
    assert body["months_requested"] == 9        # parsed and passed through


def test_predict_months_defaults_and_clamps(server):
    assert get(f"{server}/api/predict?symbol=X")[1]["months_requested"] == 6
    assert get(f"{server}/api/predict?symbol=X&months=99")[1]["months_requested"] == 12
    assert get(f"{server}/api/predict?symbol=X&months=0")[1]["months_requested"] == 1
    # garbage must not 500
    assert get(f"{server}/api/predict?symbol=X&months=abc")[1]["months_requested"] == 6


def test_predict_data_error_is_400_with_message(server):
    code, body = get(f"{server}/api/predict?symbol=BAD")
    assert code == 400
    assert "not found" in body["error"]


def test_predict_unexpected_error_is_500_without_internals(server):
    code, body = get(f"{server}/api/predict?symbol=BOOM")
    assert code == 500
    assert "error" in body
    assert "exploded" not in body["error"]      # never leak internals to the client


# ---- /api/screen ----

def test_screen_parses_commas_and_spaces(server):
    code, body = get(f"{server}/api/screen?symbols=okLo,%20iren%20mrvl")
    assert code == 200
    assert [c["symbol"] for c in body["cards"]] == ["okLo", "iren", "mrvl"]


def test_screen_empty_is_400(server):
    code, body = get(f"{server}/api/screen?symbols=")
    assert code == 400
    assert "ticker" in body["error"]
    assert get(f"{server}/api/screen")[0] == 400


# How far past the cap these tests push. Derived from MAX_SYMBOLS rather than a literal
# so raising the cap does not silently turn the truncation tests into no-ops.
OVER = 5


def _over_cap_symbols():
    return ",".join(f"T{i}" for i in range(SV.MAX_SYMBOLS + OVER))


def test_screen_reports_truncation_rather_than_hiding_it(server):
    code, body = get(f"{server}/api/screen?symbols={_over_cap_symbols()}")
    assert code == 200
    assert body["n_scored"] == SV.MAX_SYMBOLS
    assert len(body["truncated"]) == OVER
    assert body["truncated"][0] == f"T{SV.MAX_SYMBOLS}"


# ---- /api/classify ----

def test_classify_parses_commas_and_spaces(server):
    code, body = get(f"{server}/api/classify?symbols=okLo,%20iren%20mrvl")
    assert code == 200
    assert [r["symbol"] for r in body["results"]] == ["okLo", "iren", "mrvl"]


def test_classify_empty_is_400(server):
    code, body = get(f"{server}/api/classify?symbols=")
    assert code == 400
    assert "ticker" in body["error"]
    assert get(f"{server}/api/classify")[0] == 400


def test_classify_rf_defaults_clamps_and_survives_garbage(server):
    assert get(f"{server}/api/classify?symbols=X")[1]["rf"] == 0.04
    assert get(f"{server}/api/classify?symbols=X&rf=0.055")[1]["rf"] == 0.055
    assert get(f"{server}/api/classify?symbols=X&rf=99")[1]["rf"] == 0.20   # clamped
    assert get(f"{server}/api/classify?symbols=X&rf=-1")[1]["rf"] == 0.0
    assert get(f"{server}/api/classify?symbols=X&rf=abc")[1]["rf"] == 0.04  # never 500


def test_classify_reports_truncation_rather_than_hiding_it(server):
    code, body = get(f"{server}/api/classify?symbols={_over_cap_symbols()}")
    assert code == 200
    assert body["n_classified"] == SV.MAX_SYMBOLS
    assert len(body["truncated"]) == OVER


def test_classify_unexpected_error_is_500_without_internals(server):
    code, body = get(f"{server}/api/classify?symbols=BOOM")
    assert code == 500
    assert "exploded" not in body["error"]      # never leak internals to the client


# ---- /api/analyse (the unified endpoint the UI actually uses) ----

def test_analyse_parses_commas_and_spaces(server):
    code, body = get(f"{server}/api/analyse?symbols=nvda,%20coin%20spce")
    assert code == 200
    assert [r["symbol"] for r in body["rows"]] == ["nvda", "coin", "spce"]


def test_analyse_empty_is_400(server):
    assert get(f"{server}/api/analyse?symbols=")[0] == 400
    assert get(f"{server}/api/analyse")[0] == 400


def test_analyse_clamps_months_and_rf(server):
    assert get(f"{server}/api/analyse?symbols=X&months=99")[1]["months"] == 12
    assert get(f"{server}/api/analyse?symbols=X&months=0")[1]["months"] == 1
    assert get(f"{server}/api/analyse?symbols=X&months=abc")[1]["months"] == 6
    assert get(f"{server}/api/analyse?symbols=X&rf=99")[1]["rf"] == 0.20
    assert get(f"{server}/api/analyse?symbols=X&rf=abc")[1]["rf"] == 0.04


def test_analyse_attaches_a_forecast_only_for_a_lone_ticker(server):
    assert get(f"{server}/api/analyse?symbols=X")[1]["forecast"] is not None
    assert get(f"{server}/api/analyse?symbols=X,Y")[1]["forecast"] is None


def test_analyse_reports_truncation(server):
    body = get(f"{server}/api/analyse?symbols={_over_cap_symbols()}")[1]
    assert body["n_analysed"] == SV.MAX_SYMBOLS
    assert len(body["truncated"]) == OVER


def test_analyse_unexpected_error_is_500_without_internals(server):
    code, body = get(f"{server}/api/analyse?symbols=BOOM")
    assert code == 500 and "exploded" not in body["error"]


# ---- static + routing ----

def test_static_index_is_served(server):
    with urllib.request.urlopen(f"{server}/", timeout=10) as r:
        html = r.read().decode("utf-8", "replace")
    assert r.status == 200
    assert "Stock Risk Outlook" in html
    assert "NOT INVESTMENT ADVICE" in html.upper()   # the rail ships with the page


def test_unknown_api_path_is_not_json_handled(server):
    """An unrecognised path must fall through to the static handler (404), not be
    silently treated as a forecast request."""
    try:
        with urllib.request.urlopen(f"{server}/api/nope", timeout=10) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    assert code == 404
