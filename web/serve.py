r"""Run the Stock Risk Outlook locally -- no Vercel, no Node, no build step.

    ..\.venv\Scripts\python serve.py            # http://127.0.0.1:8000
    ..\.venv\Scripts\python serve.py --port 9000 --open

Serves the static files and answers /api/predict with the same engine the serverless
function uses, so what you see locally is what deploys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "api"))
# The company report is LOCAL-ONLY: it reads yfinance/pandas from src/forecast rather
# than the numpy-only vendored engines, so it has no serverless twin. Hence the repo
# root on the path here and nowhere in web/api/.
sys.path.insert(0, os.path.dirname(HERE))

from _analyse import analyse  # noqa: E402
from _classify import classify  # noqa: E402
from _engine import DataError, predict  # noqa: E402
from _screener import screen  # noqa: E402

MAX_MONTHS = 12
# The 20 this inherited was a SERVERLESS limit (Vercel caps these functions at 30s).
# Running locally there is no platform ceiling -- 20 names measure in ~2.3s -- so the
# cap exists only to stop a pathological paste, and `_analyse.FETCH_BUDGET_S` is the
# real guard against a slow upstream. A larger cohort also makes the Treynor quartile
# more meaningful, since it ranks against the eligible universe.
MAX_SYMBOLS = 200


def _log_ok(symbol: str, body: dict) -> None:
    """Console summary for a successful forecast.

    Deliberately defensive: a console line must NEVER be able to turn a good 200 into
    a 500. Indexing the payload directly here previously did exactly that.
    """
    try:
        beta = body.get("beta")
        rng = body.get("anchored_range") or {}
        lo, hi = rng.get("p5"), rng.get("p95")
        bits = [f"{symbol.upper():>6}  ok"]
        if isinstance(beta, (int, float)):
            bits.append(f"beta={beta:.2f}")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            bits.append(f"range={lo:.2f}..{hi:.2f}")
        print("  " + "  ".join(bits))
    except Exception:  # noqa: BLE001 -- logging is never worth failing a request over
        print(f"  {symbol.upper():>6}  ok")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def end_headers(self):
        # The local runner exists so that what you see matches what deploys. A browser
        # that caches app.js defeats exactly that: you edit a file, reload, and are
        # silently still running the old code. Never cache anything locally.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def _json(self, body, status):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _symbols(self, params) -> list:
        raw = (params.get("symbols") or [""])[0]
        return [s for s in raw.replace(" ", ",").split(",") if s.strip()]

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/analyse":
            params = parse_qs(urlparse(self.path).query)
            syms = self._symbols(params)
            try:
                rf = float((params.get("rf") or ["0.04"])[0])
            except ValueError:
                rf = 0.04
            rf = max(0.0, min(0.20, rf))
            try:
                months = max(1, min(MAX_MONTHS, int((params.get("months") or ["6"])[0])))
            except ValueError:
                months = 6
            t0 = time.monotonic()
            try:
                if not syms:
                    raise DataError("enter at least one ticker.")
                body, status = analyse(syms, MAX_SYMBOLS, rf=rf, months=months), 200
                # Timing is logged because a slow cohort is otherwise indistinguishable
                # from a hung one, and only the server knows which symbol cost the time.
                print(f"  analyse {body['n_analysed']}/{body['n_requested']} in "
                      f"{time.monotonic() - t0:.1f}s "
                      f"({'risk-on' if body['risk_on'] else 'risk-off'}"
                      f"{', +forecast' if body['forecast'] else ''}"
                      f"{', ' + str(len(body['failures'])) + ' failed' if body['failures'] else ''})")
            except DataError as e:
                body, status = {"error": str(e)}, 400
                print(f"  analyse 400 {e}")
            except Exception as e:  # noqa: BLE001
                body, status = {"error": "Something went wrong running the analysis."}, 500
                print(f"  analyse 500 {type(e).__name__}: {e}")
            return self._json(body, status)

        if path == "/api/screen":
            syms = self._symbols(parse_qs(urlparse(self.path).query))
            try:
                if not syms:
                    raise DataError("enter at least one ticker.")
                body, status = screen(syms, MAX_SYMBOLS), 200
                print(f"  screen {len(body['cards'])}/{body['n_requested']} scored")
            except DataError as e:
                body, status = {"error": str(e)}, 400
                print(f"  screen 400 {e}")
            except Exception as e:  # noqa: BLE001
                body, status = {"error": "Something went wrong running the screen."}, 500
                print(f"  screen 500 {type(e).__name__}: {e}")
            return self._json(body, status)

        if path == "/api/classify":
            params = parse_qs(urlparse(self.path).query)
            syms = self._symbols(params)
            try:
                rf = float((params.get("rf") or ["0.04"])[0])
            except ValueError:
                rf = 0.04
            rf = max(0.0, min(0.20, rf))     # a negative or absurd rate is a typo
            try:
                if not syms:
                    raise DataError("enter at least one ticker.")
                body, status = classify(syms, MAX_SYMBOLS, rf=rf), 200
                print(f"  classify {body['n_classified']}/{body['n_requested']} "
                      f"({'risk-on' if body['risk_on'] else 'risk-off'}, "
                      f"{body['quartile_mode']})")
            except DataError as e:
                body, status = {"error": str(e)}, 400
                print(f"  classify 400 {e}")
            except Exception as e:  # noqa: BLE001
                body, status = {"error": "Something went wrong running the classifier."}, 500
                print(f"  classify 500 {type(e).__name__}: {e}")
            return self._json(body, status)

        if path == "/api/company":
            params = parse_qs(urlparse(self.path).query)
            sym = (params.get("symbol") or [""])[0].strip()
            refresh = (params.get("refresh") or [""])[0] == "1"
            # Imported here, not at module scope: this is the only route that needs
            # yfinance and pandas, and paying their import cost on every startup
            # would slow the four routes that deliberately avoid them.
            from src.forecast.company_report import (  # noqa: PLC0415
                CompanyDataError, build_report)
            t0 = time.monotonic()
            try:
                body, status = build_report(sym, refresh=refresh), 200
                print(f"  company {sym.upper():>6}  ok in "
                      f"{time.monotonic() - t0:.1f}s "
                      f"({body['snowflake']['total']}/{body['snowflake']['max']}"
                      f"{', cached' if body['cached'] else ''})")
            except CompanyDataError as e:
                body, status = {"error": str(e)}, 400
                print(f"  company {sym.upper():>6}  400 {e}")
            except Exception as e:  # noqa: BLE001
                body, status = {"error": "Something went wrong building the report."}, 500
                print(f"  company {sym.upper():>6}  500 {type(e).__name__}: {e}")
            return self._json(body, status)

        if path != "/api/predict":
            return super().do_GET()

        params = parse_qs(urlparse(self.path).query)
        symbol = (params.get("symbol") or [""])[0]
        try:
            months = max(1, min(MAX_MONTHS, int((params.get("months") or ["6"])[0])))
        except ValueError:
            months = 6
        try:
            body, status = predict(symbol, months=months), 200
            _log_ok(symbol, body)
        except DataError as e:
            body, status = {"error": str(e)}, 400
            print(f"  {symbol.upper():>6}  400 {e}")
        except Exception as e:  # noqa: BLE001
            body, status = {"error": "Something went wrong generating the forecast."}, 500
            print(f"  {symbol.upper():>6}  500 {type(e).__name__}: {e}")

        self._json(body, status)

    def log_message(self, *args):        # keep the console to forecast lines only
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the outlook web app locally")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/"
    try:
        server = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        # Silently binding to a busy port would leave an OLD server answering with
        # stale code -- confusing and easy to miss. Fail loudly instead.
        print(f"ERROR: cannot bind {a.host}:{a.port} ({e.strerror or e}).\n"
              f"Another server is probably already running there -- stop it, or pass "
              f"--port with a free port.", file=sys.stderr)
        return 1

    print(f"Stock Risk Outlook running at {url}   (Ctrl+C to stop)")
    print("NOT INVESTMENT ADVICE - research tool only.\n")
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
