"""Vercel serverless function: GET /api/predict?symbol=ASTS&months=6"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _engine import DataError, predict  # noqa: E402

MAX_MONTHS = 12


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (Vercel/BaseHTTPRequestHandler contract)
        params = parse_qs(urlparse(self.path).query)
        symbol = (params.get("symbol") or [""])[0]
        try:
            months = max(1, min(MAX_MONTHS, int((params.get("months") or ["6"])[0])))
        except ValueError:
            months = 6

        try:
            body, status = predict(symbol, months=months), 200
        except DataError as e:
            body, status = {"error": str(e)}, 400
        except Exception:  # noqa: BLE001 -- never leak internals to the client
            body, status = {"error": "Something went wrong generating the forecast."}, 500

        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # forecasts move only with new daily bars; cache at the edge for 15 minutes
        self.send_header("Cache-Control", "public, max-age=900, s-maxage=900")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):    # keep function logs quiet
        pass
