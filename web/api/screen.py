"""Vercel serverless function: GET /api/screen?symbols=OKLO,IREN,MRVL"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _engine import DataError  # noqa: E402
from _screener import screen  # noqa: E402

MAX_SYMBOLS = 20


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        params = parse_qs(urlparse(self.path).query)
        raw = (params.get("symbols") or [""])[0]
        syms = [s for s in raw.replace(" ", ",").split(",") if s.strip()]
        try:
            if not syms:
                raise DataError("enter at least one ticker.")
            body, status = screen(syms, MAX_SYMBOLS), 200
        except DataError as e:
            body, status = {"error": str(e)}, 400
        except Exception:  # noqa: BLE001
            body, status = {"error": "Something went wrong running the screen."}, 500

        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=900, s-maxage=900")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass
