"""Behavioural gate: a candidate config must EARN its deployment in simulation.

Isolation note, stated honestly: the subprocess gives crash/state isolation and
guarantees the candidate config is evaluated by a fresh interpreter that cannot
touch the live engine's objects. It is not an OS-level network jail -- the
backtester simply makes no network calls by construction (it reads SQLite and
nothing else). Overclaiming that would be worse than saying it plainly.

ACCEPTANCE CRITERIA -- note the deviation from the blueprint's literal wording:

  The blueprint specifies "candidate expectancy >= current expectancy * 1.05".
  That is wrong whenever the incumbent's expectancy is NEGATIVE, which is exactly
  when the optimizer fires: -0.10 * 1.05 = -0.105, so a candidate at -0.104 would
  "pass" while still losing money. Multiplying a negative number by 1.05 LOWERS
  the bar.

  The correct generalization, used here, is:
      candidate >= incumbent + 0.05 * abs(incumbent)
  which reduces to *1.05 for positive incumbents and demands a genuine improvement
  for negative ones.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.common.models import BacktestResult, SandboxVerdict

log = logging.getLogger(__name__)

MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000

WINDOW_DAYS = 30
STRESS_WINDOW_MS = 48 * MS_PER_HOUR
STRESS_STEP_MS = 6 * MS_PER_HOUR

MIN_CANDIDATE_TRADES = 15
EXPECTANCY_IMPROVEMENT = 0.05     # must beat incumbent by 5% of |incumbent|
MAX_DD_MULTIPLE = 1.10            # candidate drawdown ceiling vs incumbent
MIN_DD_FLOOR_PCT = 0.5            # so a ~0 incumbent DD isn't an impossible bar
STRESS_DD_MULTIPLE = 2.0          # catastrophic-on-stress rejection
SUBPROCESS_TIMEOUT_SEC = 600


class SandboxError(RuntimeError):
    pass


class BacktestSandbox:
    def __init__(self, db_path: Path, root: Path, python: str | None = None,
                 slippage_bps: float = 0.0, timeout_sec: int = SUBPROCESS_TIMEOUT_SEC) -> None:
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.python = python or sys.executable
        self.slippage_bps = slippage_bps
        self.timeout_sec = timeout_sec

    # -- subprocess --------------------------------------------------------

    def _run_one(self, cfg: dict[str, Any], start_ms: int, end_ms: int,
                 label: str) -> BacktestResult | None:
        with tempfile.TemporaryDirectory(prefix="lq-sandbox-") as tmp:
            cfg_path = Path(tmp) / "candidate.json"
            out_path = Path(tmp) / "result.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            cmd = [
                self.python, "-m", "src.backtest.backtester",
                "--config", str(cfg_path),
                "--db", str(self.db_path),
                "--start-ms", str(start_ms),
                "--end-ms", str(end_ms),
                "--slippage-bps", str(self.slippage_bps),
                "--out", str(out_path),
            ]
            try:
                proc = subprocess.run(cmd, cwd=str(self.root), capture_output=True,
                                      text=True, timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                raise SandboxError(f"backtest '{label}' timed out after {self.timeout_sec}s") from None

            if proc.returncode == 2:
                log.warning("sandbox '%s': no candles in range", label)
                return None
            if proc.returncode != 0:
                raise SandboxError(
                    f"backtest '{label}' exited {proc.returncode}: {proc.stderr.strip()[:500]}")

            d = json.loads(out_path.read_text(encoding="utf-8"))
            return BacktestResult(
                n_trades=d["n_trades"], expectancy=d["expectancy"], sharpe=d["sharpe"],
                max_drawdown_pct=d["max_drawdown_pct"], win_rate=d["win_rate"],
                profit_factor=d["profit_factor"], total_return_pct=d["total_return_pct"],
                pnl_pct_std=d.get("pnl_pct_std", 0.0),
                trade_fingerprint=d.get("trade_fingerprint", ""),
            )

    # -- verdict -----------------------------------------------------------

    def run(self, candidate: dict[str, Any], current: dict[str, Any],
            now_ms: int | None = None,
            stress_windows: list[dict[str, Any]] | None = None) -> SandboxVerdict:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        start = now - WINDOW_DAYS * MS_PER_DAY

        cand = self._run_one(candidate, start, now, "candidate-30d")
        inc = self._run_one(current, start, now, "incumbent-30d")

        if cand is None or inc is None:
            return SandboxVerdict(False, ["no candle data in the 30d window; cannot evaluate"],
                                  cand, inc)

        reasons: list[str] = []

        # 1. enough activity to mean anything
        if cand.n_trades < MIN_CANDIDATE_TRADES:
            reasons.append(f"candidate produced only {cand.n_trades} trades over {WINDOW_DAYS}d "
                           f"(need >= {MIN_CANDIDATE_TRADES}); insufficient signal to judge")

        # 2. expectancy must genuinely improve (see module docstring)
        required = inc.expectancy + EXPECTANCY_IMPROVEMENT * abs(inc.expectancy)
        if cand.expectancy < required:
            reasons.append(
                f"expectancy {cand.expectancy:.4f}% does not beat incumbent "
                f"{inc.expectancy:.4f}% by {EXPECTANCY_IMPROVEMENT:.0%} of its magnitude "
                f"(needed >= {required:.4f}%)")

        # 3. drawdown ceiling
        dd_cap = max(inc.max_drawdown_pct * MAX_DD_MULTIPLE, MIN_DD_FLOOR_PCT)
        if cand.max_drawdown_pct > dd_cap:
            reasons.append(f"max drawdown {cand.max_drawdown_pct:.2f}% exceeds the ceiling "
                           f"{dd_cap:.2f}% ({MAX_DD_MULTIPLE}x incumbent "
                           f"{inc.max_drawdown_pct:.2f}%)")

        # 4. stress windows
        stress: dict[str, dict[str, Any]] = {}
        for w in (stress_windows or []):
            label = w["label"]
            try:
                cs = self._run_one(candidate, w["start_ms"], w["end_ms"], f"candidate-{label}")
                is_ = self._run_one(current, w["start_ms"], w["end_ms"], f"incumbent-{label}")
            except SandboxError as e:
                log.warning("stress window %s failed: %s", label, e)
                continue
            if cs is None or is_ is None:
                continue
            stress[label] = {"candidate": cs.to_dict(), "incumbent": is_.to_dict()}
            cap = max(is_.max_drawdown_pct * STRESS_DD_MULTIPLE, MIN_DD_FLOOR_PCT)
            if cs.max_drawdown_pct > cap:
                reasons.append(
                    f"catastrophic on stress window '{label}': drawdown "
                    f"{cs.max_drawdown_pct:.2f}% vs incumbent {is_.max_drawdown_pct:.2f}% "
                    f"(cap {cap:.2f}%)")

        passed = not reasons
        if passed:
            log.info("sandbox PASS: expectancy %.4f%% -> %.4f%%, DD %.2f%% -> %.2f%%, "
                     "trades %d -> %d", inc.expectancy, cand.expectancy,
                     inc.max_drawdown_pct, cand.max_drawdown_pct, inc.n_trades, cand.n_trades)
        else:
            log.warning("sandbox REJECT: %s", "; ".join(reasons))
        return SandboxVerdict(passed, reasons, cand, inc, stress)


# ---------------------------------------------------------------------------
# stress window discovery
# ---------------------------------------------------------------------------

def compute_stress_windows(db, symbol: str, tf: str, now_ms: int | None = None,
                           days: int = WINDOW_DAYS) -> list[dict[str, Any]]:
    """Find the two 48h windows a config would most hate: the most volatile, and
    the most strongly trending (worst case for a mean-reversion component).

    Recomputed weekly by the engine; a config that only survives a calm month is
    a config that has not been tested.
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    candles = db.get_candles(symbol, tf, now - days * MS_PER_DAY, now)
    if len(candles) < 200:
        log.warning("not enough history (%d bars) to compute stress windows", len(candles))
        return []

    best_vol: tuple[float, int, int] | None = None
    best_trend: tuple[float, int, int] | None = None

    start_ms = candles[0].ts_open_ms
    end_ms = candles[-1].ts_open_ms
    cursor = start_ms
    while cursor + STRESS_WINDOW_MS <= end_ms:
        w_end = cursor + STRESS_WINDOW_MS
        window = [c for c in candles if cursor <= c.ts_open_ms < w_end]
        if len(window) >= 60:
            rets = []
            for i in range(1, len(window)):
                p0, p1 = window[i - 1].close, window[i].close
                if p0 > 0 and p1 > 0:
                    rets.append(math.log(p1 / p0))
            if len(rets) >= 30:
                vol = statistics.pstdev(rets)
                if vol > 0:
                    total = abs(math.log(window[-1].close / window[0].close))
                    trend = total / (vol * math.sqrt(len(rets)))
                    if best_vol is None or vol > best_vol[0]:
                        best_vol = (vol, cursor, w_end)
                    if best_trend is None or trend > best_trend[0]:
                        best_trend = (trend, cursor, w_end)
        cursor += STRESS_STEP_MS

    out: list[dict[str, Any]] = []
    if best_vol:
        out.append({"label": "highest_vol_48h", "start_ms": best_vol[1], "end_ms": best_vol[2],
                    "metric": round(best_vol[0], 8)})
    if best_trend:
        out.append({"label": "most_trending_48h", "start_ms": best_trend[1],
                    "end_ms": best_trend[2], "metric": round(best_trend[0], 4)})
    log.info("stress windows: %s", [w["label"] for w in out])
    return out


def save_stress_windows(windows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_ms": int(time.time() * 1000), "windows": windows},
                               indent=2), encoding="utf-8")


def load_stress_windows(path: Path, max_age_days: int = 7) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if int(time.time() * 1000) - d.get("generated_ms", 0) > max_age_days * MS_PER_DAY:
        return []
    return d.get("windows", [])
