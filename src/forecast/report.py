"""Render forecasts: console table, CSV, and an optional PNG chart.

Kept free of any live-engine imports. The chart uses matplotlib with the non-
interactive "Agg" backend so it works headless (CI, servers) with no display.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path

from src.common.models import Candle
from src.forecast.features import Features
from src.forecast.result import ForecastPath

log = logging.getLogger(__name__)

DISCLAIMER = ("NOT INVESTMENT ADVICE. Statistical baselines from historical drift/"
              "volatility, not a prediction of actual price.")


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def console_table(symbol: str, feats: Features, gbm: ForecastPath,
                  arima: ForecastPath) -> str:
    """Human-readable summary. Returns the string (also convenient for tests)."""
    lines: list[str] = []
    lines.append(f"=== {symbol} multi-day price-path forecast ===")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append(f"last close        : {_fmt(feats.last_close)}   (over {feats.n_bars} daily bars)")
    rsi = "n/a" if feats.rsi is None else f"{feats.rsi:.1f}"
    mh = "n/a" if feats.macd_hist is None else f"{feats.macd_hist:+.4f}"
    lines.append(f"RSI / MACD hist   : {rsi} / {mh}")
    lines.append(f"signal score      : {feats.signal_score:+.3f}  ({feats.signal_action})")
    lines.append(f"daily drift / vol : mu={feats.mu:+.5f}  sigma={feats.sigma:.5f}")
    lines.append("")
    lines.append(f"gbm  : {gbm.note}")
    lines.append(f"arima: {arima.note}")
    lines.append("")
    hdr = (f"{'day':>4} | {'GBM p5':>10} {'GBM p50':>10} {'GBM p95':>10} "
           f"| {'ARIMA lo':>10} {'ARIMA mid':>10} {'ARIMA hi':>10}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, d in enumerate(gbm.days):
        lines.append(
            f"{d:>4} | {_fmt(gbm.lower[i]):>10} {_fmt(gbm.median[i]):>10} {_fmt(gbm.upper[i]):>10} "
            f"| {_fmt(arima.lower[i]):>10} {_fmt(arima.median[i]):>10} {_fmt(arima.upper[i]):>10}"
        )
    g_last, a_last = gbm.median[-1], arima.median[-1]
    lines.append("")
    lines.append(f"{len(gbm.days)}-day median: GBM {_fmt(g_last)} "
                 f"({_pct_change(feats.last_close, g_last):+.1f}%)   "
                 f"ARIMA {_fmt(a_last)} ({_pct_change(feats.last_close, a_last):+.1f}%)")
    return "\n".join(lines)


def _pct_change(base: float, val: float) -> float:
    return 0.0 if base <= 0 else (val / base - 1.0) * 100.0


def write_csv(path: Path, gbm: ForecastPath, arima: ForecastPath) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["day", "gbm_p5", "gbm_p25", "gbm_p50", "gbm_p75", "gbm_p95",
                    "arima_lower", "arima_median", "arima_upper"])
        for i, d in enumerate(gbm.days):
            g25 = gbm.p25[i] if gbm.p25 else ""
            g75 = gbm.p75[i] if gbm.p75 else ""
            w.writerow([d,
                        f"{gbm.lower[i]:.4f}", (f"{g25:.4f}" if g25 != "" else ""),
                        f"{gbm.median[i]:.4f}", (f"{g75:.4f}" if g75 != "" else ""),
                        f"{gbm.upper[i]:.4f}",
                        f"{arima.lower[i]:.4f}", f"{arima.median[i]:.4f}", f"{arima.upper[i]:.4f}"])
    return path


def write_chart(path: Path, symbol: str, candles: list[Candle],
                gbm: ForecastPath, arima: ForecastPath, history_tail: int = 120) -> Path | None:
    """Plot recent history + both forecasts. Returns the path, or None if matplotlib
    is unavailable (charting is optional; the CSV/console output is the source of truth)."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        log.warning("matplotlib not installed; skipping chart")
        return None

    tail = candles[-history_tail:] if len(candles) > history_tail else candles
    hist_x = list(range(-len(tail) + 1, 1))            # ... -2, -1, 0 (last close)
    hist_y = [c.close for c in tail]
    fut_x = gbm.days                                   # 1 .. horizon

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hist_x, hist_y, color="#333", lw=1.4, label=f"{symbol} close (history)")

    # GBM band + median
    ax.fill_between(fut_x, gbm.lower, gbm.upper, color="#4c78a8", alpha=0.18,
                    label="GBM 90% band")
    if gbm.p25 and gbm.p75:
        ax.fill_between(fut_x, gbm.p25, gbm.p75, color="#4c78a8", alpha=0.28,
                        label="GBM 50% band")
    ax.plot(fut_x, gbm.median, color="#4c78a8", lw=1.8, label="GBM median")

    # ARIMA median + interval
    ax.plot(fut_x, arima.median, color="#e45756", lw=1.8, ls="--", label="ARIMA median")
    ax.plot(fut_x, arima.lower, color="#e45756", lw=0.9, ls=":", alpha=0.7)
    ax.plot(fut_x, arima.upper, color="#e45756", lw=0.9, ls=":", alpha=0.7)

    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title(f"{symbol} — {len(fut_x)}-day price-path forecast\n{DISCLAIMER}", fontsize=10)
    ax.set_xlabel("trading days (0 = last close)")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def default_csv_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"forecast_{symbol}.csv"


def default_chart_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"forecast_{symbol}.png"


def _today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
