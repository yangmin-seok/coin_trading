from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jzfoAAAAASUVORK5CYII="
)


def _safe_plot_png(path: Path, plotter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        plotter(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception:
        path.write_bytes(_MINIMAL_PNG)


def create_split_equity_curves(run_dir: Path, traces: dict[str, pd.DataFrame]) -> list[str]:
    out_dir = run_dir / "plots"
    generated: list[str] = []
    for split in ("train", "valid", "test"):
        trace = traces.get(split, pd.DataFrame())
        out_path = out_dir / f"equity_curve_{split}.png"

        def _plot(ax):
            ax.set_title(f"Equity Curve ({split})")
            ax.set_xlabel("step")
            ax.set_ylabel("equity")
            if trace.empty or "equity" not in trace:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                return
            x = np.arange(len(trace))
            ax.plot(x, trace["equity"].astype(float).to_numpy(), color="#2ca02c", linewidth=1.8)
            ax.grid(alpha=0.25)

        _safe_plot_png(out_path, _plot)
        generated.append(str(out_path.relative_to(run_dir)))
    return generated


def create_common_risk_plots(run_dir: Path, traces: dict[str, pd.DataFrame]) -> dict[str, str]:
    combined = []
    for split in ("train", "valid", "test"):
        trace = traces.get(split, pd.DataFrame())
        if trace.empty:
            continue
        tmp = trace.copy()
        tmp["split"] = split
        combined.append(tmp)
    combo = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

    out_dir = run_dir / "plots"
    drawdown_path = out_dir / "drawdown_curve.png"

    def _drawdown(ax):
        ax.set_title("Drawdown Curve")
        ax.set_xlabel("step")
        ax.set_ylabel("drawdown")
        if combo.empty or "drawdown" not in combo:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        for split in ("train", "valid", "test"):
            part = combo[combo["split"] == split]
            if not part.empty:
                ax.plot(part.index.to_numpy(), part["drawdown"].astype(float).to_numpy(), label=split)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    _safe_plot_png(drawdown_path, _drawdown)

    heatmap_path = out_dir / "monthly_returns_heatmap.png"

    def _heatmap(ax):
        ax.set_title("Monthly Returns Heatmap")
        if combo.empty or "equity" not in combo or "open_time" not in combo:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        frame = combo[["open_time", "equity"]].copy()
        frame["ts"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame.sort_values("ts")
        frame["ret"] = frame["equity"].astype(float).pct_change().fillna(0.0)
        monthly = frame.groupby(frame["ts"].dt.to_period("M"))["ret"].sum()
        if monthly.empty:
            ax.text(0.5, 0.5, "No monthly data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        mat = np.full((12, max(1, len(monthly.index.year.unique()))), np.nan)
        years = sorted(monthly.index.year.unique())
        for period, value in monthly.items():
            y_idx = years.index(period.year)
            mat[period.month - 1, y_idx] = float(value)
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_yticks(np.arange(12), labels=[str(i) for i in range(1, 13)])
        ax.set_xticks(np.arange(len(years)), labels=[str(y) for y in years])
        ax.set_xlabel("year")
        ax.set_ylabel("month")
        ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    _safe_plot_png(heatmap_path, _heatmap)
    return {
        "drawdown_curve_png": str(drawdown_path.relative_to(run_dir)),
        "monthly_returns_heatmap_png": str(heatmap_path.relative_to(run_dir)),
    }


def _trade_stats(trace: pd.DataFrame) -> dict[str, float]:
    if trace.empty:
        return {"win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_holding_time": 0.0, "trade_count": 0.0}
    equity = trace.get("equity", pd.Series(dtype=float)).astype(float)
    returns = equity.pct_change().fillna(0.0)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float((returns > 0).mean())
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    expectancy = float(returns.mean())

    positions = trace.get("action_effective_pos", pd.Series([0.0] * len(trace))).astype(float)
    in_pos = positions.abs() > 1e-9
    lengths: list[int] = []
    run = 0
    for flag in in_pos:
        if flag:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)

    filled = trace.get("filled_qty", pd.Series([0.0] * len(trace))).astype(float)
    trade_count = int((filled.abs() > 1e-9).sum())

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_holding_time": float(np.mean(lengths)) if lengths else 0.0,
        "trade_count": float(trade_count),
    }


def write_trade_stats_report(run_dir: Path, trace: pd.DataFrame, overfit_warning: bool) -> str:
    stats = _trade_stats(trace)
    badge = (
        "<span style='padding:4px 8px;background:#dc2626;color:#fff;border-radius:6px;font-weight:600;'>"
        "Overfitting Warning: train/test gap is large"
        "</span>"
        if overfit_warning
        else "<span style='padding:4px 8px;background:#16a34a;color:#fff;border-radius:6px;font-weight:600;'>Generalization OK</span>"
    )
    html = f"""
<!doctype html>
<html>
<head><meta charset='utf-8'><title>Trade Stats</title></head>
<body style='font-family:Arial,sans-serif;margin:24px;'>
<h2>Trade Statistics Report</h2>
<div style='margin-bottom:16px;'>{badge}</div>
<table border='1' cellspacing='0' cellpadding='8'>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Win rate</td><td>{stats['win_rate']:.4f}</td></tr>
<tr><td>Profit factor</td><td>{stats['profit_factor']:.4f}</td></tr>
<tr><td>Expectancy</td><td>{stats['expectancy']:.6f}</td></tr>
<tr><td>Average holding time (steps)</td><td>{stats['avg_holding_time']:.2f}</td></tr>
<tr><td>Trade count</td><td>{int(stats['trade_count'])}</td></tr>
</table>
</body></html>
"""
    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trade_stats.html"
    out_path.write_text(html.strip() + "\n", encoding="utf-8")
    return str(out_path.relative_to(run_dir))


def create_benchmark_comparison(run_dir: Path, candles_df: pd.DataFrame, seed: int) -> str:
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_comparison.png"
    if candles_df.empty or "close" not in candles_df:
        out_path.write_bytes(_MINIMAL_PNG)
        return str(out_path.relative_to(run_dir))

    close = candles_df["close"].astype(float).reset_index(drop=True)
    pct = close.pct_change().fillna(0.0)

    buy_hold = (1 + pct).cumprod()

    short = close.rolling(5, min_periods=1).mean()
    long = close.rolling(20, min_periods=1).mean()
    ma_pos = (short > long).astype(float).shift(1).fillna(0.0)
    ma_curve = (1 + ma_pos * pct).cumprod()

    rng = np.random.default_rng(seed)
    rand_pos = pd.Series(rng.integers(0, 2, size=len(close)), dtype=float).shift(1).fillna(0.0)
    rand_curve = (1 + rand_pos * pct).cumprod()

    def _plot(ax):
        ax.set_title("Benchmark Comparison")
        ax.set_xlabel("step")
        ax.set_ylabel("normalized equity")
        x = np.arange(len(close))
        ax.plot(x, buy_hold.to_numpy(), label="Buy&Hold", linewidth=1.8)
        ax.plot(x, ma_curve.to_numpy(), label="MA crossover", linewidth=1.4)
        ax.plot(x, rand_curve.to_numpy(), label="Random policy", linewidth=1.2)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    _safe_plot_png(out_path, _plot)
    return str(out_path.relative_to(run_dir))


def detect_overfit(train_metrics: dict[str, Any], test_metrics: dict[str, Any], threshold: float = 0.3) -> bool:
    train_eq = float(train_metrics.get("final_equity", 0.0) or 0.0)
    test_eq = float(test_metrics.get("final_equity", 0.0) or 0.0)
    if train_eq <= 0 or test_eq <= 0:
        return False
    gap = abs(train_eq - test_eq) / max(train_eq, 1e-12)
    return bool(gap >= threshold)
