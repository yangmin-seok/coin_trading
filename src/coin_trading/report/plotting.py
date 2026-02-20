from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def render_multi_line_svg(df: pd.DataFrame, series: list[tuple[str, str]], title: str) -> str:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 64, 24, 28, 52
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    y_data = {
        name: [float(v) for v in df.get(name, pd.Series([0.0] * len(df))).fillna(0.0).tolist()]
        for name, _ in series
    }
    all_vals = [v for vals in y_data.values() for v in vals] or [0.0]
    y_min, y_max = min(all_vals), max(all_vals)
    if y_max == y_min:
        y_max = y_min + 1.0

    def _y_scale(v: float) -> float:
        ratio = (v - y_min) / (y_max - y_min)
        return pad_t + (1 - ratio) * chart_h

    scaled = {name: [_y_scale(v) for v in vals] for name, vals in y_data.items()}

    def _line(vals: list[float], color: str) -> str:
        if not vals:
            return ""
        pts = []
        n = max(1, len(vals) - 1)
        for i, y in enumerate(vals):
            xx = pad_l + (i / n) * chart_w
            pts.append(f"{xx:.2f},{y:.2f}")
        return f"<polyline fill='none' stroke='{color}' stroke-width='1.8' points='{' '.join(pts)}'/>"

    x_steps = max(1, len(df) - 1)
    x_ticks = min(6, x_steps + 1)
    y_ticks = 6

    grid_parts: list[str] = []
    x_label_parts: list[str] = []
    y_label_parts: list[str] = []

    for i in range(y_ticks):
        ratio = i / (y_ticks - 1)
        yy = pad_t + ratio * chart_h
        y_value = y_max - ratio * (y_max - y_min)
        grid_parts.append(
            f"<line x1='{pad_l}' y1='{yy:.2f}' x2='{pad_l + chart_w}' y2='{yy:.2f}' stroke='#e6e6e6' stroke-width='1'/>"
        )
        y_label_parts.append(
            f"<text x='{pad_l - 8}' y='{yy + 4:.2f}' text-anchor='end' font-size='11' fill='#666'>{y_value:.4g}</text>"
        )

    for i in range(x_ticks):
        ratio = i / (x_ticks - 1) if x_ticks > 1 else 0.0
        xx = pad_l + ratio * chart_w
        step_val = int(round(ratio * x_steps))
        grid_parts.append(
            f"<line x1='{xx:.2f}' y1='{pad_t}' x2='{xx:.2f}' y2='{pad_t + chart_h}' stroke='#f0f0f0' stroke-width='1'/>"
        )
        x_label_parts.append(
            f"<text x='{xx:.2f}' y='{pad_t + chart_h + 18}' text-anchor='middle' font-size='11' fill='#666'>{step_val}</text>"
        )

    axes = (
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l}' y1='{pad_t + chart_h}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
    )

    legend = []
    for i, (name, color) in enumerate(series):
        legend.append(f"<text x='{70 + i * 130}' y='18' font-size='12' fill='{color}'>{name}</text>")

    labels = (
        f"<text x='{pad_l + chart_w / 2:.2f}' y='{height-10}' text-anchor='middle' font-size='11' fill='#666'>step</text>"
        f"<text x='16' y='{pad_t + chart_h / 2:.2f}' font-size='11' fill='#666'>value</text>"
        f"<text x='{pad_l}' y='{pad_t - 8}' font-size='12' fill='#222'>{title}</text>"
        + "".join(legend)
    )

    lines = "".join(_line(scaled[name], color) for name, color in series)

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{''.join(grid_parts)}{axes}{''.join(x_label_parts)}{''.join(y_label_parts)}{lines}{labels}"
        "</svg>"
    )


def write_learning_curve_artifacts(history: list[dict[str, Any]], reports_dir: Path, plots_dir: Path) -> None:
    rows = [
        {
            "timesteps": h["timesteps"],
            "val_sharpe": h["val"]["sharpe"],
            "val_max_drawdown": h["val"]["max_drawdown"],
            "val_final_equity": h["val"]["final_equity"],
            "loss": 0.0,
            "entropy_loss": 0.0,
            "value_loss": 0.0,
        }
        for h in history
    ]
    frame = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports_dir / "learning_curve.csv", index=False)
    (reports_dir / "learning_curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (plots_dir / "learning_curve.svg").write_text(
        render_multi_line_svg(
            frame,
            series=[("val_sharpe", "#1f77b4"), ("val_final_equity", "#2ca02c")],
            title="Validation Sharpe / Final Equity",
        ),
        encoding="utf-8",
    )
