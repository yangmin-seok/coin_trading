from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def render_multi_line_svg(df: pd.DataFrame, series: list[tuple[str, str]], title: str) -> str:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    y_data = {
        name: [float(v) for v in df.get(name, pd.Series([0.0] * len(df))).fillna(0.0).tolist()]
        for name, _ in series
    }

    def _scale(vals: list[float]) -> list[float]:
        if not vals:
            return []
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            vmax = vmin + 1.0
        out: list[float] = []
        for v in vals:
            ratio = (v - vmin) / (vmax - vmin)
            out.append(pad_t + (1 - ratio) * chart_h)
        return out

    scaled = {name: _scale(vals) for name, vals in y_data.items()}

    def _line(vals: list[float], color: str) -> str:
        if not vals:
            return ""
        pts = []
        n = max(1, len(vals) - 1)
        for i, y in enumerate(vals):
            xx = pad_l + (i / n) * chart_w
            pts.append(f"{xx:.2f},{y:.2f}")
        return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}' />"

    guides = (
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l}' y1='{pad_t + chart_h}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
    )

    legend = []
    for i, (name, color) in enumerate(series):
        legend.append(f"<text x='{55 + i * 120}' y='18' font-size='12' fill='{color}'>{name}</text>")
    labels = f"<text x='{pad_l}' y='{height-8}' font-size='11' fill='#666'>step</text>"
    labels += "".join(legend)
    labels += f"<text x='{pad_l}' y='{pad_t + 12}' font-size='12' fill='#222'>{title}</text>"

    lines = "".join(_line(scaled[name], color) for name, color in series)

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{guides}{lines}{labels}"
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
