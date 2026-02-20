from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SERIES_STYLE: dict[str, dict[str, str]] = {
    "train": {"color": "#1f77b4", "dash": ""},
    "val": {"color": "#ff7f0e", "dash": ""},
    "test": {"color": "#2ca02c", "dash": ""},
    "baseline": {"color": "#7f7f7f", "dash": "5,3"},
    "cash_hold": {"color": "#7f7f7f", "dash": "5,3"},
    "buy_hold": {"color": "#8c564b", "dash": "3,2"},
}


def render_multi_line_svg(
    df: pd.DataFrame,
    primary_series: list[tuple[str, str]],
    secondary_series: list[tuple[str, str]],
    title: str,
    subtitle: str = "",
    annotations: list[dict[str, Any]] | None = None,
    y_format_left: str = "number",
    y_format_right: str = "number",
) -> str:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 64, 24, 28, 52
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    def _series_values(name: str) -> list[float]:
        return [float(v) for v in df.get(name, pd.Series([0.0] * len(df))).fillna(0.0).tolist()]

    primary_data = {name: _series_values(name) for name, _ in primary_series}
    secondary_data = {name: _series_values(name) for name, _ in secondary_series}

    primary_vals = [v for vals in primary_data.values() for v in vals] or [0.0]
    secondary_vals = [v for vals in secondary_data.values() for v in vals] or [0.0]

    left_min, left_max = min(primary_vals), max(primary_vals)
    right_min, right_max = min(secondary_vals), max(secondary_vals)
    if left_max == left_min:
        left_max = left_min + 1.0
    if right_max == right_min:
        right_max = right_min + 1.0

    def _y_scale(v: float, min_v: float, max_v: float) -> float:
        ratio = (v - min_v) / (max_v - min_v)
        return pad_t + (1 - ratio) * chart_h

    primary_scaled = {name: [_y_scale(v, left_min, left_max) for v in vals] for name, vals in primary_data.items()}
    secondary_scaled = {
        name: [_y_scale(v, right_min, right_max) for v in vals] for name, vals in secondary_data.items()
    }

    def _series_style(name: str, color: str) -> tuple[str, str]:
        for key, spec in SERIES_STYLE.items():
            if key in name:
                return spec["color"], spec.get("dash", "")
        return color, ""

    def _line(name: str, vals: list[float], color: str) -> str:
        if not vals:
            return ""
        line_color, dash = _series_style(name, color)
        pts = []
        n = max(1, len(vals) - 1)
        for i, y in enumerate(vals):
            xx = pad_l + (i / n) * chart_w
            pts.append(f"{xx:.2f},{y:.2f}")
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        return (
            f"<polyline fill='none' stroke='{line_color}' stroke-width='1.8'{dash_attr} points='{' '.join(pts)}'/>"
        )

    def _format_tick(value: float, mode: str) -> str:
        if mode == "percent":
            return f"{value * 100:.1f}%"
        return f"{value:.4g}"

    x_steps = max(1, len(df) - 1)
    x_ticks = min(6, x_steps + 1)
    y_ticks = 6

    grid_parts: list[str] = []
    x_label_parts: list[str] = []
    y_label_parts: list[str] = []

    for i in range(y_ticks):
        ratio = i / (y_ticks - 1)
        yy = pad_t + ratio * chart_h
        y_left = left_max - ratio * (left_max - left_min)
        y_right = right_max - ratio * (right_max - right_min)
        grid_parts.append(
            f"<line x1='{pad_l}' y1='{yy:.2f}' x2='{pad_l + chart_w}' y2='{yy:.2f}' stroke='#e6e6e6' stroke-width='1'/>"
        )
        y_label_parts.append(
            f"<text x='{pad_l - 8}' y='{yy + 4:.2f}' text-anchor='end' font-size='11' fill='#666'>{_format_tick(y_left, y_format_left)}</text>"
        )
        y_label_parts.append(
            f"<text x='{pad_l + chart_w + 8}' y='{yy + 4:.2f}' text-anchor='start' font-size='11' fill='#666'>{_format_tick(y_right, y_format_right)}</text>"
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
        f"<line x1='{pad_l + chart_w}' y1='{pad_t}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l}' y1='{pad_t + chart_h}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
    )

    legend = []
    legend_x = 70
    for name, color in primary_series:
        line_color, _ = _series_style(name, color)
        legend.append(f"<text x='{legend_x}' y='18' font-size='12' fill='{line_color}'>[L] {name}</text>")
        legend_x += 145
    for name, color in secondary_series:
        line_color, _ = _series_style(name, color)
        legend.append(f"<text x='{legend_x}' y='18' font-size='12' fill='{line_color}'>[R] {name}</text>")
        legend_x += 165

    labels = (
        f"<text x='{pad_l + chart_w / 2:.2f}' y='{height-10}' text-anchor='middle' font-size='11' fill='#666'>step</text>"
        f"<text x='16' y='{pad_t + chart_h / 2:.2f}' font-size='11' fill='#666'>{'return % [L]' if y_format_left == 'percent' else 'value [L]'}</text>"
        f"<text x='{pad_l + chart_w + 36}' y='{pad_t + chart_h / 2:.2f}' font-size='11' fill='#666'>{'return % [R]' if y_format_right == 'percent' else 'value [R]'}</text>"
        f"<text x='{pad_l}' y='{pad_t - 8}' font-size='12' fill='#222'>{title}</text>"
        f"<text x='{pad_l}' y='{pad_t + 8}' font-size='11' fill='#555'>{subtitle}</text>"
        + "".join(legend)
    )

    lines = "".join(_line(name, primary_scaled[name], color) for name, color in primary_series)
    lines += "".join(_line(name, secondary_scaled[name], color) for name, color in secondary_series)

    marker_parts: list[str] = []
    for ann in annotations or []:
        idx = int(ann.get("index", 0))
        n = max(1, len(df) - 1)
        xx = pad_l + (max(0, min(idx, n)) / n) * chart_w
        value = float(ann.get("value", 0.0))
        axis = ann.get("axis", "left")
        yy = _y_scale(value, right_min, right_max) if axis == "right" else _y_scale(value, left_min, left_max)
        color = ann.get("color", "#111")
        label = ann.get("label", "")
        marker_parts.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='3.2' fill='{color}'/>")
        marker_parts.append(
            f"<text x='{xx + 6:.2f}' y='{yy - 6:.2f}' font-size='10' fill='{color}'>{label}</text>"
        )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{''.join(grid_parts)}{axes}{''.join(x_label_parts)}{''.join(y_label_parts)}{lines}{''.join(marker_parts)}{labels}"
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
    plot_frame = pd.DataFrame(
        [
            {
                "val_sharpe": h["val"].get("sharpe", 0.0),
                "val_final_equity": h["val"].get("final_equity", 0.0),
                "val_pnl": h["val"].get("pnl", 0.0),
                "val_turnover": h["val"].get("turnover", 0.0),
                "val_total_cost": h["val"].get("total_cost", 0.0),
                "val_cost_pnl_ratio": h["val"].get("cost_pnl_ratio", 0.0),
            }
            for h in history
        ]
    )
    if not plot_frame.empty:
        base_equity = float(plot_frame["val_final_equity"].iloc[0]) if float(plot_frame["val_final_equity"].iloc[0]) != 0 else 1.0
        plot_frame["val_return_pct"] = plot_frame["val_final_equity"].astype(float) / base_equity - 1.0
        plot_frame["val_pnl_pct"] = plot_frame["val_pnl"].astype(float) / base_equity

    subtitle = ""
    annotations: list[dict[str, Any]] = []
    if history:
        best_idx = max(range(len(history)), key=lambda i: float(history[i].get("val", {}).get("sharpe", 0.0)))
        last_idx = len(history) - 1
        best_val = history[best_idx].get("val", {})
        last_val = history[last_idx].get("val", {})
        subtitle = (
            f"FinalEq={float(last_val.get('final_equity', 0.0)):.2f}, "
            f"MDD={float(last_val.get('max_drawdown', 0.0)):.2%}, "
            f"Turnover={float(last_val.get('turnover', 0.0)):.4f}, "
            f"Sharpe={float(last_val.get('sharpe', 0.0)):.3f}"
        )
        annotations = [
            {
                "index": best_idx,
                "value": float(plot_frame["val_sharpe"].iloc[best_idx]) if not plot_frame.empty else 0.0,
                "axis": "right",
                "color": "#ff7f0e",
                "label": f"best@{int(history[best_idx].get('timesteps', 0))}",
            },
            {
                "index": last_idx,
                "value": float(plot_frame["val_sharpe"].iloc[last_idx]) if not plot_frame.empty else 0.0,
                "axis": "right",
                "color": "#2ca02c",
                "label": f"final@{int(history[last_idx].get('timesteps', 0))}",
            },
        ]
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports_dir / "learning_curve.csv", index=False)
    (reports_dir / "learning_curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (plots_dir / "learning_curve.svg").write_text(
        render_multi_line_svg(
            plot_frame,
            primary_series=[("val_return_pct", "#ff7f0e"), ("val_pnl_pct", "#9467bd")],
            secondary_series=[
                ("val_sharpe", "#ff7f0e"),
                ("val_turnover", "#bcbd22"),
                ("val_cost_pnl_ratio", "#8c564b"),
            ],
            title="Validation Metrics Learning Curve",
            subtitle=subtitle,
            annotations=annotations,
            y_format_left="percent",
        ),
        encoding="utf-8",
    )
