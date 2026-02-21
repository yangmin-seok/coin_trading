from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LEARNING_CURVE_SCHEMA = {
    "version": 2,
    "fields": {
        "timesteps": "int",
        "val_sharpe": "float",
        "val_max_drawdown": "float",
        "val_final_equity": "float",
        "loss": "float|null",
        "entropy_loss": "float|null",
        "value_loss": "float|null",
        "loss_collected": "bool",
    },
    "notes": {
        "loss_fields": "null means metric unavailable/uncollected (distinct from numeric 0.0)",
        "backward_compatibility": "v1 rows without loss_collected and nested train metrics remain supported",
    },
}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_learning_curve_row(history_item: dict[str, Any]) -> dict[str, Any]:
    train_metrics = history_item.get("train", {}) if isinstance(history_item.get("train", {}), dict) else {}
    # Backward-compatible parsing order: explicit train metrics -> legacy top-level keys.
    loss = _optional_float(train_metrics.get("loss"))
    if loss is None:
        loss = _optional_float(history_item.get("loss"))
    entropy_loss = _optional_float(train_metrics.get("entropy_loss"))
    if entropy_loss is None:
        entropy_loss = _optional_float(history_item.get("entropy_loss"))
    value_loss = _optional_float(train_metrics.get("value_loss"))
    if value_loss is None:
        value_loss = _optional_float(history_item.get("value_loss"))

    explicit_flag = train_metrics.get("loss_collected")
    if isinstance(explicit_flag, bool):
        loss_collected = explicit_flag
    else:
        loss_collected = all(metric is not None for metric in (loss, entropy_loss, value_loss))

    return {
        "timesteps": int(history_item.get("timesteps", 0)),
        "val_sharpe": float(history_item.get("val", {}).get("sharpe", 0.0)),
        "val_max_drawdown": float(history_item.get("val", {}).get("max_drawdown", 0.0)),
        "val_final_equity": float(history_item.get("val", {}).get("final_equity", 0.0)),
        "loss": loss,
        "entropy_loss": entropy_loss,
        "value_loss": value_loss,
        "loss_collected": loss_collected,
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
    width, height = 980, 420
    pad_l, pad_r, pad_b = 72, 28, 56

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

    # Name mapping for recurring series. Avoid broad substring rules like 'val' that collapse colors.
    fixed_series_style: dict[str, dict[str, str]] = {
        "cash_hold_return_pct": {"color": "#7f7f7f", "dash": "5,3"},
        "buy_hold_return_pct": {"color": "#8c564b", "dash": "3,2"},
        "drawdown": {"color": "#d62728", "dash": ""},
    }

    def _series_style(name: str, color: str) -> tuple[str, str]:
        if name in fixed_series_style:
            spec = fixed_series_style[name]
            return spec["color"], spec.get("dash", "")
        return color, ""

    legend: list[str] = []
    legend_items = [("L", name, color) for name, color in primary_series] + [("R", name, color) for name, color in secondary_series]
    legend_start_x = pad_l + 4
    legend_x = legend_start_x
    legend_y = 54
    legend_line_h = 16
    max_x = width - pad_r - 120
    for side, name, color in legend_items:
        line_color, _ = _series_style(name, color)
        label = f"[{side}] {name}"
        step = max(120, 8 * len(label) + 28)
        if legend_x > legend_start_x and legend_x + step > max_x:
            legend_x = legend_start_x
            legend_y += legend_line_h
        legend.append(f"<text x='{legend_x}' y='{legend_y}' font-size='11' fill='{line_color}'>{label}</text>")
        legend_x += step

    pad_t = legend_y + 20
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    def _y_scale(v: float, min_v: float, max_v: float) -> float:
        ratio = (v - min_v) / (max_v - min_v)
        return pad_t + (1 - ratio) * chart_h

    primary_scaled = {name: [_y_scale(v, left_min, left_max) for v in vals] for name, vals in primary_data.items()}
    secondary_scaled = {
        name: [_y_scale(v, right_min, right_max) for v in vals] for name, vals in secondary_data.items()
    }

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
        return f"<polyline fill='none' stroke='{line_color}' stroke-width='2'{dash_attr} points='{' '.join(pts)}'/>"

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
            f"<text x='{pad_l - 10}' y='{yy + 4:.2f}' text-anchor='end' font-size='11' fill='#666'>{_format_tick(y_left, y_format_left)}</text>"
        )
        y_label_parts.append(
            f"<text x='{pad_l + chart_w + 10}' y='{yy + 4:.2f}' text-anchor='start' font-size='11' fill='#666'>{_format_tick(y_right, y_format_right)}</text>"
        )

    for i in range(x_ticks):
        ratio = i / (x_ticks - 1) if x_ticks > 1 else 0.0
        xx = pad_l + ratio * chart_w
        step_val = int(round(ratio * x_steps))
        grid_parts.append(
            f"<line x1='{xx:.2f}' y1='{pad_t}' x2='{xx:.2f}' y2='{pad_t + chart_h}' stroke='#f2f2f2' stroke-width='1'/>"
        )
        x_label_parts.append(
            f"<text x='{xx:.2f}' y='{pad_t + chart_h + 20}' text-anchor='middle' font-size='11' fill='#666'>{step_val}</text>"
        )

    axes = (
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l + chart_w}' y1='{pad_t}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l}' y1='{pad_t + chart_h}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
    )

    labels = (
        f"<text x='{pad_l}' y='22' font-size='13' font-weight='600' fill='#222'>{title}</text>"
        f"<text x='{pad_l}' y='38' font-size='11' fill='#555'>{subtitle}</text>"
        + "".join(legend)
        + f"<text x='{pad_l + chart_w / 2:.2f}' y='{height - 12}' text-anchor='middle' font-size='11' fill='#666'>step</text>"
        + f"<text x='18' y='{pad_t + chart_h / 2:.2f}' font-size='11' fill='#666'>{'return % [L]' if y_format_left == 'percent' else 'value [L]'}</text>"
        + f"<text x='{pad_l + chart_w + 40}' y='{pad_t + chart_h / 2:.2f}' font-size='11' fill='#666'>{'return % [R]' if y_format_right == 'percent' else 'value [R]'}</text>"
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
        marker_parts.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='3.5' fill='{color}'/>")
        anchor_end = xx > (pad_l + chart_w * 0.78)
        tx = xx - 8 if anchor_end else xx + 8
        ty = max(pad_t + 14, min(yy - 8, pad_t + chart_h - 8))
        text_anchor = 'end' if anchor_end else 'start'
        marker_parts.append(
            f"<text x='{tx:.2f}' y='{ty:.2f}' text-anchor='{text_anchor}' font-size='10' font-weight='600' fill='{color}'>{label}</text>"
        )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{''.join(grid_parts)}{axes}{''.join(x_label_parts)}{''.join(y_label_parts)}{lines}{''.join(marker_parts)}{labels}"
        "</svg>"
    )


def render_summary_cards_svg(summary: dict[str, float], title: str = "Validation Summary") -> str:
    width, height = 980, 170
    cards = [
        ("Final Equity", summary.get("final_equity", 0.0), "{:.2f}", "#1f77b4"),
        ("Sharpe", summary.get("sharpe", 0.0), "{:.3f}", "#ff7f0e"),
        ("Max Drawdown", summary.get("max_drawdown", 0.0), "{:.2%}", "#d62728"),
        ("Turnover", summary.get("turnover", 0.0), "{:.4f}", "#2ca02c"),
        ("Cost/PNL", summary.get("cost_pnl_ratio", 0.0), "{:.4f}", "#9467bd"),
    ]
    gap = 12
    left = 20
    top = 42
    card_w = int((width - left * 2 - gap * (len(cards) - 1)) / len(cards))
    card_h = 94

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='{left}' y='24' font-size='14' font-weight='600' fill='#222'>{title}</text>",
    ]

    for i, (name, value, fmt, color) in enumerate(cards):
        x = left + i * (card_w + gap)
        parts.append(f"<rect x='{x}' y='{top}' width='{card_w}' height='{card_h}' rx='8' ry='8' fill='#fafafa' stroke='#dddddd'/>")
        parts.append(f"<rect x='{x}' y='{top}' width='4' height='{card_h}' fill='{color}'/>")
        parts.append(f"<text x='{x + 12}' y='{top + 28}' font-size='11' fill='#666'>{name}</text>")
        parts.append(f"<text x='{x + 12}' y='{top + 62}' font-size='20' font-weight='700' fill='{color}'>{fmt.format(float(value))}</text>")

    parts.append('</svg>')
    return ''.join(parts)


def write_learning_curve_artifacts(history: list[dict[str, Any]], reports_dir: Path, plots_dir: Path) -> None:
    rows = [_normalize_learning_curve_row(h) for h in history]
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
    (reports_dir / "learning_curve.schema.json").write_text(json.dumps(LEARNING_CURVE_SCHEMA, indent=2), encoding="utf-8")
    (plots_dir / "learning_curve.svg").write_text(
        render_multi_line_svg(
            plot_frame,
            primary_series=[("val_return_pct", "#1f77b4"), ("val_pnl_pct", "#9467bd")],
            secondary_series=[
                ("val_sharpe", "#ff7f0e"),
                ("val_turnover", "#2ca02c"),
                ("val_cost_pnl_ratio", "#d62728"),
            ],
            title="Validation Metrics Learning Curve",
            subtitle=subtitle,
            annotations=annotations,
            y_format_left="percent",
        ),
        encoding="utf-8",
    )

    last_val = history[-1].get("val", {}) if history else {}
    (plots_dir / "learning_curve_summary.svg").write_text(
        render_summary_cards_svg(last_val, title="Validation Summary (Last Evaluation)"),
        encoding="utf-8",
    )
