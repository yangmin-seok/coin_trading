from __future__ import annotations

from pathlib import Path

import pandas as pd


class StepRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, info: dict) -> None:
        self.rows.append(info.copy())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def write_trace_artifacts(self, out_dir: str | Path) -> dict[str, Path]:
        """Write trace artifacts for quick inspection (CSV + multiple SVG charts)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe().copy()
        if df.empty:
            csv_path = out / "trace.csv"
            csv_path.write_text("", encoding="utf-8")
            reward_equity_path = out / "reward_equity.svg"
            reward_equity_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='320'></svg>", encoding="utf-8")
            drawdown_turnover_path = out / "drawdown_turnover.svg"
            drawdown_turnover_path.write_text(
                "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='320'></svg>",
                encoding="utf-8",
            )
            action_position_path = out / "action_position.svg"
            action_position_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='320'></svg>", encoding="utf-8")
            costs_path = out / "costs.svg"
            costs_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='320'></svg>", encoding="utf-8")
            return {
                "csv": csv_path,
                "svg": reward_equity_path,
                "reward_equity_svg": reward_equity_path,
                "drawdown_turnover_svg": drawdown_turnover_path,
                "action_position_svg": action_position_path,
                "costs_svg": costs_path,
            }

        if "filled_qty" in df.columns:
            def _signal(v: float) -> str:
                if v > 0:
                    return "buy"
                if v < 0:
                    return "sell"
                return "hold"

            df["signal"] = df["filled_qty"].fillna(0.0).map(_signal)

        df.insert(0, "step", range(len(df)))
        csv_path = out / "trace.csv"
        df.to_csv(csv_path, index=False)

        reward_equity_path = out / "reward_equity.svg"
        reward_equity_path.write_text(
            _render_multi_line_svg(
                df,
                series=[("reward", "#1f77b4"), ("equity", "#2ca02c")],
                title="Reward / Equity",
            ),
            encoding="utf-8",
        )
        drawdown_turnover_path = out / "drawdown_turnover.svg"
        drawdown_turnover_path.write_text(
            _render_multi_line_svg(
                df,
                series=[("drawdown", "#d62728"), ("action_effective_pos", "#9467bd")],
                title="Drawdown / Effective Position",
            ),
            encoding="utf-8",
        )
        action_position_path = out / "action_position.svg"
        action_position_path.write_text(
            _render_multi_line_svg(
                df,
                series=[("action_target_pos", "#ff7f0e"), ("action_effective_pos", "#17becf")],
                title="Target vs Effective Position",
            ),
            encoding="utf-8",
        )
        costs_path = out / "costs.svg"
        costs_path.write_text(
            _render_multi_line_svg(
                df,
                series=[("fee", "#8c564b"), ("slippage_cost", "#e377c2")],
                title="Trading Costs",
            ),
            encoding="utf-8",
        )
        return {
            "csv": csv_path,
            "svg": reward_equity_path,
            "reward_equity_svg": reward_equity_path,
            "drawdown_turnover_svg": drawdown_turnover_path,
            "action_position_svg": action_position_path,
            "costs_svg": costs_path,
        }


def _render_multi_line_svg(df: pd.DataFrame, series: list[tuple[str, str]], title: str) -> str:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 70, 20, 20, 45
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    y_data = {
        name: [float(v) for v in df.get(name, pd.Series([0.0] * len(df))).fillna(0.0).tolist()]
        for name, _ in series
    }

    def _scale(vals: list[float], invert: bool = False) -> list[float]:
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            vmax = vmin + 1.0
        out: list[float] = []
        for v in vals:
            ratio = (v - vmin) / (vmax - vmin)
            y = pad_t + (1 - ratio) * chart_h if invert else pad_t + ratio * chart_h
            out.append(y)
        return out

    scaled = {name: _scale(vals, invert=True) for name, vals in y_data.items()}

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
    ticks = "".join(
        f"<text x='{pad_l-8}' y='{pad_t + i * chart_h / 4:.1f}' text-anchor='end' font-size='10' fill='#666'>{4-i}</text>"
        for i in range(5)
    )

    legend = []
    for i, (name, color) in enumerate(series):
        legend.append(f"<text x='{55 + i * 120}' y='18' font-size='12' fill='{color}'>{name}</text>")
    labels = f"<text x='{pad_l + chart_w/2:.1f}' y='{height-8}' text-anchor='middle' font-size='11' fill='#666'>step index (x-axis)</text>"
    labels += f"<text x='14' y='{pad_t + chart_h/2:.1f}' font-size='11' fill='#666' transform='rotate(-90 14,{pad_t + chart_h/2:.1f})'>normalized metric value (y-axis)</text>"
    labels += "".join(legend)
    labels += f"<text x='{pad_l}' y='{pad_t + 12}' font-size='12' fill='#222'>{title}</text>"

    lines = "".join(_line(scaled[name], color) for name, color in series)

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{guides}{ticks}{lines}{labels}"
        "</svg>"
    )
