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
        """Write trace artifacts for quick inspection (CSV + simple SVG chart)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe().copy()
        if df.empty:
            csv_path = out / "trace.csv"
            csv_path.write_text("", encoding="utf-8")
            svg_path = out / "reward_equity.svg"
            svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='320'></svg>", encoding="utf-8")
            return {"csv": csv_path, "svg": svg_path}

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

        svg_path = out / "reward_equity.svg"
        svg_path.write_text(_render_reward_equity_svg(df), encoding="utf-8")
        return {"csv": csv_path, "svg": svg_path}


def _render_reward_equity_svg(df: pd.DataFrame) -> str:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    x = list(range(len(df)))
    rewards = [float(v) for v in df.get("reward", pd.Series([0.0] * len(df))).fillna(0.0).tolist()]
    equities = [float(v) for v in df.get("equity", pd.Series([0.0] * len(df))).fillna(0.0).tolist()]

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

    ys_reward = _scale(rewards, invert=True)
    ys_equity = _scale(equities, invert=True)

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

    labels = (
        "<text x='55' y='18' font-size='12' fill='#1f77b4'>reward</text>"
        "<text x='120' y='18' font-size='12' fill='#2ca02c'>equity</text>"
        f"<text x='{pad_l}' y='{height-8}' font-size='11' fill='#666'>step</text>"
    )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"{guides}{_line(ys_reward, '#1f77b4')}{_line(ys_equity, '#2ca02c')}{labels}"
        "</svg>"
    )
