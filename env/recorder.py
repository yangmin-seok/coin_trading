from __future__ import annotations

from pathlib import Path
import base64

import pandas as pd

from src.coin_trading.report.plotting import render_multi_line_svg


def _write_fallback_png(path: Path) -> None:
    tiny_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8t2b8AAAAASUVORK5CYII=")
    path.write_bytes(tiny_png)


def _render_reward_components_png(df: pd.DataFrame, out_path: Path) -> None:
    """Render reward components chart to PNG, with fallback when matplotlib is unavailable."""
    required = ["reward_pnl", "reward_cost", "reward_penalty", "reward"]
    present = [c for c in required if c in df.columns]
    if not present:
        _write_fallback_png(out_path)
        return

    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        _write_fallback_png(out_path)
        return

    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120)
        x = df["step"] if "step" in df.columns else range(len(df))
        colors = {
            "reward_pnl": "#1f77b4",
            "reward_cost": "#d62728",
            "reward_penalty": "#9467bd",
            "reward": "#2ca02c",
        }
        for col in present:
            ax.plot(x, df[col].fillna(0.0).astype(float), label=col, color=colors.get(col, None), linewidth=1.8)

        ax.set_title("Reward Components")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, format="png")
        plt.close(fig)
    except Exception:
        _write_fallback_png(out_path)


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
            reward_components_path = out / "reward_components_timeseries.png"
            _write_fallback_png(reward_components_path)
            return {
                "csv": csv_path,
                "svg": reward_equity_path,
                "reward_equity_svg": reward_equity_path,
                "drawdown_turnover_svg": drawdown_turnover_path,
                "action_position_svg": action_position_path,
                "costs_svg": costs_path,
                "reward_components_timeseries_png": reward_components_path,
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
            render_multi_line_svg(
                df,
                series=[("reward", "#1f77b4"), ("equity", "#2ca02c")],
                title="Reward / Equity",
            ),
            encoding="utf-8",
        )
        drawdown_turnover_path = out / "drawdown_turnover.svg"
        drawdown_turnover_path.write_text(
            render_multi_line_svg(
                df,
                series=[("drawdown", "#d62728"), ("action_effective_pos", "#9467bd")],
                title="Drawdown / Effective Position",
            ),
            encoding="utf-8",
        )
        action_position_path = out / "action_position.svg"
        action_position_path.write_text(
            render_multi_line_svg(
                df,
                series=[("action_target_pos", "#ff7f0e"), ("action_effective_pos", "#17becf")],
                title="Target vs Effective Position",
            ),
            encoding="utf-8",
        )
        costs_path = out / "costs.svg"
        costs_path.write_text(
            render_multi_line_svg(
                df,
                series=[("fee", "#8c564b"), ("slippage_cost", "#e377c2")],
                title="Trading Costs",
            ),
            encoding="utf-8",
        )
        reward_components_path = out / "reward_components_timeseries.png"
        _render_reward_components_png(df, reward_components_path)
        return {
            "csv": csv_path,
            "svg": reward_equity_path,
            "reward_equity_svg": reward_equity_path,
            "drawdown_turnover_svg": drawdown_turnover_path,
            "action_position_svg": action_position_path,
            "costs_svg": costs_path,
            "reward_components_timeseries_png": reward_components_path,
        }

