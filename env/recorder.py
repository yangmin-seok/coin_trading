from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coin_trading.report.plotting import render_multi_line_svg


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
        return {
            "csv": csv_path,
            "svg": reward_equity_path,
            "reward_equity_svg": reward_equity_path,
            "drawdown_turnover_svg": drawdown_turnover_path,
            "action_position_svg": action_position_path,
            "costs_svg": costs_path,
        }

