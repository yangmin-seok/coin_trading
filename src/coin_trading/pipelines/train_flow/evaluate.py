from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.coin_trading.config.schema import AppConfig
from src.coin_trading.pipelines.train_flow.env import build_env


def _cost_report(trace: pd.DataFrame) -> dict[str, float]:
    filled = trace["filled_qty"].astype(float).abs() if "filled_qty" in trace else pd.Series(dtype=float)
    fee = trace["fee"].astype(float) if "fee" in trace else pd.Series(dtype=float)
    slippage = trace["slippage_cost"].astype(float) if "slippage_cost" in trace else pd.Series(dtype=float)
    turnover_total = float(filled.sum()) if not filled.empty else 0.0
    fee_slippage_total = float((fee + slippage).sum()) if not fee.empty or not slippage.empty else 0.0
    cost_per_turnover = fee_slippage_total / turnover_total if turnover_total > 0 else 0.0

    return {
        "turnover_total": turnover_total,
        "fee_slippage_total": fee_slippage_total,
        "cost_per_turnover": float(cost_per_turnover),
    }


def rollout_model(
    model: Any,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: AppConfig,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    eval_env = build_env(candles_df, features_df, cfg)
    obs, _ = eval_env.reset(seed=cfg.seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

    trace = eval_env.env.recorder.to_dataframe()
    if trace.empty:
        empty_cost = {"turnover_total": 0.0, "fee_slippage_total": 0.0, "cost_per_turnover": 0.0}
        return {
            "steps": 0,
            "final_equity": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "win_rate": 0.0,
            "total_cost": 0.0,
            "pnl": 0.0,
            "cost_pnl_ratio": 0.0,
            "cost_report": empty_cost,
        }

    reward = trace["reward"].astype(float)
    sharpe = 0.0
    if reward.std(ddof=0) > 0:
        sharpe = float((reward.mean() / reward.std(ddof=0)) * np.sqrt(252.0))

    cost_report = _cost_report(trace)
    total_cost = float(cost_report["fee_slippage_total"])
    pnl = float(trace["equity"].iloc[-1] - trace["equity"].iloc[0])
    cost_pnl_ratio = total_cost / abs(pnl) if abs(pnl) > 1e-12 else 0.0

    metrics = {
        "steps": int(len(trace)),
        "final_equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": float(trace["drawdown"].max()) if "drawdown" in trace else 0.0,
        "turnover": float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0,
        "win_rate": float((reward > 0).mean()),
        "total_cost": total_cost,
        "pnl": pnl,
        "cost_pnl_ratio": float(cost_pnl_ratio),
        "cost_report": cost_report,
    }
    if artifacts_dir is not None:
        files = eval_env.env.recorder.write_trace_artifacts(artifacts_dir)
        metrics["artifacts"] = {k: str(v) for k, v in files.items()}
    return metrics
