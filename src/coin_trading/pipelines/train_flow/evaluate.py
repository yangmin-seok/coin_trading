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


def _baseline_equities(candles_df: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    if candles_df.empty or "open" not in candles_df:
        return {"cash_hold": initial_equity, "buy_hold": initial_equity}
    open_prices = candles_df["open"].astype(float).reset_index(drop=True)
    if open_prices.empty or open_prices.iloc[0] <= 0:
        return {"cash_hold": initial_equity, "buy_hold": initial_equity}
    buy_hold = initial_equity * float(open_prices.iloc[-1] / open_prices.iloc[0])
    return {"cash_hold": initial_equity, "buy_hold": buy_hold}


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
            "trade_count": 0,
            "no_trade_ratio": 1.0,
            "avg_abs_position": 0.0,
            "baseline_equity": {"cash_hold": 0.0, "buy_hold": 0.0},
            "excess_vs_baseline": {"cash_hold": 0.0, "buy_hold": 0.0},
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
    trade_count = int((trace["filled_qty"].astype(float).abs() > 1e-12).sum()) if "filled_qty" in trace else 0
    no_trade_ratio = float((trace["no_trade"].astype(float)).mean()) if "no_trade" in trace else 0.0
    avg_abs_position = float(trace["position_abs"].astype(float).mean()) if "position_abs" in trace else 0.0

    final_equity = float(trace["equity"].iloc[-1])
    baseline_equity = _baseline_equities(candles_df, float(trace["equity"].iloc[0]))
    excess_vs_baseline = {
        "cash_hold": final_equity - float(baseline_equity["cash_hold"]),
        "buy_hold": final_equity - float(baseline_equity["buy_hold"]),
    }

    metrics = {
        "steps": int(len(trace)),
        "final_equity": final_equity,
        "sharpe": sharpe,
        "max_drawdown": float(trace["drawdown"].max()) if "drawdown" in trace else 0.0,
        "turnover": float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0,
        "win_rate": float((reward > 0).mean()),
        "total_cost": total_cost,
        "pnl": pnl,
        "cost_pnl_ratio": float(cost_pnl_ratio),
        "trade_count": trade_count,
        "no_trade_ratio": no_trade_ratio,
        "avg_abs_position": avg_abs_position,
        "baseline_equity": baseline_equity,
        "excess_vs_baseline": excess_vs_baseline,
        "cost_report": cost_report,
    }
    if artifacts_dir is not None:
        files = eval_env.env.recorder.write_trace_artifacts(artifacts_dir)
        metrics["artifacts"] = {k: str(v) for k, v in files.items()}
    return metrics
