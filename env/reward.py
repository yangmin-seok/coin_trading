from __future__ import annotations

import math
from typing import Any


def _drawdown_penalty(drawdown: float, lambda_dd: float, dd_limit: float) -> float:
    return lambda_dd * max(0.0, drawdown - dd_limit)


def compute_reward_components(
    equity_t: float,
    equity_prev: float,
    turnover: float,
    drawdown: float,
    *,
    reward_type: str = "log_return_regularized",
    lambda_turnover: float = 0.001,
    lambda_dd: float = 0.1,
    dd_limit: float = 0.2,
    inactivity_threshold: float = 1e-4,
    inactivity_penalty: float = 0.0,
    target_position_utilization: float = 0.15,
    lambda_under_utilization: float = 0.0,
    dsr_beta: float = 0.05,
    dsr_scale: float = 1.0,
    downside_beta: float = 0.05,
    lambda_downside: float = 0.0,
    current_position_abs: float = 0.0,
    state: dict[str, Any] | None = None,
) -> tuple[float, float, float, float, dict[str, float]]:
    if equity_t <= 0 or equity_prev <= 0:
        raise ValueError("equity must be positive")

    pnl = math.log(equity_t / equity_prev)
    cost = lambda_turnover * turnover
    dd_penalty = _drawdown_penalty(drawdown, lambda_dd, dd_limit)

    if reward_type == "log_return_regularized":
        position_abs = min(1.0, max(0.0, current_position_abs))
        under_utilization = max(0.0, target_position_utilization - position_abs)
        inactivity = inactivity_penalty if turnover < inactivity_threshold else 0.0
        extra_penalty = lambda_under_utilization * under_utilization + inactivity
        reward = pnl - cost - dd_penalty - extra_penalty
        penalties = {
            "drawdown": dd_penalty,
            "inactivity": inactivity,
            "under_utilization": lambda_under_utilization * under_utilization,
            "downside": 0.0,
        }
        return reward, pnl, cost, dd_penalty + extra_penalty, penalties

    if reward_type == "differential_sharpe":
        reward_state = state if state is not None else {}
        prev_mean = float(reward_state.get("mean_return", 0.0))
        prev_var = max(float(reward_state.get("var_return", 1e-8)), 1e-8)
        diff_sharpe = (pnl - prev_mean) / math.sqrt(prev_var + 1e-8)

        new_mean = (1.0 - dsr_beta) * prev_mean + dsr_beta * pnl
        centered = pnl - new_mean
        new_var = (1.0 - dsr_beta) * prev_var + dsr_beta * (centered * centered)
        reward_state["mean_return"] = new_mean
        reward_state["var_return"] = max(new_var, 1e-8)

        reward = dsr_scale * diff_sharpe - cost - dd_penalty
        penalties = {
            "drawdown": dd_penalty,
            "inactivity": 0.0,
            "under_utilization": 0.0,
            "downside": 0.0,
        }
        return reward, pnl, cost, dd_penalty, penalties

    if reward_type == "downside_risk":
        reward_state = state if state is not None else {}
        prev_downside_sq = float(reward_state.get("downside_sq_ema", 0.0))
        downside = max(0.0, -pnl)
        downside_sq_ema = (1.0 - downside_beta) * prev_downside_sq + downside_beta * (downside * downside)
        reward_state["downside_sq_ema"] = downside_sq_ema

        downside_vol = math.sqrt(max(downside_sq_ema, 1e-10))
        downside_penalty = lambda_downside * downside_vol
        reward = pnl - cost - dd_penalty - downside_penalty
        penalties = {
            "drawdown": dd_penalty,
            "inactivity": 0.0,
            "under_utilization": 0.0,
            "downside": downside_penalty,
        }
        return reward, pnl, cost, dd_penalty + downside_penalty, penalties

    raise ValueError(f"unsupported reward_type: {reward_type}")


def compute_reward(
    equity_t: float,
    equity_prev: float,
    turnover: float,
    drawdown: float,
    **kwargs: Any,
) -> float:
    reward, _, _, _, _ = compute_reward_components(equity_t, equity_prev, turnover, drawdown, **kwargs)
    return reward
