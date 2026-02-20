from __future__ import annotations

import math


def compute_reward_components(
    equity_t: float,
    equity_prev: float,
    turnover: float,
    drawdown: float,
    lambda_turnover: float,
    lambda_dd: float,
    dd_limit: float,
) -> tuple[float, float, float, float]:
    if equity_t <= 0 or equity_prev <= 0:
        raise ValueError("equity must be positive")
    pnl = math.log(equity_t / equity_prev)
    cost = lambda_turnover * turnover
    penalty = lambda_dd * max(0.0, drawdown - dd_limit)
    reward = pnl - cost - penalty
    return reward, pnl, cost, penalty


def compute_reward(equity_t: float, equity_prev: float, turnover: float, drawdown: float, lambda_turnover: float, lambda_dd: float, dd_limit: float) -> float:
    reward, _, _, _ = compute_reward_components(equity_t, equity_prev, turnover, drawdown, lambda_turnover, lambda_dd, dd_limit)
    return reward
