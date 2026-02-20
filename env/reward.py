from __future__ import annotations

import math


def compute_reward(equity_t: float, equity_prev: float, turnover: float, drawdown: float, lambda_turnover: float, lambda_dd: float, dd_limit: float) -> float:
    if equity_t <= 0 or equity_prev <= 0:
        raise ValueError("equity must be positive")
    reward = math.log(equity_t / equity_prev)
    reward -= lambda_turnover * turnover
    reward -= lambda_dd * max(0.0, drawdown - dd_limit)
    return reward
