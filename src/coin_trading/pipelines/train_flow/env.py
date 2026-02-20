from __future__ import annotations

import pandas as pd

from env.execution_model import ExecutionModel
from src.coin_trading.agents.sb3_env import GymTradingEnv
from src.coin_trading.config.schema import AppConfig


def build_env(candles_df: pd.DataFrame, features_df: pd.DataFrame, cfg: AppConfig) -> GymTradingEnv:
    return GymTradingEnv(
        candles_df,
        features_df,
        ExecutionModel(
            fee_rate=cfg.execution.fee_rate,
            slippage_bps=cfg.execution.slippage_bps,
            max_step_change=cfg.execution.max_step_change,
            min_delta=cfg.execution.min_delta,
        ),
        seed=cfg.seed,
        lambda_turnover=cfg.reward.lambda_turnover,
        lambda_dd=cfg.reward.lambda_dd,
        dd_limit=cfg.reward.dd_limit,
    )
