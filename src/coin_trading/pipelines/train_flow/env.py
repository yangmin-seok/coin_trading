from __future__ import annotations

import pandas as pd

from env.execution_model import ExecutionModel
from src.coin_trading.agents.sb3_env import GymTradingEnv
from src.coin_trading.config.schema import AppConfig


def build_env(candles_df: pd.DataFrame, features_df: pd.DataFrame, cfg: AppConfig) -> GymTradingEnv:
    return GymTradingEnv(candles_df, features_df, ExecutionModel(), seed=cfg.seed)
