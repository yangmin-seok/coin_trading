from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coin_trading.config.schema import AppConfig
from src.coin_trading.pipelines.train_flow.orchestrator import run


def ensure_training_candles(cfg: AppConfig, data_root: Path = Path("data/processed")) -> tuple[pd.DataFrame, bool, bool]:
    raise RuntimeError(
        "ensure_training_candles is deprecated in pipelines.train; use src.coin_trading.pipelines.train_flow.data.ensure_training_candles"
    )


def summarize_dataset_for_training(candles_df: pd.DataFrame, cfg: AppConfig) -> dict:
    raise RuntimeError(
        "summarize_dataset_for_training is deprecated in pipelines.train; use src.coin_trading.pipelines.train_flow.data.summarize_dataset"
    )


if __name__ == "__main__":
    print(run())
