from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train_flow.data import (
    build_walkforward_splits,
    ensure_training_candles,
    plan_walkforward_splits,
    split_by_date,
    summarize_dataset,
)
from src.coin_trading.pipelines.train_flow.env import build_env
from src.coin_trading.pipelines.train_flow.features import compute_features


def test_summarize_dataset(sample_candles: pd.DataFrame):
    cfg = load_config()
    summary = summarize_dataset(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)
    assert 0.0 <= summary["features"]["nan_ratio_mean"] <= 1.0


def test_split_by_date_filters_range(sample_candles: pd.DataFrame):
    dates = pd.to_datetime(sample_candles["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    start = dates.iloc[10]
    end = dates.iloc[30]

    split = split_by_date(sample_candles, (start, end))

    assert len(split) > 0
    split_dates = pd.to_datetime(split["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    assert split_dates.min() >= start
    assert split_dates.max() <= end


def test_ensure_training_candles_bootstraps_when_missing(tmp_path: Path):
    cfg = load_config()
    candles, bootstrapped, persisted = ensure_training_candles(cfg, data_root=tmp_path)

    assert bootstrapped is True
    assert len(candles) > 0
    assert persisted in {True, False}


def test_build_env_reflects_execution_and_reward_config(sample_candles: pd.DataFrame):
    cfg = load_config()
    cfg.seed = 11
    cfg.train.seed = 29
    cfg.execution.fee_rate = 0.0025
    cfg.execution.slippage_bps = 4.0
    cfg.execution.max_step_change = 0.15
    cfg.execution.min_delta = 0.03
    cfg.reward.lambda_turnover = 0.005
    cfg.reward.lambda_dd = 0.25
    cfg.reward.dd_limit = 0.12

    env = build_env(sample_candles, compute_features(sample_candles), cfg)

    assert env.env.execution_model.fee_rate == cfg.execution.fee_rate
    assert env.env.execution_model.slippage_bps == cfg.execution.slippage_bps
    assert env.env.execution_model.max_step_change == cfg.execution.max_step_change
    assert env.env.execution_model.min_delta == cfg.execution.min_delta
    assert env.env.lambda_turnover == cfg.reward.lambda_turnover
    assert env.env.lambda_dd == cfg.reward.lambda_dd
    assert env.env.dd_limit == cfg.reward.dd_limit
    assert env._seed == cfg.train.seed


def test_build_walkforward_splits_accepts_custom_step_days(sample_candles: pd.DataFrame):
    _ = sample_candles
    split = {
        "train": ("2022-01-01", "2024-12-31"),
        "val": ("2025-01-01", "2025-06-30"),
        "test": ("2025-07-01", "2025-12-31"),
    }

    empty_df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    default_splits = build_walkforward_splits(empty_df, split, target_runs=3)
    custom_step_splits = build_walkforward_splits(empty_df, split, target_runs=3, step_days=30)

    assert len(custom_step_splits) >= len(default_splits)


def test_plan_walkforward_splits_reports_shortage_reason(sample_candles: pd.DataFrame):
    _ = sample_candles
    split = {
        "train": ("2022-01-01", "2024-12-31"),
        "val": ("2025-01-01", "2025-06-30"),
        "test": ("2025-07-01", "2025-12-31"),
    }

    empty_df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    plan = plan_walkforward_splits(empty_df, split, target_runs=3, min_folds=3)

    assert len(plan["splits"]) >= 1
    assert "data_coverage" in plan["policy"]
    assert plan["policy"]["desired_runs"] == 3
    if plan["policy"]["actual_runs"] < plan["policy"]["desired_runs"]:
        assert plan["policy"]["insufficient_reason"] is not None
