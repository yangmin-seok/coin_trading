from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train_flow.data import (
    build_walkforward_splits,
    compute_walkforward_capacity,
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


def test_compute_walkforward_capacity_counts_possible_runs(sample_candles: pd.DataFrame):
    cfg = load_config()
    split = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}

    capacity = compute_walkforward_capacity(pd.DataFrame(), split)

    assert capacity["possible_runs"] >= 1
    assert capacity["step_days"] >= 1
    assert capacity["base_test_end"] == cfg.split.test[1]
