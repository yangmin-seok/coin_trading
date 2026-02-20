from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train_flow.data import (
    build_walkforward_splits,
    ensure_training_candles,
    summarize_dataset,
    validate_split_policy,
)
from src.coin_trading.pipelines.train_flow.features import (
    compute_features,
    fit_feature_scaler,
    transform_with_scaler,
    validate_rolling_features_no_lookahead,
)




def _multi_day_candles(days: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(days):
        open_time = i * 86_400_000
        rows.append({
            "open_time": open_time,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 10.0,
            "close_time": open_time + 86_399_999,
        })
    return pd.DataFrame(rows)


def _sample_split() -> dict[str, tuple[str, str]]:
    return {
        "train": ("1970-01-01", "1970-01-03"),
        "val": ("1970-01-04", "1970-01-05"),
        "test": ("1970-01-06", "1970-01-07"),
    }


def test_summarize_dataset_for_training(sample_candles):
    cfg = load_config()
    summary = summarize_dataset(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)


def test_ensure_training_candles_bootstraps_when_missing(tmp_path: Path):
    cfg = load_config()
    candles, bootstrapped, persisted = ensure_training_candles(cfg, data_root=tmp_path)
    assert bootstrapped is True
    assert len(candles) > 0
    assert persisted in {True, False}


def test_split_policy_validation_and_walkforward(sample_candles):
    candles = _multi_day_candles()
    policy = validate_split_policy(_sample_split(), candles, min_days={"train": 2, "val": 1, "test": 1})
    assert policy["ordered_non_overlapping"] is True

    splits = build_walkforward_splits(candles, _sample_split(), target_runs=3, min_days={"train": 2, "val": 1, "test": 1})
    assert len(splits) >= 2


def test_split_policy_rejects_overlap(sample_candles):
    candles = _multi_day_candles()
    split = _sample_split()
    split["val"] = ("1970-01-03", "1970-01-05")
    with pytest.raises(ValueError, match="overlap"):
        validate_split_policy(split, candles, min_days={"train": 2, "val": 1, "test": 1})


def test_lookahead_validation_and_scaler_policy(sample_candles):
    train_df = sample_candles.iloc[:80].reset_index(drop=True)
    val_df = sample_candles.iloc[80:100].reset_index(drop=True)

    train_features = compute_features(train_df)
    val_features = compute_features(val_df)

    assert validate_rolling_features_no_lookahead(train_df, train_features)["passed"] is True

    scaler = fit_feature_scaler(train_features, split_name="train")
    scaled_train = transform_with_scaler(train_features, scaler)
    scaled_val = transform_with_scaler(val_features, scaler)
    assert isinstance(scaled_train, pd.DataFrame)
    assert isinstance(scaled_val, pd.DataFrame)

    with pytest.raises(ValueError, match="train split"):
        fit_feature_scaler(train_features, split_name="val")
