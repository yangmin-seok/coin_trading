from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train_flow.data import ensure_training_candles, split_by_date, summarize_dataset


def test_summarize_dataset(sample_candles: pd.DataFrame):
    cfg = load_config()
    summary = summarize_dataset(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)
    assert 0.0 <= summary["features"]["nan_ratio_mean"] <= 1.0


@pytest.mark.parametrize(("start_idx", "end_idx"), [(10, 30)])
def test_split_by_date_filters_range(sample_candles: pd.DataFrame, start_idx: int, end_idx: int):
    dates = pd.to_datetime(sample_candles["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    start = dates.iloc[start_idx]
    end = dates.iloc[end_idx]

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
