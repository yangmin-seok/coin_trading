from __future__ import annotations

from config.loader import load_config
from pipelines.train import summarize_dataset_for_training


def test_summarize_dataset_for_training(sample_candles):
    cfg = load_config()
    summary = summarize_dataset_for_training(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)
    assert 0.0 <= summary["features"]["nan_ratio_mean"] <= 1.0
