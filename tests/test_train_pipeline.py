from __future__ import annotations

import json
from pathlib import Path

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train import ensure_training_candles, run, summarize_dataset_for_training


def test_summarize_dataset_for_training(sample_candles):
    cfg = load_config()
    summary = summarize_dataset_for_training(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)
    assert 0.0 <= summary["features"]["nan_ratio_mean"] <= 1.0


def test_ensure_training_candles_bootstraps_when_missing(tmp_path: Path):
    cfg = load_config()
    candles, bootstrapped, persisted = ensure_training_candles(cfg, data_root=tmp_path)
    assert bootstrapped is True
    assert len(candles) > 0
    assert sorted(candles.columns.tolist()) == sorted(["open_time", "open", "high", "low", "close", "volume", "close_time"])
    assert persisted in {True, False}


def test_train_run_writes_dependency_block_or_training_artifacts():
    run_id = run()
    run_dir = Path("runs") / run_id
    assert run_dir.exists()


    assert (run_dir / "plots").exists()
    assert (run_dir / "reports").exists()
    assert (run_dir / "artifacts").exists()
    assert (run_dir / "artifacts" / "config.yaml").exists()
    metadata = json.loads((run_dir / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    assert "seed" in metadata
    assert "git_sha" in metadata
    assert "start_time_utc" in metadata
    assert set(metadata["data_range"].keys()) == {"train", "val", "test"}
    train_manifest = json.loads((run_dir / "train_manifest.json").read_text(encoding="utf-8"))
    model_train = json.loads((run_dir / "reports" / "model_train_summary.json").read_text(encoding="utf-8"))

    assert train_manifest["status"] in {"ready", "blocked_missing_dependencies"}

    if train_manifest["status"] == "ready":
        assert model_train["enabled"] is True
        assert model_train["model"] in {"SB3-PPO", "SB3-SAC"}
        assert (run_dir / "plots" / "learning_curve.csv").exists()
        assert (run_dir / "plots" / "learning_curve.json").exists()
        assert (run_dir / "plots" / "learning_curve.svg").exists()
        assert (run_dir / "reports" / "metrics.json").exists()
        assert (run_dir / "artifacts" / "model.zip").exists()
        assert (run_dir / "reports" / "val_trace" / "reward_equity.svg").exists()
        assert (run_dir / "reports" / "test_trace" / "reward_equity.svg").exists()
    else:
        assert model_train["enabled"] is False
        assert model_train["reason"] == "missing_dependencies"

    data_manifest = (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    assert '"bootstrap_generated": ' in data_manifest
    assert '"bootstrap_persisted": ' in data_manifest
