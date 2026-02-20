from __future__ import annotations

import json
from pathlib import Path

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train import (
    _build_walkforward_splits,
    _enforce_split_policy,
    _validate_no_lookahead,
    ensure_training_candles,
    run,
    summarize_dataset_for_training,
)

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train import (
    ensure_training_candles,
    preprocess_training_candles,
    run,
    summarize_dataset_for_training,
)

def test_summarize_dataset_for_training(sample_candles):
    cfg = load_config()
    summary = summarize_dataset_for_training(sample_candles, cfg)

    assert summary["rows"] == len(sample_candles)
    assert summary["coverage"]["start_open_time"] <= summary["coverage"]["end_open_time"]
    assert set(summary["splits"].keys()) == {"train", "val", "test"}
    assert summary["features"]["rows"] == len(sample_candles)
    assert 0.0 <= summary["features"]["nan_ratio_mean"] <= 1.0

def test_preprocess_training_candles_forbids_price_forward_fill(sample_candles):
    with pytest.raises(ValueError, match="Forward fill"):
        preprocess_training_candles(sample_candles, price_fill_method="ffill")

def test_preprocess_training_candles_raises_on_price_nan(sample_candles):
    df = sample_candles.copy()
    df.loc[df.index[0], "close"] = None
    with pytest.raises(ValueError, match="price columns"):
        preprocess_training_candles(df)

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
        assert model_train["folds"] >= 2
        assert len(model_train["walkforward"]) == model_train["folds"]
        for fold in model_train["walkforward"]:
            if fold.get("enabled"):
                fold_dir = run_dir / fold["fold_name"]
                assert (fold_dir / "learning_curve.csv").exists()
                assert (fold_dir / "learning_curve.json").exists()
                assert (fold_dir / "learning_curve.svg").exists()
                assert (fold_dir / "evaluation_metrics.json").exists()
                assert fold["lookahead_validation"]["passed"] is True
    else:
        assert model_train["enabled"] is False
        assert model_train["reason"] == "missing_dependencies"

    data_manifest = (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    assert '"bootstrap_generated": ' in data_manifest
    assert '"bootstrap_persisted": ' in data_manifest


def test_split_policy_and_walkforward_defaults(sample_candles):
    cfg = load_config()
    policy = _enforce_split_policy(cfg, sample_candles)
    assert set(policy["split"].keys()) == {"train", "val", "test"}
    splits = _build_walkforward_splits(cfg, sample_candles)
    assert len(splits) >= 2


def test_lookahead_validation_passes(sample_candles):
    result = _validate_no_lookahead(sample_candles)
    assert result["passed"] is True
