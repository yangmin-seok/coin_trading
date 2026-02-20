from __future__ import annotations

import json
from pathlib import Path

from config.loader import load_config
from pipelines.train import ensure_training_candles, run, run_training_probe, summarize_dataset_for_training


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


def test_training_probe_writes_reward_artifacts(sample_candles, tmp_path: Path):
    summary = run_training_probe(sample_candles, tmp_path)
    assert summary["enabled"] is True
    assert summary["epochs"] == 1
    assert summary["model"] == "VolTarget-baseline"

    trace_path = tmp_path / summary["artifacts"]["trace_csv"]
    svg_path = tmp_path / summary["artifacts"]["reward_equity_svg"]
    assert trace_path.exists()
    assert svg_path.exists()


def test_train_run_creates_ready_manifest_with_bootstrap():
    run_id = run()
    run_dir = Path("runs") / run_id
    assert run_dir.exists()

    train_manifest = json.loads((run_dir / "train_manifest.json").read_text(encoding="utf-8"))
    assert train_manifest["status"] == "ready"
    assert train_manifest["epochs"] == 1
    assert train_manifest["model"] == "VolTarget-baseline"
    assert "probe" in train_manifest

    data_manifest = (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    assert '"bootstrap_generated": ' in data_manifest
    assert '"bootstrap_persisted": ' in data_manifest
