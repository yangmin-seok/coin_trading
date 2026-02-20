from __future__ import annotations

from pathlib import Path

from src.coin_trading.pipelines.run_manager import write_data_manifest, write_feature_manifest, write_meta, write_train_manifest


def test_run_manager_writes_manifests(tmp_path: Path):
    write_meta(tmp_path)
    write_data_manifest(tmp_path, {"symbol": "BTCUSDT"})
    write_feature_manifest(tmp_path, {"feature_set_version": "v1"})
    write_train_manifest(tmp_path, {"status": "ready"})

    assert (tmp_path / "artifacts" / "metadata.json").exists()
    assert (tmp_path / "data_manifest.json").exists()
    assert (tmp_path / "feature_manifest.json").exists()
    assert (tmp_path / "train_manifest.json").exists()
