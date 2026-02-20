from __future__ import annotations

from pathlib import Path
import re

from src.coin_trading.pipelines.run_manager import (
    RUN_ID_FORMAT,
    make_run_id,
    write_data_manifest,
    write_feature_manifest,
    write_meta,
    write_train_manifest,
)


def test_run_manager_writes_manifests(tmp_path: Path):
    write_meta(tmp_path)
    write_data_manifest(tmp_path, {"symbol": "BTCUSDT"})
    write_feature_manifest(tmp_path, {"feature_set_version": "v1"})
    write_train_manifest(tmp_path, {"status": "ready"})

    assert (tmp_path / "artifacts" / "metadata.json").exists()
    assert (tmp_path / "data_manifest.json").exists()
    assert (tmp_path / "feature_manifest.json").exists()
    assert (tmp_path / "train_manifest.json").exists()



def test_make_run_id_default_format(monkeypatch):
    monkeypatch.setattr("src.coin_trading.pipelines.run_manager.git_sha", lambda: "abcdef123456")

    run_id = make_run_id()

    assert re.fullmatch(r"\d{8}_\d{6}Z_abcdef1", run_id)


def test_make_run_id_with_option(monkeypatch):
    monkeypatch.setattr("src.coin_trading.pipelines.run_manager.git_sha", lambda: "abcdef123456")

    run_id = make_run_id(option="train flow@night")

    assert run_id.endswith("__train_flow_night")
    assert RUN_ID_FORMAT == "<YYYYMMDD_HHMMSSZ>_<git_sha7>[__<option>]"
