from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.coin_trading.pipelines.train_flow import orchestrator


def test_train_manifest_created_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_run_id: str,
    patched_meta,
    patched_train_sb3,
):
    monkeypatch.chdir(tmp_path)
    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    assert (run_dir / "artifacts" / "train_manifest.json").exists()
    manifest = json.loads((run_dir / "artifacts" / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {"ready", "blocked_missing_dependencies", "blocked_no_training_data"}
