from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.coin_trading.pipelines.train_flow import orchestrator


def test_dependency_block_path_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_run_id: str,
    patched_meta,
):
    monkeypatch.chdir(tmp_path)

    def _raise_missing_deps(*_args, **_kwargs):
        raise RuntimeError("stable-baselines3/gymnasium is required")

    monkeypatch.setattr(orchestrator, "train_sb3", _raise_missing_deps)
    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    manifest = json.loads((run_dir / "artifacts" / "train_manifest.json").read_text(encoding="utf-8"))
    model_summary = json.loads((run_dir / "reports" / "model_train_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "blocked_missing_dependencies"
    assert model_summary["enabled"] is False
    assert model_summary["results"][0]["summary"]["reason"] == "missing_dependencies"
