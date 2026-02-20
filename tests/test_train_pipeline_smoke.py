from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.coin_trading.pipelines.train_flow import orchestrator




@pytest.fixture
def patched_meta(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orchestrator, "write_meta", lambda run_dir: (run_dir / "meta.json").write_text("{}", encoding="utf-8"))




@pytest.fixture
def patched_train_sb3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestrator,
        "train_sb3",
        lambda *_args, **_kwargs: {"enabled": False, "reason": "insufficient_split_rows"},
    )


@pytest.fixture
def fixed_run_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orchestrator, "make_run_id", lambda *args, **kwargs: "smoke_train_run")
    return "smoke_train_run"


def test_train_entry_execution_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta, patched_train_sb3):
    monkeypatch.chdir(tmp_path)
    run_id = orchestrator.run()

    assert run_id == fixed_run_id
    assert (tmp_path / "runs" / fixed_run_id).exists()
    assert (tmp_path / "runs" / fixed_run_id / "plots").exists()
    assert (tmp_path / "runs" / fixed_run_id / "reports").exists()
    assert (tmp_path / "runs" / fixed_run_id / "artifacts").exists()


def test_train_manifest_created_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta, patched_train_sb3):
    monkeypatch.chdir(tmp_path)
    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    assert (run_dir / "artifacts" / "train_manifest.json").exists()
    manifest = json.loads((run_dir / "artifacts" / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {"ready", "blocked_missing_dependencies", "blocked_no_training_data"}


def test_dependency_block_path_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta):
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


def test_train_module_run_returns_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.coin_trading.pipelines import train
    from src.coin_trading.pipelines.train_flow import orchestrator as train_orchestrator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_orchestrator, "make_run_id", lambda *args, **kwargs: "entry_run_id")
    monkeypatch.setattr(
        train_orchestrator,
        "write_meta",
        lambda run_dir: (run_dir / "meta.json").write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        train_orchestrator,
        "train_sb3",
        lambda *_args, **_kwargs: {"enabled": False, "reason": "insufficient_split_rows"},
    )

    run_id = train.run()

    assert run_id == "entry_run_id"
