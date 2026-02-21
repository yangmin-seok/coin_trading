from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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
    monkeypatch.setattr(orchestrator, "make_run_id", lambda: "smoke_train_run")
    return "smoke_train_run"


def test_train_entry_execution_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta, patched_train_sb3):
    monkeypatch.chdir(tmp_path)
    run_id = orchestrator.run()

    assert run_id == fixed_run_id
    assert (tmp_path / "runs" / fixed_run_id).exists()
    assert (tmp_path / "runs" / fixed_run_id / "plots").exists()
    assert (tmp_path / "runs" / fixed_run_id / "reports").exists()
    assert (tmp_path / "runs" / fixed_run_id / "artifacts").exists()
    assert (tmp_path / "runs" / fixed_run_id / "artifacts" / "config.yaml").exists()


def test_train_manifest_created_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta, patched_train_sb3):
    monkeypatch.chdir(tmp_path)
    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    assert (run_dir / "artifacts" / "train_manifest.json").exists()
    manifest = json.loads((run_dir / "artifacts" / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {"ready", "blocked_missing_dependencies", "blocked_no_training_data"}
    assert manifest["artifacts"]["train_summary_report"] == "reports/model_train_summary.json"
    assert manifest["artifacts"]["config"] == "artifacts/config.yaml"
    assert manifest["artifacts"]["metadata"] == "artifacts/metadata.json"


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


def test_walkforward_shortfall_reason_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        orchestrator,
        "plan_walkforward_splits",
        lambda _candles_df, split, target_runs, min_folds=3: {"splits": [split]},
    )
    monkeypatch.setattr(
        orchestrator,
        "train_sb3",
        lambda *_args, **_kwargs: {"enabled": False, "reason": "insufficient_split_rows"},
    )

    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    model_summary = json.loads((run_dir / "reports" / "model_train_summary.json").read_text(encoding="utf-8"))

    assert model_summary["walkforward_requested"] > model_summary["walkforward_runs"]
    assert model_summary["walkforward_coverage_check"]["next_fold_required_test_end"] is not None
    assert model_summary["walkforward_coverage_check"]["possible_runs"] >= 1
    assert model_summary["walkforward_shortfall"] is not None
    assert model_summary["walkforward_shortfall"]["reason"] == "insufficient_data_coverage_for_requested_walkforward"
    assert len(model_summary["walkforward_shortfall"]["alternatives"]) >= 2


def test_walkforward_split_reduction_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta):
    monkeypatch.chdir(tmp_path)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2026-03-31", tz="UTC")
    open_times = pd.date_range(start=start, end=end, freq="1D")
    candles_df = pd.DataFrame(
        {
            "open_time": (open_times.view("int64") // 1_000_000).astype(int),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "close_time": (open_times.view("int64") // 1_000_000).astype(int) + 60_000,
        }
    )
    monkeypatch.setattr(orchestrator, "ensure_training_candles", lambda _cfg, allow_bootstrap=None: (candles_df, False, False, None))
    monkeypatch.setattr(
        orchestrator,
        "plan_walkforward_splits",
        lambda _candles_df, split, target_runs, min_folds=3: {"splits": [split] * target_runs},
    )
    monkeypatch.setattr(
        orchestrator,
        "train_sb3",
        lambda *_args, **_kwargs: {"enabled": False, "reason": "insufficient_split_rows"},
    )

    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    model_summary = json.loads((run_dir / "reports" / "model_train_summary.json").read_text(encoding="utf-8"))

    assert model_summary["walkforward_coverage_check"]["satisfied"] is False
    assert model_summary["walkforward_coverage_adjustment"] is not None
    assert model_summary["walkforward_coverage_adjustment"]["action"] == "split_reduction"


def test_train_module_run_returns_run_id_without_type_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """완료 기준: run()이 TypeError 없이 run_id를 반환해야 한다."""
    from src.coin_trading.pipelines import train
    from src.coin_trading.pipelines.train_flow import orchestrator as train_orchestrator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_orchestrator, "make_run_id", lambda: "entry_run_id")
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


def test_walkforward_shortfall_abort_policy_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta):
    monkeypatch.chdir(tmp_path)

    from src.coin_trading.config.loader import load_config as _load_config

    cfg = _load_config()
    cfg.train.walkforward_shortfall_policy = "abort"

    monkeypatch.setattr(orchestrator, "load_config", lambda: cfg)
    monkeypatch.setattr(
        orchestrator,
        "plan_walkforward_splits",
        lambda _candles_df, split, target_runs, min_folds=3: {"splits": [split]},
    )

    with pytest.raises(RuntimeError, match="walkforward shortfall"):
        orchestrator.run()


def test_live_mode_fails_fast_when_bootstrap_disallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_meta):
    from src.coin_trading.config.loader import load_config as _load_config

    cfg = _load_config()
    cfg.mode = "live"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "load_config", lambda: cfg)
    monkeypatch.setattr(orchestrator, "make_run_id", lambda: "live_failfast_run")

    with pytest.raises(RuntimeError, match="bootstrap is disabled"):
        orchestrator.run()


def test_manifest_records_bootstrap_persist_failure_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_run_id: str, patched_meta, patched_train_sb3):
    monkeypatch.chdir(tmp_path)
    from src.coin_trading.config.loader import load_config as _load_config

    cfg = _load_config()
    start = pd.Timestamp(cfg.split.train[0], tz="UTC")
    end = pd.Timestamp(cfg.split.test[1], tz="UTC")
    open_times = pd.date_range(start=start, end=end, freq="1D")
    candles_df = pd.DataFrame(
        {
            "open_time": (open_times.view("int64") // 1_000_000).astype(int),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "close_time": (open_times.view("int64") // 1_000_000).astype(int) + 60_000,
        }
    )

    monkeypatch.setattr(
        orchestrator,
        "ensure_training_candles",
        lambda _cfg, allow_bootstrap=None: (candles_df, True, False, "PermissionError: denied"),
    )

    orchestrator.run()

    run_dir = tmp_path / "runs" / fixed_run_id
    data_manifest = json.loads((run_dir / "artifacts" / "data_manifest.json").read_text(encoding="utf-8"))
    train_manifest = json.loads((run_dir / "artifacts" / "train_manifest.json").read_text(encoding="utf-8"))
    model_summary = json.loads((run_dir / "reports" / "model_train_summary.json").read_text(encoding="utf-8"))

    assert data_manifest["bootstrap_allowed"] is True
    assert data_manifest["bootstrap_generated"] is True
    assert data_manifest["bootstrap_persisted"] is False
    assert data_manifest["bootstrap_persist_failure_reason"] == "PermissionError: denied"
    assert train_manifest["demo_smoke_only"] is True
    assert model_summary["demo_smoke_only"] is True
