from __future__ import annotations

from pathlib import Path

import pytest

from src.coin_trading.pipelines.train_flow import orchestrator


def test_train_entry_execution_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_run_id: str,
    patched_meta,
    patched_train_sb3,
):
    monkeypatch.chdir(tmp_path)
    run_id = orchestrator.run()

    assert run_id == fixed_run_id
    assert (tmp_path / "runs" / fixed_run_id).exists()
    assert (tmp_path / "runs" / fixed_run_id / "plots").exists()
    assert (tmp_path / "runs" / fixed_run_id / "reports").exists()
    assert (tmp_path / "runs" / fixed_run_id / "artifacts").exists()
