from __future__ import annotations

import json
from pathlib import Path

from src.coin_trading.report.run_doctor import main, run_doctor


REQUIRED_FILES = [
    "artifacts/metadata.json",
    "artifacts/train_manifest.json",
    "reports/model_train_summary.json",
    "artifacts/data_manifest.json",
    "artifacts/dataset_summary.json",
]



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def test_run_doctor_test3_like_warnings(tmp_path: Path):
    run_dir = tmp_path / "runs" / "Test3Like"

    _write_json(
        run_dir / "artifacts/metadata.json",
        {
            "git_sha": "abc123",
            "git_dirty": True,
            "libs": {},
        },
    )
    _write_json(
        run_dir / "artifacts/train_manifest.json",
        {
            "model_train": {
                "walkforward_requested": 3,
                "walkforward_runs": 1,
                "walkforward_shortfall": {
                    "reason": "insufficient_data_coverage_for_requested_walkforward",
                    "suggestion": "collect more data",
                },
            }
        },
    )
    _write_json(
        run_dir / "reports/model_train_summary.json",
        {
            "walkforward_requested": 3,
            "walkforward_runs": 1,
            "walkforward_shortfall": {
                "reason": "insufficient_data_coverage_for_requested_walkforward",
                "suggestion": "collect more data",
            },
        },
    )
    _write_json(
        run_dir / "artifacts/data_manifest.json",
        {
            "bootstrap_generated": True,
            "bootstrap_persisted": False,
        },
    )
    _write_json(run_dir / "artifacts/dataset_summary.json", {"rows": 720})

    report = run_doctor(run_dir)
    codes = {issue.code: issue for issue in report.issues}

    assert "walkforward_shortfall" in codes
    assert codes["walkforward_shortfall"].details["reason"] == "insufficient_data_coverage_for_requested_walkforward"
    assert "bootstrap_state" in codes
    assert codes["bootstrap_state"].level == "HIGH"
    assert "git_dirty" in codes
    assert "missing_lib_versions" in codes



def test_run_doctor_cli_json_output(tmp_path: Path, capsys):
    run_dir = tmp_path / "runs" / "good"
    for rel in REQUIRED_FILES:
        _write_json(run_dir / rel, {"ok": True})

    output_json = tmp_path / "doctor" / "report.json"
    exit_code = main([str(run_dir), "--json-out", str(output_json)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run Doctor Report" in captured.out
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_dir"] == str(run_dir)
    assert isinstance(payload["issues"], list)
