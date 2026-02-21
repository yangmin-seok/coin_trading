from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "artifacts/metadata.json",
    "artifacts/train_manifest.json",
    "reports/model_train_summary.json",
    "artifacts/data_manifest.json",
    "artifacts/dataset_summary.json",
]


@dataclass
class DoctorIssue:
    level: str
    code: str
    message: str
    details: dict[str, Any]


@dataclass
class DoctorReport:
    run_dir: str
    ok: bool
    issues: list[DoctorIssue]



def _read_json(path: Path) -> tuple[dict[str, Any] | None, DoctorIssue | None]:
    if not path.exists():
        return None, DoctorIssue(
            level="HIGH",
            code="missing_file",
            message=f"Required file is missing: {path}",
            details={"path": str(path)},
        )
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, DoctorIssue(
            level="HIGH",
            code="json_parse_error",
            message=f"Failed to parse JSON: {path}",
            details={"path": str(path), "error": str(exc)},
        )



def _pick_walkforward_payload(train_manifest: dict[str, Any], model_train_summary: dict[str, Any]) -> dict[str, Any]:
    if isinstance(model_train_summary, dict) and model_train_summary:
        return model_train_summary
    return train_manifest.get("model_train", {}) if isinstance(train_manifest, dict) else {}



def _extract_walkforward(payload: dict[str, Any]) -> tuple[int | None, int | None, dict[str, Any] | None]:
    requested = payload.get("walkforward_requested")
    actual = payload.get("walkforward_runs")
    if actual is None and isinstance(payload.get("results"), list):
        actual = len(payload["results"])
    shortfall = payload.get("walkforward_shortfall")
    if not isinstance(shortfall, dict):
        shortfall = None
    return requested, actual, shortfall



def run_doctor(run_dir: Path) -> DoctorReport:
    issues: list[DoctorIssue] = []
    payloads: dict[str, dict[str, Any]] = {}

    for rel in REQUIRED_FILES:
        full_path = run_dir / rel
        data, issue = _read_json(full_path)
        if issue:
            issues.append(issue)
            continue
        payloads[rel] = data or {}

    metadata = payloads.get("artifacts/metadata.json", {})
    train_manifest = payloads.get("artifacts/train_manifest.json", {})
    model_train_summary = payloads.get("reports/model_train_summary.json", {})
    data_manifest = payloads.get("artifacts/data_manifest.json", {})

    walkforward_payload = _pick_walkforward_payload(train_manifest, model_train_summary)
    requested, actual, shortfall = _extract_walkforward(walkforward_payload)
    if requested is not None and actual is not None and requested > actual:
        issues.append(
            DoctorIssue(
                level="WARN",
                code="walkforward_shortfall",
                message=f"Walkforward runs are short: requested={requested}, actual={actual}.",
                details={
                    "requested": requested,
                    "actual": actual,
                    "reason": (shortfall or {}).get("reason") or walkforward_payload.get("reason"),
                    "suggestion": (shortfall or {}).get("suggestion") or walkforward_payload.get("message"),
                },
            )
        )

    if data_manifest.get("bootstrap_generated") is True or data_manifest.get("bootstrap_persisted") is False:
        issues.append(
            DoctorIssue(
                level="HIGH",
                code="bootstrap_state",
                message="Bootstrap data flags detected. Treat this run as potentially non-production quality.",
                details={
                    "bootstrap_generated": data_manifest.get("bootstrap_generated"),
                    "bootstrap_persisted": data_manifest.get("bootstrap_persisted"),
                },
            )
        )

    if metadata.get("git_dirty") is True:
        issues.append(
            DoctorIssue(
                level="WARN",
                code="git_dirty",
                message="Run metadata indicates a dirty git tree at runtime.",
                details={"git_dirty": True},
            )
        )

    libs = metadata.get("libs")
    if not isinstance(libs, dict) or not libs:
        issues.append(
            DoctorIssue(
                level="WARN",
                code="missing_lib_versions",
                message="Run metadata has empty or missing libs; reproducibility may be limited.",
                details={"libs": libs},
            )
        )

    return DoctorReport(run_dir=str(run_dir), ok=not any(i.level == "HIGH" for i in issues), issues=issues)



def _render_human(report: DoctorReport) -> str:
    lines = [f"Run Doctor Report: {report.run_dir}", f"Overall: {'OK' if report.ok else 'NEEDS_ATTENTION'}"]
    if not report.issues:
        lines.append("No issues found.")
        return "\n".join(lines)

    lines.append("Issues:")
    for issue in report.issues:
        lines.append(f"- [{issue.level}] {issue.code}: {issue.message}")
        if issue.details:
            details_txt = ", ".join(f"{k}={v}" for k, v in issue.details.items() if v is not None)
            if details_txt:
                lines.append(f"  details: {details_txt}")
    return "\n".join(lines)



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect run artifacts and report quality/reproducibility warnings.")
    parser.add_argument("run_dir", type=Path, help="Path to run directory, e.g. runs/<RUN_ID>")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path for machine-readable JSON report")
    args = parser.parse_args(argv)

    report = run_doctor(args.run_dir)
    print(_render_human(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"run_dir": report.run_dir, "ok": report.ok, "issues": [asdict(i) for i in report.issues]}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
