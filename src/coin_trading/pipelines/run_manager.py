from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sha = git_sha()[:7]
    return f"{ts}_{sha}"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def write_meta(run_dir: Path, extra: dict[str, Any] | None = None) -> None:
    meta = {
        "git_sha": git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libs": {},
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        meta.update(extra)

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_data_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "data_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_feature_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "feature_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_train_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "train_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def implementation_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    return h.hexdigest()
