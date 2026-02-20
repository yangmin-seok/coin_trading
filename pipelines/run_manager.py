from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def make_run_id(mode: str, symbol: str, interval: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sha = git_sha()[:7]
    return f"{ts}_{sha}_{mode}_{symbol}_{interval}_seed{seed}"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_meta(run_dir: Path) -> None:
    meta = {
        "git_sha": git_sha(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def implementation_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    return h.hexdigest()
