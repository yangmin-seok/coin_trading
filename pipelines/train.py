from __future__ import annotations

from pathlib import Path

from config.loader import load_config
from pipelines.run_manager import make_run_id, write_meta


def run() -> str:
    cfg = load_config()
    run_id = make_run_id(cfg.mode, cfg.symbol, cfg.interval, cfg.seed)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(Path("config/default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    write_meta(run_dir)
    return run_id


if __name__ == "__main__":
    print(run())
