from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.coin_trading.config.loader import load_config
from src.coin_trading.features.definitions import FEATURE_COLUMNS
from src.coin_trading.pipelines.run_manager import (
    implementation_hash,
    make_run_id,
    write_data_manifest,
    write_feature_manifest,
    write_meta,
    write_train_manifest,
)
from src.coin_trading.pipelines.train_flow.data import ensure_training_candles, split_by_date, summarize_dataset
from src.coin_trading.pipelines.train_flow.train import train_sb3


def run() -> str:
    cfg = load_config()
    run_id = make_run_id()
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    for directory in (plots_dir, reports_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    default_config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    (artifacts_dir / "config.yaml").write_text(default_config_path.read_text(encoding="utf-8"), encoding="utf-8")

    candles_df, bootstrapped, bootstrap_persisted = ensure_training_candles(cfg)
    dataset_summary = summarize_dataset(candles_df, cfg)

    write_meta(
        run_dir,
        {
            "seed": cfg.seed,
            "start_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_range": {k: v["range"] for k, v in dataset_summary["splits"].items()},
        },
    )

    train_df = split_by_date(candles_df, cfg.split.train)
    val_df = split_by_date(candles_df, cfg.split.val)
    test_df = split_by_date(candles_df, cfg.split.test)

    status = "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data"
    if dataset_summary["rows"] > 0:
        try:
            train_summary = train_sb3(train_df, val_df, test_df, cfg, run_dir)
        except RuntimeError as exc:
            status = "blocked_missing_dependencies"
            train_summary = {"enabled": False, "reason": "missing_dependencies", "message": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive fallback for runtime errors in optional training stack
            status = "blocked_training_error"
            train_summary = {"enabled": False, "reason": "training_error", "message": str(exc)}
    else:
        train_summary = {"enabled": False, "reason": "no_data"}

    (reports_dir / "model_train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

    write_data_manifest(
        run_dir,
        {
            "exchange": cfg.exchange,
            "market": cfg.market,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "processed": {"time_unit": "ms"},
            "bootstrap_generated": bool(bootstrapped),
            "bootstrap_persisted": bool(bootstrap_persisted),
            "dataset": dataset_summary,
            "artifacts": {
                "config": "artifacts/config.yaml",
                "metadata": "artifacts/metadata.json",
                "dataset_summary": "reports/dataset_summary.json",
            },
        },
    )
    write_feature_manifest(
        run_dir,
        {
            "feature_set_version": cfg.features.version,
            "windows": cfg.features.windows.model_dump(by_alias=True),
            "columns": [{"name": c, "dtype": "float64"} for c in FEATURE_COLUMNS],
            "implementation_hash": implementation_hash(
                [
                    Path(__file__).resolve().parents[2] / "features" / "common.py",
                    Path(__file__).resolve().parents[2] / "features" / "definitions.py",
                    Path(__file__).resolve().parents[2] / "features" / "offline.py",
                ]
            ),
            "artifact_paths": {
                "manifest": "feature_manifest.json",
                "train_manifest": "train_manifest.json",
            },
        },
    )
    write_train_manifest(
        run_dir,
        {
            "status": status,
            "missing": [] if status == "ready" else [train_summary.get("message", "training unavailable")],
            "split_rows": {k: v["rows"] for k, v in dataset_summary["splits"].items()},
            "epochs": 0,
            "model": train_summary.get("model", "none"),
            "model_train": train_summary,
            "artifacts": {
                "train_summary_report": "reports/model_train_summary.json",
            },
        },
    )
    (reports_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id
