from __future__ import annotations

import json
import inspect
from datetime import datetime, timedelta, timezone
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
from src.coin_trading.pipelines.train_flow.data import (
    build_walkforward_splits,
    ensure_training_candles,
    split_by_date,
    summarize_dataset,
    validate_split_policy,
)
from src.coin_trading.pipelines.train_flow.train import train_sb3


def _write_meta_compat(run_dir: Path, extra: dict[str, object]) -> None:
    """Call write_meta while remaining compatible with legacy monkeypatch stubs.

    Some tests replace ``write_meta`` with a lambda that only accepts ``run_dir``.
    Keep passing enriched metadata when supported, and gracefully fallback otherwise.
    """

    params = inspect.signature(write_meta).parameters
    if "extra" in params:
        write_meta(run_dir, extra=extra)
        return
    write_meta(run_dir)


def run() -> str:
    cfg = load_config()
    run_id = make_run_id()
    run_dir = Path("runs") / run_id
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    for directory in (plots_dir, reports_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    default_config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    (artifacts_dir / "config.yaml").write_text(default_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_meta_compat(
        run_dir,
        extra={
            "mode": cfg.mode,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "seed": cfg.train.seed if cfg.train.seed is not None else cfg.seed,
        },
    )

    candles_df, bootstrapped, bootstrap_persisted = ensure_training_candles(cfg)
    dataset_summary = summarize_dataset(candles_df, cfg)

    base_split = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    split_policy = validate_split_policy(base_split, candles_df)
    wf_splits = build_walkforward_splits(candles_df, base_split, target_runs=cfg.train.walkforward_runs)

    walkforward_shortfall = None
    if dataset_summary["rows"] > 0 and len(wf_splits) < cfg.train.walkforward_runs:
        val_start = datetime.fromisoformat(cfg.split.val[0]).replace(tzinfo=timezone.utc)
        val_end = datetime.fromisoformat(cfg.split.val[1]).replace(tzinfo=timezone.utc)
        test_end = datetime.fromisoformat(cfg.split.test[1]).replace(tzinfo=timezone.utc)
        step_days = max(1, (val_end - val_start).days + 1)
        next_fold_test_end = test_end + timedelta(days=step_days * len(wf_splits))
        data_end = datetime.fromtimestamp(int(candles_df["open_time"].max()) / 1000, tz=timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        walkforward_shortfall = {
            "reason": "insufficient_data_coverage_for_requested_walkforward",
            "requested": cfg.train.walkforward_runs,
            "actual": len(wf_splits),
            "data_end": data_end.strftime("%Y-%m-%d"),
            "next_fold_required_test_end": next_fold_test_end.strftime("%Y-%m-%d"),
            "suggestion": "collect more data or reduce val/test ranges to increase walkforward folds",
        }

    status = "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data"
    wf_results = []

    if dataset_summary["rows"] > 0:
        for idx, split_cfg in enumerate(wf_splits, start=1):
            train_df = split_by_date(candles_df, split_cfg["train"])
            val_df = split_by_date(candles_df, split_cfg["val"])
            test_df = split_by_date(candles_df, split_cfg["test"])
            fold_dir = run_dir / f"walkforward_{idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            try:
                fold_summary = train_sb3(train_df, val_df, test_df, cfg, fold_dir)
            except RuntimeError as exc:
                status = "blocked_missing_dependencies"
                fold_summary = {"enabled": False, "reason": "missing_dependencies", "message": str(exc)}
            wf_results.append(
                {
                    "fold": idx,
                    "split": {k: list(v) for k, v in split_cfg.items()},
                    "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
                    "summary": fold_summary,
                }
            )
            if status == "blocked_missing_dependencies":
                break
    else:
        wf_results.append({"fold": 1, "summary": {"enabled": False, "reason": "no_data"}})

    primary_summary = wf_results[0]["summary"] if wf_results else {}
    train_summary = {
        "enabled": status == "ready",
        "walkforward_runs": len(wf_results),
        "walkforward_requested": cfg.train.walkforward_runs,
        "walkforward_shortfall": walkforward_shortfall,
        "results": wf_results,
        "model": primary_summary.get("model", "none"),
        "reason": primary_summary.get("reason"),
        "message": primary_summary.get("message"),
    }
    if status == "blocked_missing_dependencies" and wf_results:
        train_summary["reason"] = "missing_dependencies"
        train_summary["message"] = wf_results[0]["summary"].get("message", "training unavailable")

    (reports_dir / "model_train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

    write_data_manifest(
        artifacts_dir,
        {
            "exchange": cfg.exchange,
            "market": cfg.market,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "processed": {"time_unit": "ms"},
            "bootstrap_generated": bool(bootstrapped),
            "bootstrap_persisted": bool(bootstrap_persisted),
            "dataset": dataset_summary,
            "split_policy": split_policy,
        },
    )
    write_feature_manifest(
        artifacts_dir,
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
                "manifest": "artifacts/feature_manifest.json",
                "train_manifest": "artifacts/train_manifest.json",
            },
        },
    )
    write_train_manifest(
        artifacts_dir,
        {
            "status": status,
            "missing": [] if status == "ready" else [train_summary.get("message", "training unavailable")],
            "split_rows": {k: v["rows"] for k, v in dataset_summary["splits"].items()},
            "epochs": 0,
            "model": train_summary.get("model", "none"),
            "model_train": train_summary,
            "artifacts": {
                "train_summary_report": "reports/model_train_summary.json",
                "config": "artifacts/config.yaml",
                "metadata": "artifacts/metadata.json",
            },
        },
    )
    (artifacts_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id
