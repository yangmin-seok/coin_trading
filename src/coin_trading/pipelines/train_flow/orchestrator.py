from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.coin_trading.config.loader import load_config
from src.coin_trading.features.definitions import FEATURE_COLUMNS
from src.coin_trading.pipelines.reporting import (
    create_benchmark_comparison,
    create_common_risk_plots,
    create_split_equity_curves,
    detect_overfit,
    write_trade_stats_report,
)
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

LOGGER = logging.getLogger(__name__)


def _extract_traces(fold_summary: dict[str, object]) -> dict[str, pd.DataFrame]:
    raw = fold_summary.pop("trace_frames", {})
    traces: dict[str, pd.DataFrame] = {}
    for split in ("train", "valid", "test"):
        trace = raw.get(split) if isinstance(raw, dict) else None
        traces[split] = trace if isinstance(trace, pd.DataFrame) else pd.DataFrame()
    return traces


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
    write_meta(run_dir)

    candles_df, bootstrapped, bootstrap_persisted = ensure_training_candles(cfg)
    dataset_summary = summarize_dataset(candles_df, cfg)

    base_split = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    split_policy = validate_split_policy(base_split, candles_df)
    wf_splits = build_walkforward_splits(candles_df, base_split, target_runs=cfg.train.walkforward_runs)

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

            traces = _extract_traces(fold_summary)
            has_any_trace = any(not trace.empty for trace in traces.values())
            if not has_any_trace:
                LOGGER.warning("No train/valid/test traces for walkforward_%02d. Generating placeholder reporting artifacts.", idx)

            split_curve_files = create_split_equity_curves(fold_dir / "plots", traces)
            risk_plot_files = create_common_risk_plots(fold_dir / "plots", traces)
            overfit_warning = detect_overfit(
                fold_summary.get("train_metrics", {}) if isinstance(fold_summary.get("train_metrics", {}), dict) else {},
                fold_summary.get("test_metrics", {}) if isinstance(fold_summary.get("test_metrics", {}), dict) else {},
            )
            trade_trace = traces.get("test", pd.DataFrame())
            if trade_trace.empty:
                LOGGER.warning("Empty test trace for walkforward_%02d. Trade stats report will use fallback values.", idx)
            trade_stats_report = write_trade_stats_report(fold_dir, trade_trace, overfit_warning=overfit_warning)
            benchmark_png = create_benchmark_comparison(fold_dir / "plots", test_df, seed=cfg.seed)

            fold_reporting = {
                "plots": {
                    "split_equity_curves": [f"plots/{name}" for name in split_curve_files],
                    "drawdown_curve_png": f"plots/{risk_plot_files['drawdown_curve_png']}",
                    "monthly_returns_heatmap_png": f"plots/{risk_plot_files['monthly_returns_heatmap_png']}",
                    "benchmark_comparison_png": f"plots/{benchmark_png}",
                },
                "reports": {
                    "trade_stats_html": trade_stats_report,
                },
                "trace_fallback": {
                    "used": not has_any_trace,
                    "message": "placeholder plots/statistics generated due to empty traces" if not has_any_trace else "none",
                },
            }

            artifacts = fold_summary.get("artifacts", {}) if isinstance(fold_summary.get("artifacts"), dict) else {}
            artifacts["reporting"] = fold_reporting
            fold_summary["artifacts"] = artifacts

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
        "results": wf_results,
        "model": primary_summary.get("model", "none"),
        "reason": primary_summary.get("reason"),
        "message": primary_summary.get("message"),
    }
    if status == "blocked_missing_dependencies" and wf_results:
        train_summary["reason"] = "missing_dependencies"
        train_summary["message"] = wf_results[0].get("summary", {}).get("message", "")

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
                "walkforward_reports": [f"walkforward_{result['fold']:02d}/reports/trade_stats.html" for result in wf_results],
                "walkforward_plots_dir": [f"walkforward_{result['fold']:02d}/plots" for result in wf_results],
            },
        },
    )
    (artifacts_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id
