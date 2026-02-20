from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.coin_trading.config.schema import AppConfig
from src.coin_trading.pipelines.train_flow.env import build_env
from src.coin_trading.pipelines.train_flow.evaluate import rollout_model
from src.coin_trading.pipelines.train_flow.features import compute_features
from src.coin_trading.pipelines.reporting import (
    create_benchmark_comparison,
    create_common_risk_plots,
    create_split_equity_curves,
    detect_overfit,
    write_trade_stats_report,
)
from src.coin_trading.report.plotting import write_learning_curve_artifacts


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def build_sb3_algo(algo_name: str, env, cfg: AppConfig):
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise RuntimeError("stable-baselines3/gymnasium is required for train mode. Please install project dependencies.") from exc

    if algo_name == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.train.learning_rate,
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
            n_steps=cfg.train.n_steps,
            seed=cfg.train.seed if cfg.train.seed is not None else cfg.seed,
            verbose=0,
        )
    if algo_name == "sac":
        return SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.train.learning_rate,
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
            seed=cfg.train.seed if cfg.train.seed is not None else cfg.seed,
            verbose=0,
        )
    raise ValueError(f"unsupported algo: {algo_name}")


def train_sb3(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: AppConfig, run_dir: Path) -> dict[str, Any]:
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    seed_everything(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    train_features = compute_features(train_df)
    val_features = compute_features(val_df)
    test_features = compute_features(test_df) if not test_df.empty else pd.DataFrame()

    train_env = build_env(train_df, train_features, cfg)
    model = build_sb3_algo(cfg.train.algo, train_env, cfg)

    if cfg.train.resume_from:
        model = model.__class__.load(cfg.train.resume_from, env=train_env)

    total = cfg.train.total_timesteps
    interval = min(cfg.train.eval_interval, total)
    ckpt_interval = cfg.train.checkpoint_interval
    best_sharpe = -1e9
    stale = 0
    trained = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[str] = []

    while trained < total:
        chunk = min(interval, total - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained += chunk
        val_metrics = rollout_model(model, val_df, val_features, cfg)
        history.append({"timesteps": trained, "val": val_metrics})

        if val_metrics["sharpe"] > best_sharpe:
            best_sharpe = val_metrics["sharpe"]
            stale = 0
            model.save(str(run_dir / "best_model"))
        else:
            stale += 1

        if trained % ckpt_interval == 0 or trained == total:
            ckpt_name = f"checkpoint_{trained}.zip"
            model.save(str(run_dir / ckpt_name.replace(".zip", "")))
            checkpoints.append(ckpt_name)

        if cfg.train.early_stop > 0 and stale >= cfg.train.early_stop:
            break

    best_model_path = run_dir / "best_model.zip"
    best_model = model.__class__.load(str(best_model_path), env=train_env) if best_model_path.exists() else model

    trace_root = run_dir / "reports" / "traces"
    train_final = rollout_model(best_model, train_df, train_features, cfg, trace_root / "train", include_trace=True)
    val_final = rollout_model(best_model, val_df, val_features, cfg, trace_root / "valid", include_trace=True)
    test_metrics = (
        rollout_model(best_model, test_df, test_features, cfg, trace_root / "test", include_trace=True)
        if not test_df.empty
        else {"enabled": False, "trace": pd.DataFrame()}
    )

    traces = {
        "train": train_final.pop("trace", pd.DataFrame()),
        "valid": val_final.pop("trace", pd.DataFrame()),
        "test": test_metrics.pop("trace", pd.DataFrame()),
    }

    overfit_warning = detect_overfit(train_final, test_metrics) if test_metrics.get("enabled", True) else False
    equity_curves = create_split_equity_curves(run_dir, traces)
    risk_plots = create_common_risk_plots(run_dir, traces)
    benchmark_plot = create_benchmark_comparison(run_dir, test_df if not test_df.empty else val_df, cfg.seed)
    trade_stats_report = write_trade_stats_report(run_dir, traces.get("test") if not traces.get("test", pd.DataFrame()).empty else traces.get("valid", pd.DataFrame()), overfit_warning)

    write_learning_curve_artifacts(history, run_dir)

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "best_model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "train_metrics": train_final,
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "overfit_warning": overfit_warning,
        "artifacts": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "learning_curve_svg": "learning_curve.svg",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
            "trace_dirs": {"train": "reports/traces/train", "valid": "reports/traces/valid", "test": "reports/traces/test"},
            "split_equity_curves": equity_curves,
            **risk_plots,
            "benchmark_comparison_png": benchmark_plot,
            "trade_stats_report_html": trade_stats_report,
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps({"history": history, "train": train_final, "val": val_final, "test": test_metrics, "overfit_warning": overfit_warning}, indent=2),
        encoding="utf-8",
    )
    return summary
