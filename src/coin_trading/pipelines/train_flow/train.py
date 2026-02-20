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
from src.coin_trading.pipelines.train_flow.features import (
    compute_features,
    fit_feature_scaler,
    transform_with_scaler,
    validate_rolling_features_no_lookahead,
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

    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    train_features_raw = compute_features(train_df)
    val_features_raw = compute_features(val_df)
    test_features_raw = compute_features(test_df) if not test_df.empty else pd.DataFrame()

    validate_rolling_features_no_lookahead(train_df, train_features_raw)
    validate_rolling_features_no_lookahead(val_df, val_features_raw)
    if not test_df.empty:
        validate_rolling_features_no_lookahead(test_df, test_features_raw)

    scaler = fit_feature_scaler(train_features_raw, split_name="train")
    train_features = transform_with_scaler(train_features_raw, scaler)
    val_features = transform_with_scaler(val_features_raw, scaler)
    test_features = transform_with_scaler(test_features_raw, scaler) if not test_df.empty else pd.DataFrame()

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
            model.save(str(artifacts_dir / "model"))
        else:
            stale += 1

        if trained % ckpt_interval == 0 or trained == total:
            ckpt_name = f"checkpoint_{trained}.zip"
            model.save(str(artifacts_dir / ckpt_name.replace(".zip", "")))
            checkpoints.append(f"artifacts/{ckpt_name}")

        if cfg.train.early_stop > 0 and stale >= cfg.train.early_stop:
            break

    best_model_path = artifacts_dir / "model.zip"
    best_model = model.__class__.load(str(best_model_path), env=train_env) if best_model_path.exists() else model

    val_final = rollout_model(best_model, val_df, val_features, cfg, reports_dir / "val_trace")
    test_metrics = rollout_model(best_model, test_df, test_features, cfg, reports_dir / "test_trace") if not test_df.empty else {"enabled": False}

    write_learning_curve_artifacts(history, reports_dir, plots_dir)

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "artifacts/model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "artifacts": {
            "learning_curve_csv": "reports/learning_curve.csv",
            "learning_curve_json": "reports/learning_curve.json",
            "learning_curve_svg": "plots/learning_curve.svg",
            "best_model": "artifacts/model.zip" if best_model_path.exists() else None,
            "val_trace_dir": "reports/val_trace",
            "test_trace_dir": "reports/test_trace",
            "evaluation_metrics": "artifacts/metrics.json",
        },
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps({"history": history, "val": val_final, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    return summary
