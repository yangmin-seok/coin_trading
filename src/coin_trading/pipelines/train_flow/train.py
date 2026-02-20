from __future__ import annotations

import json
import random
from itertools import product
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

EPSILON = 1e-9


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


def _compute_model_selection_score(metrics: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    final_equity = float(metrics.get("final_equity", 0.0) or 0.0)
    turnover = float(metrics.get("turnover", 0.0) or 0.0)
    turnover_cap = float(cfg.execution.max_step_change)

    turnover_excess = max(0.0, turnover - turnover_cap)
    penalty = turnover_excess * 100.0
    score = sharpe - penalty

    return {
        "score": float(score),
        "sharpe": sharpe,
        "final_equity": final_equity,
        "turnover": turnover,
        "turnover_cap": turnover_cap,
        "turnover_penalty": float(penalty),
        "turnover_excess": float(turnover_excess),
    }


def _is_better_model_candidate(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True

    cand_score = float(candidate["score"])
    curr_score = float(current["score"])
    if cand_score > curr_score + EPSILON:
        return True
    if abs(cand_score - curr_score) <= EPSILON:
        cand_sharpe = float(candidate["sharpe"])
        curr_sharpe = float(current["sharpe"])
        if cand_sharpe > curr_sharpe + EPSILON:
            return True
        if abs(cand_sharpe - curr_sharpe) <= EPSILON:
            cand_equity = float(candidate["final_equity"])
            curr_equity = float(current["final_equity"])
            if cand_equity > curr_equity + EPSILON:
                return True
    return False


def _select_training_schedule(cfg: AppConfig) -> tuple[int, int, int, int]:
    total = max(int(cfg.train.total_timesteps), 10_000)
    interval = min(max(int(cfg.train.eval_interval), 2_000), total)
    early_stop = max(int(cfg.train.early_stop), 5)
    checkpoint_interval = max(int(cfg.train.checkpoint_interval), interval)
    return total, interval, early_stop, checkpoint_interval


def _unique(values: list[float]) -> list[float]:
    seen = set()
    result: list[float] = []
    for value in values:
        key = float(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _build_experiment_grid(cfg: AppConfig) -> list[dict[str, float]]:
    axes = cfg.exploration.axes
    lambdas = _unique(axes.lambda_turnover)
    min_deltas = _unique(axes.min_delta)
    max_steps = _unique(axes.max_step_change)
    grid = [
        {
            "lambda_turnover": float(lambda_turnover),
            "min_delta": float(min_delta),
            "max_step_change": float(max_step_change),
        }
        for lambda_turnover, min_delta, max_step_change in product(lambdas, min_deltas, max_steps)
    ]
    baseline = {
        "lambda_turnover": float(cfg.reward.lambda_turnover),
        "min_delta": float(cfg.execution.min_delta),
        "max_step_change": float(cfg.execution.max_step_change),
    }
    if baseline in grid:
        grid.remove(baseline)
    return [baseline, *grid]


def _run_single_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> dict[str, Any]:
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

    total, interval, early_stop_rounds, ckpt_interval = _select_training_schedule(cfg)
    best_candidate: dict[str, Any] | None = None
    stale = 0
    trained = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[str] = []

    while trained < total:
        chunk = min(interval, total - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained += chunk
        val_metrics = rollout_model(model, val_df, val_features, cfg)
        candidate = _compute_model_selection_score(val_metrics, cfg)
        candidate_record = {"timesteps": trained, "val": val_metrics, "selection": candidate}
        history.append(candidate_record)

        if _is_better_model_candidate(candidate, best_candidate):
            best_candidate = {"timesteps": trained, **candidate}
            stale = 0
            model.save(str(artifacts_dir / "model"))
        else:
            stale += 1

        if trained % ckpt_interval == 0 or trained == total:
            ckpt_name = f"checkpoint_{trained}.zip"
            model.save(str(artifacts_dir / ckpt_name.replace(".zip", "")))
            checkpoints.append(f"artifacts/{ckpt_name}")

        if early_stop_rounds > 0 and stale >= early_stop_rounds:
            break

    best_model_path = artifacts_dir / "model.zip"
    best_model = model.__class__.load(str(best_model_path), env=train_env) if best_model_path.exists() else model

    val_final = rollout_model(best_model, val_df, val_features, cfg, reports_dir / "val_trace")
    test_metrics = rollout_model(best_model, test_df, test_features, cfg, reports_dir / "test_trace") if not test_df.empty else {"enabled": False}

    write_learning_curve_artifacts(history, reports_dir, plots_dir)

    selection_summary = {
        "criteria": "maximize composite score (sharpe - turnover_penalty), then higher sharpe, then higher final_equity",
        "best_candidate": best_candidate,
        "training_schedule": {
            "configured": {
                "total_timesteps": cfg.train.total_timesteps,
                "eval_interval": cfg.train.eval_interval,
                "early_stop": cfg.train.early_stop,
                "checkpoint_interval": cfg.train.checkpoint_interval,
            },
            "effective": {
                "total_timesteps": total,
                "eval_interval": interval,
                "early_stop": early_stop_rounds,
                "checkpoint_interval": ckpt_interval,
            },
        },
    }
    metrics_payload = {
        "history": history,
        "selection": selection_summary,
        "val": val_final,
        "test": test_metrics,
    }

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "artifacts/model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "selection": metrics_payload["selection"],
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
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return summary


def _is_better_candidate(candidate: dict[str, Any], current: dict[str, Any] | None, baseline: dict[str, Any]) -> bool:
    if current is None:
        return True

    cand_test = candidate.get("test_metrics", {})
    curr_test = current.get("test_metrics", {})
    base_test = baseline.get("test_metrics", {})

    cand_eq = float(cand_test.get("final_equity", 0.0) or 0.0)
    curr_eq = float(curr_test.get("final_equity", 0.0) or 0.0)
    base_eq = float(base_test.get("final_equity", 0.0) or 0.0)

    cand_cost = float(cand_test.get("total_cost", float("inf")) or float("inf"))
    curr_cost = float(curr_test.get("total_cost", float("inf")) or float("inf"))
    base_cost = float(base_test.get("total_cost", float("inf")) or float("inf"))

    cand_meets = cand_eq > base_eq and cand_cost < base_cost
    curr_meets = curr_eq > base_eq and curr_cost < base_cost
    if cand_meets != curr_meets:
        return cand_meets
    if cand_eq != curr_eq:
        return cand_eq > curr_eq
    return cand_cost < curr_cost


def train_sb3(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: AppConfig, run_dir: Path) -> dict[str, Any]:
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    experiments = _build_experiment_grid(cfg)
    experiment_results: list[dict[str, Any]] = []

    for idx, axes in enumerate(experiments, start=1):
        exp_cfg = cfg.model_copy(deep=True)
        exp_cfg.reward.lambda_turnover = axes["lambda_turnover"]
        exp_cfg.execution.min_delta = axes["min_delta"]
        exp_cfg.execution.max_step_change = axes["max_step_change"]

        exp_dir = run_dir / f"experiment_{idx:02d}"
        summary = _run_single_experiment(train_df, val_df, test_df, exp_cfg, exp_dir)
        experiment_results.append(
            {
                "id": idx,
                "axes": axes,
                "run_dir": str(exp_dir.relative_to(run_dir)),
                "summary": summary,
            }
        )

    baseline = experiment_results[0]["summary"]
    selected: dict[str, Any] | None = None
    for candidate in experiment_results:
        if _is_better_candidate(candidate["summary"], selected["summary"] if selected else None, baseline):
            selected = candidate

    if selected is None:
        return {"enabled": False, "reason": "no_experiment_result"}

    selected_summary = selected["summary"]
    selected_summary["selected_experiment"] = {
        "id": selected["id"],
        "axes": selected["axes"],
        "run_dir": selected["run_dir"],
        "criteria": "maximize test final_equity while requiring lower total_cost than baseline when possible",
    }
    selected_summary["experiments"] = [
        {
            "id": item["id"],
            "axes": item["axes"],
            "run_dir": item["run_dir"],
            "test_final_equity": item["summary"].get("test_metrics", {}).get("final_equity", 0.0),
            "test_total_cost": item["summary"].get("test_metrics", {}).get("total_cost", 0.0),
            "test_cost_pnl_ratio": item["summary"].get("test_metrics", {}).get("cost_pnl_ratio", 0.0),
        }
        for item in experiment_results
    ]
    return selected_summary
