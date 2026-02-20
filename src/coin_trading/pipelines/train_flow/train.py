from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.coin_trading.agents.baselines import BuyAndHold, MACrossover, RandomPolicy
from src.coin_trading.config.schema import AppConfig
from src.coin_trading.pipelines.reporting import detect_overfit, write_trade_stats_report
from src.coin_trading.pipelines.train_flow.env import build_env
from src.coin_trading.pipelines.train_flow.evaluate import rollout_model
from src.coin_trading.pipelines.train_flow.features import compute_features
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


def _metrics_from_trace(trace: pd.DataFrame) -> dict[str, Any]:
    if trace.empty:
        return {"steps": 0, "final_equity": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "win_rate": 0.0}
    reward = trace["reward"].astype(float)
    sharpe = 0.0
    if reward.std(ddof=0) > 0:
        sharpe = float((reward.mean() / reward.std(ddof=0)) * np.sqrt(252.0))
    return {
        "steps": int(len(trace)),
        "final_equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": float(trace["drawdown"].max()) if "drawdown" in trace else 0.0,
        "turnover": float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0,
        "win_rate": float((reward > 0).mean()),
    }


def _run_baseline(
    policy,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: AppConfig,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    env = build_env(candles_df, features_df, cfg)
    obs, _ = env.reset(seed=cfg.seed)
    policy.reset()
    done = False
    while not done:
        t = int(env.env.t)
        row = candles_df.iloc[min(t, len(candles_df) - 1)]
        feat_row = features_df.iloc[min(t, len(features_df) - 1)] if not features_df.empty else {}
        action = float(policy.act(obs, {"close": float(row.get("close", 0.0)), "logret_1": float(feat_row.get("logret_1", 0.0))}))
        obs, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
        done = terminated or truncated

    trace = env.env.recorder.to_dataframe()
    metrics = _metrics_from_trace(trace)
    if artifacts_dir is not None:
        files = env.env.recorder.write_trace_artifacts(artifacts_dir)
        metrics["artifacts"] = {k: str(v) for k, v in files.items()}
    return metrics


def _build_cost_scenarios(cfg: AppConfig) -> dict[str, AppConfig]:
    scenarios = {"base": cfg}
    for fee, name in ((0.0004, "cost_0.04pct"), (0.0008, "cost_0.08pct")):
        cfg_s = cfg.model_copy(deep=True)
        cfg_s.execution.fee_rate = fee
        scenarios[name] = cfg_s
    return scenarios


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

    train_metrics = rollout_model(best_model, train_df, train_features, cfg, run_dir / "train_trace")
    val_final = rollout_model(best_model, val_df, val_features, cfg, run_dir / "val_trace")
    test_metrics = rollout_model(best_model, test_df, test_features, cfg, run_dir / "test_trace") if not test_df.empty else {"enabled": False}

    baseline_window_df = test_df if not test_df.empty else val_df
    baseline_window_features = test_features if not test_df.empty else val_features
    baseline_comparison = {
        "buy_and_hold": _run_baseline(BuyAndHold(), baseline_window_df, baseline_window_features, cfg),
        "ma_crossover": _run_baseline(MACrossover(), baseline_window_df, baseline_window_features, cfg),
        "random": _run_baseline(RandomPolicy(seed=cfg.seed), baseline_window_df, baseline_window_features, cfg),
    }

    sensitivity: dict[str, Any] = {"scenarios": {}}
    scenarios = _build_cost_scenarios(cfg)
    for name, cfg_s in scenarios.items():
        rl_metrics = rollout_model(best_model, baseline_window_df, baseline_window_features, cfg_s)
        sensitivity["scenarios"][name] = {
            "rl": {
                "tail_risk": rl_metrics.get("max_drawdown", 0.0),
                "max_drawdown": rl_metrics.get("max_drawdown", 0.0),
                "turnover": rl_metrics.get("turnover", 0.0),
                "final_equity": rl_metrics.get("final_equity", 0.0),
                "sharpe": rl_metrics.get("sharpe", 0.0),
            },
            "baselines": {
                "buy_and_hold": _run_baseline(BuyAndHold(), baseline_window_df, baseline_window_features, cfg_s),
                "ma_crossover": _run_baseline(MACrossover(), baseline_window_df, baseline_window_features, cfg_s),
                "random": _run_baseline(RandomPolicy(seed=cfg.seed), baseline_window_df, baseline_window_features, cfg_s),
            },
        }
    (run_dir / "baseline_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")

    overfit_warning = bool(test_metrics.get("enabled", True)) and detect_overfit(train_metrics, test_metrics)
    stats_trace_path = run_dir / ("test_trace" if bool(test_metrics.get("enabled", True)) else "val_trace") / "trace.csv"
    trade_stats_report = write_trade_stats_report(run_dir, pd.read_csv(stats_trace_path), overfit_warning)

    write_learning_curve_artifacts(history, run_dir)

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "artifacts/model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "baseline_comparison": baseline_comparison,
        "overfit_warning": overfit_warning,
        "reports": {"trade_stats_html": trade_stats_report},
        "artifacts": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "learning_curve_svg": "learning_curve.svg",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
            "train_trace_dir": "train_trace",
            "val_trace_dir": "val_trace",
            "test_trace_dir": "test_trace",
            "baseline_sensitivity_json": "baseline_sensitivity.json",
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "train": train_metrics,
                "val": val_final,
                "test": test_metrics,
                "baselines": baseline_comparison,
                "overfit_warning": overfit_warning,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
