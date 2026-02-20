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
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    turnover = float(metrics.get("turnover", 0.0) or 0.0)
    total_cost = float(metrics.get("total_cost", 0.0) or 0.0)
    no_trade_ratio = float(metrics.get("no_trade_ratio", 0.0) or 0.0)
    baseline = metrics.get("baseline_equity", {})
    baseline_buy_hold = float(baseline.get("buy_hold", 0.0) or 0.0)

    equity_alpha = (final_equity - baseline_buy_hold) / max(abs(baseline_buy_hold), 1.0)
    turnover_cap = float(cfg.execution.max_step_change)
    turnover_penalty = max(0.0, turnover - turnover_cap) * 10.0
    inactivity_penalty = max(0.0, no_trade_ratio - 0.7) * 3.0
    cost_penalty = total_cost / max(final_equity, 1.0)

    score = (1.5 * sharpe) + (1.0 * equity_alpha) - (1.2 * max_drawdown) - (0.8 * turnover_penalty) - (1.2 * cost_penalty) - (0.8 * inactivity_penalty)

    return {
        "score": float(score),
        "sharpe": sharpe,
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "total_cost": total_cost,
        "no_trade_ratio": no_trade_ratio,
        "equity_alpha": float(equity_alpha),
        "turnover_cap": turnover_cap,
        "turnover_penalty": float(turnover_penalty),
        "cost_penalty": float(cost_penalty),
        "inactivity_penalty": float(inactivity_penalty),
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
        "criteria": "maximize multi-objective score (risk-adjusted return + alpha - drawdown/cost/turnover/inactivity penalties)",
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
        "reward_type": cfg.reward.type,
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


def _aggregate_reward_runs(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for exp in experiments:
        grouped.setdefault(str(exp["reward_type"]), []).append(exp)

    table: list[dict[str, Any]] = []
    for reward_type, rows in grouped.items():
        tests = [r["summary"].get("test_metrics", {}) for r in rows if r.get("summary", {}).get("enabled")]
        if not tests:
            continue

        def _mean(metric: str) -> float:
            vals = [float(t.get(metric, 0.0) or 0.0) for t in tests]
            return float(np.mean(vals)) if vals else 0.0

        def _mean_nested(parent: str, child: str) -> float:
            vals = [float((t.get(parent, {}) or {}).get(child, 0.0) or 0.0) for t in tests]
            return float(np.mean(vals)) if vals else 0.0

        table.append(
            {
                "reward_type": reward_type,
                "runs": len(tests),
                "final_equity_mean": _mean("final_equity"),
                "sharpe_mean": _mean("sharpe"),
                "turnover_mean": _mean("turnover"),
                "total_cost_mean": _mean("total_cost"),
                "max_drawdown_mean": _mean("max_drawdown"),
                "trade_count_mean": _mean("trade_count"),
                "no_trade_ratio_mean": _mean("no_trade_ratio"),
                "avg_abs_position_mean": _mean("avg_abs_position"),
                "excess_vs_cash_mean": _mean_nested("excess_vs_baseline", "cash_hold"),
                "excess_vs_buy_hold_mean": _mean_nested("excess_vs_baseline", "buy_hold"),
            }
        )
    table.sort(key=lambda x: x["reward_type"])
    return table


def _is_better_candidate(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    c = candidate.get("test_metrics", {})
    k = current.get("test_metrics", {})

    c_score = (1.2 * float(c.get("sharpe", 0.0) or 0.0)) + (0.001 * float(c.get("final_equity", 0.0) or 0.0)) - (1.0 * float(c.get("max_drawdown", 0.0) or 0.0)) - (0.5 * float(c.get("cost_pnl_ratio", 0.0) or 0.0))
    k_score = (1.2 * float(k.get("sharpe", 0.0) or 0.0)) + (0.001 * float(k.get("final_equity", 0.0) or 0.0)) - (1.0 * float(k.get("max_drawdown", 0.0) or 0.0)) - (0.5 * float(k.get("cost_pnl_ratio", 0.0) or 0.0))
    if c_score > k_score + EPSILON:
        return True
    if abs(c_score - k_score) <= EPSILON:
        return float(c.get("final_equity", 0.0) or 0.0) > float(k.get("final_equity", 0.0) or 0.0)
    return False


def train_sb3(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: AppConfig, run_dir: Path) -> dict[str, Any]:
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    reward_types = list(dict.fromkeys(cfg.reward.comparison_types or [cfg.reward.type]))
    repeats = int(cfg.reward.comparison_repeats)
    experiment_results: list[dict[str, Any]] = []

    idx = 0
    for reward_type in reward_types:
        for repeat_idx in range(1, repeats + 1):
            idx += 1
            exp_cfg = cfg.model_copy(deep=True)
            exp_cfg.reward.type = reward_type
            exp_cfg.train.seed = (cfg.train.seed if cfg.train.seed is not None else cfg.seed) + repeat_idx - 1
            exp_dir = run_dir / f"experiment_{idx:02d}_{reward_type}_r{repeat_idx}"
            summary = _run_single_experiment(train_df, val_df, test_df, exp_cfg, exp_dir)
            experiment_results.append(
                {
                    "id": idx,
                    "reward_type": reward_type,
                    "repeat": repeat_idx,
                    "seed": exp_cfg.train.seed,
                    "run_dir": str(exp_dir.relative_to(run_dir)),
                    "summary": summary,
                }
            )

    selected: dict[str, Any] | None = None
    for candidate in experiment_results:
        if _is_better_candidate(candidate["summary"], selected["summary"] if selected else None):
            selected = candidate

    if selected is None:
        return {"enabled": False, "reason": "no_experiment_result"}

    reward_table = _aggregate_reward_runs(experiment_results)
    selected_summary = selected["summary"]
    selected_summary["selected_experiment"] = {
        "id": selected["id"],
        "reward_type": selected["reward_type"],
        "repeat": selected["repeat"],
        "seed": selected["seed"],
        "run_dir": selected["run_dir"],
        "criteria": "multi-objective adoption rule: return + risk-adjusted + cost + consistency",
    }
    selected_summary["comparison_protocol"] = {
        "scope": [
            "Risk-adjusted RL trading rewards",
            "Differential Sharpe approximation (online)",
            "Downside-risk / Sortino-style penalization",
        ],
        "fixed_split": True,
        "repeats_per_reward": repeats,
        "reward_types": reward_types,
        "metrics": ["final_equity", "sharpe", "turnover", "total_cost", "max_drawdown", "trade_count"],
        "required_baselines": ["cash_hold", "buy_hold"],
        "bias_metrics": ["no_trade_ratio", "avg_abs_position"],
    }
    selected_summary["reward_comparison_table"] = reward_table
    selected_summary["experiments"] = [
        {
            "id": item["id"],
            "reward_type": item["reward_type"],
            "repeat": item["repeat"],
            "seed": item["seed"],
            "run_dir": item["run_dir"],
            "test_final_equity": item["summary"].get("test_metrics", {}).get("final_equity", 0.0),
            "test_sharpe": item["summary"].get("test_metrics", {}).get("sharpe", 0.0),
            "test_turnover": item["summary"].get("test_metrics", {}).get("turnover", 0.0),
            "test_total_cost": item["summary"].get("test_metrics", {}).get("total_cost", 0.0),
            "test_max_drawdown": item["summary"].get("test_metrics", {}).get("max_drawdown", 0.0),
            "test_trade_count": item["summary"].get("test_metrics", {}).get("trade_count", 0.0),
            "test_no_trade_ratio": item["summary"].get("test_metrics", {}).get("no_trade_ratio", 0.0),
            "test_avg_abs_position": item["summary"].get("test_metrics", {}).get("avg_abs_position", 0.0),
            "test_excess_vs_cash": item["summary"].get("test_metrics", {}).get("excess_vs_baseline", {}).get("cash_hold", 0.0),
            "test_excess_vs_buy_hold": item["summary"].get("test_metrics", {}).get("excess_vs_baseline", {}).get("buy_hold", 0.0),
        }
        for item in experiment_results
    ]
    selected_summary["adoption_criteria"] = {
        "description": "Do not adopt on return only. Select reward balancing profitability, risk, trading cost, and consistency across repeats.",
        "objective_axes": ["return", "cost_efficiency", "risk_control", "consistency"],
        "decision_rule": {
            "must_have": [
                "non-negative excess_vs_cash_mean",
                "stable no_trade_ratio_mean (avoid degenerate no-trade policy)",
            ],
            "optimize": ["sharpe_mean", "final_equity_mean", "total_cost_mean", "max_drawdown_mean"],
        },
    }
    return selected_summary
