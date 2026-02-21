from __future__ import annotations

import json
import random
import re
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




def _collect_torch_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    try:
        import torch

        info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    except Exception:
        pass
    return info


def _normalize_requested_device(requested_device: str) -> str:
    req = str(requested_device or "auto").strip().lower()
    alias_map = {
        "gpu": "cuda",
    }
    req = alias_map.get(req, req)
    if req in {"auto", "cpu", "cuda"}:
        return req
    if re.fullmatch(r"cuda:\d+", req):
        return req
    raise ValueError(f"unsupported train.device='{requested_device}'. Expected one of: auto, cpu, cuda, cuda:N")


def _resolve_training_device(requested_device: str) -> str:
    req = _normalize_requested_device(requested_device)
    torch_info = _collect_torch_device_info()
    cuda_available = bool(torch_info["cuda_available"])
    cuda_count = int(torch_info["cuda_device_count"])

    if req == "auto":
        return "cuda:0" if cuda_available and cuda_count > 0 else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if not (cuda_available and cuda_count > 0):
            raise RuntimeError("train.device='cuda' requested but CUDA is unavailable")
        return "cuda:0"

    idx = int(req.split(":", 1)[1])
    if not (cuda_available and cuda_count > 0):
        raise RuntimeError(f"train.device='{req}' requested but CUDA is unavailable")
    if idx >= cuda_count:
        raise RuntimeError(f"train.device='{req}' requested but only {cuda_count} CUDA device(s) detected")
    return f"cuda:{idx}"


def build_sb3_algo(algo_name: str, env, cfg: AppConfig, resolved_device: str):
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
            device=resolved_device,
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
            device=resolved_device,
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
    turnover_cap = float(cfg.reward.selection_turnover_target)
    inactivity_target = float(cfg.reward.selection_inactivity_target)
    turnover_penalty = max(0.0, turnover - turnover_cap)
    inactivity_penalty = max(0.0, no_trade_ratio - inactivity_target)
    drawdown_penalty = max(0.0, max_drawdown)
    cost_penalty = total_cost / max(final_equity, 1.0)

    score = (
        (1.5 * sharpe)
        + (1.0 * equity_alpha)
        - (cfg.reward.selection_drawdown_penalty_weight * drawdown_penalty)
        - (cfg.reward.selection_turnover_penalty_weight * turnover_penalty)
        - (1.2 * cost_penalty)
        - (cfg.reward.selection_inactivity_penalty_weight * inactivity_penalty)
    )

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
        "inactivity_target": inactivity_target,
        "turnover_penalty": float(turnover_penalty),
        "drawdown_penalty": float(drawdown_penalty),
        "cost_penalty": float(cost_penalty),
        "inactivity_penalty": float(inactivity_penalty),
        "weights": {
            "drawdown": float(cfg.reward.selection_drawdown_penalty_weight),
            "turnover": float(cfg.reward.selection_turnover_penalty_weight),
            "inactivity": float(cfg.reward.selection_inactivity_penalty_weight),
            "cost": 1.2,
        },
    }


def _score_with_weights(metrics: dict[str, Any], cfg: AppConfig, weights: dict[str, float]) -> dict[str, Any]:
    mod_cfg = cfg.model_copy(deep=True)
    mod_cfg.reward.selection_turnover_penalty_weight = float(weights["turnover"])
    mod_cfg.reward.selection_inactivity_penalty_weight = float(weights["inactivity"])
    mod_cfg.reward.selection_drawdown_penalty_weight = float(weights["drawdown"])
    scored = _compute_model_selection_score(metrics, mod_cfg)
    scored["weights"] = {
        "turnover": float(weights["turnover"]),
        "inactivity": float(weights["inactivity"]),
        "drawdown": float(weights["drawdown"]),
        "cost": 1.2,
    }
    return scored


def _iter_penalty_weights(cfg: AppConfig) -> list[dict[str, float]]:
    if not cfg.reward.penalty_sweep_enabled:
        return [
            {
                "turnover": float(cfg.reward.selection_turnover_penalty_weight),
                "inactivity": float(cfg.reward.selection_inactivity_penalty_weight),
                "drawdown": float(cfg.reward.selection_drawdown_penalty_weight),
            }
        ]

    if cfg.reward.penalty_sweep_mode == "grid":
        combinations: list[dict[str, float]] = []
        for turn in cfg.reward.turnover_penalty_grid:
            for ina in cfg.reward.inactivity_penalty_grid:
                for dd in cfg.reward.drawdown_penalty_grid:
                    combinations.append({"turnover": float(turn), "inactivity": float(ina), "drawdown": float(dd)})
        return combinations

    rng = random.Random(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    turn_vals = cfg.reward.turnover_penalty_grid
    ina_vals = cfg.reward.inactivity_penalty_grid
    dd_vals = cfg.reward.drawdown_penalty_grid
    combinations = []
    for _ in range(int(cfg.reward.penalty_sweep_trials)):
        combinations.append(
            {
                "turnover": float(rng.choice(turn_vals)),
                "inactivity": float(rng.choice(ina_vals)),
                "drawdown": float(rng.choice(dd_vals)),
            }
        )
    return combinations


def _select_best_penalty_weight(metrics: dict[str, Any], cfg: AppConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    best: dict[str, Any] | None = None
    for weights in _iter_penalty_weights(cfg):
        scored = _score_with_weights(metrics, cfg, weights)
        candidates.append(scored)
        if best is None or _is_better_model_candidate(scored, best):
            best = scored
    return (best if best is not None else _compute_model_selection_score(metrics, cfg), candidates)


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

    requested_device = _normalize_requested_device(cfg.train.device)
    resolved_device = _resolve_training_device(requested_device)
    torch_device_info = _collect_torch_device_info()

    train_env = build_env(train_df, train_features, cfg)
    model = build_sb3_algo(cfg.train.algo, train_env, cfg, resolved_device)

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
        candidate, sweep_candidates = _select_best_penalty_weight(val_metrics, cfg)
        candidate_record = {"timesteps": trained, "val": val_metrics, "selection": candidate}
        if cfg.reward.penalty_sweep_enabled:
            candidate_record["selection_sweep"] = sweep_candidates
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
    device_metadata = {
        "requested": requested_device,
        "resolved": resolved_device,
        **torch_device_info,
    }
    metrics_payload = {
        "history": history,
        "selection": selection_summary,
        "val": val_final,
        "test": test_metrics,
        "training_device": device_metadata,
    }

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "device": resolved_device,
        "training_device": device_metadata,
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
            "learning_curve_summary_svg": "plots/learning_curve_summary.svg",
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
    c = candidate.get("val_metrics", {})
    k = current.get("val_metrics", {})

    c_score = (1.2 * float(c.get("sharpe", 0.0) or 0.0)) + (0.001 * float(c.get("final_equity", 0.0) or 0.0)) - (1.0 * float(c.get("max_drawdown", 0.0) or 0.0)) - (0.5 * float(c.get("cost_pnl_ratio", 0.0) or 0.0)) - (0.3 * float(c.get("turnover", 0.0) or 0.0))
    k_score = (1.2 * float(k.get("sharpe", 0.0) or 0.0)) + (0.001 * float(k.get("final_equity", 0.0) or 0.0)) - (1.0 * float(k.get("max_drawdown", 0.0) or 0.0)) - (0.5 * float(k.get("cost_pnl_ratio", 0.0) or 0.0)) - (0.3 * float(k.get("turnover", 0.0) or 0.0))
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
        "criteria": "multi-objective adoption rule on validation: return + risk-adjusted + cost + turnover + consistency",
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
            "val_final_equity": item["summary"].get("val_metrics", {}).get("final_equity", 0.0),
            "val_sharpe": item["summary"].get("val_metrics", {}).get("sharpe", 0.0),
            "val_turnover": item["summary"].get("val_metrics", {}).get("turnover", 0.0),
            "val_total_cost": item["summary"].get("val_metrics", {}).get("total_cost", 0.0),
            "val_max_drawdown": item["summary"].get("val_metrics", {}).get("max_drawdown", 0.0),
            "val_trade_count": item["summary"].get("val_metrics", {}).get("trade_count", 0.0),
            "val_no_trade_ratio": item["summary"].get("val_metrics", {}).get("no_trade_ratio", 0.0),
            "val_avg_abs_position": item["summary"].get("val_metrics", {}).get("avg_abs_position", 0.0),
            "val_excess_vs_cash": item["summary"].get("val_metrics", {}).get("excess_vs_baseline", {}).get("cash_hold", 0.0),
            "val_excess_vs_buy_hold": item["summary"].get("val_metrics", {}).get("excess_vs_baseline", {}).get("buy_hold", 0.0),
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
