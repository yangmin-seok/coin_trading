from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.coin_trading.agents.baselines import BuyAndHold, MACrossover, RandomPolicy
from src.coin_trading.agents.sb3_env import GymTradingEnv
from src.coin_trading.config.loader import load_config
from src.coin_trading.config.schema import AppConfig
from src.coin_trading.features.definitions import FEATURE_COLUMNS
from src.coin_trading.features.offline import compute_offline
from src.coin_trading.pipelines.run_manager import (
    implementation_hash,
    make_run_id,
    write_data_manifest,
    write_feature_manifest,
    write_meta,
    write_train_manifest,
)
from data.io import write_candles_parquet
from env.execution_model import ExecutionModel


def _train_data_glob(cfg: AppConfig) -> str:
    return (
        f"exchange={cfg.exchange}/market={cfg.market}/symbol={cfg.symbol}/"
        f"interval={cfg.interval}/date=*/part-*.parquet"
    )


def load_training_candles(cfg: AppConfig, data_root: Path = Path("data/processed")) -> pd.DataFrame:
    files = sorted(data_root.glob(_train_data_glob(cfg)))
    if not files:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    candles = pd.read_parquet(files)
    candles = candles.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return candles


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"unsupported interval: {interval}")


def _generate_bootstrap_candles(cfg: AppConfig, candles_per_split: int = 240) -> pd.DataFrame:
    step_ms = _interval_to_ms(cfg.interval)
    rng = np.random.default_rng(cfg.seed)
    split_starts = [cfg.split.train[0], cfg.split.val[0], cfg.split.test[0]]
    rows: list[dict[str, float | int]] = []
    price = 100.0
    for split_start in split_starts:
        start_ts = int(pd.Timestamp(split_start, tz="UTC").timestamp() * 1000)
        for i in range(candles_per_split):
            open_time = start_ts + i * step_ms
            close_time = open_time + step_ms - 1
            open_price = price
            ret = float(rng.normal(0, 0.002))
            close_price = max(0.1, open_price * (1 + ret))
            high = max(open_price, close_price) * (1 + abs(float(rng.normal(0, 0.0008))))
            low = min(open_price, close_price) * (1 - abs(float(rng.normal(0, 0.0008))))
            volume = float(8 + abs(rng.normal(0, 1.5)))
            rows.append(
                {
                    "open_time": int(open_time),
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close_price),
                    "volume": float(volume),
                    "close_time": int(close_time),
                }
            )
            price = close_price
    return pd.DataFrame(rows)


def ensure_training_candles(cfg: AppConfig, data_root: Path = Path("data/processed")) -> tuple[pd.DataFrame, bool, bool]:
    candles_df = load_training_candles(cfg, data_root=data_root)
    if not candles_df.empty:
        return candles_df, False, False

    bootstrap_df = _generate_bootstrap_candles(cfg)
    try:
        write_candles_parquet(
            bootstrap_df,
            root=data_root,
            exchange=cfg.exchange,
            market=cfg.market,
            symbol=cfg.symbol,
            interval=cfg.interval,
        )
        return load_training_candles(cfg, data_root=data_root), True, True
    except (ImportError, ModuleNotFoundError):
        return bootstrap_df.sort_values("open_time").reset_index(drop=True), True, False


def _split_by_date(candles_df: pd.DataFrame, split_range: tuple[str, str]) -> pd.DataFrame:
    if candles_df.empty:
        return candles_df.copy()
    dates = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    mask = (dates >= split_range[0]) & (dates <= split_range[1])
    return candles_df.loc[mask].reset_index(drop=True)


def summarize_dataset_for_training(candles_df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    if candles_df.empty:
        return {
            "rows": 0,
            "coverage": None,
            "splits": {"train": {"rows": 0}, "val": {"rows": 0}, "test": {"rows": 0}},
            "features": {"rows": 0, "nan_ratio_mean": None},
        }

    dates = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    def _rows_in_range(start: str, end: str) -> int:
        return int(((dates >= start) & (dates <= end)).sum())

    features_df = compute_offline(candles_df)
    feature_nan_ratio = features_df[FEATURE_COLUMNS].isna().mean()

    return {
        "rows": int(len(candles_df)),
        "coverage": {
            "start_open_time": int(candles_df["open_time"].iloc[0]),
            "end_open_time": int(candles_df["open_time"].iloc[-1]),
        },
        "splits": {
            "train": {"range": list(cfg.split.train), "rows": _rows_in_range(*cfg.split.train)},
            "val": {"range": list(cfg.split.val), "rows": _rows_in_range(*cfg.split.val)},
            "test": {"range": list(cfg.split.test), "rows": _rows_in_range(*cfg.split.test)},
        },
        "features": {
            "rows": int(len(features_df)),
            "nan_ratio_mean": float(feature_nan_ratio.mean()),
            "nan_ratio_by_feature": {k: float(v) for k, v in feature_nan_ratio.to_dict().items()},
        },
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _build_sb3_algo(algo_name: str, env: GymTradingEnv, cfg: AppConfig):
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


def _compute_trace_metrics(trace: pd.DataFrame) -> dict[str, Any]:
    if trace.empty:
        return {
            "steps": 0,
            "final_equity": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "tail_risk": 0.0,
            "win_rate": 0.0,
        }

    reward = trace["reward"].astype(float)
    sharpe = 0.0
    if reward.std(ddof=0) > 0:
        sharpe = float((reward.mean() / reward.std(ddof=0)) * np.sqrt(252.0))

    tail_risk = float(np.percentile(reward, 5)) if len(reward) > 0 else 0.0
    return {
        "steps": int(len(trace)),
        "final_equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": float(trace["drawdown"].max()) if "drawdown" in trace else 0.0,
        "turnover": float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0,
        "tail_risk": tail_risk,
        "win_rate": float((reward > 0).mean()),
    }


def _run_policy_rollout(
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    execution_model: ExecutionModel,
    action_fn,
    seed: int,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    eval_env = GymTradingEnv(candles_df, features_df, execution_model, seed=seed)
    obs, _ = eval_env.reset(seed=seed)
    done = False
    while not done:
        action = float(action_fn(obs, eval_env.env))
        obs, _, terminated, truncated, _ = eval_env.step(np.array([action], dtype=np.float32))
        done = terminated or truncated

    trace = eval_env.env.recorder.to_dataframe()
    metrics = _compute_trace_metrics(trace)
    if artifacts_dir is not None:
        files = eval_env.env.recorder.write_trace_artifacts(artifacts_dir)
        metrics["artifacts"] = {k: str(v) for k, v in files.items()}
    return metrics


def _rollout_model(
    model: Any,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: AppConfig,
    execution_model: ExecutionModel,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    return _run_policy_rollout(
        candles_df,
        features_df,
        execution_model,
        action_fn=lambda obs, _env: model.predict(obs, deterministic=True)[0],
        seed=cfg.seed,
        artifacts_dir=artifacts_dir,
    )


def _rollout_baseline(
    policy: Any,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: AppConfig,
    execution_model: ExecutionModel,
) -> dict[str, Any]:
    policy.reset()
    return _run_policy_rollout(
        candles_df,
        features_df,
        execution_model,
        action_fn=lambda obs, env: policy.act(obs, {"close": float(env.candles.loc[env.t, "close"])}),
        seed=cfg.seed,
    )


def _evaluate_baselines_and_sensitivity(
    model: Any,
    val_df: pd.DataFrame,
    val_features: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> dict[str, Any]:
    scenarios = [
        {
            "name": "base",
            "fee_rate": float(cfg.execution.fee_rate),
            "slippage_bps": float(cfg.execution.slippage_bps),
            "max_step_change": float(cfg.execution.max_step_change),
            "min_delta": float(cfg.execution.min_delta),
        },
        {
            "name": "cost_0.04pct",
            "fee_rate": 0.0004,
            "slippage_bps": 4.0,
            "max_step_change": float(cfg.execution.max_step_change),
            "min_delta": float(cfg.execution.min_delta),
        },
        {
            "name": "cost_0.08pct",
            "fee_rate": 0.0008,
            "slippage_bps": 8.0,
            "max_step_change": float(cfg.execution.max_step_change),
            "min_delta": float(cfg.execution.min_delta),
        },
    ]

    baseline_policies = {
        "buy_and_hold": BuyAndHold(),
        "ma_crossover": MACrossover(),
        "random": RandomPolicy(seed=cfg.seed),
    }

    output: dict[str, Any] = {"scenarios": {}}
    for scenario in scenarios:
        exec_model = ExecutionModel(
            fee_rate=scenario["fee_rate"],
            slippage_bps=scenario["slippage_bps"],
            max_step_change=scenario["max_step_change"],
            min_delta=scenario["min_delta"],
        )
        rl_metrics = _rollout_model(model, val_df, val_features, cfg, exec_model)
        baselines = {
            name: _rollout_baseline(policy, val_df, val_features, cfg, exec_model)
            for name, policy in baseline_policies.items()
        }
        output["scenarios"][scenario["name"]] = {
            "cost": scenario,
            "rl": rl_metrics,
            "baselines": baselines,
        }

    base = output["scenarios"]["base"]
    rl_equity = float(base["rl"]["final_equity"])
    worse_than_all = all(rl_equity < float(m["final_equity"]) for m in base["baselines"].values())
    output["rl_warning"] = (
        {
            "level": "warning",
            "code": "RL_BASELINE_UNDERPERFORM",
            "message": "RL policy underperforms Buy&Hold, MA crossover, and Random on the same window. Check reward/env design.",
        }
        if worse_than_all
        else None
    )
    (run_dir / "baseline_sensitivity.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _render_learning_curve_svg(history: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 900, 340
    pad_l, pad_r, pad_t, pad_b = 70, 20, 20, 45
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    sharpe = [float(item["val"]["sharpe"]) for item in history]
    equity = [float(item["val"]["final_equity"]) for item in history]

    def _scale(vals: list[float]) -> list[float]:
        if not vals:
            return []
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            vmax = vmin + 1.0
        out: list[float] = []
        for v in vals:
            ratio = (v - vmin) / (vmax - vmin)
            out.append(pad_t + (1 - ratio) * chart_h)
        return out

    y1 = _scale(sharpe)
    y2 = _scale(equity)

    def _line(vals: list[float], color: str) -> str:
        if not vals:
            return ""
        n = max(1, len(vals) - 1)
        pts = [f"{pad_l + (i / n) * chart_w:.2f},{yy:.2f}" for i, yy in enumerate(vals)]
        return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}' />"

    y_ticks = "".join(
        f"<text x='{pad_l-8}' y='{pad_t + i * chart_h / 4:.1f}' text-anchor='end' font-size='10' fill='#666'>{4-i}</text>"
        for i in range(5)
    )

    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"<line x1='{pad_l}' y1='{pad_t + chart_h}' x2='{pad_l + chart_w}' y2='{pad_t + chart_h}' stroke='#999'/>"
        f"{_line(y1, '#1f77b4')}{_line(y2, '#2ca02c')}"
        f"{y_ticks}"
        f"<text x='55' y='18' font-size='12' fill='#1f77b4'>val_sharpe</text>"
        f"<text x='170' y='18' font-size='12' fill='#2ca02c'>val_final_equity</text>"
        f"<text x='{pad_l + chart_w/2:.1f}' y='{height-8}' text-anchor='middle' font-size='11' fill='#666'>evaluation step index (x-axis)</text>"
        f"<text x='14' y='{pad_t + chart_h/2:.1f}' font-size='11' fill='#666' transform='rotate(-90 14,{pad_t + chart_h/2:.1f})'>normalized metric value (y-axis)</text>"
        f"</svg>"
    )
    out_path.write_text(svg, encoding="utf-8")


def _train_sb3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> dict[str, Any]:
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    _seed_everything(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    train_features = compute_offline(train_df)
    val_features = compute_offline(val_df)
    test_features = compute_offline(test_df) if not test_df.empty else pd.DataFrame()

    base_exec_model = ExecutionModel(
        fee_rate=cfg.execution.fee_rate,
        slippage_bps=cfg.execution.slippage_bps,
        max_step_change=cfg.execution.max_step_change,
        min_delta=cfg.execution.min_delta,
    )
    train_env = GymTradingEnv(train_df, train_features, base_exec_model, seed=cfg.seed)
    model = _build_sb3_algo(cfg.train.algo, train_env, cfg)

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
        val_metrics = _rollout_model(model, val_df, val_features, cfg, base_exec_model)
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

    val_final = _rollout_model(best_model, val_df, val_features, cfg, base_exec_model, run_dir / "val_trace")
    test_metrics = (
        _rollout_model(best_model, test_df, test_features, cfg, base_exec_model, run_dir / "test_trace")
        if not test_df.empty
        else {"enabled": False}
    )
    baseline_eval = _evaluate_baselines_and_sensitivity(best_model, val_df, val_features, cfg, run_dir)

    curve_rows = [
        {
            "timesteps": h["timesteps"],
            "val_sharpe": h["val"]["sharpe"],
            "val_max_drawdown": h["val"]["max_drawdown"],
            "val_final_equity": h["val"]["final_equity"],
            "loss": 0.0,
            "entropy_loss": 0.0,
            "value_loss": 0.0,
        }
        for h in history
    ]
    pd.DataFrame(curve_rows).to_csv(run_dir / "learning_curve.csv", index=False)
    (run_dir / "learning_curve.json").write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")
    _render_learning_curve_svg(history, run_dir / "learning_curve.svg")

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "best_model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "baseline_comparison": baseline_eval,
        "artifacts": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "learning_curve_svg": "learning_curve.svg",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
            "val_trace_dir": "val_trace",
            "test_trace_dir": "test_trace",
            "baseline_sensitivity": "baseline_sensitivity.json",
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps({"history": history, "val": val_final, "test": test_metrics, "baseline": baseline_eval}, indent=2),
        encoding="utf-8",
    )
    return summary


def run() -> str:
    cfg = load_config()
    run_id = make_run_id(cfg.mode, cfg.symbol, cfg.interval, cfg.seed)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    default_config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    (run_dir / "config.yaml").write_text(default_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_meta(run_dir)

    candles_df, bootstrapped, bootstrap_persisted = ensure_training_candles(cfg)
    dataset_summary = summarize_dataset_for_training(candles_df, cfg)

    train_df = _split_by_date(candles_df, cfg.split.train)
    val_df = _split_by_date(candles_df, cfg.split.val)
    test_df = _split_by_date(candles_df, cfg.split.test)

    status = "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data"
    if dataset_summary["rows"] > 0:
        try:
            train_summary = _train_sb3(train_df, val_df, test_df, cfg, run_dir)
        except RuntimeError as exc:
            status = "blocked_missing_dependencies"
            train_summary = {"enabled": False, "reason": "missing_dependencies", "message": str(exc)}
    else:
        train_summary = {"enabled": False, "reason": "no_data"}

    (run_dir / "model_train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

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
                    Path(__file__).resolve().parents[1] / "features" / "common.py",
                    Path(__file__).resolve().parents[1] / "features" / "definitions.py",
                    Path(__file__).resolve().parents[1] / "features" / "offline.py",
                ]
            ),
        },
    )
    write_train_manifest(
        run_dir,
        {
            "status": status,
            "missing": [] if status == "ready" else ["stable-baselines3/gymnasium dependencies"],
            "split_rows": {k: v["rows"] for k, v in dataset_summary["splits"].items()},
            "epochs": 0,
            "model": train_summary.get("model", "none"),
            "model_train": train_summary,
        },
    )
    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id


if __name__ == "__main__":
    print(run())
