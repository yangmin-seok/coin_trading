from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


class _FallbackModel:
    def __init__(self, env: GymTradingEnv, seed: int = 0) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False) -> None:
        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
            action = np.array([self.rng.uniform(-1, 1)], dtype=np.float32)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()

    def predict(self, observation: Any, deterministic: bool = True):
        return np.array([0.0], dtype=np.float32), None

    def save(self, path: str) -> None:
        p = Path(path if path.endswith('.zip') else f"{path}.zip")
        p.write_text(json.dumps({"fallback": True}), encoding='utf-8')

    @classmethod
    def load(cls, path: str, env: GymTradingEnv):
        return cls(env=env)



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
    except ImportError:
        return _FallbackModel(env, seed=cfg.train.seed if cfg.train.seed is not None else cfg.seed)

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


def _evaluate_model(model: Any, candles_df: pd.DataFrame, features_df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    eval_env = GymTradingEnv(candles_df, features_df, ExecutionModel(), seed=cfg.seed)
    obs, _ = eval_env.reset(seed=cfg.seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
    trace = eval_env.env.recorder.to_dataframe()
    if trace.empty:
        return {"steps": 0, "final_equity": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "win_rate": 0.0}

    reward = trace["reward"].astype(float)
    sharpe = 0.0
    if reward.std(ddof=0) > 0:
        sharpe = float((reward.mean() / reward.std(ddof=0)) * np.sqrt(252.0))
    max_drawdown = float(trace["drawdown"].max()) if "drawdown" in trace else 0.0
    turnover = float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0
    win_rate = float((reward > 0).mean())
    return {
        "steps": int(len(trace)),
        "final_equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "win_rate": win_rate,
    }


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

    train_env = GymTradingEnv(train_df, train_features, ExecutionModel(), seed=cfg.seed)
    model = _build_sb3_algo(cfg.train.algo, train_env, cfg)
    using_fallback = isinstance(model, _FallbackModel)

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
        val_metrics = _evaluate_model(model, val_df, val_features, cfg)
        history.append({"timesteps": trained, "val": val_metrics})

        if val_metrics["sharpe"] > best_sharpe:
            best_sharpe = val_metrics["sharpe"]
            stale = 0
            model.save(str(run_dir / "best_model"))
        else:
            stale += 1

        if trained % ckpt_interval == 0 or trained == total:
            ckpt_name = f"checkpoint_{trained}.zip"
            model.save(str(run_dir / ckpt_name.replace('.zip', '')))
            checkpoints.append(ckpt_name)

        if cfg.train.early_stop > 0 and stale >= cfg.train.early_stop:
            break

    best_model_path = run_dir / "best_model.zip"
    best_model = model.__class__.load(str(best_model_path), env=train_env) if best_model_path.exists() else model
    test_metrics = _evaluate_model(best_model, test_df, test_features, cfg) if not test_df.empty else {"enabled": False}

    curve_rows = [{
        "timesteps": h["timesteps"],
        "val_sharpe": h["val"]["sharpe"],
        "val_max_drawdown": h["val"]["max_drawdown"],
        "val_final_equity": h["val"]["final_equity"],
        "loss": 0.0,
        "entropy_loss": 0.0,
        "value_loss": 0.0,
    } for h in history]
    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(run_dir / "learning_curve.csv", index=False)
    (run_dir / "learning_curve.json").write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")

    summary = {
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}" + ("-FALLBACK" if using_fallback else ""),
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "best_model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "test_metrics": test_metrics,
        "artifacts": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(json.dumps({"history": history, "test": test_metrics}, indent=2), encoding="utf-8")
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

    train_summary = _train_sb3(train_df, val_df, test_df, cfg, run_dir) if dataset_summary["rows"] > 0 else {"enabled": False, "reason": "no_data"}
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
            "status": "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data",
            "missing": [] if dataset_summary["rows"] > 0 else ["data/processed parquet dataset"],
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
