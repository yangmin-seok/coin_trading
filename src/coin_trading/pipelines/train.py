from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.coin_trading.agents.baselines import VolTarget
from src.coin_trading.config.loader import load_config
from src.coin_trading.config.schema import AppConfig
from data.io import write_candles_parquet
from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv
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


def _generate_bootstrap_candles(cfg: AppConfig, candles_per_split: int = 160) -> pd.DataFrame:
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


def summarize_dataset_for_training(candles_df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    if candles_df.empty:
        return {
            "rows": 0,
            "coverage": None,
            "splits": {
                "train": {"rows": 0},
                "val": {"rows": 0},
                "test": {"rows": 0},
            },
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


def run_training_probe(candles_df: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    """Run 1-epoch baseline rollout and export reward/equity artifacts."""
    if candles_df.empty:
        return {"enabled": False, "reason": "no_data"}

    features_df = compute_offline(candles_df)
    env = TradingEnv(candles_df, features_df, ExecutionModel())
    policy = VolTarget()
    policy.reset()

    total_reward = 0.0
    steps = 0
    done = False
    env.reset()

    max_steps = min(max(len(candles_df) - 1, 0), 1000)
    while not done and steps < max_steps:
        t = env.t
        info_for_policy = {
            "close": float(candles_df.loc[t, "close"]),
            "logret_1": float(features_df.loc[t, "logret_1"]) if pd.notna(features_df.loc[t, "logret_1"]) else 0.0,
        }
        action = policy.act(obs=None, info=info_for_policy)
        _, reward, done, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

    artifacts = env.recorder.write_trace_artifacts(run_dir / "train_probe")
    mean_reward = total_reward / steps if steps > 0 else 0.0
    final_equity = float(env.state.equity)

    summary = {
        "enabled": True,
        "epochs": 1,
        "model": "VolTarget-baseline",
        "steps": int(steps),
        "total_reward": float(total_reward),
        "mean_reward": float(mean_reward),
        "final_equity": final_equity,
        "artifacts": {
            "trace_csv": str(artifacts["csv"].relative_to(run_dir)),
            "reward_equity_svg": str(artifacts["svg"].relative_to(run_dir)),
        },
    }
    (run_dir / "train_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    probe_summary = run_training_probe(candles_df, run_dir)

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
            "epochs": probe_summary.get("epochs", 0),
            "model": probe_summary.get("model", "none"),
            "probe": probe_summary,
        },
    )
    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id


if __name__ == "__main__":
    print(run())
