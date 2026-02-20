from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from agents.policy_wrapper import create_policy
from config.loader import load_config
from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv
from features.definitions import FEATURE_COLUMNS
from features.offline import compute_offline
from pipelines.run_manager import (
    implementation_hash,
    make_run_id,
    write_data_manifest,
    write_feature_manifest,
    write_meta,
)


def _make_synthetic_candles(n: int, seed: int, interval: str) -> pd.DataFrame:
    if interval.endswith("m"):
        step_ms = int(interval[:-1]) * 60_000
    elif interval.endswith("h"):
        step_ms = int(interval[:-1]) * 3_600_000
    else:
        step_ms = 300_000

    rng = np.random.default_rng(seed)
    open_time = np.arange(n, dtype=np.int64) * step_ms
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = np.full(n, 10.0)
    close_time = open_time + step_ms - 1
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
            "close_time": close_time,
        }
    )


def _run_baseline_episode(env: TradingEnv, policy_name: str) -> dict[str, float | int | str]:
    policy = create_policy(policy_name)
    obs = env.reset()
    policy.reset()

    done = False
    steps = 0
    reward_sum = 0.0
    while not done:
        info_for_policy = {
            "close": float(env.candles.loc[env.t, "close"]),
            "logret_1": float(env.features.loc[env.t, "logret_1"]),
        }
        action = policy.act(obs, info_for_policy)
        obs, reward, done, info = env.step(action)
        reward_sum += float(reward)
        steps += 1

    return {
        "policy": policy_name,
        "steps": steps,
        "reward_sum": reward_sum,
        "final_equity": float(info["equity"]),
        "max_drawdown": float(max((r.get("drawdown", 0.0) for r in env.recorder.rows), default=0.0)),
    }


def run() -> str:
    cfg = load_config()
    run_id = make_run_id(cfg.mode, cfg.symbol, cfg.interval, cfg.seed)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(Path("config/default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    write_meta(run_dir)

    candles = _make_synthetic_candles(n=500, seed=cfg.seed, interval=cfg.interval)
    feats = compute_offline(candles)
    env = TradingEnv(
        candles,
        feats,
        ExecutionModel(
            fee_rate=cfg.execution.fee_rate,
            slippage_bps=cfg.execution.slippage_bps,
            max_step_change=cfg.execution.max_step_change,
            min_delta=cfg.execution.min_delta,
        ),
        lambda_turnover=cfg.reward.lambda_turnover,
        lambda_dd=cfg.reward.lambda_dd,
        dd_limit=cfg.reward.dd_limit,
    )

    summary = _run_baseline_episode(env, policy_name="buy_and_hold")
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_data_manifest(
        run_dir,
        {
            "exchange": cfg.exchange,
            "market": cfg.market,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "source": "synthetic",
            "rows": len(candles),
            "processed": {"time_unit": "ms"},
        },
    )
    write_feature_manifest(
        run_dir,
        {
            "feature_set_version": cfg.features.version,
            "columns": [{"name": c, "dtype": "float64"} for c in FEATURE_COLUMNS],
            "implementation_hash": implementation_hash([Path("features/common.py"), Path("features/definitions.py")]),
        },
    )
    return run_id


if __name__ == "__main__":
    print(run())
