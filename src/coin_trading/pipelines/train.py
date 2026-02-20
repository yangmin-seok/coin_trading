from __future__ import annotations

import base64
import json
import os
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
    git_sha,
    implementation_hash,
    make_run_id,
    write_data_manifest,
    write_feature_manifest,
    write_train_manifest,
)
from src.coin_trading.pipelines.reporting import (
    create_benchmark_comparison,
    create_common_risk_plots,
    create_split_equity_curves,
    detect_overfit,
    write_trade_stats_report,
)
from data.io import write_candles_parquet
from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv


PRICE_COLUMNS = ["open", "high", "low", "close"]
MAX_ALLOWED_FEATURE_MISSING_RATIO = 0.05


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


def preprocess_training_candles(candles_df: pd.DataFrame, price_fill_method: str | None = None) -> pd.DataFrame:
    if price_fill_method in {"ffill", "pad", "forward_fill"}:
        raise ValueError("Forward fill for price columns is forbidden. Use raw market data without price forward fill.")

    clean = candles_df.copy()
    if clean[PRICE_COLUMNS].isna().any(axis=None):
        raise ValueError("NaN detected in price columns. Forward filling price series is forbidden by preprocessing policy.")
    return clean


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





def _default_split_ranges_for_5m(anchor_end: pd.Timestamp) -> dict[str, tuple[str, str]]:
    test_end = anchor_end.normalize()
    test_start = test_end - pd.DateOffset(months=6) + pd.DateOffset(days=1)
    val_end = test_start - pd.DateOffset(days=1)
    val_start = val_end - pd.DateOffset(months=6) + pd.DateOffset(days=1)
    train_end = val_start - pd.DateOffset(days=1)
    train_start = train_end - pd.DateOffset(months=24) + pd.DateOffset(days=1)
    return {
        "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        "val": (val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
        "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
    }


def _month_span(split_range: tuple[str, str]) -> float:
    start = pd.Timestamp(split_range[0], tz="UTC")
    end = pd.Timestamp(split_range[1], tz="UTC")
    return (end - start).days / 30.44


def _enforce_split_policy(cfg: AppConfig, candles_df: pd.DataFrame) -> dict[str, Any]:
    if candles_df.empty or cfg.interval != "5m":
        return {
            "applied": False,
            "split": {
                "train": cfg.split.train,
                "val": cfg.split.val,
                "test": cfg.split.test,
            },
        }
    train_months = _month_span(cfg.split.train)
    val_months = _month_span(cfg.split.val)
    test_months = _month_span(cfg.split.test)
    valid = train_months >= 24.0 and 3.0 <= val_months <= 6.5 and 3.0 <= test_months <= 6.5
    if valid:
        return {
            "applied": False,
            "split": {
                "train": cfg.split.train,
                "val": cfg.split.val,
                "test": cfg.split.test,
            },
        }
    anchor_end = pd.to_datetime(candles_df["open_time"].max(), unit="ms", utc=True)
    policy = _default_split_ranges_for_5m(anchor_end)
    return {"applied": True, "split": policy}


def _fit_feature_scaler(train_features: pd.DataFrame) -> dict[str, pd.Series]:
    mu = train_features[FEATURE_COLUMNS].mean()
    sigma = train_features[FEATURE_COLUMNS].std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"mu": mu, "sigma": sigma}


def _apply_feature_scaler(features: pd.DataFrame, scaler: dict[str, pd.Series]) -> pd.DataFrame:
    scaled = features.copy()
    scaled[FEATURE_COLUMNS] = (scaled[FEATURE_COLUMNS] - scaler["mu"]) / scaler["sigma"]
    return scaled


def _validate_no_lookahead(candles_df: pd.DataFrame, sample_points: int = 10) -> dict[str, Any]:
    if candles_df.empty or len(candles_df) < 2:
        return {"checked": 0, "passed": True}
    full_features = compute_offline(candles_df)
    indices = np.linspace(1, len(candles_df) - 1, num=min(sample_points, len(candles_df) - 1), dtype=int)
    for idx in sorted(set(int(i) for i in indices)):
        prefix_features = compute_offline(candles_df.iloc[: idx + 1].reset_index(drop=True))
        a = full_features.iloc[idx][FEATURE_COLUMNS].astype(float)
        b = prefix_features.iloc[-1][FEATURE_COLUMNS].astype(float)
        if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
            raise ValueError(f"lookahead_detected_at_index={idx}")
    return {"checked": int(len(set(indices.tolist()))), "passed": True}


def _build_regime_coverage(candles_df: pd.DataFrame, split_df: pd.DataFrame) -> dict[str, Any]:
    if split_df.empty:
        return {"rows": 0, "years": {}, "volatility": {}}
    work = split_df.copy()
    work["year"] = pd.to_datetime(work["open_time"], unit="ms", utc=True).dt.year.astype(str)
    close = work["close"].astype(float)
    work["logret"] = np.log(close / close.shift(1))
    vol = work["logret"].rolling(96, min_periods=10).std()
    q = vol.quantile([0.33, 0.66]).fillna(0.0).to_list()
    bins = [-np.inf, q[0], q[1], np.inf]
    labels = ["low", "mid", "high"]
    work["vol_regime"] = pd.cut(vol, bins=bins, labels=labels)
    year_counts = {str(k): int(v) for k, v in work["year"].value_counts().sort_index().items()}
    vol_counts = {str(k): int(v) for k, v in work["vol_regime"].value_counts(dropna=True).items()}
    return {"rows": int(len(work)), "years": year_counts, "volatility": vol_counts}


def _build_walkforward_splits(cfg: AppConfig, candles_df: pd.DataFrame) -> list[dict[str, tuple[str, str]]]:
    policy = _enforce_split_policy(cfg, candles_df)["split"]
    train_start, train_end = policy["train"]
    val_start, val_end = policy["val"]
    test_start, test_end = policy["test"]
    base = {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }
    runs = []
    shifts = [2 * i for i in range(max(1, cfg.train.walkforward_runs))]
    for shift in reversed(shifts):
        if shift == 0:
            runs.append(base)
            continue
        s = {
            k: (
                (pd.Timestamp(v[0], tz="UTC") - pd.DateOffset(months=shift)).strftime("%Y-%m-%d"),
                (pd.Timestamp(v[1], tz="UTC") - pd.DateOffset(months=shift)).strftime("%Y-%m-%d"),
            )
            for k, v in base.items()
        }
        runs.append(s)
    return runs

def _split_by_date(candles_df: pd.DataFrame, split_range: tuple[str, str]) -> pd.DataFrame:
    if candles_df.empty:
        return candles_df.copy()
    dates = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    mask = (dates >= split_range[0]) & (dates <= split_range[1])
    return candles_df.loc[mask].reset_index(drop=True)


def summarize_dataset_for_training(candles_df: pd.DataFrame, cfg: AppConfig, split_policy: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    if candles_df.empty:
        return {
            "rows": 0,
            "coverage": None,
            "splits": {"train": {"rows": 0}, "val": {"rows": 0}, "test": {"rows": 0}},
            "features": {"rows": 0, "nan_ratio_mean": None},
        }

    split_policy = split_policy or {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
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
            "train": {"range": list(split_policy["train"]), "rows": _rows_in_range(*split_policy["train"])},
            "val": {"range": list(split_policy["val"]), "rows": _rows_in_range(*split_policy["val"])},
            "test": {"range": list(split_policy["test"]), "rows": _rows_in_range(*split_policy["test"])},
        },
        "features": {
            "rows": int(len(features_df)),
            "nan_ratio_mean": float(feature_nan_ratio.mean()),
            "nan_ratio_by_feature": {k: float(v) for k, v in feature_nan_ratio.to_dict().items()},
        },
    }


def _write_fallback_png(path: Path) -> None:
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8t2b8AAAAASUVORK5CYII="
    )
    path.write_bytes(tiny_png)


def _save_line_chart_png(x: list[int], series: dict[str, list[float]], out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_fallback_png(out_path)
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, vals in series.items():
        ax.plot(x, vals, label=name)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_feature_visualizations(features_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = out_dir / "feature_corr_heatmap.png"
    importance_path = out_dir / "feature_importance_proxy.png"

    subset = features_df[FEATURE_COLUMNS].fillna(0.0).astype(float) if not features_df.empty else pd.DataFrame(columns=FEATURE_COLUMNS)
    corr = subset.corr().fillna(0.0) if not subset.empty else pd.DataFrame(np.eye(len(FEATURE_COLUMNS)), index=FEATURE_COLUMNS, columns=FEATURE_COLUMNS)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_fallback_png(heatmap_path)
        _write_fallback_png(importance_path)
        importance = {c: 0.0 for c in FEATURE_COLUMNS}
        return {
            "feature_corr_heatmap": str(heatmap_path.name),
            "feature_importance_proxy": str(importance_path.name),
            "importance_proxy": importance,
            "max_abs_corr_offdiag": 0.0,
        }

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=7)
    ax.set_title("Feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)

    target = subset["logret_1"].shift(-1).fillna(0.0) if "logret_1" in subset else pd.Series([0.0] * len(subset))
    importance: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        vals = subset[col] if col in subset else pd.Series([0.0] * len(subset))
        coef = float(np.corrcoef(vals.to_numpy(), target.to_numpy())[0, 1]) if len(vals) > 1 else 0.0
        if not np.isfinite(coef):
            coef = 0.0
        importance[col] = abs(coef)

    order = sorted(importance, key=importance.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = list(range(len(order)))
    ax.bar(x_pos, [importance[k] for k in order], color="#2a9d8f")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
    ax.set_title("Feature importance proxy (|corr with next logret_1|)")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=160)
    plt.close(fig)

    abs_corr = corr.abs().to_numpy()
    max_abs_corr = 0.0
    if abs_corr.size:
        mask = ~np.eye(abs_corr.shape[0], dtype=bool)
        max_abs_corr = float(abs_corr[mask].max()) if mask.any() else 0.0

    return {
        "feature_corr_heatmap": str(heatmap_path.name),
        "feature_importance_proxy": str(importance_path.name),
        "importance_proxy": importance,
        "max_abs_corr_offdiag": max_abs_corr,
    }


def _run_transaction_cost_impact_experiment(candles_df: pd.DataFrame, features_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transaction_cost_impact.png"
    if candles_df.empty or len(candles_df) < 5:
        _write_fallback_png(out_path)
        return {"artifact": out_path.name, "warning": "insufficient_rows"}

    sample_len = int(min(120, len(candles_df)))
    sample_candles = candles_df.iloc[:sample_len].reset_index(drop=True)
    sample_features = features_df.iloc[:sample_len].reset_index(drop=True)

    series: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    for label, model in {
        "cost_on": ExecutionModel(),
        "cost_off": ExecutionModel(fee_rate=0.0, slippage_bps=0.0),
    }.items():
        env = TradingEnv(sample_candles, sample_features, model)
        env.reset()
        done = False
        i = 0
        rewards: list[float] = []
        equities: list[float] = []
        while not done:
            action = 0.9 if (i % 8) < 4 else 0.1
            _, reward, done, info = env.step(action)
            if "equity" in info:
                rewards.append(float(reward))
                equities.append(float(info["equity"]))
            i += 1
        series[f"{label}_equity"] = equities
        summary[f"{label}_final_equity"] = equities[-1] if equities else 0.0
        summary[f"{label}_avg_reward"] = float(np.mean(rewards)) if rewards else 0.0

    x = list(range(max((len(v) for v in series.values()), default=0)))
    aligned = {k: (v + [v[-1]] * (len(x) - len(v)) if v else [0.0] * len(x)) for k, v in series.items()}
    _save_line_chart_png(x, aligned, out_path, "Transaction cost on/off impact")
    return {"artifact": out_path.name, **summary, "sample_rows": sample_len}


def _compute_warning_flags(features_info: dict[str, Any], reward_metrics: dict[str, Any]) -> dict[str, Any]:
    max_abs_corr = float(features_info.get("max_abs_corr_offdiag", 0.0))
    reward_abs_max = float(reward_metrics.get("reward_abs_max", 0.0))
    reward_std = float(reward_metrics.get("reward_std", 0.0))
    return {
        "feature_redundancy_high": max_abs_corr >= 0.95,
        "reward_scale_outlier": (reward_abs_max >= 0.2) or (reward_std >= 0.05),
        "thresholds": {
            "feature_redundancy_max_abs_corr": 0.95,
            "reward_abs_max": 0.2,
            "reward_std": 0.05,
        },
        "observed": {
            "max_abs_corr_offdiag": max_abs_corr,
            "reward_abs_max": reward_abs_max,
            "reward_std": reward_std,
        },
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write_metadata(
    artifacts_dir: Path,
    *,
    seed: int,
    start_time_utc: str,
    git_sha_value: str,
    data_range: dict[str, list[str]],
) -> None:
    metadata = {
        "seed": seed,
        "git_sha": git_sha_value,
        "start_time_utc": start_time_utc,
        "data_range": data_range,
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
        "reward_abs_max": float(reward.abs().max()),
        "reward_std": float(reward.std(ddof=0)),
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
    return metrics, trace


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


def _safe_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64") if values else pd.Series(dtype="float64")


def _render_training_artifacts(
    run_dir: Path,
    train_progress: list[dict[str, float]],
    step_trace: list[dict[str, float]],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if plt is None:
        return artifacts

    progress_df = pd.DataFrame(train_progress)
    trace_df = pd.DataFrame(step_trace)

    if not progress_df.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(progress_df["timesteps"], progress_df["episode_reward"], color="#1f77b4")
        axes[0].set_ylabel("episode reward")
        axes[0].grid(alpha=0.2)

        axes[1].plot(progress_df["timesteps"], progress_df["avg_return"], color="#2ca02c")
        axes[1].set_ylabel("avg return")
        axes[1].grid(alpha=0.2)

        axes[2].plot(progress_df["timesteps"], progress_df["explained_variance"], color="#d62728")
        axes[2].set_ylabel("explained variance")
        axes[2].set_xlabel("timesteps")
        axes[2].grid(alpha=0.2)

        fig.tight_layout()
        train_curve_path = run_dir / "train_curve.png"
        fig.savefig(train_curve_path, dpi=120)
        plt.close(fig)
        artifacts["train_curve_png"] = train_curve_path.name

        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        components = ["policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction"]
        colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
        for ax, key, color in zip(axes, components, colors):
            ax.plot(progress_df["timesteps"], progress_df[key], color=color)
            ax.set_ylabel(key)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("timesteps")
        fig.tight_layout()
        loss_components_path = run_dir / "loss_components.png"
        fig.savefig(loss_components_path, dpi=120)
        plt.close(fig)
        artifacts["loss_components_png"] = loss_components_path.name

    if not trace_df.empty and "action" in trace_df:
        action_values = trace_df["action"].astype(float)
        effective_pos = trace_df.get("effective_position", pd.Series([0.0] * len(trace_df))).astype(float)
        leverage = trace_df.get("leverage", pd.Series([0.0] * len(trace_df))).astype(float)
        switch_series = np.sign(effective_pos).diff().fillna(0.0).ne(0.0)
        switch_frequency = float(switch_series.mean()) if len(switch_series) > 0 else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(action_values, bins=20, color="#1f77b4", alpha=0.85)
        axes[0].set_title("Action histogram")
        axes[0].set_xlabel("action")
        axes[0].set_ylabel("count")

        sign_counts = pd.Series(np.sign(effective_pos)).value_counts().reindex([-1.0, 0.0, 1.0], fill_value=0)
        axes[1].bar(["short", "flat", "long"], sign_counts.values, color="#ff7f0e", alpha=0.85)
        axes[1].set_title(f"Position sign counts\nswitch freq={switch_frequency:.3f}")
        axes[1].set_ylabel("count")
        fig.tight_layout()
        action_dist_path = run_dir / "action_distribution.png"
        fig.savefig(action_dist_path, dpi=120)
        plt.close(fig)
        artifacts["action_distribution_png"] = action_dist_path.name

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(effective_pos.values, label="effective_position", color="#2ca02c", linewidth=1.5)
        ax.plot(leverage.values, label="leverage", color="#d62728", linewidth=1.2, alpha=0.9)
        ax.set_title("Position / Leverage exposure")
        ax.set_xlabel("step")
        ax.set_ylabel("exposure")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        exposure_path = run_dir / "leverage_or_position_exposure.png"
        fig.savefig(exposure_path, dpi=120)
        plt.close(fig)
        artifacts["leverage_or_position_exposure_png"] = exposure_path.name

    return artifacts


def _build_warning_events(train_progress: list[dict[str, float]]) -> dict[str, Any]:
    progress_df = pd.DataFrame(train_progress)
    warnings: list[dict[str, Any]] = []
    thresholds = {
        "entropy_drop_ratio": 0.4,
        "explained_variance_negative_streak": 3,
        "clip_fraction_high": 0.30,
    }

    if progress_df.empty:
        return {"thresholds": thresholds, "events": warnings}

    entropy_series = _safe_series(progress_df["entropy_loss"].astype(float).tolist())
    if len(entropy_series) >= 2:
        entropy_abs = entropy_series.abs()
        baseline = float(entropy_abs.iloc[0])
        latest = float(entropy_abs.iloc[-1])
        if baseline > 0 and latest < baseline * thresholds["entropy_drop_ratio"]:
            warnings.append(
                {
                    "type": "entropy_drop",
                    "timesteps": int(progress_df["timesteps"].iloc[-1]),
                    "baseline": baseline,
                    "latest": latest,
                }
            )

    explained = progress_df["explained_variance"].astype(float)
    negative_run = 0
    max_negative_run = 0
    for value in explained:
        if value < 0:
            negative_run += 1
        else:
            max_negative_run = max(max_negative_run, negative_run)
            negative_run = 0
    max_negative_run = max(max_negative_run, negative_run)
    if max_negative_run >= thresholds["explained_variance_negative_streak"]:
        warnings.append(
            {
                "type": "explained_variance_negative_streak",
                "max_streak": int(max_negative_run),
                "threshold": int(thresholds["explained_variance_negative_streak"]),
            }
        )

    clip_series = progress_df["clip_fraction"].astype(float)
    high_clip = progress_df.loc[clip_series > thresholds["clip_fraction_high"], ["timesteps", "clip_fraction"]]
    if not high_clip.empty:
        warnings.append(
            {
                "type": "high_clip_fraction",
                "count": int(len(high_clip)),
                "threshold": float(thresholds["clip_fraction_high"]),
                "samples": [
                    {"timesteps": int(row["timesteps"]), "clip_fraction": float(row["clip_fraction"])}
                    for _, row in high_clip.head(10).iterrows()
                ],
            }
        )

    return {"thresholds": thresholds, "events": warnings}


def _train_sb3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
    fold_name: str = "base",
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    _seed_everything(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    lookahead_validation = _validate_no_lookahead(train_df)
    train_features_raw = compute_offline(train_df)
    val_features_raw = compute_offline(val_df)
    test_features_raw = compute_offline(test_df) if not test_df.empty else pd.DataFrame()
    scaler = _fit_feature_scaler(train_features_raw)
    train_features = _apply_feature_scaler(train_features_raw, scaler)
    val_features = _apply_feature_scaler(val_features_raw, scaler)
    test_features = _apply_feature_scaler(test_features_raw, scaler) if not test_df.empty else pd.DataFrame()

    base_exec_model = ExecutionModel(
        fee_rate=cfg.execution.fee_rate,
        slippage_bps=cfg.execution.slippage_bps,
        max_step_change=cfg.execution.max_step_change,
        min_delta=cfg.execution.min_delta,
    )
    train_env = GymTradingEnv(train_df, train_features, base_exec_model, seed=cfg.seed)
    model = _build_sb3_algo(cfg.train.algo, train_env, cfg)

    from stable_baselines3.common.callbacks import BaseCallback

    train_progress: list[dict[str, float]] = []
    step_trace: list[dict[str, float]] = []

    class TrainingMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._episode_rewards: list[float] = []
            self._current_episode_reward = 0.0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            actions = self.locals.get("actions")
            reward_values = self.locals.get("rewards")
            reward = 0.0
            if reward_values is not None:
                reward = float(np.asarray(reward_values).reshape(-1)[0])
            self._current_episode_reward += reward

            action_scalar = 0.0
            if actions is not None:
                action_scalar = float(np.asarray(actions).reshape(-1)[0])

            info = infos[0] if infos else {}
            equity = float(info.get("equity", 0.0))
            position_value = float(info.get("position_value", 0.0))
            leverage = abs(position_value) / max(equity, 1e-12)
            step_trace.append(
                {
                    "timesteps": float(self.num_timesteps),
                    "action": action_scalar,
                    "effective_position": float(info.get("action_effective_pos", 0.0)),
                    "target_position": float(info.get("action_target_pos", action_scalar)),
                    "leverage": float(leverage),
                    "reward": float(reward),
                }
            )

            dones = self.locals.get("dones")
            done = bool(np.asarray(dones).reshape(-1)[0]) if dones is not None else False
            if done:
                self._episode_rewards.append(float(self._current_episode_reward))
                self._current_episode_reward = 0.0
            return True

        def _on_rollout_end(self) -> None:
            values = getattr(self.model.logger, "name_to_value", {})
            episode_reward = self._episode_rewards[-1] if self._episode_rewards else self._current_episode_reward
            avg_return = float(np.mean(self._episode_rewards[-20:])) if self._episode_rewards else float(self._current_episode_reward)
            train_progress.append(
                {
                    "timesteps": float(self.num_timesteps),
                    "episode_reward": float(episode_reward),
                    "avg_return": avg_return,
                    "explained_variance": float(values.get("train/explained_variance", 0.0)),
                    "policy_loss": float(values.get("train/policy_gradient_loss", 0.0)),
                    "value_loss": float(values.get("train/value_loss", 0.0)),
                    "entropy_loss": float(values.get("train/entropy_loss", 0.0)),
                    "approx_kl": float(values.get("train/approx_kl", 0.0)),
                    "clip_fraction": float(values.get("train/clip_fraction", 0.0)),
                }
            )

    callback = TrainingMetricsCallback()

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
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback)
        trained += chunk
        val_metrics = _rollout_model(model, val_df, val_features, cfg, base_exec_model)
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
            checkpoints.append(ckpt_name)

        if cfg.train.early_stop > 0 and stale >= cfg.train.early_stop:
            break

    best_model_path = artifacts_dir / "model.zip"
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
    pd.DataFrame(curve_rows).to_csv(plots_dir / "learning_curve.csv", index=False)
    (plots_dir / "learning_curve.json").write_text(json.dumps(curve_rows, indent=2), encoding="utf-8")
    _render_learning_curve_svg(history, plots_dir / "learning_curve.svg")

    train_artifacts = _render_training_artifacts(run_dir, train_progress, step_trace)
    warning_metrics = _build_warning_events(train_progress)
    metrics_payload = {
        "train_progress": train_progress,
        "action_trace": step_trace,
        "warning_events": warning_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary = {
        "fold_name": fold_name,
        "lookahead_validation": lookahead_validation,
        "enabled": True,
        "model": f"SB3-{cfg.train.algo.upper()}",
        "algo": cfg.train.algo,
        "steps": int(trained),
        "best_model": "artifacts/model.zip" if best_model_path.exists() else None,
        "checkpoints": checkpoints,
        "history": history,
        "val_metrics": val_final,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": baseline_eval,
        "artifacts": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "learning_curve_svg": "learning_curve.svg",
            "metrics_json": "metrics.json",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
            "train_trace_dir": "train_trace",
            "val_trace_dir": "val_trace",
            "test_trace_dir": "test_trace",
            "baseline_sensitivity": "baseline_sensitivity.json",
        },
        "overfit_warning": overfit_warning,
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps({"history": history, "val": val_final, "test": test_metrics, "baseline": baseline_eval}, indent=2),
        encoding="utf-8",
    )
    return summary


def run() -> str:
    cfg = load_config()
    start_time_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = make_run_id()
    run_dir = Path("runs") / run_id
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/default.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    (artifacts_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    _write_metadata(
        artifacts_dir,
        seed=cfg.train.seed if cfg.train.seed is not None else cfg.seed,
        start_time_utc=start_time_utc,
        git_sha_value=git_sha(),
        data_range={
            "train": list(cfg.split.train),
            "val": list(cfg.split.val),
            "test": list(cfg.split.test),
        },
    )

    candles_df, bootstrapped, bootstrap_persisted = ensure_training_candles(cfg)
    split_policy_info = _enforce_split_policy(cfg, candles_df)
    split_policy = split_policy_info["split"]
    dataset_summary = summarize_dataset_for_training(candles_df, cfg, split_policy=split_policy)

    status = "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data"
    walkforward_results: list[dict[str, Any]] = []
    if dataset_summary["rows"] > 0:
        try:
            for idx, split in enumerate(_build_walkforward_splits(cfg, candles_df), start=1):
                train_df = _split_by_date(candles_df, split["train"])
                val_df = _split_by_date(candles_df, split["val"])
                test_df = _split_by_date(candles_df, split["test"])
                fold_summary = _train_sb3(train_df, val_df, test_df, cfg, run_dir / f"wf_{idx}", fold_name=f"wf_{idx}")
                fold_summary["split"] = split
                fold_summary["regime_coverage"] = {
                    "train": _build_regime_coverage(candles_df, train_df),
                    "val": _build_regime_coverage(candles_df, val_df),
                    "test": _build_regime_coverage(candles_df, test_df),
                }
                walkforward_results.append(fold_summary)
            train_summary = {"enabled": True, "walkforward": walkforward_results, "folds": len(walkforward_results)}
        except RuntimeError as exc:
            status = "blocked_missing_dependencies"
            train_summary = {"enabled": False, "reason": "missing_dependencies", "message": str(exc)}
    else:
        train_summary = {"enabled": False, "reason": "no_data"}

    (reports_dir / "model_train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

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
            "split_policy": split_policy_info,
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
    warning_flags = _compute_warning_flags(feature_viz, train_summary.get("val_metrics", {}))
    metrics_payload = {
        "warning_flags": warning_flags,
        "feature_visualization": {
            "feature_corr_heatmap": feature_viz["feature_corr_heatmap"],
            "feature_importance_proxy": feature_viz["feature_importance_proxy"],
        },
        "transaction_cost_impact": cost_impact,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    write_train_manifest(
        run_dir,
        {
            "status": status,
            "missing": [] if status == "ready" else ["stable-baselines3/gymnasium dependencies"],
            "split_rows": {k: v["rows"] for k, v in dataset_summary["splits"].items()},
            "epochs": 0,
            "model": train_summary.get("model", "none"),
            "model_train": train_summary,
            "artifacts": {
                "feature_corr_heatmap": feature_viz["feature_corr_heatmap"],
                "feature_importance_proxy": feature_viz["feature_importance_proxy"],
                "transaction_cost_impact": cost_impact["artifact"],
            },
        },
    )
    (reports_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id


if __name__ == "__main__":
    print(run())
