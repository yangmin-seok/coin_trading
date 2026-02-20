from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import binascii
import struct
import zlib

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

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
from data.io import write_candles_parquet
from env.execution_model import ExecutionModel


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


def _write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].astype(np.uint8).tobytes() for row in range(height))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(raw, level=6))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(header + ihdr + idat + iend)


def _sparkline_image(series: pd.Series, width: int = 1200, height: int = 420, line_color: tuple[int, int, int] = (31, 119, 180)) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    vals = series.to_numpy(dtype=float)
    if len(vals) < 2:
        return img
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if ymax == ymin:
        ymax = ymin + 1.0
    xs = np.linspace(20, width - 20, len(vals)).astype(int)
    ys = (height - 20 - ((vals - ymin) / (ymax - ymin) * (height - 40))).astype(int)
    ys = np.clip(ys, 0, height - 1)
    for i in range(1, len(xs)):
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(steps + 1):
            x = int(x0 + (x1 - x0) * t / steps)
            y = int(y0 + (y1 - y0) * t / steps)
            img[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = line_color
    return img


def _hist_image(values: np.ndarray, width: int = 1200, height: int = 420, bins: int = 60) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    peak = hist.max() if len(hist) else 0.0
    if peak <= 0:
        return img
    bar_w = max(1, (width - 40) // bins)
    for idx, count in enumerate(hist):
        h = int((count / peak) * (height - 40))
        x0 = 20 + idx * bar_w
        x1 = min(width - 20, x0 + bar_w - 1)
        y0 = height - 20 - h
        img[y0 : height - 20, x0:x1] = (148, 103, 189)
    return img


def generate_data_diagnostics(candles_df: pd.DataFrame, run_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    data_coverage_path = plots_dir / "data_coverage.png"
    price_volume_path = plots_dir / "price_volume_overview.png"
    returns_distribution_path = plots_dir / "returns_distribution.png"
    missing_heatmap_path = plots_dir / "missingness_heatmap.png"

    if candles_df.empty:
        blank = np.full((420, 1200, 3), 255, dtype=np.uint8)
        for p in [data_coverage_path, price_volume_path, returns_distribution_path, missing_heatmap_path]:
            _write_png(p, blank)
        warnings.append("Dataset is empty; diagnostics plots contain no data.")
        (reports_dir / "data_quality.html").write_text(
            "<html><body><h2>Data Quality Warnings</h2><ul><li>Dataset is empty.</li></ul></body></html>",
            encoding="utf-8",
        )
        return {"warnings": warnings, "plots": [str(p.relative_to(run_dir)) for p in [data_coverage_path, price_volume_path, returns_distribution_path, missing_heatmap_path]]}

    df = candles_df.sort_values("open_time").reset_index(drop=True)
    step_ms = _interval_to_ms(cfg.interval)
    diffs = df["open_time"].diff().fillna(step_ms)
    gap_counts = ((diffs // step_ms) - 1).clip(lower=0)

    rows_per_day = df.assign(date=pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date).groupby("date").size().astype(float)
    coverage_img = _sparkline_image(diffs.astype(float), line_color=(31, 119, 180))
    coverage_img[220:] = _sparkline_image(rows_per_day, width=1200, height=200, line_color=(44, 160, 44))[:200]
    _write_png(data_coverage_path, coverage_img)

    price_img = _sparkline_image(df["close"].astype(float), line_color=(31, 119, 180))
    vol_img = _sparkline_image(df["volume"].astype(float), line_color=(255, 127, 14))
    price_img[260:] = vol_img[:160]
    _write_png(price_volume_path, price_img)

    log_ret = np.log(df["close"].astype(float)).diff().dropna().to_numpy(dtype=float)
    tail = np.abs(log_ret)
    hist_img = _hist_image(log_ret)
    tail_img = _hist_image(tail)
    returns_img = np.concatenate([hist_img[:, :600], tail_img[:, 600:]], axis=1)
    _write_png(returns_distribution_path, returns_img)

    features_df = compute_offline(df)
    missing_matrix = features_df[FEATURE_COLUMNS].isna().astype(np.uint8).to_numpy().T
    heat_h, heat_w = missing_matrix.shape
    scale_y = max(1, 420 // max(1, heat_h))
    scale_x = max(1, 1200 // max(1, heat_w))
    heat = np.kron(missing_matrix, np.ones((scale_y, scale_x), dtype=np.uint8))
    heat = heat[:420, :1200]
    heat_rgb = np.full((heat.shape[0], heat.shape[1], 3), 255, dtype=np.uint8)
    heat_rgb[heat == 1] = (68, 1, 84)
    _write_png(missing_heatmap_path, heat_rgb)

    missing_gaps = int(gap_counts.sum())
    if missing_gaps > 0:
        warnings.append(f"Detected candle gaps: missing candles={missing_gaps}.")

    feature_missing_ratio = features_df[FEATURE_COLUMNS].isna().mean()
    over_threshold = feature_missing_ratio[feature_missing_ratio > MAX_ALLOWED_FEATURE_MISSING_RATIO]
    if not over_threshold.empty:
        detail = ", ".join(f"{name}: {ratio:.2%}" for name, ratio in over_threshold.items())
        warnings.append(f"Abnormal feature missingness ratio detected (> {MAX_ALLOWED_FEATURE_MISSING_RATIO:.0%}): {detail}")

    warning_items = "".join(f"<li>{w}</li>" for w in warnings) if warnings else "<li>No warnings.</li>"
    (reports_dir / "data_quality.html").write_text(
        (
            "<html><body>"
            "<h2>Data Quality Warnings</h2>"
            f"<p>run_id={run_dir.name}</p>"
            f"<ul>{warning_items}</ul>"
            "</body></html>"
        ),
        encoding="utf-8",
    )

    return {
        "warnings": warnings,
        "plots": [
            str(data_coverage_path.relative_to(run_dir)),
            str(price_volume_path.relative_to(run_dir)),
            str(returns_distribution_path.relative_to(run_dir)),
            str(missing_heatmap_path.relative_to(run_dir)),
        ],
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


def _rollout_model(model: Any, candles_df: pd.DataFrame, features_df: pd.DataFrame, cfg: AppConfig, artifacts_dir: Path | None = None) -> dict[str, Any]:
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
    metrics = {
        "steps": int(len(trace)),
        "final_equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": float(trace["drawdown"].max()) if "drawdown" in trace else 0.0,
        "turnover": float(trace["filled_qty"].abs().mean()) if "filled_qty" in trace else 0.0,
        "win_rate": float((reward > 0).mean()),
    }
    if artifacts_dir is not None:
        files = eval_env.env.recorder.write_trace_artifacts(artifacts_dir)
        metrics["artifacts"] = {k: str(v) for k, v in files.items()}
    return metrics


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
    plots_dir: Path,
    reports_dir: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    if train_df.empty or val_df.empty:
        return {"enabled": False, "reason": "insufficient_split_rows"}

    _seed_everything(cfg.train.seed if cfg.train.seed is not None else cfg.seed)
    train_features = compute_offline(train_df)
    val_features = compute_offline(val_df)
    test_features = compute_offline(test_df) if not test_df.empty else pd.DataFrame()

    train_env = GymTradingEnv(train_df, train_features, ExecutionModel(), seed=cfg.seed)
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
        val_metrics = _rollout_model(model, val_df, val_features, cfg)
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

    val_final = _rollout_model(best_model, val_df, val_features, cfg, reports_dir / "val_trace")
    test_metrics = _rollout_model(best_model, test_df, test_features, cfg, reports_dir / "test_trace") if not test_df.empty else {"enabled": False}

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
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_json": "learning_curve.json",
            "learning_curve_svg": "learning_curve.svg",
            "metrics_json": "metrics.json",
            "best_model": "best_model.zip" if best_model_path.exists() else None,
            "val_trace_dir": "val_trace",
            "test_trace_dir": "test_trace",
            **train_artifacts,
        },
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps({"history": history, "val": val_final, "test": test_metrics}, indent=2),
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
    candles_df = preprocess_training_candles(candles_df)
    diagnostics = generate_data_diagnostics(candles_df, run_dir, cfg)
    dataset_summary = summarize_dataset_for_training(candles_df, cfg)

    train_df = _split_by_date(candles_df, cfg.split.train)
    val_df = _split_by_date(candles_df, cfg.split.val)
    test_df = _split_by_date(candles_df, cfg.split.test)

    status = "ready" if dataset_summary["rows"] > 0 else "blocked_no_training_data"
    if dataset_summary["rows"] > 0:
        try:
            train_summary = _train_sb3(train_df, val_df, test_df, cfg, plots_dir, reports_dir, artifacts_dir)
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
            "diagnostics": diagnostics,
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
    (reports_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return run_id


if __name__ == "__main__":
    print(run())
