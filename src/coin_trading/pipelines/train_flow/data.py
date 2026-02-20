"""Training-data utilities used by train_flow.

공개 API:
- load_training_candles
- ensure_training_candles
- split_by_date
- validate_split_policy
- build_walkforward_splits
- plan_walkforward_splits
- summarize_dataset

비공개 API:
- 이름이 ``_`` 로 시작하는 내부 헬퍼(예: ``_train_data_glob``, ``_generate_bootstrap_candles``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.io import write_candles_parquet
from src.coin_trading.config.schema import AppConfig
from src.coin_trading.features.definitions import FEATURE_COLUMNS
from src.coin_trading.features.offline import compute_offline


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


def _generate_bootstrap_candles(
    cfg: AppConfig,
    candles_per_split: dict[str, int] | None = None,
) -> pd.DataFrame:
    candles_per_split = candles_per_split or {"train": 480, "val": 120, "test": 120}
    step_ms = _interval_to_ms(cfg.interval)
    rng = np.random.default_rng(cfg.seed)
    split_starts = {
        "train": cfg.split.train[0],
        "val": cfg.split.val[0],
        "test": cfg.split.test[0],
    }
    rows: list[dict[str, float | int]] = []
    price = 100.0
    for split_name, split_start in split_starts.items():
        split_rows = int(candles_per_split.get(split_name, 120))
        start_ts = int(pd.Timestamp(split_start, tz="UTC").timestamp() * 1000)
        for i in range(split_rows):
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


def split_by_date(candles_df: pd.DataFrame, split_range: tuple[str, str]) -> pd.DataFrame:
    if candles_df.empty:
        return candles_df.copy()
    dates = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    mask = (dates >= split_range[0]) & (dates <= split_range[1])
    return candles_df.loc[mask].reset_index(drop=True)


def validate_split_policy(
    split: dict[str, tuple[str, str]],
    candles_df: pd.DataFrame,
    min_days: dict[str, int] | None = None,
) -> dict[str, Any]:
    min_days = min_days or {"train": 60, "val": 14, "test": 14}
    required = ("train", "val", "test")
    for key in required:
        if key not in split:
            raise ValueError(f"missing split: {key}")

    ts_split = {
        key: (pd.Timestamp(split[key][0], tz="UTC"), pd.Timestamp(split[key][1], tz="UTC"))
        for key in required
    }
    for key, (start_ts, end_ts) in ts_split.items():
        if start_ts > end_ts:
            raise ValueError(f"invalid split order for {key}: start must be <= end")
        period_days = int((end_ts - start_ts).days) + 1
        if period_days < int(min_days.get(key, 1)):
            raise ValueError(f"{key} split is shorter than minimum period: {period_days} < {min_days[key]}")

    if not (ts_split["train"][1] < ts_split["val"][0] and ts_split["val"][1] < ts_split["test"][0]):
        raise ValueError("split overlap is not allowed and ranges must be strictly ordered train < val < test")

    if not candles_df.empty:
        data_start = pd.to_datetime(candles_df["open_time"].min(), unit="ms", utc=True).normalize()
        data_end = pd.to_datetime(candles_df["open_time"].max(), unit="ms", utc=True).normalize()
        for key, (start_ts, _) in ts_split.items():
            if start_ts < data_start or start_ts > data_end:
                raise ValueError(
                    f"{key} split start must be inside data coverage: data={data_start.date()}..{data_end.date()}"
                )

    return {
        "split": {k: [v[0], v[1]] for k, v in split.items()},
        "min_days": {k: int(v) for k, v in min_days.items()},
        "ordered_non_overlapping": True,
    }


def build_walkforward_splits(
    candles_df: pd.DataFrame,
    split: dict[str, tuple[str, str]],
    target_runs: int,
    step_days: int | None = None,
    min_days: dict[str, int] | None = None,
) -> list[dict[str, tuple[str, str]]]:
    if target_runs < 1:
        raise ValueError("target_runs must be >= 1")

    validate_split_policy(split, candles_df, min_days=min_days)
    base_train_start = pd.Timestamp(split["train"][0], tz="UTC")
    train_end = pd.Timestamp(split["train"][1], tz="UTC")
    val_start = pd.Timestamp(split["val"][0], tz="UTC")
    val_end = pd.Timestamp(split["val"][1], tz="UTC")
    test_start = pd.Timestamp(split["test"][0], tz="UTC")
    test_end = pd.Timestamp(split["test"][1], tz="UTC")

    val_days = (val_end - val_start).days + 1
    test_days = (test_end - test_start).days + 1
    step_days = max(1, int(step_days) if step_days is not None else val_days)

    if candles_df.empty:
        data_end = test_end + pd.Timedelta(days=step_days * (target_runs - 1))
    else:
        data_end = pd.to_datetime(candles_df["open_time"].max(), unit="ms", utc=True).normalize()

    splits: list[dict[str, tuple[str, str]]] = []
    for i in range(max(target_runs, 1)):
        shift = pd.Timedelta(days=i * step_days)
        run_split = {
            "train": (base_train_start.strftime("%Y-%m-%d"), (train_end + shift).strftime("%Y-%m-%d")),
            "val": ((val_start + shift).strftime("%Y-%m-%d"), (val_end + shift).strftime("%Y-%m-%d")),
            "test": ((test_start + shift).strftime("%Y-%m-%d"), (test_end + shift).strftime("%Y-%m-%d")),
        }
        run_test_end = pd.Timestamp(run_split["test"][1], tz="UTC")
        if run_test_end > data_end:
            break
        splits.append(run_split)

    if not splits:
        splits.append(split)
    return splits




def compute_walkforward_capacity(
    candles_df: pd.DataFrame,
    split: dict[str, tuple[str, str]],
    step_days: int | None = None,
    data_end_override: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """현재 split/step 기준으로 데이터가 수용 가능한 최대 fold 수를 계산한다."""

    validate_split_policy(split, candles_df)
    val_start = pd.Timestamp(split["val"][0], tz="UTC")
    val_end = pd.Timestamp(split["val"][1], tz="UTC")
    test_end = pd.Timestamp(split["test"][1], tz="UTC")

    val_days = int((val_end - val_start).days) + 1
    chosen_step_days = max(1, int(step_days) if step_days is not None else val_days)

    if data_end_override is not None:
        data_end = data_end_override
    elif candles_df.empty:
        data_end = test_end
    else:
        data_end = pd.to_datetime(candles_df["open_time"].max(), unit="ms", utc=True).normalize()

    forward_days = int((data_end - test_end).days)
    feasible_forward_days = max(0, forward_days)
    possible_runs = 1 + (feasible_forward_days // chosen_step_days)

    return {
        "possible_runs": int(max(1, possible_runs)),
        "step_days": int(chosen_step_days),
        "val_days": int(val_days),
        "forward_days": int(forward_days),
        "data_end": data_end.strftime("%Y-%m-%d"),
        "base_test_end": test_end.strftime("%Y-%m-%d"),
    }


def plan_walkforward_splits(
    candles_df: pd.DataFrame,
    split: dict[str, tuple[str, str]],
    target_runs: int,
    min_folds: int = 3,
    step_days: int | None = None,
    min_days: dict[str, int] | None = None,
) -> dict[str, Any]:
    """워크포워드 split 정책을 계산하고 부족 사유를 함께 반환한다.

    정책 우선순위:
    - C: step 이동 단위 분리(기본값은 val 기간)
    - B: val/test 기간 축소(기본 75% → 50% → 25%)
    - A: 그래도 부족하면 데이터 커버리지 확장 필요로 판단
    """

    min_days = min_days or {"train": 60, "val": 14, "test": 14}
    validate_split_policy(split, candles_df, min_days=min_days)

    val_start = pd.Timestamp(split["val"][0], tz="UTC")
    val_end = pd.Timestamp(split["val"][1], tz="UTC")
    test_start = pd.Timestamp(split["test"][0], tz="UTC")
    test_end = pd.Timestamp(split["test"][1], tz="UTC")

    val_days = int((val_end - val_start).days) + 1
    test_days = int((test_end - test_start).days) + 1
    desired_folds = max(int(target_runs), int(min_folds))

    if candles_df.empty:
        data_start = pd.Timestamp(split["train"][0], tz="UTC")
        data_end = test_end + pd.Timedelta(days=(desired_folds - 1) * max(1, val_days))
    else:
        data_start = pd.to_datetime(candles_df["open_time"].min(), unit="ms", utc=True).normalize()
        data_end = pd.to_datetime(candles_df["open_time"].max(), unit="ms", utc=True).normalize()

    notes: list[str] = []
    chosen_step_days = max(1, int(step_days) if step_days is not None else val_days)
    policy_split = {k: tuple(v) for k, v in split.items()}

    def _folds_for(candidate_split: dict[str, tuple[str, str]], candidate_step_days: int) -> int:
        return len(
            build_walkforward_splits(
                candles_df,
                candidate_split,
                target_runs=desired_folds,
                step_days=candidate_step_days,
                min_days=min_days,
            )
        )

    folds = _folds_for(policy_split, chosen_step_days)

    # C) step 단위를 val 기간과 분리해서 재설정
    if folds < desired_folds and desired_folds > 1:
        total_forward_days = int((data_end - test_end).days)
        candidate_step = total_forward_days // (desired_folds - 1)
        if candidate_step >= 1:
            adjusted_step = min(chosen_step_days, candidate_step)
            if adjusted_step < chosen_step_days:
                notes.append(
                    f"option_C_applied: step_days {chosen_step_days} -> {adjusted_step} (val_days={val_days})"
                )
                chosen_step_days = adjusted_step
                folds = _folds_for(policy_split, chosen_step_days)

    # B) val/test 기간 축소(반기→분기 등)
    if folds < desired_folds:
        for ratio in (0.75, 0.5, 0.25):
            new_val_days = max(int(min_days.get("val", 1)), int(round(val_days * ratio)))
            new_test_days = max(int(min_days.get("test", 1)), int(round(test_days * ratio)))
            cand_val_end = val_start + pd.Timedelta(days=new_val_days - 1)
            cand_test_start = cand_val_end + pd.Timedelta(days=1)
            cand_test_end = cand_test_start + pd.Timedelta(days=new_test_days - 1)
            candidate_split = {
                "train": (split["train"][0], (val_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
                "val": (val_start.strftime("%Y-%m-%d"), cand_val_end.strftime("%Y-%m-%d")),
                "test": (cand_test_start.strftime("%Y-%m-%d"), cand_test_end.strftime("%Y-%m-%d")),
            }
            try:
                validate_split_policy(candidate_split, candles_df, min_days=min_days)
            except ValueError:
                continue
            candidate_folds = _folds_for(candidate_split, chosen_step_days)
            if candidate_folds > folds:
                notes.append(
                    "option_B_applied: "
                    f"val_days {val_days} -> {new_val_days}, test_days {test_days} -> {new_test_days}"
                )
                policy_split = candidate_split
                folds = candidate_folds
            if folds >= desired_folds:
                break

    final_splits = build_walkforward_splits(
        candles_df,
        policy_split,
        target_runs=desired_folds,
        step_days=chosen_step_days,
        min_days=min_days,
    )
    insufficient_reason = None
    if len(final_splits) < desired_folds:
        insufficient_reason = (
            f"requested={desired_folds}, actual={len(final_splits)}; "
            "data coverage is insufficient for current split policy. "
            "option_A_required: extend training data collection range."
        )

    return {
        "splits": final_splits,
        "policy": {
            "requested_runs": int(target_runs),
            "minimum_required_runs": int(min_folds),
            "desired_runs": int(desired_folds),
            "actual_runs": int(len(final_splits)),
            "data_coverage": {
                "start": data_start.strftime("%Y-%m-%d"),
                "end": data_end.strftime("%Y-%m-%d"),
                "days": int((data_end - data_start).days) + 1,
            },
            "base_lengths_days": {"val": int(val_days), "test": int(test_days)},
            "selected_step_days": int(chosen_step_days),
            "selected_split": {k: list(v) for k, v in policy_split.items()},
            "adjustment_notes": notes,
            "insufficient_reason": insufficient_reason,
        },
    }


def summarize_dataset(candles_df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
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
