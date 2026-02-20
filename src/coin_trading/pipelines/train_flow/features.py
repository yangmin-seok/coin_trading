from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.coin_trading.features.definitions import FEATURE_COLUMNS
from src.coin_trading.features.offline import compute_offline


@dataclass
class SplitFeatureScaler:
    mean: pd.Series
    std: pd.Series
    fitted_split: str


def compute_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Single feature interface shared by offline training and online replay parity."""
    return compute_offline(candles_df)


def validate_rolling_features_no_lookahead(candles_df: pd.DataFrame, features_df: pd.DataFrame) -> dict[str, object]:
    if candles_df.empty or len(candles_df) < 4:
        return {"passed": True, "checked_rows": int(len(candles_df))}

    pivot = max(1, len(candles_df) // 2)
    mutated = candles_df.copy()
    mutated.loc[mutated.index[pivot:], "close"] = mutated.loc[mutated.index[pivot:], "close"] * 1.05
    mutated_features = compute_features(mutated)

    baseline = features_df[FEATURE_COLUMNS].iloc[:pivot].fillna(0.0).to_numpy(dtype=float)
    challenger = mutated_features[FEATURE_COLUMNS].iloc[:pivot].fillna(0.0).to_numpy(dtype=float)
    if not np.allclose(baseline, challenger, atol=1e-12, rtol=1e-9):
        raise ValueError("lookahead detected: rolling features changed when only future candles were perturbed")

    return {"passed": True, "checked_rows": int(pivot)}


def fit_feature_scaler(train_features: pd.DataFrame, split_name: str = "train") -> SplitFeatureScaler:
    if split_name != "train":
        raise ValueError("scaler fit must be performed on train split after split-by-date")
    mean = train_features[FEATURE_COLUMNS].mean()
    std = train_features[FEATURE_COLUMNS].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return SplitFeatureScaler(mean=mean, std=std, fitted_split=split_name)


def transform_with_scaler(features_df: pd.DataFrame, scaler: SplitFeatureScaler) -> pd.DataFrame:
    out = features_df.copy()
    out[FEATURE_COLUMNS] = (out[FEATURE_COLUMNS] - scaler.mean) / scaler.std
    return out
