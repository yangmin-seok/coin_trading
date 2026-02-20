from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    type: Literal["log_return"] = "log_return"
    lambda_turnover: float = Field(ge=0.0, default=0.001)
    lambda_dd: float = Field(ge=0.0, default=0.1)
    dd_limit: float = Field(ge=0.0, le=1.0, default=0.2)


class ExecutionConfig(BaseModel):
    fee_rate: float = Field(ge=0.0, default=0.001)
    slippage_bps: float = Field(ge=0.0, default=2.0)
    max_step_change: float = Field(ge=0.0, le=1.0, default=0.25)
    min_delta: float = Field(ge=0.0, le=1.0, default=0.01)


class FeatureWindowsConfig(BaseModel):
    return_: int = Field(alias="return", ge=2, default=20)
    rsi: int = Field(ge=2, default=14)
    bb: int = Field(ge=2, default=20)
    macd_fast: int = Field(ge=2, default=12)
    macd_slow: int = Field(ge=2, default=26)
    macd_signal: int = Field(ge=2, default=9)


class FeaturesConfig(BaseModel):
    version: str = "v1"
    windows: FeatureWindowsConfig = Field(default_factory=FeatureWindowsConfig)


class SplitConfig(BaseModel):
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


class TrainConfig(BaseModel):
    algo: Literal["ppo", "sac"] = "ppo"
    total_timesteps: int = Field(ge=256, default=1024)
    learning_rate: float = Field(gt=0.0, default=3e-4)
    batch_size: int = Field(gt=0, default=64)
    gamma: float = Field(gt=0.0, le=1.0, default=0.99)
    n_steps: int = Field(gt=0, default=128)
    seed: int | None = Field(default=None)
    eval_interval: int = Field(gt=0, default=256)
    early_stop: int = Field(ge=0, default=3)
    checkpoint_interval: int = Field(gt=0, default=512)
    resume_from: str | None = None
    walkforward_runs: int = Field(ge=2, le=10, default=3)


class ExplorationAxesConfig(BaseModel):
    lambda_turnover: list[float] = Field(default_factory=lambda: [0.001, 0.003, 0.005])
    min_delta: list[float] = Field(default_factory=lambda: [0.01, 0.03, 0.05])
    max_step_change: list[float] = Field(default_factory=lambda: [0.25, 0.15, 0.10])


class ExplorationConfig(BaseModel):
    axes: ExplorationAxesConfig = Field(default_factory=ExplorationAxesConfig)


class AppConfig(BaseModel):
    mode: Literal["demo", "live", "backtest"] = "demo"
    exchange: Literal["binance"] = "binance"
    market: Literal["spot"] = "spot"
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    seed: int = 3
    timezone_storage: Literal["UTC"] = "UTC"
    report_timezone: str = "Asia/Seoul"
    reward: RewardConfig = Field(default_factory=RewardConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    split: SplitConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
