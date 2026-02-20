from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    type: Literal["log_return_regularized", "differential_sharpe", "downside_risk"] = "log_return_regularized"
    comparison_types: list[Literal["log_return_regularized", "differential_sharpe", "downside_risk"]] = Field(
        default_factory=lambda: ["log_return_regularized", "differential_sharpe", "downside_risk"]
    )
    comparison_repeats: int = Field(ge=1, le=10, default=3)
    lambda_turnover: float = Field(ge=0.0, default=0.001)
    lambda_dd: float = Field(ge=0.0, default=0.1)
    dd_limit: float = Field(ge=0.0, le=1.0, default=0.2)
    inactivity_threshold: float = Field(ge=0.0, le=1.0, default=1e-4)
    inactivity_penalty: float = Field(ge=0.0, default=0.0005)
    target_position_utilization: float = Field(ge=0.0, le=1.0, default=0.15)
    lambda_under_utilization: float = Field(ge=0.0, default=0.001)
    dsr_beta: float = Field(gt=0.0, le=1.0, default=0.05)
    dsr_scale: float = Field(gt=0.0, default=1.0)
    downside_beta: float = Field(gt=0.0, le=1.0, default=0.05)
    lambda_downside: float = Field(ge=0.0, default=0.5)
    selection_turnover_penalty_weight: float = Field(ge=0.0, default=1.0)
    selection_inactivity_penalty_weight: float = Field(ge=0.0, default=1.0)
    selection_drawdown_penalty_weight: float = Field(ge=0.0, default=1.0)
    selection_turnover_target: float = Field(ge=0.0, default=0.25)
    selection_inactivity_target: float = Field(ge=0.0, le=1.0, default=0.7)
    penalty_sweep_enabled: bool = False
    penalty_sweep_mode: Literal["grid", "random"] = "grid"
    penalty_sweep_trials: int = Field(ge=1, le=200, default=12)
    turnover_penalty_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5])
    inactivity_penalty_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5])
    drawdown_penalty_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5])


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
    device: str = "cuda:0"
    total_timesteps: int = Field(ge=256, default=20_000)
    learning_rate: float = Field(gt=0.0, default=3e-4)
    batch_size: int = Field(gt=0, default=64)
    gamma: float = Field(gt=0.0, le=1.0, default=0.99)
    n_steps: int = Field(gt=0, default=128)
    seed: int | None = Field(default=None)
    eval_interval: int = Field(gt=0, default=2_000)
    early_stop: int = Field(ge=0, default=8)
    checkpoint_interval: int = Field(gt=0, default=5_000)
    resume_from: str | None = None
    walkforward_runs: int = Field(ge=2, le=10, default=3)
    walkforward_adjustment_scenario: Literal["auto", "extend_data", "reduce_val_test", "off"] = "auto"
    walkforward_shortfall_policy: Literal["warn_and_continue", "abort"] = "warn_and_continue"
    walkforward_extend_days: int = Field(ge=0, default=0)


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
