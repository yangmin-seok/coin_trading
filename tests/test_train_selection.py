from __future__ import annotations

import pandas as pd
import pytest

from src.coin_trading.config.loader import load_config
from src.coin_trading.pipelines.train_flow.evaluate import _reward_component_contributions
from src.coin_trading.pipelines.train_flow.train import _is_better_candidate, _select_best_penalty_weight


def test_reward_component_contributions_are_reconstructable() -> None:
    trace = pd.DataFrame(
        {
            "reward": [0.2, -0.05],
            "reward_pnl": [0.3, 0.0],
            "reward_cost": [0.05, 0.05],
            "reward_penalty": [0.05, 0.0],
            "reward_penalty_drawdown": [0.05, 0.0],
            "reward_penalty_inactivity": [0.0, 0.0],
            "reward_penalty_under_utilization": [0.0, 0.0],
            "reward_penalty_downside": [0.0, 0.0],
        }
    )

    contrib = _reward_component_contributions(trace)

    assert contrib["reward_total"] == pytest.approx(0.15)
    assert contrib["pnl_total"] == pytest.approx(0.3)
    assert contrib["turnover_cost_total"] == pytest.approx(0.1)
    assert contrib["penalty_total"] == pytest.approx(0.05)
    assert abs(contrib["reconstruction_error"]) < 1e-12


def test_candidate_selection_uses_validation_metrics_not_test_metrics() -> None:
    candidate = {
        "val_metrics": {"sharpe": 1.0, "final_equity": 10050.0, "max_drawdown": 0.1, "cost_pnl_ratio": 0.1, "turnover": 0.1},
        "test_metrics": {"sharpe": -10.0, "final_equity": 1.0, "max_drawdown": 1.0, "cost_pnl_ratio": 10.0, "turnover": 10.0},
    }
    current = {
        "val_metrics": {"sharpe": 0.1, "final_equity": 10000.0, "max_drawdown": 0.3, "cost_pnl_ratio": 0.2, "turnover": 0.3},
        "test_metrics": {"sharpe": 10.0, "final_equity": 100000.0, "max_drawdown": 0.0, "cost_pnl_ratio": 0.0, "turnover": 0.0},
    }

    assert _is_better_candidate(candidate, current) is True


def test_penalty_weight_sweep_returns_best_and_candidates() -> None:
    cfg = load_config()
    cfg.reward.penalty_sweep_enabled = True
    cfg.reward.penalty_sweep_mode = "grid"
    cfg.reward.turnover_penalty_grid = [0.5, 1.0]
    cfg.reward.inactivity_penalty_grid = [1.0]
    cfg.reward.drawdown_penalty_grid = [1.0]

    metrics = {
        "sharpe": 1.0,
        "final_equity": 10100.0,
        "max_drawdown": 0.1,
        "turnover": 0.5,
        "total_cost": 10.0,
        "no_trade_ratio": 0.6,
        "baseline_equity": {"buy_hold": 10000.0},
    }

    best, candidates = _select_best_penalty_weight(metrics, cfg)

    assert len(candidates) == 2
    assert "weights" in best
