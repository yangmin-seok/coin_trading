from __future__ import annotations

from env.execution_model import ExecutionModel


def test_execution_model_limits_delta():
    model = ExecutionModel(max_step_change=0.1, min_delta=0.01)
    result = model.execute_target(
        target_pos=1.0,
        current_pos=0.0,
        cash=10_000,
        position_qty=0.0,
        equity=10_000,
        next_open=100,
    )
    assert result.filled_qty > 0
    assert result.new_position_qty == result.filled_qty
