from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionResult:
    filled_qty: float
    fill_price: float
    fee: float
    slippage_cost: float
    new_cash: float
    new_position_qty: float


class ExecutionModel:
    def __init__(self, fee_rate: float = 0.001, slippage_bps: float = 2.0, max_step_change: float = 0.25, min_delta: float = 0.01) -> None:
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.max_step_change = max_step_change
        self.min_delta = min_delta

    def execute_target(self, target_pos: float, current_pos: float, cash: float, position_qty: float, equity: float, next_open: float) -> ExecutionResult:
        clipped_target = min(1.0, max(0.0, target_pos))
        delta = max(-self.max_step_change, min(self.max_step_change, clipped_target - current_pos))
        if abs(delta) < self.min_delta:
            delta = 0.0
        effective_target = current_pos + delta

        target_position_value = effective_target * equity
        current_position_value = position_qty * next_open
        delta_value = target_position_value - current_position_value
        filled_qty = delta_value / next_open if next_open else 0.0

        slip = next_open * (self.slippage_bps / 10_000)
        fill_price = next_open + (slip if filled_qty > 0 else -slip)
        fee = abs(filled_qty * fill_price) * self.fee_rate
        new_position_qty = position_qty + filled_qty
        new_cash = cash - filled_qty * fill_price - fee
        slippage_cost = abs(filled_qty) * abs(fill_price - next_open)
        return ExecutionResult(filled_qty, fill_price, fee, slippage_cost, new_cash, new_position_qty)
