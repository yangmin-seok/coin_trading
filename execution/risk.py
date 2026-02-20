from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RiskLimits:
    max_position_ratio: float = 1.0
    min_position_ratio: float = 0.0
    max_notional_per_trade: float = 5_000.0
    min_notional: float = 5.0
    min_qty: float = 0.0
    step_size: float = 0.000001


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    target_position_ratio: float
    order_qty: float
    order_notional: float
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate_target(
        self,
        target_position_ratio: float,
        current_position_ratio: float,
        equity: float,
        price: float,
    ) -> RiskDecision:
        reasons: list[str] = []
        clamped_target = float(max(self.limits.min_position_ratio, min(self.limits.max_position_ratio, target_position_ratio)))
        if clamped_target != target_position_ratio:
            reasons.append("target_clamped")

        delta_ratio = clamped_target - current_position_ratio
        order_notional = abs(delta_ratio) * max(equity, 0.0)
        if order_notional > self.limits.max_notional_per_trade:
            capped_delta = self.limits.max_notional_per_trade / max(equity, 1e-12)
            delta_ratio = capped_delta if delta_ratio >= 0 else -capped_delta
            clamped_target = current_position_ratio + delta_ratio
            order_notional = abs(delta_ratio) * max(equity, 0.0)
            reasons.append("notional_capped")

        if price <= 0:
            return RiskDecision(False, current_position_ratio, 0.0, 0.0, reasons + ["invalid_price"])

        raw_qty = (delta_ratio * equity) / price
        step = max(self.limits.step_size, 1e-12)
        stepped_qty = int(raw_qty / step) * step

        if abs(stepped_qty) < self.limits.min_qty:
            return RiskDecision(False, current_position_ratio, 0.0, 0.0, reasons + ["below_min_qty"])

        stepped_notional = abs(stepped_qty) * price
        if stepped_notional < self.limits.min_notional:
            return RiskDecision(False, current_position_ratio, 0.0, 0.0, reasons + ["below_min_notional"])

        return RiskDecision(True, clamped_target, stepped_qty, stepped_notional, reasons)
