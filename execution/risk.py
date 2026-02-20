from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    target: float
    reason: str = ""


@dataclass(slots=True)
class RiskManager:
    """Stateless risk checks for target-position trading."""

    max_position: float = 1.0
    min_position: float = 0.0
    max_drawdown: float = 0.3
    max_leverage: float = 1.0

    def clamp_target(self, target: float) -> float:
        return max(self.min_position, min(self.max_position, float(target)))

    def approve_target(self, target: float, *, equity: float, peak_equity: float, notional: float = 0.0) -> RiskDecision:
        if equity <= 0:
            return RiskDecision(approved=False, target=0.0, reason="equity_non_positive")

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = max(0.0, (peak_equity - equity) / peak_equity)
        if drawdown >= self.max_drawdown:
            return RiskDecision(approved=False, target=0.0, reason="drawdown_limit")

        leverage = abs(notional) / equity if equity > 0 else float("inf")
        if leverage > self.max_leverage:
            return RiskDecision(approved=False, target=0.0, reason="leverage_limit")

        return RiskDecision(approved=True, target=self.clamp_target(target), reason="ok")
