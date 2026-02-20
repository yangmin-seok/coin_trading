from __future__ import annotations

from execution.orders import OrderExecutor, OrderRequest
from execution.risk import RiskLimits, RiskManager


def test_risk_manager_applies_limits():
    rm = RiskManager(RiskLimits(max_notional_per_trade=100.0, min_notional=10.0, step_size=0.1))
    out = rm.evaluate_target(target_position_ratio=1.0, current_position_ratio=0.0, equity=1_000.0, price=100.0)
    assert out.approved
    assert out.order_notional <= 100.0


def test_order_executor_rejects_invalid_quantity():
    ex = OrderExecutor()
    out = ex.place_market_order(OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.0), mark_price=100.0)
    assert not out.accepted
