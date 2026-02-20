from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"


@dataclass(slots=True)
class OrderResult:
    accepted: bool
    executed_qty: float
    avg_price: float
    reason: str = ""


class OrderExecutor:
    def place_market_order(self, req: OrderRequest, mark_price: float) -> OrderResult:
        if req.quantity <= 0:
            return OrderResult(False, 0.0, mark_price, "non_positive_qty")
        if mark_price <= 0:
            return OrderResult(False, 0.0, mark_price, "invalid_price")
        return OrderResult(True, req.quantity, mark_price, "filled")
