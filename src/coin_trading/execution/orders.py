from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.coin_trading.execution.state import PortfolioState


class OrderClient(Protocol):
    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any: ...


@dataclass(slots=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    reduce_only: bool = False


@dataclass(slots=True)
class OrderManager:
    client: OrderClient

    def create_market_order(self, symbol: str, side: str, quantity: float) -> dict[str, Any]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        norm_side = side.upper()
        if norm_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        return self.client._request(
            "POST",
            "/v3/order",
            params={
                "symbol": symbol.upper(),
                "side": norm_side,
                "type": "MARKET",
                "quantity": f"{quantity:.8f}",
            },
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self.client._request(
            "DELETE",
            "/v3/order",
            params={"symbol": symbol.upper(), "orderId": int(order_id)},
            signed=True,
        )

    def target_to_intent(self, symbol: str, *, current_position: float, target_position: float, equity: float, price: float) -> OrderIntent | None:
        if equity <= 0 or price <= 0:
            return None
        delta_pos = float(target_position) - float(current_position)
        if abs(delta_pos) < 1e-9:
            return None

        notional = abs(delta_pos) * equity
        qty = notional / price
        if qty <= 0:
            return None
        side = "BUY" if delta_pos > 0 else "SELL"
        return OrderIntent(symbol=symbol.upper(), side=side, quantity=qty)

    def apply_execution_report(self, state: PortfolioState, report: dict[str, Any]) -> None:
        status = str(report.get("X", "")).upper()
        if status not in {"FILLED", "PARTIALLY_FILLED"}:
            return

        side = str(report.get("S", "")).upper()
        qty = float(report.get("l", 0.0) or 0.0)  # last filled qty
        price = float(report.get("L", 0.0) or 0.0)  # last filled price
        commission = float(report.get("n", 0.0) or 0.0)

        if qty <= 0 or price <= 0:
            return

        signed_qty = qty if side == "BUY" else -qty
        state.position_qty += signed_qty
        state.cash -= signed_qty * price
        state.cash -= commission
        state.mark_to_market(price)
