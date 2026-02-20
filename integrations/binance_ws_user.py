from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UserStreamEvent:
    event_type: str
    event_time: int
    payload: dict[str, Any]


def normalize_user_event(message: dict[str, Any]) -> UserStreamEvent:
    payload = message.get("data", message)
    event_type = str(payload.get("e", "unknown"))
    event_time = int(payload.get("E", 0))
    return UserStreamEvent(event_type=event_type, event_time=event_time, payload=payload)


class UserEventRouter:
    def route(self, event: UserStreamEvent) -> dict[str, Any]:
        p = event.payload
        if event.event_type == "executionReport":
            return {
                "kind": "order_update",
                "symbol": p.get("s"),
                "order_id": p.get("i"),
                "status": p.get("X"),
                "side": p.get("S"),
                "last_executed_qty": float(p.get("l", 0.0)),
                "last_executed_price": float(p.get("L", 0.0)),
            }
        if event.event_type == "outboundAccountPosition":
            balances: dict[str, float] = {}
            for b in p.get("B", []):
                asset = str(b.get("a"))
                balances[asset] = float(b.get("f", 0.0)) + float(b.get("l", 0.0))
            return {"kind": "balance_update", "balances": balances}
        return {"kind": "unknown", "raw": p}
