from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class UserStreamEvent:
    event_type: Literal["outboundAccountPosition", "executionReport", "balanceUpdate", "other"]
    payload: dict[str, Any]


def extract_user_payload(message: dict[str, Any]) -> dict[str, Any]:
    """Handle raw or combined stream payloads."""
    if "data" in message and isinstance(message["data"], dict):
        return message["data"]
    return message


@dataclass(slots=True)
class BinanceUserWSConfig:
    reconnect_backoff_s: float = 1.0
    max_retries: int = 10


class BinanceUserWS:
    def __init__(self, queue: asyncio.Queue[UserStreamEvent], config: BinanceUserWSConfig | None = None) -> None:
        self.queue = queue
        self.config = config or BinanceUserWSConfig()
        self._running = False

    async def handle_message(self, message: dict[str, Any]) -> UserStreamEvent:
        payload = extract_user_payload(message)
        event_type = str(payload.get("e", "other"))
        if event_type not in {"outboundAccountPosition", "executionReport", "balanceUpdate"}:
            event_type = "other"
        event = UserStreamEvent(event_type=event_type, payload=payload)
        await self.queue.put(event)
        return event
