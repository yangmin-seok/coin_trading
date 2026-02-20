from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol


@dataclass(slots=True)
class UserStreamEvent:
    event_type: str
    event_time: int
    payload: dict[str, Any]


class UserMessageStream(Protocol):
    async def recv(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


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


class UserStreamRunner:
    def __init__(
        self,
        stream_factory: Callable[[], Awaitable[UserMessageStream]],
        on_event: Callable[[dict[str, Any]], None],
        *,
        backoff_s: float = 0.5,
        max_backoff_s: float = 5.0,
    ) -> None:
        self.stream_factory = stream_factory
        self.on_event = on_event
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s

    async def run(self, stop_event: asyncio.Event) -> None:
        delay = self.backoff_s
        while not stop_event.is_set():
            stream: UserMessageStream | None = None
            try:
                stream = await self.stream_factory()
                delay = self.backoff_s
                while not stop_event.is_set():
                    msg = await stream.recv()
                    if msg is None:
                        await asyncio.sleep(0.1)
                        continue
                    event = normalize_user_event(msg)
                    self.on_event(UserEventRouter().route(event))
            except Exception:
                await asyncio.sleep(delay)
                delay = min(self.max_backoff_s, delay * 2)
            finally:
                if stream is not None:
                    await stream.close()
