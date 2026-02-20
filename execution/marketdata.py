from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"unsupported interval: {interval}")


@dataclass(slots=True)
class CandleClosedEvent:
    symbol: str
    interval: str
    open_time_ms: int
    close_time_ms: int
    o: float
    h: float
    l: float
    c: float
    v: float
    num_trades: int
    is_final: bool
    source: Literal["ws", "gapfill"]


class StateStore(Protocol):
    def load_last_ts(self, key: str) -> int | None: ...

    def save_last_ts(self, key: str, value: int) -> None: ...


class MemoryStateStore:
    def __init__(self) -> None:
        self._state: dict[str, int] = {}

    def load_last_ts(self, key: str) -> int | None:
        return self._state.get(key)

    def save_last_ts(self, key: str, value: int) -> None:
        self._state[key] = value


class RestKlineClient(Protocol):
    async def fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]: ...


class MessageStream(Protocol):
    async def recv(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


class GapFiller:
    def __init__(self, symbol: str, interval: str, rest_client: RestKlineClient):
        self.symbol = symbol
        self.interval = interval
        self.interval_ms = interval_to_ms(interval)
        self.rest_client = rest_client

    async def fill(self, expected_next_ms: int, incoming_open_time_ms: int) -> list[CandleClosedEvent]:
        if incoming_open_time_ms <= expected_next_ms:
            return []
        end_ms = incoming_open_time_ms - self.interval_ms
        if end_ms < expected_next_ms:
            return []

        klines = await self.rest_client.fetch_klines(self.symbol, self.interval, expected_next_ms, end_ms)
        events: list[CandleClosedEvent] = []
        for k in klines:
            events.append(
                CandleClosedEvent(
                    symbol=self.symbol,
                    interval=self.interval,
                    open_time_ms=int(k["open_time"]),
                    close_time_ms=int(k["close_time"]),
                    o=float(k["open"]),
                    h=float(k["high"]),
                    l=float(k["low"]),
                    c=float(k["close"]),
                    v=float(k["volume"]),
                    num_trades=int(k.get("num_trades", 0)),
                    is_final=True,
                    source="gapfill",
                )
            )
        events.sort(key=lambda e: e.open_time_ms)
        return events


class MarketDataWS:
    def __init__(
        self,
        symbol: str,
        interval: str,
        queue: asyncio.Queue[CandleClosedEvent],
        state_store: StateStore,
        gap_filler: GapFiller | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.queue = queue
        self.state_store = state_store
        self.gap_filler = gap_filler
        self.interval_ms = interval_to_ms(interval)
        self._state_key = f"{self.symbol}:{self.interval}:last_ts"
        self.last_ts = state_store.load_last_ts(self._state_key)
        self._running = False
        self._reader_task: asyncio.Task | None = None

    async def start(
        self,
        stream: MessageStream,
        payload_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if self._running:
            return
        self._running = True
        extractor = payload_extractor or (lambda x: x)
        self._reader_task = asyncio.create_task(self._reader_loop(stream, extractor))

    async def stop(self, stream: MessageStream | None = None) -> None:
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if stream is not None:
            await stream.close()

    async def _reader_loop(self, stream: MessageStream, extractor: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        while self._running:
            message = await stream.recv()
            if message is None:
                await asyncio.sleep(0.1)
                continue
            payload = extractor(message)
            await self.handle_kline_message(payload)

    async def handle_kline_message(self, payload: dict) -> None:
        k = payload.get("k", {})
        if not k.get("x", False):
            return

        event = CandleClosedEvent(
            symbol=self.symbol,
            interval=self.interval,
            open_time_ms=int(k["t"]),
            close_time_ms=int(k["T"]),
            o=float(k["o"]),
            h=float(k["h"]),
            l=float(k["l"]),
            c=float(k["c"]),
            v=float(k["v"]),
            num_trades=int(k.get("n", 0)),
            is_final=True,
            source="ws",
        )

        if self.last_ts is None:
            await self.queue.put(event)
            self._update_last_ts(event.open_time_ms)
            return

        expected_next = self.last_ts + self.interval_ms
        if event.open_time_ms < expected_next:
            return

        if event.open_time_ms > expected_next and self.gap_filler is not None:
            filled_events = await self.gap_filler.fill(expected_next, event.open_time_ms)
            for filled in filled_events:
                if self.last_ts is not None and filled.open_time_ms <= self.last_ts:
                    continue
                await self.queue.put(filled)
                self._update_last_ts(filled.open_time_ms)

        if self.last_ts is None or event.open_time_ms > self.last_ts:
            await self.queue.put(event)
            self._update_last_ts(event.open_time_ms)

    def _update_last_ts(self, ts: int) -> None:
        self.last_ts = ts
        self.state_store.save_last_ts(self._state_key, ts)
