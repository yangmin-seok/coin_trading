from __future__ import annotations

import asyncio

from execution.marketdata import MarketDataWS, MemoryStateStore


class FakeStream:
    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    async def recv(self):
        if not self._messages:
            await asyncio.sleep(0.01)
            return None
        return self._messages.pop(0)

    async def close(self):
        self.closed = True


def test_marketdata_reader_loop_puts_closed_event():
    async def _run():
        q: asyncio.Queue = asyncio.Queue()
        ws = MarketDataWS("BTCUSDT", "5m", q, MemoryStateStore())
        msg = {"k": {"t": 0, "T": 299999, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "x": True}}
        stream = FakeStream([msg])
        await ws.start(stream)
        await asyncio.sleep(0.05)
        await ws.stop(stream)
        assert not q.empty()
        ev = await q.get()
        assert ev.open_time_ms == 0

    asyncio.run(_run())
