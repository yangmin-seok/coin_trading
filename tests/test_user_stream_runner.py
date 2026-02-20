from __future__ import annotations

import asyncio

from integrations.binance_ws_user import UserStreamRunner


class FlakyFactory:
    def __init__(self):
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return FakeStream([{"e": "executionReport", "E": 1, "s": "BTCUSDT", "i": 1, "X": "FILLED", "S": "BUY", "l": "0.1", "L": "100"}])


class FakeStream:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(0.05)
        return None

    async def close(self):
        self.closed = True


def test_user_stream_runner_reconnects_and_routes_event():
    async def _run():
        routed: list[dict] = []
        stop = asyncio.Event()
        factory = FlakyFactory()
        runner = UserStreamRunner(factory, on_event=routed.append, backoff_s=0.01, max_backoff_s=0.02)

        async def stopper():
            await asyncio.sleep(0.2)
            stop.set()

        task = asyncio.create_task(runner.run(stop))
        await stopper()
        await task

        assert factory.calls >= 2
        assert routed
        assert routed[0]["kind"] == "order_update"

    asyncio.run(_run())
