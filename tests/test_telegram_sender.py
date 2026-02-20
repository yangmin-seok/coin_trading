from __future__ import annotations

import json

import integrations.telegram as telegram_mod
from integrations.telegram import TelegramSender


class DummyResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_telegram_sender_dedup_and_retry(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary")
        return DummyResp({"ok": True})

    monkeypatch.setattr(telegram_mod, "urlopen", fake_urlopen)

    sender = TelegramSender("token", "chat", max_retries=1, backoff_s=0.0)
    out1 = sender.send_text("hello", dedup_key="k1")
    out2 = sender.send_text("hello", dedup_key="k1")

    assert out1["ok"] is True
    assert out2["deduped"] is True
    assert calls["n"] == 2
