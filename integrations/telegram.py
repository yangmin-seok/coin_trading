from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen


@dataclass(slots=True)
class TelegramSender:
    bot_token: str
    chat_id: str
    timeout_s: float = 5.0
    max_retries: int = 2
    backoff_s: float = 0.5
    dedup_ttl_s: float = 30.0
    _recent_keys: dict[str, float] = field(default_factory=dict)

    def _cleanup_keys(self, now: float) -> None:
        expired = [k for k, ts in self._recent_keys.items() if now - ts >= self.dedup_ttl_s]
        for k in expired:
            self._recent_keys.pop(k, None)

    def send_text(self, text: str, dedup_key: str | None = None) -> dict[str, Any]:
        now = time.monotonic()
        self._cleanup_keys(now)
        if dedup_key and dedup_key in self._recent_keys:
            return {"ok": True, "deduped": True}

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        req = Request(url=url, method="POST", data=payload, headers={"Content-Type": "application/json"})

        err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                    out = json.loads(body)
                    if dedup_key:
                        self._recent_keys[dedup_key] = time.monotonic()
                    return out
            except Exception as exc:
                err = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError("telegram send failed") from err
