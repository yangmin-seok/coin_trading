from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


@dataclass(slots=True)
class TelegramSender:
    bot_token: str
    chat_id: str
    timeout_s: float = 5.0

    def send_text(self, text: str) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        req = Request(url=url, method="POST", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
