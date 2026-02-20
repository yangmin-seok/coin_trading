from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(slots=True)
class TelegramSender:
    bot_token: str
    chat_id: str
    timeout_s: float = 5.0

    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_text(self, text: str, disable_notification: bool = False) -> dict:
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": disable_notification,
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.api_url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "coin-trading/0.1"},
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
