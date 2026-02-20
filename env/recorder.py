from __future__ import annotations


class StepRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, info: dict) -> None:
        self.rows.append(info.copy())
