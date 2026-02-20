from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from config.schema import AppConfig


ENV_PREFIX = "COIN_TRADING__"


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = data
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def load_config(path: str | Path = "config/default.yaml") -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        keys = env_key[len(ENV_PREFIX) :].lower().split("__")
        _set_nested(data, keys, env_value)

    return AppConfig.model_validate(data)
