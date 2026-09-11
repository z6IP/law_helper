"""Data-driven retrieval policy loaded from versioned JSON configuration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.config import BASE_DIR


@lru_cache(maxsize=1)
def get_policy() -> dict:
    path = BASE_DIR / "config" / "policy.json"
    if not path.is_file():
        raise FileNotFoundError(f"检索策略配置不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def clear_policy_cache() -> None:
    get_policy.cache_clear()
