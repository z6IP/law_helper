"""Versioned prompt templates loaded from the project prompts directory."""
from __future__ import annotations

from functools import lru_cache
from string import Formatter
from pathlib import Path

from app.config import BASE_DIR, get_settings


def _prompt_directory() -> Path:
    prompt_dir = Path(get_settings().prompt_dir)
    return prompt_dir if prompt_dir.is_absolute() else BASE_DIR / prompt_dir


@lru_cache(maxsize=None)
def _read_prompt(name: str) -> str:
    """Read one prompt template and fail loudly when it is missing."""
    path = _prompt_directory() / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"提示词模板不存在：{path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(name: str, **values: object) -> str:
    """Render a named template with explicit variables only."""
    template = _read_prompt(name)
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing = sorted(fields - values.keys())
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"提示词模板 {name!r} 缺少变量：{missing_text}")
    return template.format_map(values)


def clear_prompt_cache() -> None:
    """Clear cached templates for tests and controlled runtime reloads."""
    _read_prompt.cache_clear()
