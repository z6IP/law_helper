"""本地文件会话存储：data/sessions/{id}.json，原子写入 + 线程锁。

工程约定：单用户本地工具，无用户隔离与鉴权；
损坏的会话文件在 list_all 时静默跳过，不影响其他会话。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime

from pydantic import ValidationError

from app.config import get_settings
from app.errors import SessionError
from app.schemas import SessionData

_LOCK = threading.Lock()


def _session_path(session_id: str) -> str:
    """会话文件路径；只允许字母数字与连字符，防路径穿越。"""
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch == "-")
    if not safe:
        raise SessionError("非法的会话 ID")
    settings = get_settings()
    os.makedirs(settings.sessions_full_dir, exist_ok=True)
    return str(settings.sessions_full_dir / f"{safe}.json")


def save(session_id: str, title: str, messages: list[dict]) -> str:
    """upsert 会话（tmp + os.replace 原子写），返回 updated_at（ISO 时间）。"""
    path = _session_path(session_id)
    updated_at = datetime.now().isoformat(timespec="seconds")
    data = {
        "id": session_id,
        "title": title,
        "updated_at": updated_at,
        "messages": messages,
    }
    with _LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    return updated_at


def list_all() -> list[dict]:
    """全部会话，按 updated_at 降序；损坏文件静默跳过。"""
    settings = get_settings()
    directory = settings.sessions_full_dir
    if not os.path.isdir(directory):
        return []
    out: list[dict] = []
    with _LOCK:
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, encoding="utf-8") as f:
                    data = SessionData.model_validate(json.load(f))
                out.append(data.model_dump())
            except (OSError, json.JSONDecodeError, ValidationError):
                continue
    out.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return out


def delete(session_id: str) -> bool:
    """删除会话文件，返回是否存在。"""
    path = _session_path(session_id)
    with _LOCK:
        if os.path.exists(path):
            os.remove(path)
            return True
    return False
