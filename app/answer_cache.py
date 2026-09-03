"""问题 → 回答缓存（SQLite 持久化）。

命中则跳过「检索 → 重排 → 生成」整条链路，直接返回已缓存的回答与引用法条，
从而节省 token 与耗时；未命中/命中情况均通过 trace 的 cache_hit 字段记录，
供 Dashboard 统计缓存命中率。

约束：仅对「无历史的单轮问题」启用，避免多轮追问因上下文差异误命中。
缓存键为归一化（去首尾空白、合并连续空白、小写）后问题文本的 SHA-256。
法条语料更新（ingest）成功后清空缓存，防止回答基于过时法条。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DB = BASE_DIR / "logs" / "answer_cache.db"

_LOCK = threading.Lock()


def _normalize(question: str) -> str:
    return " ".join((question or "").strip().split()).lower()


def _key(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()


def _init_db() -> None:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY,"
            " question TEXT,"
            " answer TEXT,"
            " refs TEXT,"
            " created_at TEXT"
            ")"
        )
        conn.commit()
    finally:
        conn.close()


# 模块加载即建表
_init_db()


def get(question: str) -> dict | None:
    """命中返回 {"answer": str, "references": list[dict]}，否则返回 None。"""
    k = _key(question)
    with _LOCK:
        conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT answer, refs FROM cache WHERE key = ?", (k,)
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    try:
        references = json.loads(row[1])
    except (json.JSONDecodeError, TypeError):
        references = []
    return {"answer": row[0], "references": references}


def put(question: str, answer: str, references: list[dict]) -> None:
    """写入（或覆盖）一条缓存。"""
    k = _key(question)
    with _LOCK:
        conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, question, answer, refs, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    k,
                    _normalize(question),
                    answer,
                    json.dumps(references, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def clear() -> None:
    """清空全部缓存（ingest 法条更新后失效）。"""
    with _LOCK:
        conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        try:
            conn.execute("DELETE FROM cache")
            conn.commit()
        finally:
            conn.close()