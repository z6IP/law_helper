"""会话存储兼容层：底层使用 SQLite（app/session_db.py），删除时级联清理附件。

工程约定：单用户本地工具，无用户隔离与鉴权。
"""
from __future__ import annotations

from pathlib import Path

from app import session_db, upload_store


def _ensure_db() -> None:
    """确保 SQLite 数据库已初始化。"""
    session_db.init_db()


def save(session_id: str, title: str, messages: list[dict]) -> str:
    """保存/更新会话，返回 updated_at（ISO 时间）。"""
    _ensure_db()
    return session_db.save(session_id, title, messages)


def list_all() -> list[dict]:
    """全部会话，按 updated_at 降序。"""
    _ensure_db()
    return session_db.list_all()


def delete(session_id: str) -> bool:
    """删除会话及其附件；返回是否删除成功。"""
    _ensure_db()
    ok, stored_names = session_db.delete(session_id)
    if ok and stored_names:
        upload_store.delete_attachments(stored_names)
    return ok


def exists(session_id: str) -> bool:
    """判断会话是否已持久化。"""
    _ensure_db()
    return session_db.load(session_id) is not None


def migrate_from_json(json_dir: Path | None = None) -> int:
    """将历史 JSON 文件一次性导入 SQLite，返回导入会话数。"""
    _ensure_db()
    return session_db.migrate_from_json(json_dir)
