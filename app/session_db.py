"""SQLite 会话持久化：sessions / messages / attachments 三表关联。

删除 sessions 记录时，通过外键级联自动删除 messages 和 attachments 记录；
上层再按 attachments 中记录的 stored_name 删除物理文件。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

_DB_PATH: Path | None = None
_LOCK = threading.Lock()


def _db_path() -> Path:
    """返回 SQLite 数据库路径；默认放在 data/sessions 同级目录。"""
    global _DB_PATH
    if _DB_PATH is None:
        settings = get_settings()
        _DB_PATH = settings.sessions_full_dir.parent / "law_helper.db"
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _set_db_path_for_test(path: Path) -> None:
    """测试专用：覆盖默认数据库路径。"""
    global _DB_PATH
    _DB_PATH = path


def _connect() -> sqlite3.Connection:
    """创建启用外键与 WAL 模式的连接。"""
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    """初始化数据库表结构（幂等）。"""
    with _LOCK, _connect() as conn:
        _create_tables(conn)


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            references_json TEXT,
            reasoning TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_idx INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL UNIQUE,
            url TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_attachments_session
            ON attachments(session_id);
        """
    )
    conn.commit()


def save(session_id: str, title: str, messages: list[dict]) -> str:
    """保存/更新会话、消息及附件元数据，返回 updated_at。"""
    updated_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    with _LOCK, _connect() as conn, conn:  # 事务
        conn.execute(
            """
            INSERT INTO sessions (id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (session_id, title, updated_at),
        )
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM attachments WHERE session_id = ?",
            (session_id,),
        )
        for idx, msg in enumerate(messages):
            conn.execute(
                """
                INSERT INTO messages
                    (session_id, role, content, references_json, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    json.dumps(msg.get("references", []) or [], ensure_ascii=False),
                    msg.get("reasoning"),
                    float(idx),
                ),
            )
            for att in msg.get("attachments", []) or []:
                stored_name = _extract_stored_name(att.get("url", ""))
                if not stored_name:
                    continue
                conn.execute(
                    """
                    INSERT INTO attachments
                        (session_id, message_idx, original_name, stored_name, url)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        idx,
                        att.get("name", ""),
                        stored_name,
                        att.get("url", ""),
                    ),
                )
    return updated_at


def _extract_stored_name(url: str) -> str | None:
    """从附件 URL 中解析出磁盘上的 stored_name。"""
    if not url:
        return None
    prefix = "/api/v1/uploads/"
    if url.startswith(prefix):
        return url[len(prefix) :]
    # 兜底：取 URL 路径最后一段
    if "/" in url:
        return os.path.basename(url)
    return url or None


def list_all() -> list[dict]:
    """返回所有会话（按 updated_at 降序），包含完整消息。"""
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
        )
        sessions = []
        for row in cur.fetchall():
            sessions.append(
                _load_session(conn, row["id"], row["title"], row["updated_at"])
            )
        return sessions


def _load_session(
    conn: sqlite3.Connection, session_id: str, title: str, updated_at: str
) -> dict:
    """加载单个会话的消息和附件。"""
    cur = conn.execute(
        """
        SELECT role, content, references_json, reasoning, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY created_at ASC
        """,
        (session_id,),
    )
    messages: list[dict] = []
    for row in cur.fetchall():
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "references": (
                    json.loads(row["references_json"])
                    if row["references_json"]
                    else []
                ),
                "reasoning": row["reasoning"],
            }
        )

    # 把附件还原到对应消息
    cur = conn.execute(
        "SELECT message_idx, original_name, url FROM attachments WHERE session_id = ?",
        (session_id,),
    )
    for row in cur.fetchall():
        idx = int(row["message_idx"])
        if 0 <= idx < len(messages):
            messages[idx].setdefault("attachments", []).append(
                {
                    "name": row["original_name"],
                    "type": _guess_attachment_type(row["url"]),
                    "url": row["url"],
                }
            )

    return {
        "id": session_id,
        "title": title,
        "updated_at": updated_at,
        "messages": messages,
    }


def _guess_attachment_type(url: str) -> str:
    """根据 URL 后缀猜测附件类型。"""
    ext = Path(url).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    return "image" if ext in image_exts else "document"


def load(session_id: str) -> dict | None:
    """按 id 读取单个会话，不存在时返回 None。"""
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT id, title, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return _load_session(conn, row["id"], row["title"], row["updated_at"])


def is_attachment_accessible(stored_name: str, allowed_session_ids: set[str]) -> bool:
    """判断附件是否属于给定的会话之一。"""
    if not allowed_session_ids:
        return False
    with _LOCK, _connect() as conn:
        placeholders = ",".join("?" * len(allowed_session_ids))
        row = conn.execute(
            f"SELECT 1 FROM attachments WHERE stored_name = ? AND session_id IN ({placeholders})",
            (stored_name, *allowed_session_ids),
        ).fetchone()
        return row is not None


def delete(session_id: str) -> tuple[bool, list[str]]:
    """删除会话；返回 (是否删除成功, 需要清理的附件 stored_name 列表)。

    attachments 表由 ON DELETE CASCADE 自动清理，这里只负责收集待删除物理文件名。
    """
    with _LOCK, _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not exists:
            return False, []
        cur = conn.execute(
            "SELECT stored_name FROM attachments WHERE session_id = ?",
            (session_id,),
        )
        stored_names = [row["stored_name"] for row in cur.fetchall()]
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return True, stored_names


def migrate_from_json(json_dir: Path | None = None) -> int:
    """将历史 JSON 文件一次性导入 SQLite，返回导入会话数。"""
    if json_dir is None:
        json_dir = get_settings().sessions_full_dir

    if not json_dir.is_dir():
        return 0

    imported = 0
    for path in sorted(json_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            session_id = data.get("id")
            title = data.get("title", "新对话")
            messages = data.get("messages", [])
            if not session_id:
                continue
            save(session_id, title, messages)
            imported += 1
        except (OSError, json.JSONDecodeError):
            logger.exception("迁移会话文件失败: %s", path)
            continue
    return imported
