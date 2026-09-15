"""认证数据持久化：users / access_tokens / usage_daily 三表。

与 app/session_db.py 同目录、同惯例（SQLite WAL + threading.Lock），
专供免登录认证模块使用，不引入任何新依赖。

- users：匿名用户身份（user_id、首次/最近访问时间、请求计数、状态）
- access_tokens：授权令牌（只存 SHA-256 哈希，不落明文）
- usage_daily：按 (user_id, date) 聚合的每日聊天用量，支撑配额与统计
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import USER_DATA_DIR

logger = logging.getLogger(__name__)

_DB_PATH: Path | None = None
_LOCK = threading.Lock()
_INITED = False


def _db_path() -> Path:
    """返回认证数据库路径：~/.law_helper/session_history/auth.db。"""
    global _DB_PATH
    if _DB_PATH is None:
        db_dir = USER_DATA_DIR / "session_history"
        db_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = db_dir / "auth.db"
    return _DB_PATH


def _set_db_path_for_test(path: Path | None) -> None:
    """测试专用：覆盖默认数据库路径（None 恢复默认）。"""
    global _DB_PATH, _INITED
    _DB_PATH = path
    _INITED = False


def _connect() -> sqlite3.Connection:
    """创建启用外键与 WAL 模式的连接；首次连接时自动建表（幂等）。"""
    global _INITED
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    if not _INITED:
        _create_tables(conn)
        _INITED = True
    return conn


def init_db() -> None:
    """初始化数据库表结构（幂等）。"""
    with _LOCK, closing(_connect()) as conn:
        _create_tables(conn)


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS access_tokens (
            token_hash TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            note TEXT,
            revoked INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS usage_daily (
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_access_tokens_expires
            ON access_tokens(expires_at);
        """
    )
    conn.commit()


def _now_iso() -> str:
    """UTC ISO 时间（秒级），与 session_db 保持一致。"""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def touch_user(user_id: str) -> None:
    """注册新用户或更新已有用户的最近访问时间与请求计数。"""
    now = _now_iso()
    with _LOCK, closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, first_seen, last_seen, request_count, status)
            VALUES (?, ?, ?, 1, 'active')
            ON CONFLICT(user_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                request_count = request_count + 1
            """,
            (user_id, now, now),
        )
        conn.commit()


def user_exists(user_id: str) -> bool:
    """判断用户是否已注册（active）。"""
    with _LOCK, closing(_connect()) as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        return row is not None


def create_token(token_hash: str, expires_at: str | None, note: str) -> None:
    """写入一枚令牌（只存哈希）。"""
    with _LOCK, closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO access_tokens (token_hash, created_at, expires_at, note, revoked)
            VALUES (?, ?, ?, ?, 0)
            """,
            (token_hash, _now_iso(), expires_at, note),
        )
        conn.commit()


def get_token(token_hash: str) -> dict | None:
    """按哈希读取令牌（含有效期与吊销状态）。"""
    with _LOCK, closing(_connect()) as conn:
        row = conn.execute(
            "SELECT token_hash, created_at, expires_at, note, revoked "
            "FROM access_tokens WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def revoke_token(token_hash: str) -> bool:
    """吊销令牌（软删除，保留审计痕迹）。"""
    with _LOCK, closing(_connect()) as conn:
        cur = conn.execute(
            "UPDATE access_tokens SET revoked = 1 WHERE token_hash = ?",
            (token_hash,),
        )
        conn.commit()
        return cur.rowcount > 0


def list_tokens() -> list[dict]:
    """列出全部令牌（不含明文，按创建时间倒序）。"""
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT token_hash, created_at, expires_at, note, revoked "
            "FROM access_tokens ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def increment_usage(user_id: str, date: str) -> int:
    """递增某用户某天的聊天计数并返回新值（原子 UPSERT）。"""
    with _LOCK, closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO usage_daily (user_id, date, count)
            VALUES (?, ?, 1)
            ON CONFLICT(user_id, date) DO UPDATE SET count = count + 1
            """,
            (user_id, date),
        )
        row = conn.execute(
            "SELECT count FROM usage_daily WHERE user_id = ? AND date = ?",
            (user_id, date),
        ).fetchone()
        conn.commit()
        return int(row["count"])


def get_usage(user_id: str, date: str) -> int:
    """读取某用户某天的聊天计数。"""
    with _LOCK, closing(_connect()) as conn:
        row = conn.execute(
            "SELECT count FROM usage_daily WHERE user_id = ? AND date = ?",
            (user_id, date),
        ).fetchone()
        return int(row["count"]) if row else 0
