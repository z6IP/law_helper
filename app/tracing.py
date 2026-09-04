"""轻量全链路 trace 记录器（自研，零外部依赖）。

设计要点：
- 以「一个请求 / 一次预热 / 一次入库」为一个 trace，分配唯一 trace_id；
- span 表示一段有开始与耗时的步骤，event 表示瞬时事实（如候选数、命中数）；
- 通过 contextvars 让 trace_id / 当前 span 在同一执行上下文内自动传递，无需显式传参。

存储：双写
- logs/trace.jsonl：append-only 明细，每行一个 JSON，易 grep / 手写分析；
- logs/trace.db（SQLite）：结构化存取，供 Dashboard 按 trace_id / token / 缓存命中聚合查询。
  表结构：traces（一行一个 trace 的摘要）与 spans（一行一个 span 明细）。

记录结构（JSONL 每行一个 JSON 对象）：
  {"kind": "trace|span|event", "trace_id": ..., "span_id": ...,
   "parent_id": ..., "name": ..., "start": ISO 时间,
   "duration_ms": float | null, "status": "ok|error", "attributes": {...}}

无 trace 上下文时（如直接 `python app/ingestion.py`），span/event 为 no-op，不产生输出。
"""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

# 项目根目录（app/ 的上一级），trace 文件与 logs/backend.log 同级存放
BASE_DIR = Path(__file__).resolve().parent.parent
TRACE_FILE = BASE_DIR / "logs" / "trace.jsonl"
TRACE_DB = BASE_DIR / "logs" / "trace.db"

# 当前请求的 trace_id 与当前 span_id（跨模块自动传递，不污染全局状态）
_TRACE_ID = ContextVar("trace_id", default=None)
_SPAN_ID = ContextVar("span_id", default=None)
# trace 开始时间（perf 用于精确计算耗时，ISO 字符串用于展示排序）
_TRACE_START_PERF = ContextVar("trace_start_perf", default=None)
_TRACE_STARTED_AT = ContextVar("trace_started_at", default=None)

# 追加写文件 / DB 写串行化，避免并发请求交错脏写
_LOCK = threading.Lock()


def _now() -> str:
    """毫秒级 ISO 时间戳，用于按时间排序同一 trace 内的事件。"""
    return datetime.now().isoformat(timespec="milliseconds")


_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    kind TEXT,
    question TEXT,
    session_id TEXT,
    cache_hit INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ok',
    started_at TEXT,
    ended_at TEXT,
    duration_ms REAL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT,
    span_id TEXT,
    parent_id TEXT,
    name TEXT,
    started_at TEXT,
    duration_ms REAL,
    status TEXT,
    attributes TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_started_at ON traces(started_at);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
"""


def _init_db() -> None:
    """建表（幂等），并开启 WAL 以提升并发读写性能。"""
    TRACE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


# 模块加载即建表，保证任何写操作前数据库已就绪
_init_db()


def _db_execute(sql: str, params: tuple = ()) -> None:
    """执行单条写语句（调用方需持有 _LOCK）。"""
    conn = sqlite3.connect(TRACE_DB, check_same_thread=False)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _emit(record: dict) -> None:
    """追加一行 JSON 到 trace 文件；default=str 兜底 numpy/torch 等非 JSON 类型。"""
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK, open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _add_tokens(trace_id: str, prompt: int | None, completion: int | None, total: int | None) -> None:
    """把一次 LLM 调用的 token 累加到所属 trace 的汇总列。"""
    with _LOCK:
        _db_execute(
            "UPDATE traces SET prompt_tokens = prompt_tokens + ?, "
            "completion_tokens = completion_tokens + ?, total_tokens = total_tokens + ? "
            "WHERE trace_id = ?",
            (prompt or 0, completion or 0, total or 0, trace_id),
        )


def start_trace(**attrs) -> str:
    """开始一个新 trace，返回 trace_id；后续同上下文内的 span/event 自动归属。

    若已存在 trace 上下文，不覆盖（保持最外层 trace 为根），直接返回 None。
    支持的关键 attrs：kind / question / session_id / cache_hit（命中缓存时为 True）。
    """
    if _TRACE_ID.get() is not None:
        return _TRACE_ID.get()
    trace_id = uuid.uuid4().hex[:16]
    started_at = _now()
    _TRACE_ID.set(trace_id)
    _SPAN_ID.set(None)
    _TRACE_START_PERF.set(time.perf_counter())
    _TRACE_STARTED_AT.set(started_at)

    _emit({
        "kind": "trace",
        "trace_id": trace_id,
        "span_id": None,
        "parent_id": None,
        "name": "trace",
        "start": started_at,
        "duration_ms": None,
        "status": "ok",
        "attributes": attrs,
    })

    kind = attrs.get("kind")
    question = attrs.get("question")
    session_id = attrs.get("session_id")
    cache_hit = 1 if attrs.get("cache_hit") else 0
    with _LOCK:
        _db_execute(
            "INSERT OR REPLACE INTO traces "
            "(trace_id, kind, question, session_id, cache_hit, status, started_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (trace_id, kind, question, session_id, cache_hit, "ok", started_at),
        )
    return trace_id


def finish_trace(status: str = "ok") -> None:
    """结束当前 trace：回填 ended_at 与耗时（自 start 起）。调用方在请求收尾时调用。"""
    trace_id = _TRACE_ID.get()
    if trace_id is None:
        return
    start_perf = _TRACE_START_PERF.get()
    ended_at = _now()
    duration_ms = (
        round((time.perf_counter() - start_perf) * 1000.0, 2)
        if start_perf is not None
        else None
    )
    with _LOCK:
        _db_execute(
            "UPDATE traces SET status = ?, ended_at = ?, duration_ms = ? WHERE trace_id = ?",
            (status, ended_at, duration_ms, trace_id),
        )


def event(name: str, **attrs) -> None:
    """记录一个瞬时事件（无耗时），归属当前 trace / 当前 span。"""
    trace_id = _TRACE_ID.get()
    if trace_id is None:
        return
    # llm.tokens 事件同时累加到 trace 的 token 汇总列，供 Dashboard 直接读数
    if name == "llm.tokens":
        _add_tokens(
            trace_id,
            attrs.get("prompt_tokens"),
            attrs.get("completion_tokens"),
            attrs.get("total_tokens"),
        )
    _emit({
        "kind": "event",
        "trace_id": trace_id,
        "span_id": _SPAN_ID.get(),
        "parent_id": None,
        "name": name,
        "start": _now(),
        "duration_ms": None,
        "status": "ok",
        "attributes": attrs,
    })


@contextmanager
def span(name: str, **attrs):
    """记录一段带耗时的步骤；异常时 status 置为 error 并透传异常。

    支持嵌套：子 span 的 parent_id 自动指向当前 span，构成调用树。
    """
    trace_id = _TRACE_ID.get()
    if trace_id is None:
        yield
        return

    parent_id = _SPAN_ID.get()
    span_id = uuid.uuid4().hex[:8]
    start_wall = _now()
    start_perf = time.perf_counter()
    token = _SPAN_ID.set(span_id)

    status = "ok"
    record_attrs = dict(attrs)
    try:
        yield
    except Exception:
        status = "error"
        exc_info = sys.exc_info()
        record_attrs["error"] = f"{exc_info[0].__name__}: {exc_info[1]}"
        raise
    finally:
        _SPAN_ID.reset(token)
        duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
        _emit({
            "kind": "span",
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id,
            "name": name,
            "start": start_wall,
            "duration_ms": duration_ms,
            "status": status,
            "attributes": record_attrs,
        })
        with _LOCK:
            _db_execute(
                "INSERT INTO spans "
                "(trace_id, span_id, parent_id, name, started_at, duration_ms, status, attributes) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    trace_id,
                    span_id,
                    parent_id,
                    name,
                    start_wall,
                    duration_ms,
                    status,
                    json.dumps(record_attrs, ensure_ascii=False, default=str),
                ),
            )


def query_traces(limit: int = 200, offset: int = 0) -> dict:
    """只读查询：返回 trace 摘要列表与汇总统计（供 Dashboard 使用）。

    排序按开始时间倒序，最多返回 limit 条。
    """
    conn = sqlite3.connect(TRACE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT trace_id, kind, question, session_id, cache_hit, status, "
            "started_at, duration_ms, prompt_tokens, completion_tokens, total_tokens "
            "FROM traces ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        agg = conn.execute(
            "SELECT COUNT(*) AS total_count, COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "COALESCE(SUM(cache_hit), 0) AS cache_hit_count, "
            "COALESCE(AVG(duration_ms), 0) AS avg_duration_ms FROM traces"
        ).fetchone()
    finally:
        conn.close()

    traces = [
        {
            "trace_id": r["trace_id"],
            "kind": r["kind"],
            "question": r["question"],
            "session_id": r["session_id"],
            "cache_hit": bool(r["cache_hit"]),
            "status": r["status"],
            "started_at": r["started_at"] or "",
            "duration_ms": r["duration_ms"] or 0,
            "prompt_tokens": r["prompt_tokens"] or 0,
            "completion_tokens": r["completion_tokens"] or 0,
            "total_tokens": r["total_tokens"] or 0,
        }
        for r in rows
    ]
    return {
        "traces": traces,
        "total_count": agg["total_count"] or 0,
        "total_tokens": agg["total_tokens"] or 0,
        "cache_hit_count": agg["cache_hit_count"] or 0,
        "avg_duration_ms": round(agg["avg_duration_ms"] or 0, 2),
    }