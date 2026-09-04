"""session_store 兼容层测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import session_db, session_store, upload_store


@pytest.fixture
def temp_db(tmp_path: Path):
    db_path = tmp_path / "law_helper.db"
    uploads_path = tmp_path / "uploads"
    uploads_path.mkdir(parents=True, exist_ok=True)

    session_db._set_db_path_for_test(db_path)
    original_uploads_dir = upload_store.UPLOADS_DIR
    upload_store.UPLOADS_DIR = uploads_path

    session_store._ensure_db()

    yield tmp_path

    session_db._set_db_path_for_test(None)
    upload_store.UPLOADS_DIR = original_uploads_dir


def test_session_store_save_list_delete(temp_db: Path):
    """session_store 兼容层能保存、列出、删除会话。"""
    uploads_path = temp_db / "uploads"
    (uploads_path / "img.webp").write_bytes(b"fake")

    session_store.save(
        "sid-1",
        "测试",
        [
            {
                "role": "user",
                "content": "hi",
                "attachments": [
                    {"name": "img.webp", "type": "image", "url": "/api/v1/uploads/img.webp"}
                ],
            }
        ],
    )

    sessions = session_store.list_all()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "sid-1"

    ok = session_store.delete("sid-1")
    assert ok is True
    assert len(session_store.list_all()) == 0
    assert not (uploads_path / "img.webp").exists()


def test_session_store_migrate_from_json(temp_db: Path):
    """session_store 能迁移历史 JSON 文件。"""
    sessions_dir = temp_db / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "sid-old",
        "title": "旧会话",
        "updated_at": "2024-01-01T00:00:00",
        "messages": [{"role": "user", "content": "hello"}],
    }
    (sessions_dir / "sid-old.json").write_text(json.dumps(data), encoding="utf-8")

    count = session_store.migrate_from_json(sessions_dir)
    assert count == 1

    session = session_db.load("sid-old")
    assert session is not None
    assert session["title"] == "旧会话"
