"""session_db / upload_store / session_store 测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import session_db, upload_store


@pytest.fixture
def temp_db(tmp_path: Path):
    """为每个测试提供独立的 SQLite 数据库和 uploads 目录。"""
    db_path = tmp_path / "law_helper.db"
    uploads_path = tmp_path / "uploads"
    uploads_path.mkdir(parents=True, exist_ok=True)

    # 覆盖数据库路径
    session_db._set_db_path_for_test(db_path)
    # 覆盖上传目录
    original_uploads_dir = upload_store.UPLOADS_DIR
    upload_store.UPLOADS_DIR = uploads_path

    session_db.init_db()

    yield tmp_path

    # 恢复
    session_db._set_db_path_for_test(None)
    upload_store.UPLOADS_DIR = original_uploads_dir


def _create_upload(uploads_path: Path, stored_name: str) -> None:
    """创建一个虚拟上传文件。"""
    (uploads_path / stored_name).write_bytes(b"fake-image-data")


def test_save_and_list_all(temp_db: Path):
    """保存会话后能正确列出。"""
    session_db.save("sid-1", "测试会话", [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ])
    sessions = session_db.list_all()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "sid-1"
    assert sessions[0]["title"] == "测试会话"
    assert len(sessions[0]["messages"]) == 2


def test_delete_cascade_with_attachments(temp_db: Path):
    """删除会话时级联删除附件元数据并返回 stored_name。"""
    uploads_path = temp_db / "uploads"
    _create_upload(uploads_path, "abc123.webp")

    session_db.save("sid-2", "带附件会话", [
        {
            "role": "user",
            "content": "看图",
            "attachments": [
                {
                    "name": "test.webp",
                    "type": "image",
                    "url": "/api/v1/uploads/abc123.webp",
                }
            ],
        },
        {"role": "assistant", "content": "收到"},
    ])

    ok, stored_names = session_db.delete("sid-2")
    assert ok is True
    assert stored_names == ["abc123.webp"]
    assert session_db.load("sid-2") is None

    # 物理文件应被 session_store.delete 删除；这里验证 upload_store 能删除
    upload_store.delete_attachments(stored_names)
    assert not (uploads_path / "abc123.webp").exists()


def test_delete_without_attachments(temp_db: Path):
    """删除无附件会话不应报错。"""
    session_db.save("sid-3", "纯文本会话", [
        {"role": "user", "content": "hello"},
    ])
    ok, stored_names = session_db.delete("sid-3")
    assert ok is True
    assert stored_names == []


def test_migrate_from_json(temp_db: Path):
    """JSON 迁移应正确导入会话及附件。"""
    json_dir = temp_db / "sessions"
    json_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "sid-migrate",
        "title": "迁移会话",
        "updated_at": "2024-01-01T00:00:00",
        "messages": [
            {
                "role": "user",
                "content": "迁移测试",
                "attachments": [
                    {"name": "a.png", "type": "image", "url": "/api/v1/uploads/mig.png"}
                ],
            }
        ],
    }
    (json_dir / "sid-migrate.json").write_text(json.dumps(data), encoding="utf-8")

    count = session_db.migrate_from_json(json_dir)
    assert count == 1

    session = session_db.load("sid-migrate")
    assert session is not None
    assert session["title"] == "迁移会话"
    assert len(session["messages"]) == 1
    assert session["messages"][0]["attachments"][0]["url"] == "/api/v1/uploads/mig.png"
