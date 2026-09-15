"""免登录认证模块测试：匿名身份、令牌、门禁、每日配额。"""
from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app import auth, auth_db


class FakeRequest:
    """轻量 Request 替身：仅暴露 auth 模块依赖的 session 与 state。"""

    def __init__(self):
        self.session: dict = {}
        self.state = types.SimpleNamespace()


def _settings(auth_mode: str = "open", quota: int = 50):
    return types.SimpleNamespace(auth_mode=auth_mode, user_daily_chat_quota=quota)


@pytest.fixture
def temp_db(tmp_path):
    """每个测试使用独立的认证数据库。"""
    db_path = tmp_path / "auth.db"
    auth_db._set_db_path_for_test(db_path)
    auth_db.init_db()
    yield
    auth_db._set_db_path_for_test(None)


# ---------- 匿名身份 ----------

def test_get_or_create_user_reuses_identity(temp_db):
    req = FakeRequest()
    uid1 = auth.get_or_create_user(req)
    uid2 = auth.get_or_create_user(req)
    assert uid1 == uid2
    assert req.session["user_id"] == uid1
    assert auth_db.user_exists(uid1) is True


def test_get_or_create_user_distinct_per_session(temp_db):
    uid1 = auth.get_or_create_user(FakeRequest())
    uid2 = auth.get_or_create_user(FakeRequest())
    assert uid1 != uid2


# ---------- 令牌生成 / 兑换 / 吊销 / 过期 ----------

def test_token_generate_and_consume(temp_db):
    info = auth.generate_token("alice", 30)
    assert info["token"]
    assert info["expires_at"] is not None

    # 明文不落库，只存哈希
    assert auth_db.get_token(info["token_hash"])["revoked"] == 0

    req = FakeRequest()
    assert auth.consume_token(req, info["token"]) is True
    assert req.session["authorized"] is True


def test_consume_invalid_token(temp_db):
    req = FakeRequest()
    assert auth.consume_token(req, "nonexistent-token") is False
    assert "authorized" not in req.session


def test_consume_revoked_token(temp_db):
    info = auth.generate_token("bob", 30)
    assert auth.revoke_token(info["token_hash"]) is True
    req = FakeRequest()
    assert auth.consume_token(req, info["token"]) is False


def test_consume_expired_token(temp_db):
    token = "expired-plain-token"
    token_hash = auth._hash_token(token)
    expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    auth_db.create_token(token_hash, expired, "expired")
    req = FakeRequest()
    assert auth.consume_token(req, token) is False


def test_revoke_unknown_token(temp_db):
    assert auth.revoke_token("no-such-hash") is False


# ---------- 门禁 ----------

def test_require_authorized_user_open_mode(temp_db, monkeypatch):
    """open 模式退化为仅返回匿名身份，不做门禁。"""
    monkeypatch.setattr("app.auth.get_settings", lambda: _settings(auth_mode="open"))
    req = FakeRequest()
    user_id = auth.require_authorized_user(req)
    assert user_id


def test_require_authorized_user_token_denies_unauthorized(temp_db, monkeypatch):
    monkeypatch.setattr("app.auth.get_settings", lambda: _settings(auth_mode="token"))
    req = FakeRequest()
    with pytest.raises(HTTPException) as exc:
        auth.require_authorized_user(req)
    assert exc.value.status_code == 403


def test_require_authorized_user_token_allows_authorized(temp_db, monkeypatch):
    monkeypatch.setattr("app.auth.get_settings", lambda: _settings(auth_mode="token"))
    req = FakeRequest()
    req.session["authorized"] = True
    user_id = auth.require_authorized_user(req)
    assert user_id


# ---------- 每日配额 ----------

def test_daily_quota_exceeded_returns_429(temp_db, monkeypatch):
    monkeypatch.setattr("app.auth.get_settings", lambda: _settings(auth_mode="token", quota=2))
    req = FakeRequest()
    auth.enforce_daily_chat_quota(req)  # 第 1 次
    auth.enforce_daily_chat_quota(req)  # 第 2 次
    with pytest.raises(HTTPException) as exc:
        auth.enforce_daily_chat_quota(req)  # 第 3 次 -> 429
    assert exc.value.status_code == 429


def test_daily_quota_zero_means_unlimited(temp_db, monkeypatch):
    monkeypatch.setattr("app.auth.get_settings", lambda: _settings(auth_mode="open", quota=0))
    req = FakeRequest()
    for _ in range(5):
        auth.enforce_daily_chat_quota(req)  # 不抛异常
