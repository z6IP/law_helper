"""免登录用户认证：匿名身份、令牌授权、门禁依赖与每日配额。

设计（详见 docs/auth.md）：
- get_or_create_user：首次访问时为访客生成稳定 user_id 写入 starsessions 会话并落库，
  身份随 Cookie 持久化（request.state 缓存，单次请求内只计数一次）
- require_authorized_user：门禁依赖，AUTH_MODE=token 时校验授权标记，未授权返回 403；
  AUTH_MODE=open 时退化为仅返回身份（保留单用户本地体验）
- 令牌：secrets 生成，库中只存 SHA-256 哈希；兑换成功后种授权标记
- enforce_daily_chat_quota：服务端每日配额（usage_daily 计数），超限返回 429
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from app import auth_db
from app.config import get_settings


def _hash_token(token: str) -> str:
    """令牌明文 -> SHA-256 哈希（库中只存哈希，DB 泄露不致令牌泄露）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_or_create_user(request: Request) -> str:
    """返回当前访客的稳定匿名 user_id（首次访问生成并落库）。

    用 request.state 缓存，保证单次请求内（限流 key、门禁、配额多处复用）
    只进行一次注册/计数，避免 request_count 重复累加。
    """
    cached = getattr(request.state, "_user_id", None)
    if cached is not None:
        return cached

    user_id = request.session.get("user_id")
    if not user_id:
        user_id = uuid.uuid4().hex
        request.session["user_id"] = user_id
    auth_db.touch_user(user_id)
    request.state._user_id = user_id
    return user_id


def is_authorized(request: Request) -> bool:
    """当前浏览器会话是否已通过令牌授权。"""
    return bool(request.session.get("authorized"))


def require_authorized_user(request: Request) -> str:
    """门禁依赖：token 模式下未授权访客无法访问核心接口。

    返回 user_id，供端点用于限流/审计。
    """
    user_id = get_or_create_user(request)
    if get_settings().auth_mode == "open":
        return user_id
    if not is_authorized(request):
        raise HTTPException(status_code=403, detail="未授权访问，请使用有效链接")
    return user_id


def enforce_daily_chat_quota(request: Request) -> str:
    """每日聊天配额依赖：递增计数并在超限时返回 429。

    配额为 0 表示不限制。日期按服务器本地时区计（个人部署直觉优先）。
    """
    user_id = get_or_create_user(request)
    quota = get_settings().user_daily_chat_quota
    if quota <= 0:
        return user_id
    today = datetime.now().date().isoformat()
    count = auth_db.increment_usage(user_id, today)
    if count > quota:
        raise HTTPException(
            status_code=429,
            detail=f"今日配额已用完（{quota} 次/天），请明日再试",
        )
    return user_id


def generate_token(note: str, days: int) -> dict:
    """生成一枚授权令牌，返回含明文 token 的信息（仅此一次展示明文）。

    明文 token 用于拼授权链接；库中只存哈希。
    """
    token = secrets.token_urlsafe(32)
    expires_at = (
        (_utc_now() + timedelta(days=days)).isoformat(timespec="seconds")
        if days > 0
        else None
    )
    token_hash = _hash_token(token)
    auth_db.create_token(token_hash, expires_at, note)
    return {
        "token": token,
        "token_hash": token_hash,
        "expires_at": expires_at,
        "note": note,
    }


def consume_token(request: Request, token: str) -> bool:
    """校验并消费授权令牌：有效则在会话标记 authorized=True 并返回 True。

    对无效/已吊销/已过期令牌统一返回 False（不区分原因，避免探测）。
    """
    token_hash = _hash_token(token)
    record = auth_db.get_token(token_hash)
    if record is None or record["revoked"]:
        return False
    expires_at = record["expires_at"]
    if expires_at:
        try:
            if datetime.fromisoformat(expires_at) < _utc_now():
                return False
        except ValueError:
            return False
    request.session["authorized"] = True
    return True


def list_tokens() -> list[dict]:
    """列出全部令牌（不含明文）。"""
    return auth_db.list_tokens()


def revoke_token(token_hash: str) -> bool:
    """吊销令牌，返回是否命中。"""
    return auth_db.revoke_token(token_hash)
