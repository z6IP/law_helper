"""Cloudflare Turnstile 人机验证（防刷兜底，可选启用）。

背景：匿名 user_id 的限流/每日配额绑定 Cookie，机器人清 Cookie 即换新身份、重置配额；
Turnstile 在浏览器端完成无感人机挑战（Managed 模式），服务端 siteverify 校验一次性 token，
补上"清 Cookie 重置配额"的漏洞。

启用条件：TURNSTILE_SITE_KEY 与 TURNSTILE_SECRET_KEY 同时配置（默认空 = 禁用）。

规范依据：developers.cloudflare.com/turnstile/spin/prompt.md
- 校验 success === true、action 匹配、hostname 白名单（可选）、token 长度 0 < len <= 2048
- siteverify 10 秒超时；请求失败统一视为未通过（fail-closed），不暴露内部异常
"""
from __future__ import annotations

import logging

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

_SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
# 稳定 action：与前端 widget 的 data-action 一致（1-32 字符，仅字母数字下划线连字符）
EXPECTED_ACTION = "chat"
_MAX_TOKEN_LEN = 2048
# 内置默认验证脚本源（TURNSTILE_SCRIPT_SRC 留空时使用）
DEFAULT_SCRIPT_SRC = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"


def is_enabled() -> bool:
    """sitekey 与 secret 同时配置时才启用验证。"""
    s = get_settings()
    return bool(s.turnstile_site_key and s.turnstile_secret_key)


def script_sources() -> list[str]:
    """验证脚本地址候选列表（按配置顺序去重，供前端依次回退加载）。

    首个为主源；仅当前一个加载失败才尝试下一个，故默认单源不引入额外等待。
    """
    raw = get_settings().turnstile_script_src or DEFAULT_SCRIPT_SRC
    seen: dict[str, None] = {}
    for item in raw.split(","):
        url = item.strip()
        if url:
            seen.setdefault(url, None)
    return list(seen) or [DEFAULT_SCRIPT_SRC]


def verify(token: str | None, remote_ip: str | None = None) -> bool:
    """校验一次性 Turnstile token。

    - 未启用（配置缺失）：恒 True，行为不变（功能开关 fail-open）
    - 启用但 token 缺失/非法：False
    - siteverify 网络异常/非 2xx：False（fail-closed，避免绕过）

    所有拒绝路径记 warning 日志（仅事件类型与长度，不含 token 明文与 secret），
    便于运维区分「脚本被网络阻断导致正常用户被拦」与「真实刷量」。
    """
    if not is_enabled():
        return True
    if not isinstance(token, str) or not (0 < len(token) <= _MAX_TOKEN_LEN):
        logger.warning(
            "Turnstile 拒绝：token 缺失或长度非法（len=%s）",
            len(token) if isinstance(token, str) else None,
        )
        return False

    settings = get_settings()
    hostnames = {h.strip() for h in settings.turnstile_hostnames.split(",") if h.strip()}

    try:
        resp = requests.post(
            _SITEVERIFY_URL,
            data={
                "secret": settings.turnstile_secret_key,
                "response": token,
                **({"remoteip": remote_ip} if remote_ip else {}),
            },
            timeout=10,
        )
        resp.raise_for_status()
        outcome = resp.json()
    except Exception:  # noqa: BLE001
        logger.exception("Turnstile siteverify 调用失败")
        return False

    if outcome.get("success") is not True:
        logger.warning(
            "Turnstile 拒绝：siteverify 未通过（error-codes=%s）",
            outcome.get("error-codes"),
        )
        return False
    if outcome.get("action") != EXPECTED_ACTION:
        logger.warning(
            "Turnstile 拒绝：action 不匹配（期望 %s，实际 %r）",
            EXPECTED_ACTION,
            outcome.get("action"),
        )
        return False
    # hostname 白名单：仅当配置时校验（生产建议配置）
    if hostnames and outcome.get("hostname") not in hostnames:
        logger.warning(
            "Turnstile 拒绝：hostname 不在白名单（实际 %r）",
            outcome.get("hostname"),
        )
        return False
    return True
