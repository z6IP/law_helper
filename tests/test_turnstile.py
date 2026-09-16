"""Cloudflare Turnstile 人机验证模块测试。"""
from __future__ import annotations

import types

import pytest

from app import turnstile


def _settings(
    site_key: str = "",
    secret_key: str = "",
    hostnames: str = "",
    script_src: str = "",
):
    return types.SimpleNamespace(
        turnstile_site_key=site_key,
        turnstile_secret_key=secret_key,
        turnstile_hostnames=hostnames,
        turnstile_script_src=script_src,
    )


class _FakeResp:
    """siteverify 响应替身。"""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def patched(monkeypatch):
    def _patch(settings, fake_resp=None):
        monkeypatch.setattr("app.turnstile.get_settings", lambda: settings)
        if fake_resp is not None:
            monkeypatch.setattr("app.turnstile.requests.post", lambda *a, **k: fake_resp)
    return _patch


def test_is_enabled(monkeypatch):
    monkeypatch.setattr("app.turnstile.get_settings", lambda: _settings())
    assert turnstile.is_enabled() is False
    monkeypatch.setattr("app.turnstile.get_settings", lambda: _settings(site_key="s", secret_key="t"))
    assert turnstile.is_enabled() is True


def test_verify_disabled_passes_through(monkeypatch):
    """未启用（缺 secret）时恒通过，行为不变。"""
    monkeypatch.setattr("app.turnstile.get_settings", lambda: _settings(site_key="s"))
    assert turnstile.verify("whatever-token") is True


def test_script_sources_default_when_unset(monkeypatch):
    """未配置脚本源时回退到内置默认源（单源，不会引入额外等待）。"""
    monkeypatch.setattr("app.turnstile.get_settings", lambda: _settings())
    assert turnstile.script_sources() == [turnstile.DEFAULT_SCRIPT_SRC]


def test_script_sources_ordered_dedup_and_trim(monkeypatch):
    """多源按配置顺序去重、去空白，首个为主源。"""
    monkeypatch.setattr(
        "app.turnstile.get_settings",
        lambda: _settings(
            script_src=" https://a.example/api.js ,https://b.example/api.js,https://a.example/api.js "
        ),
    )
    assert turnstile.script_sources() == [
        "https://a.example/api.js",
        "https://b.example/api.js",
    ]


def test_script_sources_blank_falls_back_to_default(monkeypatch):
    """只配空白字符时回退到内置默认源，避免前端拿到空列表导致加载逻辑空转。"""
    monkeypatch.setattr("app.turnstile.get_settings", lambda: _settings(script_src=" , "))
    assert turnstile.script_sources() == [turnstile.DEFAULT_SCRIPT_SRC]


def test_verify_missing_token_fails(patched):
    patched(_settings(site_key="s", secret_key="t"))
    assert turnstile.verify(None) is False
    assert turnstile.verify("") is False
    assert turnstile.verify("x" * 2049) is False  # 超长


def test_verify_success_and_action_match(patched):
    patched(
        _settings(site_key="s", secret_key="t"),
        _FakeResp({"success": True, "action": "chat", "hostname": "lawhelper.xyz"}),
    )
    assert turnstile.verify("valid-token", "1.2.3.4") is True


def test_verify_action_mismatch_fails(patched):
    patched(
        _settings(site_key="s", secret_key="t"),
        _FakeResp({"success": True, "action": "signup", "hostname": "lawhelper.xyz"}),
    )
    assert turnstile.verify("valid-token") is False


def test_verify_hostname_not_whitelisted_fails(patched):
    patched(
        _settings(site_key="s", secret_key="t", hostnames="lawhelper.xyz"),
        _FakeResp({"success": True, "action": "chat", "hostname": "evil.example"}),
    )
    assert turnstile.verify("valid-token") is False


def test_verify_hostname_empty_whitelist_skips(patched):
    """未配置 hostnames 白名单时跳过 hostname 校验。"""
    patched(
        _settings(site_key="s", secret_key="t"),
        _FakeResp({"success": True, "action": "chat", "hostname": "any.example"}),
    )
    assert turnstile.verify("valid-token") is True


def test_verify_network_error_fails_closed(patched, monkeypatch):
    patched(_settings(site_key="s", secret_key="t"))
    def _boom(*a, **k):
        raise ConnectionError("network down")
    monkeypatch.setattr("app.turnstile.requests.post", _boom)
    assert turnstile.verify("valid-token") is False
