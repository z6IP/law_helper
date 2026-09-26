"""DashScope embedding 分支参数构造测试（不发起真实网络请求）。

守护点：qwen3-vl-embedding 走 MultiModalEmbedding 时，input 必须是元素列表
（[{"text": ...}]）而非 {"contents": [...]}，且必须显式传 dimension，否则默认
2560 维会与配置维度不一致导致向量库写入失败。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.embeddings import EmbeddingModel


class _FakeDashScope:
    """记录调用参数的 dashscope 替身。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.MultiModalEmbedding = SimpleNamespace(call=self._multi_modal)
        self.TextEmbedding = SimpleNamespace(call=self._text)

    def _multi_modal(self, **kwargs):
        self.calls.append(("multi_modal", kwargs))
        return kwargs

    def _text(self, **kwargs):
        self.calls.append(("text", kwargs))
        return kwargs


def _settings(model_id: str, dimensions: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_model_id=model_id, embedding_dimensions=dimensions
    )


def test_vl_branch_passes_element_list_and_dimension():
    fake = _FakeDashScope()

    EmbeddingModel()._call_dashscope_embedding(
        fake, _settings("qwen3-vl-embedding"), ["甲", "乙"], is_vl=True
    )

    kind, kwargs = fake.calls[0]
    assert kind == "multi_modal"
    assert kwargs["model"] == "qwen3-vl-embedding"
    assert kwargs["input"] == [{"text": "甲"}, {"text": "乙"}]
    assert kwargs["dimension"] == 1024


def test_text_branch_passes_string_list_and_dimension():
    fake = _FakeDashScope()

    EmbeddingModel()._call_dashscope_embedding(
        fake, _settings("qwen3.7-text-embedding"), ["甲"], is_vl=False
    )

    kind, kwargs = fake.calls[0]
    assert kind == "text"
    assert kwargs["input"] == ["甲"]
    assert kwargs["dimension"] == 1024


@pytest.mark.parametrize("dimensions", [768, 1024, 2560])
def test_dimension_always_forwarded(dimensions):
    """维度必填（config.py 无默认值），任何取值都应原样透传。"""
    fake = _FakeDashScope()

    EmbeddingModel()._call_dashscope_embedding(
        fake, _settings("qwen3-vl-embedding", dimensions), ["甲"], is_vl=True
    )

    assert fake.calls[0][1]["dimension"] == dimensions
