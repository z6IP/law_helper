"""qwen3.7-text-embedding 向量模型封装（阿里云百炼 OpenAI 兼容 API）。

工程约束：Embedding 模型通过阿里云百炼 API 调用，不在本地加载权重，
避免内存占用与模型下载依赖。返回向量已做 L2 归一化，可直接用于余弦相似度计算。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.errors import ConfigError
from app.tracing import event, span


class EmbeddingModel:
    """qwen3.7-text-embedding 语义向量封装（懒加载单例，API 调用）。"""

    # 百炼 embedding API 单次请求最大行数（qwen3.7-text-embedding 为 20）
    _BATCH_SIZE = 20

    def __init__(self) -> None:
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
            raise ConfigError("请在 .env 中配置有效的 OPENAI_API_KEY")
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2 归一化向量，使点积等于余弦相似度。"""
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        settings = get_settings()
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        with span(
            "embedding.documents",
            model=settings.embedding_model_id,
            count=len(texts),
        ):
            # 分批调用（单次最多 _BATCH_SIZE 行）
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                resp = self._client.embeddings.create(
                    model=settings.embedding_model_id,
                    input=batch,
                    dimensions=settings.embedding_dimensions,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    event(
                        "embedding.tokens",
                        model=settings.embedding_model_id,
                        prompt_tokens=usage.prompt_tokens,
                        total_tokens=usage.total_tokens,
                    )
                # 按 index 排序，确保与输入顺序一致
                data = sorted(resp.data, key=lambda d: d.index)
                for d in data:
                    all_vectors.append(self._normalize(d.embedding))
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        settings = get_settings()
        with span(
            "embedding.query",
            model=settings.embedding_model_id,
        ):
            resp = self._client.embeddings.create(
                model=settings.embedding_model_id,
                input=text,
                dimensions=settings.embedding_dimensions,
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:
                event(
                    "embedding.tokens",
                    model=settings.embedding_model_id,
                    prompt_tokens=usage.prompt_tokens,
                    total_tokens=usage.total_tokens,
                )
            return self._normalize(resp.data[0].embedding)


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
