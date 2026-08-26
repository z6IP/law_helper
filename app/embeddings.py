"""BGE-M3 向量模型封装。

工程约束：Embedding 模型仅在「后端进程」加载，前端 Streamlit 不直接加载，
避免模型内存重复占用；模型权重通过 ModelScope 镜像下载，规避 HuggingFace.co 直连超时。
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.config import get_settings
from app.errors import ConfigError


def _download_model(model_id: str) -> str:
    """通过 ModelScope 下载模型，返回本地目录路径。"""
    settings = get_settings()
    cache_dir = str(settings.modelscope_full_dir)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("未安装 modelscope，请先 `pip install modelscope`") from exc

    return snapshot_download(model_id, cache_dir=cache_dir)


class EmbeddingModel:
    """BGE-M3 语义向量封装（懒加载单例）。"""

    def __init__(self) -> None:
        self._model = None
        self._model_dir: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._model_dir = _download_model(settings.embedding_model_id)
        self._model = SentenceTransformer(self._model_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        vector = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()