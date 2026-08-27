"""BGE 向量模型封装（bge-small-zh-v1.5，轻量 CPU 友好）。

工程约束：Embedding 模型仅在「后端进程」加载，前端 Streamlit 不直接加载，
避免模型内存重复占用；模型权重通过 ModelScope 镜像下载，规避 HuggingFace.co 直连超时。
CPU 推理优化：torch.inference_mode() + model.eval() + 最优线程数。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.errors import ConfigError


def _get_cached_model_path(model_id: str) -> str | None:
    """检查模型是否已缓存，返回缓存路径（如有）。

    ModelScope snapshot_download 的存储结构：
      {cache_dir}/models/{model_id}/snapshots/{revision}/

    我们检查 snapshots/master 目录是否包含 config.json。
    """
    settings = get_settings()
    cache_dir = Path(str(settings.modelscope_full_dir))
    # ModelScope 在 cache_dir 下会再创建 models/ 子目录
    model_cache = cache_dir / "models" / model_id.replace("/", "--") / "snapshots" / "master"
    if model_cache.is_dir() and (model_cache / "config.json").is_file():
        return str(model_cache)
    return None


def _download_model(model_id: str) -> str:
    """通过 ModelScope 下载模型，返回本地目录路径。

    优化：如果模型已缓存则直接返回，跳过 snapshot_download 的验证进度条。
    """
    # 优先检查本地缓存
    cached = _get_cached_model_path(model_id)
    if cached:
        return cached

    settings = get_settings()
    cache_dir = str(settings.modelscope_full_dir)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("未安装 modelscope，请先 `pip install modelscope`") from exc

    return snapshot_download(model_id, cache_dir=cache_dir)


def _optimize_torch_for_cpu() -> None:
    """设置 PyTorch CPU 推理最优参数。"""
    import torch

    # 限制线程数，避免过度并行导致的 overhead
    # 对小模型（134M）来说 4-8 线程通常最优
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    optimal_threads = min(max(n_cores // 2, 2), 8)
    torch.set_num_threads(optimal_threads)
    # 启用 cuDNN benchmark（对 CPU 无影响但无害）
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


_optimize_torch_for_cpu()


class EmbeddingModel:
    """bge-small-zh-v1.5 语义向量封装（懒加载单例，CPU 友好）。"""

    def __init__(self) -> None:
        self._model = None
        self._model_dir: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._model_dir = _download_model(settings.embedding_model_id)
        # 直接从本地路径加载，跳过网络检查
        self._model = SentenceTransformer(self._model_dir, local_files_only=True)
        # 确保模型在 eval 模式（禁用 dropout 等）
        if hasattr(self._model, "eval"):
            self._model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        import torch

        with torch.inference_mode():
            vectors = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        import torch

        with torch.inference_mode():
            vector = self._model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
        return vector.tolist()


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
