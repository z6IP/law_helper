"""嵌入模型封装：支持阿里云百炼 OpenAI 兼容 API、DashScope 原生 SDK 与本地 sentence-transformers 三模式。

工程约束：
- API 模式（embedding_backend=api）：通过阿里云百炼 OpenAI 兼容 API 调用，
  模型与维度由 .env 显式配置。
- DashScope 原生模式（embedding_backend=dashscope）：通过 dashscope SDK 调用，
  支持 qwen3.7-text-embedding、qwen3-vl-embedding 等模型；自动按模型名选择
  TextEmbedding 或 MultiModalEmbedding。复用 DASHSCOPE_API_KEY（未配置时回退 OPENAI_API_KEY）。
- 本地模式（embedding_backend=local，离线/零 token 场景）：使用 sentence-transformers
  加载本地模型，默认使用 BGE-base-zh-v1.5（768 维），适配内存受限环境（2GB RAM）。

所有模式返回向量均做 L2 归一化，可直接用于余弦相似度计算。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.errors import ConfigError, RetrievalError
from app.tracing import event, span


class EmbeddingModel:
    """嵌入模型封装（懒加载单例，支持本地 / OpenAI-API / DashScope 三模式）。"""

    # API 模式：百炼 embedding API 单次请求最大行数。
    # text-embedding-v3 / v4 为 10，qwen3.7-text-embedding 为 20。
    # 取 10 作为安全下限，兼容所有模型，避免切换模型时触发 400 错误。
    _BATCH_SIZE = 10

    def __init__(self) -> None:
        self._client = None
        self._local_model = None
        self._backend: str | None = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        settings = get_settings()
        if settings.embedding_backend == "local":
            from sentence_transformers import SentenceTransformer

            # 优先从项目内 models/ 目录加载，避免依赖 HuggingFace 全局缓存。
            # 这样部署到云服务器时只需把项目目录整体上传，无需额外配置缓存路径，
            # 也无需联网下载。模型目录约定：{BASE_DIR}/models/{model_local_name}/
            from app.config import BASE_DIR

            model_id = settings.embedding_local_model  # 例如 "BAAI/bge-base-zh-v1.5"
            # 路径安全校验：model_id 来自 .env 配置，防止路径遍历字符导致越权加载
            if ".." in model_id:
                raise ConfigError(
                    f"embedding_local_model 不允许包含路径遍历字符 '..'：{model_id}"
                )
            local_dir = BASE_DIR / "models" / model_id.split("/")[-1]
            load_path = str(local_dir) if local_dir.exists() else model_id

            self._local_model = SentenceTransformer(
                load_path,
                device="cpu",  # 2GB RAM 环境强制 CPU
            )
            self._backend = "local"
            event(
                "embedding.local_loaded",
                model=load_path,
                dim=settings.embedding_dimensions,
                from_project_dir=local_dir.exists(),
            )
            return
        if settings.embedding_backend == "dashscope":
            import dashscope

            api_key = settings.dashscope_api_key or settings.openai_api_key
            if not api_key or api_key.startswith("your_"):
                raise ConfigError(
                    "请在 .env 中配置有效的 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）"
                )
            dashscope.api_key = api_key
            self._backend = "dashscope"
            event(
                "embedding.dashscope_loaded",
                model=settings.embedding_model_id,
                dim=settings.embedding_dimensions,
            )
            return

        # API 模式（OpenAI 兼容，兜底）
        from openai import OpenAI

        if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
            raise ConfigError("请在 .env 中配置有效的 OPENAI_API_KEY")
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
        self._backend = "api"

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

        model_name = (
            settings.embedding_local_model
            if self._backend == "local"
            else settings.embedding_model_id
        )

        if self._backend == "local":
            return self._embed_documents_local(texts, model_name)
        if self._backend == "dashscope":
            return self._embed_documents_dashscope(texts, settings, model_name)
        return self._embed_documents_api(texts, settings, model_name)

    def _embed_documents_local(
        self, texts: list[str], model_name: str
    ) -> list[list[float]]:
        """本地模式：用 sentence-transformers 编码文档。"""
        with span("embedding.documents", model=model_name, count=len(texts)):
            # sentence-transformers 内部已分批处理，无需手动分批
            embeddings = self._local_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # 手动归一化保持一致
            )
            # 确保是 2D 数组
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return [self._normalize(emb.tolist()) for emb in embeddings]

    def _embed_documents_api(
        self, texts: list[str], settings, model_name: str
    ) -> list[list[float]]:
        """API 模式：分批调用百炼 embedding API。"""
        all_vectors: list[list[float]] = []
        with span("embedding.documents", model=model_name, count=len(texts)):
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
        model_name = (
            settings.embedding_local_model
            if self._backend == "local"
            else settings.embedding_model_id
        )

        if self._backend == "local":
            return self._embed_query_local(text, model_name)
        if self._backend == "dashscope":
            return self._embed_query_dashscope(text, settings, model_name)
        return self._embed_query_api(text, settings, model_name)

    def _embed_query_local(self, text: str, model_name: str) -> list[float]:
        """本地模式：用 sentence-transformers 编码查询。"""
        with span("embedding.query", model=model_name):
            embedding = self._local_model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # 手动归一化保持一致
            )
            return self._normalize(embedding.tolist())

    def _embed_query_api(self, text: str, settings, model_name: str) -> list[float]:
        """API 模式：调用百炼 embedding API。"""
        with span("embedding.query", model=model_name):
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

    def _embed_documents_dashscope(
        self, texts: list[str], settings, model_name: str
    ) -> list[list[float]]:
        """DashScope 原生模式：批量调用文本或多模态 embedding。"""
        import dashscope
        from http import HTTPStatus

        is_vl = "vl" in settings.embedding_model_id.lower()
        all_vectors: list[list[float]] = []
        with span("embedding.documents", model=model_name, count=len(texts)):
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                resp = self._call_dashscope_embedding(dashscope, settings, batch, is_vl)

                if resp.status_code != HTTPStatus.OK:
                    raise RetrievalError(
                        f"DashScope embedding 错误 {resp.status_code}: "
                        f"{getattr(resp, 'message', resp)}"
                    )

                for vec in self._extract_dashscope_embeddings(resp, is_vl):
                    all_vectors.append(self._normalize(vec))

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    total_tokens = getattr(usage, "total_tokens", None)
                    if total_tokens is not None:
                        event(
                            "embedding.tokens",
                            model=settings.embedding_model_id,
                            total_tokens=total_tokens,
                        )

        return all_vectors

    def _embed_query_dashscope(
        self, text: str, settings, model_name: str
    ) -> list[float]:
        """DashScope 原生模式：单条查询 embedding。"""
        import dashscope
        from http import HTTPStatus

        is_vl = "vl" in settings.embedding_model_id.lower()
        with span("embedding.query", model=model_name):
            resp = self._call_dashscope_embedding(dashscope, settings, [text], is_vl)

            if resp.status_code != HTTPStatus.OK:
                raise RetrievalError(
                    f"DashScope embedding 错误 {resp.status_code}: "
                    f"{getattr(resp, 'message', resp)}"
                )

            embeddings = self._extract_dashscope_embeddings(resp, is_vl)
            return self._normalize(embeddings[0])

    def _call_dashscope_embedding(
        self, dashscope, settings, texts: list[str], is_vl: bool
    ):
        """构造 DashScope embedding 请求。

        - 视觉模型（模型名含 "vl"，如 qwen3-vl-embedding）走 MultiModalEmbedding，
          输入按 contents 格式传入文本；
        - 文本模型走 TextEmbedding，支持 dimension 参数。
        """
        if is_vl:
            return dashscope.MultiModalEmbedding.call(
                model=settings.embedding_model_id,
                input={"contents": [{"text": t} for t in texts]},
            )
        kwargs = {
            "model": settings.embedding_model_id,
            "input": texts,
        }
        if settings.embedding_dimensions:
            kwargs["dimension"] = settings.embedding_dimensions
        return dashscope.TextEmbedding.call(**kwargs)

    @staticmethod
    def _extract_dashscope_embeddings(resp, is_vl: bool) -> list[list[float]]:
        """从 DashScope 响应中按输入顺序提取向量。

        TextEmbedding 用 text_index，MultiModalEmbedding 用 index 对齐输入顺序。
        """
        embeddings = resp.output.get("embeddings", [])
        key = "index" if is_vl else "text_index"
        sorted_embeddings = sorted(embeddings, key=lambda d: d.get(key, 0))
        return [d["embedding"] for d in sorted_embeddings]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
