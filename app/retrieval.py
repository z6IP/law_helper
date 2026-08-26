"""混合检索：BM25 + 向量语义 + RRF 融合。

工程约束：
- 使用 RRF 融合，BM25_WEIGHT=0.5、RRF_LAMBDA=60；
- 向量检索与 BM25 关键词召回双路，结果用于后续重排序。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.errors import RetrievalError
from app.ingestion import COLLECTION_NAME


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut(text) if t.strip()]


def _rrf(rank: int) -> float:
    settings = get_settings()
    return 1.0 / (settings.rrf_lambda + rank)


class RetrievalEngine:
    """BM25 + 向量双路召回（RRF 融合）。"""

    def __init__(self) -> None:
        self._loaded = False
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_full_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        self._ids = result["ids"]
        self._documents = result["documents"] or []
        self._metadatas = result["metadatas"] or []
        emb = result.get("embeddings")
        self._embeddings = np.asarray(emb) if emb is not None and len(emb) > 0 else None

        if self._documents:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._documents])
        self._loaded = True

    def search(self, query: str, top_k: int) -> list[dict]:
        """返回融合排序后的 top_k 结果。

        每项结构：{"id", "text", "metadata", "score"}
        """
        self._ensure_loaded()
        if not self._documents:
            return []

        n = len(self._documents)

        # 1) BM25 得分
        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))

        # 2) 向量余弦相似度
        q_vec = np.asarray(get_embedding_model().embed_query(query))
        if self._embeddings is None:
            vec_scores = np.zeros(n)
        else:
            vec_scores = self._embeddings @ q_vec  # 已归一化，点积即余弦

        # 3) 各自排序（rank 从 0 开始 -> RRF 用 +1）
        bm25_rank = _rank_from_scores(-bm25_scores)
        vec_rank = _rank_from_scores(-vec_scores)

        settings = get_settings()
        w_bm25 = settings.bm25_weight
        w_vec = 1.0 - settings.bm25_weight

        scores = np.asarray(
            [w_bm25 * _rrf(bm25_rank[i]) + w_vec * _rrf(vec_rank[i]) for i in range(n)]
        )
        order = np.argsort(-scores)[:top_k]

        return [
            {
                "id": self._ids[i],
                "text": self._documents[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in order
        ]


def _rank_from_scores(desc_scores: np.ndarray) -> np.ndarray:
    """将（降序优先的）得分数组转换为每个元素的 rank（0 起）。"""
    order = np.argsort(desc_scores)
    rank = np.empty_like(order, dtype=int)
    rank[order] = np.arange(len(order))
    return rank


@lru_cache
def get_retrieval_engine() -> RetrievalEngine:
    engine = RetrievalEngine()
    engine._ensure_loaded()
    return engine