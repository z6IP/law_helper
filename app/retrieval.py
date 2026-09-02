"""混合检索：BM25 + 向量语义 + RRF 融合。

工程约束：
- 使用 RRF 融合，BM25_WEIGHT=0.5、RRF_LAMBDA=60；
- 向量检索与 BM25 关键词召回双路，结果用于后续重排序。
- 动态 embedding 维度检测：如果 ChromaDB 中存储的向量维度
  与当前 embedding 模型输出维度不匹配，会在加载时自动检测并标记，
  由上层（main.py startup）负责重建索引。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.errors import RetrievalError
from app.ingestion import COLLECTION_NAME
from app.tracing import event


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut(text) if t.strip()]


def _rrf(rank: int) -> float:
    settings = get_settings()
    return 1.0 / (settings.rrf_lambda + rank)


def _get_embedding_dim() -> int:
    """探测当前 embedding 模型的输出维度。"""
    test_vec = get_embedding_model().embed_query("维度检测")
    return len(test_vec)


class RetrievalEngine:
    """BM25 + 向量双路召回（RRF 融合）。"""

    def __init__(self) -> None:
        self._loaded = False
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._dim_mismatch: bool = False  # 维度不匹配标记

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_full_dir))
        # HNSW 索引配置：ef_construction 影响构建速度，ef_search 影响查询速度
        # 小数据集（72 条法条）用较小参数即可，加速构建和查询
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 16,
                "hnsw:M": 16,
            },
        )
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        self._ids = result["ids"]
        self._documents = result["documents"] or []
        self._metadatas = result["metadatas"] or []
        emb = result.get("embeddings")
        self._embeddings = np.asarray(emb) if emb is not None and len(emb) > 0 else None

        # 维度检测：如果已有向量维度与当前模型不匹配，标记
        if self._embeddings is not None and len(self._embeddings) > 0:
            expected_dim = _get_embedding_dim()
            actual_dim = self._embeddings.shape[1]
            if actual_dim != expected_dim:
                event(
                    "retrieval.dim_mismatch",
                    stored_dim=actual_dim,
                    expected_dim=expected_dim,
                )
                self._dim_mismatch = True

        if self._documents and not self._dim_mismatch:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._documents])
        self._loaded = True

    @property
    def needs_rebuild(self) -> bool:
        """索引是否需要重建（维度不匹配）。"""
        return self._dim_mismatch

    def search(self, query: str, top_k: int) -> list[dict]:
        """返回融合排序后的 top_k 结果。

        每项结构：{"id", "text", "metadata", "score"}
        """
        self._ensure_loaded()
        if not self._documents or self._dim_mismatch:
            # 维度不匹配时回退到仅 BM25 搜索
            if not self._documents or self._bm25 is None:
                return []
            n = len(self._documents)
            bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
            order = np.argsort(-bm25_scores)[:top_k]
            return [
                {
                    "id": self._ids[i],
                    "text": self._documents[i],
                    "metadata": self._metadatas[i],
                    "score": float(bm25_scores[i]),
                }
                for i in order
            ]

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