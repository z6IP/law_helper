"""CrossEncoder 重排序。

工程约束：重排序结果取 top 文档（RERANK_TOP_N，默认 5），使用 bge-reranker-v2-m3。
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.embeddings import _download_model
from app.errors import RetrievalError


class Reranker:
    """bge-reranker-v2-m3 CrossEncoder 封装（懒加载单例）。"""

    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        model_dir = _download_model(settings.rerank_model_id)
        self._model = CrossEncoder(model_dir)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int,
        min_score: float | None = None,
    ) -> list[dict]:
        """对候选结果打分并排序，返回 top_n。

        若指定 min_score，则丢弃 rerank 得分低于该阈值的候选；
        当所有候选均不达标时返回空列表，由调用方决定如何回应。
        """
        if not candidates:
            return []
        self._ensure_loaded()

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [scores]

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        result = [
            {**c, "rerank_score": float(s)}
            for c, s in ranked
            if min_score is None or float(s) >= min_score
        ]
        return result[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()