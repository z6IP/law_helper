"""CrossEncoder 重排序（bge-reranker-v2-m3，CPU 友好）。

CPU 推理优化：torch.inference_mode() + model.eval() + batch 预测。
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
        # local_files_only=True: 直接从本地加载，跳过网络检查
        self._model = CrossEncoder(model_dir, local_files_only=True)
        # 确保 eval 模式
        if hasattr(self._model, "eval"):
            self._model.eval()

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
        # 使用 torch.inference_mode() 加速推理（比 no_grad() 更快）
        import torch

        with torch.inference_mode():
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