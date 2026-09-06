"""qwen3.7-text-rerank 重排序（阿里云百炼 DashScope API）。

通过 DashScope 专用接口调用，对候选文档按相关性打分排序。
"""
from __future__ import annotations

from functools import lru_cache

import requests

from app.config import get_settings
from app.errors import RetrievalError
from app.tracing import event, span


# 默认排序任务指令（问答检索场景）
_DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."


class Reranker:
    """qwen3.7-text-rerank 重排序封装（懒加载单例，API 调用）。"""

    def __init__(self) -> None:
        self._endpoint: str | None = None
        self._api_key: str | None = None

    def _ensure_loaded(self) -> None:
        if self._endpoint is not None:
            return
        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
            raise RetrievalError("请在 .env 中配置有效的 OPENAI_API_KEY")
        # DashScope rerank 专用接口
        self._endpoint = (
            settings.dashscope_api_base.rstrip("/")
            + "/services/rerank/text-rerank/text-rerank"
        )
        self._api_key = settings.openai_api_key

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

        settings = get_settings()
        documents = [c["text"] for c in candidates]

        payload = {
            "model": settings.rerank_model_id,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "instruct": _DEFAULT_INSTRUCT,
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with span(
            "rerank.predict",
            model=settings.rerank_model_id,
            candidates=len(candidates),
        ):
            try:
                resp = requests.post(
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise RetrievalError(f"重排序 API 调用失败：{exc}") from exc

            data = resp.json()

        # 错误响应：{"code": "...", "message": "..."}
        if "code" in data and data["code"]:
            raise RetrievalError(
                f"重排序 API 返回错误：{data.get('message', data['code'])}"
            )

        results = data.get("output", {}).get("results", [])
        usage = data.get("usage")
        if usage:
            event(
                "rerank.tokens",
                model=settings.rerank_model_id,
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        # results 已按 relevance_score 降序排列
        scored: list[dict] = []
        for r in results:
            idx = r["index"]
            score = float(r["relevance_score"])
            if min_score is None or score >= min_score:
                scored.append({**candidates[idx], "rerank_score": score})

        return scored[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
