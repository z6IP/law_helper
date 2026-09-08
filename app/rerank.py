"""qwen3.7-text-rerank 重排序（阿里云百炼 DashScope API）。

通过 DashScope 专用接口调用，对候选文档按相关性打分排序。

新增 apply_role_adjustment：基于法条 section_header 的规则化角色排序调整。
研究报告 Phase 3 提供两种方案（LLM 判断 / 规则替代），此处采用规则方案：
- 不调 LLM，零成本、零偏见风险（研究报告 5.1.3 / 5.2.7）
- 只调整排序，不删除任何法条（研究报告 Phase 3 验证标准）
- 稳定排序：同角色内保持 rerank_score 降序
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import requests

from app.config import get_settings
from app.errors import RetrievalError
from app.tracing import event, span


# 默认排序任务指令（问答检索场景）
_DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."

# qwen3.7-text-rerank 单次请求最大文档数（API 硬约束）
_MAX_RERANK_DOCS = 500


# ── 法条角色分类（规则法，替代 LLM 判断）──
# 优先级：定义性 > 实体性 > 程序性
# 角色判定基于 section_header 关键词，覆盖 statute/ 实际章节分布
_ROLE_PRIORITY = {"definition": 0, "substantive": 1, "procedural": 2}

# 定义性章节关键词：规定法律概念、处罚种类、立法目的、适用范围等
_DEFINITION_KEYWORDS = (
    "总则", "一般规定", "基本规定", "术语和定义", "处罚的种类和适用",
    "民事权利", "行政强制的种类和设定", "记分分值", "附则",
)

# 程序性章节关键词：规定执行程序、救济途径、调查取证、复议诉讼等
_PROCEDURAL_KEYWORDS = (
    "程序", "执行", "复议", "诉讼", "管辖", "送达",
    "调查", "受案", "报案", "认定与复核", "检验", "鉴定",
    "强制执行", "简易程序", "期间计算", "备案审查",
    "损害赔偿调解", "执法监督", "处罚程序",
)


def _classify_role(metadata: dict) -> str:
    """根据法条 metadata.section_header 判定结构角色。

    返回 "definition" / "substantive" / "procedural"：
    - definition：定义性（总则、处罚种类、术语定义等）
    - procedural：程序性（执行、救济、调查、复议诉讼等）
    - substantive：实体性（具体行为及法律后果，默认）

    合规：只判断文本结构角色，不判断法律正确性，不输出法律相关性判断。
    """
    section = (metadata or {}).get("section_header", "") or ""
    # 同时匹配章与节部分（"第七章 / 第一节 ..." 结构），任意一段命中即归类
    parts = [p.strip() for p in section.split("/")]
    for p in parts:
        if any(k in p for k in _DEFINITION_KEYWORDS):
            return "definition"
    for p in parts:
        if any(k in p for k in _PROCEDURAL_KEYWORDS):
            return "procedural"
    return "substantive"


def apply_role_adjustment(contexts: list[dict]) -> list[dict]:
    """对 rerank 后的 contexts 做角色优先级稳定排序。

    - 优先级：定义性 > 实体性 > 程序性
    - 稳定排序：同角色内保持 rerank_score 降序（不破坏 qwen3.7-text-rerank 结果）
    - 不删除任何法条，只调整排序
    - 空 contexts 直接返回，无副作用

    合规：不替换 qwen3.7-text-rerank 在线模型的结果，只在其后追加角色排序。
    """
    if not contexts:
        return contexts
    sorted_ctx = sorted(
        contexts,
        key=lambda c: (
            _ROLE_PRIORITY[_classify_role(c.get("metadata", {}))],
            -c.get("rerank_score", 0.0),
        ),
    )
    # 仅当排序实际改变了顺序时记录事件，避免同角色内微调也触发
    if len(sorted_ctx) > 1 and sorted_ctx != contexts:
        roles = [_classify_role(c.get("metadata", {})) for c in sorted_ctx]
        event("role_adjustment.applied", roles=roles)
    return sorted_ctx


class Reranker:
    """qwen3.7-text-rerank 重排序封装（懒加载单例，API 调用）。"""

    def __init__(self) -> None:
        self._endpoint: str | None = None
        self._api_key: str | None = None

    def _ensure_loaded(self) -> None:
        if self._endpoint is not None:
            return
        settings = get_settings()
        # RERANK_API_KEY 留空时回退到 OPENAI_API_KEY（阿里云同一 key）
        api_key = settings.rerank_api_key or settings.openai_api_key
        if not api_key:
            raise RetrievalError("请在 .env 中配置 RERANK_API_KEY 或 OPENAI_API_KEY")
        # Cloudflare 格式：URL 含模型名 /ai/run/@cf/...
        if settings.rerank_payload_format == "cloudflare":
            self._endpoint = (
                settings.rerank_api_base.rstrip("/")
                + "/ai/run/"
                + settings.rerank_model_id
            )
        else:
            # 其他格式：rerank_api_base + rerank_endpoint_path
            self._endpoint = (
                settings.rerank_api_base.rstrip("/")
                + settings.rerank_endpoint_path
            )
        self._api_key = api_key

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

        同 source 限流 + 同 section 去重：让 API 返回全部排序结果，客户端做两层过滤：
        1. 同 source + 同 section_header 只保留 top1（避免 4.1 与表1 内容重复）；
        2. 同 source 最多保留 2 条（避免单一法规挤占 top_n 位置），
        同时保留立法法第五章等多条款场景的覆盖能力。
        """
        if not candidates:
            return []
        self._ensure_loaded()

        settings = get_settings()
        # 防御性截断：qwen3.7-text-rerank 单次请求最大文档数 500
        # 上游 multi_query_search 已截断到 top_k*5，此处是兜底
        if len(candidates) > _MAX_RERANK_DOCS:
            candidates = candidates[:_MAX_RERANK_DOCS]
        documents = [c["text"] for c in candidates]

        # 让 API 返回全部候选的排序结果，便于客户端做同 source 去重
        # （API 按 document 数计费，不按返回数；返回更多不增加费用）
        api_top_n = len(documents)
        # 根据 rerank_payload_format 选择请求结构
        fmt = settings.rerank_payload_format
        if fmt == "cloudflare":
            # Cloudflare 格式：contexts 数组（每项为 {text: ...}）
            payload = {
                "query": query,
                "contexts": [{"text": d} for d in documents],
                "top_k": api_top_n,
            }
        elif fmt == "dashscope":
            # DashScope 嵌套格式：{model, input:{query, documents}, parameters:{...}}
            payload = {
                "model": settings.rerank_model_id,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": api_top_n,
                    "instruct": _DEFAULT_INSTRUCT,
                },
            }
        else:
            # OpenAI 兼容扁平格式：{model, query, documents, top_n}
            payload = {
                "model": settings.rerank_model_id,
                "query": query,
                "documents": documents,
                "top_n": api_top_n,
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

        # 错误响应：{"code": "...", "message": "..."} 或 {"error": {...}} 或 Cloudflare {"success": false, "errors": [...]}
        if "code" in data and data["code"]:
            raise RetrievalError(
                f"重排序 API 返回错误：{data.get('message', data['code'])}"
            )
        if "error" in data:
            err = data["error"]
            raise RetrievalError(
                f"重排序 API 返回错误：{err.get('message', err) if isinstance(err, dict) else err}"
            )
        # Cloudflare 错误格式：{"success": false, "errors": ["..."]}
        if data.get("success") is False:
            errors = data.get("errors", [])
            raise RetrievalError(
                f"重排序 API 返回错误：{errors if errors else '未知错误'}"
            )

        # 响应解析：兼容多种格式
        # Cloudflare: {result: {response: [{id, score}]}}
        # DashScope: {output: {results: [{index, relevance_score}]}}
        # OpenAI 兼容: {results: [{index, relevance_score}]} 或 {data: [...]}
        if fmt == "cloudflare":
            results = (data.get("result") or {}).get("response", [])
            # Cloudflare 用 id 而非 index，统一映射为 index
            for r in results:
                if "index" not in r and "id" in r:
                    r["index"] = r["id"]
        else:
            results = (
                data.get("output", {}).get("results")
                or data.get("results")
                or data.get("data", [])
            )
        usage = data.get("usage")
        if usage:
            event(
                "rerank.tokens",
                model=settings.rerank_model_id,
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        # results 已按 relevance_score 降序排列
        # 同 source 限流 + 同 section 去重：
        # - 同 source + 同 section_header 只保留 top1（避免 4.1 与表1 内容重复）
        # - 同 source 最多保留 3 条（多路检索后候选已均衡，放宽限流保留更多信息）
        MAX_PER_SOURCE = 3
        source_counts: dict[str, int] = {}
        seen_sections: set[tuple[str, str]] = set()
        scored: list[dict] = []
        for r in results:
            idx = r["index"]
            # 兼容两种字段名：DashScope 用 relevance_score，OpenAI 兼容可能用 score
            score = float(r.get("relevance_score", r.get("score", 0.0)))
            if min_score is not None and score < min_score:
                continue
            meta = candidates[idx].get("metadata", {})
            source = meta.get("source", "")
            section = meta.get("section_header", "")
            section_key = (source, section)
            role = _classify_role(meta)
            # 同 source + 同 section 已有更高分的条款，跳过
            # 例外：定义性条款（role=definition）不受同 section 去重限制
            # 同一"处罚的种类和适用"章下第十条（种类定义）与第十六条（适用规则）内容不同
            if role != "definition" and section_key in seen_sections:
                continue
            # 同 source 已达上限
            if source_counts.get(source, 0) >= MAX_PER_SOURCE:
                continue
            seen_sections.add(section_key)
            source_counts[source] = source_counts.get(source, 0) + 1
            scored.append({**candidates[idx], "rerank_score": score})

        # 跨法规优先：top_n 截断时，若结果中某 source 占多条且 overflow 中有未覆盖 source，
        # 用未覆盖 source 的候选替换多余的同 source 候选，保证 top_n 内 source 多样性
        # 例外：定义性条款（role=definition）不被替换，避免高分的定义性条款
        # （如治安管理处罚法第十条）被低分的其他法条挤掉
        if len(scored) > top_n:
            kept = list(scored[:top_n])
            overflow = list(scored[top_n:])
            src_counts = Counter(c.get("metadata", {}).get("source", "") for c in kept)
            # 用 overflow 中未覆盖 source 的候选替换 kept 中多占 source 的候选
            replaced = True
            while replaced and overflow:
                replaced = False
                # 找 kept 中出现 >1 次且非定义性的候选
                for i, c in enumerate(kept):
                    c_src = c.get("metadata", {}).get("source", "")
                    if src_counts.get(c_src, 0) > 1:
                        if _classify_role(c.get("metadata", {})) == "definition":
                            continue
                        # 找 overflow 中未覆盖 source 的候选
                        for j, ov in enumerate(overflow):
                            ov_src = ov.get("metadata", {}).get("source", "")
                            if src_counts.get(ov_src, 0) == 0:
                                kept[i] = ov
                                overflow[j] = c  # 被替换出的条目放回 overflow
                                src_counts = Counter(c.get("metadata", {}).get("source", "") for c in kept)
                                replaced = True
                                break
                        if replaced:
                            break
            scored = kept

        # 定义性条款配额保障：top_n 截断后若无定义性条款，
        # 从 scored 中取 rerank_score 最高的定义性条款插入 top_n 末尾（替换最后一条）。
        # 场景：qwen3-rerank 对"处罚种类"等定义性条款打分偏低（与具体案件语义相关性弱），
        # 但法律推理中定义性条款是论证核心（如第十条支撑"拘留需有明确法律授权"）。
        # 配额只保底1条，不破坏 rerank 结果主体顺序。
        if len(scored) > top_n:
            top = list(scored[:top_n])
            has_def = any(
                _classify_role(c.get("metadata", {})) == "definition" for c in top
            )
            if not has_def:
                # 从 overflow 中找分数最高的定义性条款
                overflow = scored[top_n:]
                def_candidates = [
                    c for c in overflow
                    if _classify_role(c.get("metadata", {})) == "definition"
                ]
                if def_candidates:
                    best_def = max(
                        def_candidates, key=lambda c: c.get("rerank_score", 0.0)
                    )
                    top[-1] = best_def  # 替换 top_n 最后一条
                    event(
                        "rerank.definition_quota",
                        article=best_def.get("metadata", {}).get("article_no", ""),
                        source=best_def.get("metadata", {}).get("source", ""),
                        rerank_score=best_def.get("rerank_score", 0.0),
                    )
                    scored = top

        return scored[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
