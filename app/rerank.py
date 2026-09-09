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

# 各 rerank 模型单次请求最大文档数（API 硬约束，超过返回 HTTP 400）
# 依据：https://help.aliyun.com/zh/model-studio/developer-reference/general-text-sorting-model
# - qwen3.7-text-rerank / qwen3-rerank：500 条
# - qwen3-vl-rerank：文本 100 条（图片 40 / 视频 4）
# - gte-rerank-v2：30000 条
_RERANK_MAX_DOCS = {
    "qwen3.7-text-rerank": 500,
    "qwen3-rerank": 500,
    "qwen3-vl-rerank": 100,
    "gte-rerank-v2": 30000,
}
# 未知模型默认取最严格的文本上限，避免 400
_DEFAULT_MAX_RERANK_DOCS = 100


# ── 法条角色分类（规则法，替代 LLM 判断）──
# 优先级：定义性 == 行为定义+量刑 > 实体性 > 程序性
# 角色判定基于 section_header 关键词，覆盖 statute/ 实际章节分布
# B2 方案：definition 与 behavior_penalty 同优先级，由 rerank_score 决定顺序
#   - 概念查询（如"什么是行政拘留"）→ 第十条 rerank 分数高，自然排前
#   - 行为查询（如"摩托车上高速被拘留"）→ 第二十六条等 rerank 分数高，自然排前
_ROLE_PRIORITY = {
    "definition": 0,
    "behavior_penalty": 0,
    "substantive": 1,
    "procedural": 2,
}

# 定义性章节关键词：规定法律概念、处罚种类、立法目的、适用范围等
# 注："处罚的种类和适用"命中治安管理处罚法第二章 16 条，其中第十条是真正
# 的处罚种类定义，其余 15 条（时效、年龄、减轻等）混入是已知副作用，靠
# rerank 模型语义匹配自然挤出，不进一步收紧关键词以避免漏召回第十条。
_DEFINITION_KEYWORDS = (
    "总则", "一般规定", "基本规定", "术语和定义", "处罚的种类和适用",
    "民事权利", "行政强制的种类和设定", "记分分值", "附则",
)

# 行为定义+量刑章节关键词：治安管理处罚法第三章各节
# 句式特征："有下列行为之一的，处...拘留/罚款"——具体行为+具体量刑
# 仅治安管理处罚法第三章命中（64 条），精确无歧义
# 覆盖：扰乱公共秩序、妨害公共安全、侵犯人身财产、妨害社会管理
_BEHAVIOR_PENALTY_KEYWORDS = (
    "行为和处罚",
)

# 查询概念词：用于配额保障逻辑，检查 top_n 中是否有正文含概念词的"种类定义"条款
# 与 retrieval.py 的 _DEFINITION_KEYWORDS 保持一致
# 当查询含这些词时，top_n 中应有正文同时含概念词和"种类"的 definition 条款（如第十条）
_CONCEPT_KEYWORDS = (
    "拘留", "处罚", "种类", "定义", "什么是", "概念",
    "罚款", "警告", "吊销", "暂扣", "驱逐",
)

# 程序性章节关键词：规定执行程序、救济途径、调查取证、复议诉讼等
_PROCEDURAL_KEYWORDS = (
    "程序", "执行", "复议", "诉讼", "管辖", "送达",
    "调查", "受案", "报案", "认定与复核", "检验", "鉴定",
    "强制执行", "简易程序", "期间计算", "备案审查",
    "损害赔偿调解", "执法监督", "处罚程序",
)


def _is_protected_role(role: str) -> bool:
    """是否享有三层保护（不受同 section 去重/不被跨法规替换/配额保底）。

    definition 与 behavior_penalty 同享保护：
    - definition：法律概念定义（如第十条"处罚种类"）
    - behavior_penalty：行为定义+量刑（如第二十六条"扰乱公共秩序...处拘留"）
    """
    return role in ("definition", "behavior_penalty")


def _classify_role(metadata: dict) -> str:
    """根据法条 metadata.section_header 判定结构角色。

    返回 "definition" / "behavior_penalty" / "substantive" / "procedural"：
    - behavior_penalty：行为定义+量刑（治安管理处罚法第三章"行为和处罚"各节）
    - definition：定义性（总则、处罚种类、术语定义等）
    - procedural：程序性（执行、救济、调查、复议诉讼等）
    - substantive：实体性（具体行为及法律后果，默认）

    判定优先级：behavior_penalty > definition > procedural > substantive
    治安管理处罚法第三章 section_header 为 "第三章 ...行为和处罚 / 第N节 XX的行为和处罚"，
    split("/") 后两段都含"行为和处罚"，先判定 behavior_penalty 即返回，不会落到 definition。

    合规：只判断文本结构角色，不判断法律正确性，不输出法律相关性判断。
    """
    section = (metadata or {}).get("section_header", "") or ""
    # 同时匹配章与节部分（"第七章 / 第一节 ..." 结构），任意一段命中即归类
    parts = [p.strip() for p in section.split("/")]
    for p in parts:
        if any(k in p for k in _BEHAVIOR_PENALTY_KEYWORDS):
            return "behavior_penalty"
    for p in parts:
        if any(k in p for k in _DEFINITION_KEYWORDS):
            return "definition"
    for p in parts:
        if any(k in p for k in _PROCEDURAL_KEYWORDS):
            return "procedural"
    return "substantive"


def apply_role_adjustment(contexts: list[dict]) -> list[dict]:
    """对 rerank 后的 contexts 做角色优先级稳定排序。

    - 优先级：定义性 == 行为定义+量刑 > 实体性 > 程序性
    - 稳定排序：同角色内保持 rerank_score 降序（不破坏 qwen3.7-text-rerank 结果）
    - 不删除任何法条，只调整排序
    - 空 contexts 直接返回，无副作用
    - B2 方案：definition 与 behavior_penalty 同优先级，由 rerank_score 决定顺序

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
        # 统一使用 OPENAI_API_KEY（阿里云百炼，同一 key 通用于 LLM/Embedding/Rerank）
        api_key = settings.openai_api_key
        if not api_key:
            raise RetrievalError("请在 .env 中配置 OPENAI_API_KEY")
        # 阿里云百炼 DashScope 原生 rerank 接口
        self._endpoint = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
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
        # 防御性截断：按当前 rerank 模型的单次请求文档上限截断
        # 上游 multi_query_search 已截断到 top_k*5，此处是兜底
        max_docs = _RERANK_MAX_DOCS.get(settings.rerank_model_id, _DEFAULT_MAX_RERANK_DOCS)
        if len(candidates) > max_docs:
            candidates = candidates[:max_docs]
        documents = [c["text"] for c in candidates]

        # 让 API 返回全部候选的排序结果，便于客户端做同 source 去重
        # （API 按 document 数计费，不按返回数；返回更多不增加费用）
        api_top_n = len(documents)
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

        # 错误响应：{"code": "...", "message": "..."} 或 {"error": {...}}
        if "code" in data and data["code"]:
            raise RetrievalError(
                f"重排序 API 返回错误：{data.get('message', data['code'])}"
            )
        if "error" in data:
            err = data["error"]
            raise RetrievalError(
                f"重排序 API 返回错误：{err.get('message', err) if isinstance(err, dict) else err}"
            )

        # 响应解析：DashScope {output: {results: [{index, relevance_score}]}}
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
            # 受保护条款放宽 min_score 阈值（×0.75），避免第十条"处罚种类包括拘留"
            # 被 min_score=0.4 过滤（rerank 给 0.3995），导致配额保障逻辑找不到它
            effective_min = min_score * 0.75 if _is_protected_role(
                _classify_role(candidates[idx].get("metadata", {}))
            ) else min_score
            if min_score is not None and score < effective_min:
                continue
            meta = candidates[idx].get("metadata", {})
            source = meta.get("source", "")
            section = meta.get("section_header", "")
            section_key = (source, section)
            role = _classify_role(meta)
            # 同 source + 同 section 已有更高分的条款，跳过
            # 例外：受保护条款（definition + behavior_penalty）不受同 section 去重限制
            # 同一"处罚的种类和适用"章下第十条（种类定义）与第十六条（适用规则）内容不同；
            # 治安管理处罚法第三章同节内多条行为+量刑条款内容也不同
            if not _is_protected_role(role) and section_key in seen_sections:
                continue
            # 同 source 已达上限
            # 例外：受保护条款（definition + behavior_penalty）不受同 source 限流限制
            # 原因：治安管理处罚法有 25 条 definition 条款（第一章 9 + 第二章 16），
            # MAX_PER_SOURCE=3 会只保留 3 条，第十条（处罚种类定义）被挤出 scored 列表，
            # 无法通过配额保障进入 top_n
            if source_counts.get(source, 0) >= MAX_PER_SOURCE and not _is_protected_role(role):
                continue
            # behavior_penalty 限流：top_n 中最多 2 条 behavior_penalty 条款
            # 原因：behavior_penalty 条款（第三章 64 条）正文含"拘留"字眼，rerank 给高分，
            # 会占满 top_n 前 3 位（如第三十六条危险物质、第五十九条损毁财物、第七十六条妨害社会管理），
            # 挤出道交法相关条款（道交法实施条例第八十三条、违法行为处理程序规定等）
            bp_count = sum(1 for s in scored if _classify_role(s.get("metadata", {})) == "behavior_penalty")
            if role == "behavior_penalty" and bp_count >= 2:
                continue
            seen_sections.add(section_key)
            source_counts[source] = source_counts.get(source, 0) + 1
            scored.append({**candidates[idx], "rerank_score": score})

        # 跨法规优先：top_n 截断时，若结果中某 source 占多条且 overflow 中有未覆盖 source，
        # 用未覆盖 source 的候选替换多余的同 source 候选，保证 top_n 内 source 多样性
        # 例外：受保护条款（definition + behavior_penalty）不被替换，避免高分的定义性/
        # 行为+量刑条款（如治安管理处罚法第十条/第二十六条）被低分的其他法条挤掉
        # 从后往前找替换目标：保留同 source 中 rerank 分数更高的条款
        # 原因：scored 按 rerank_score 降序排列，同 source 的多条中排在前面的分数更高，
        # 从后往前替换可以保留高分条款（如道交法第九十条 0.6181 不被交强险第一条
        # 0.4581 替换，而是替换同 source 排在后面的第六十二条 0.5112）
        if len(scored) > top_n:
            kept = list(scored[:top_n])
            overflow = list(scored[top_n:])
            src_counts = Counter(c.get("metadata", {}).get("source", "") for c in kept)
            # 用 overflow 中未覆盖 source 的候选替换 kept 中多占 source 的候选
            replaced = True
            while replaced and overflow:
                replaced = False
                # 从后往前找 kept 中出现 >1 次且非受保护的候选（保留高分条款）
                for i in range(len(kept) - 1, -1, -1):
                    c = kept[i]
                    c_src = c.get("metadata", {}).get("source", "")
                    if src_counts.get(c_src, 0) > 1:
                        if _is_protected_role(_classify_role(c.get("metadata", {}))):
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
            # 保留 overflow 供配额保障逻辑使用，不丢弃
            # 配额保障逻辑检查 len(scored) > top_n 来决定是否触发
            scored = kept + overflow

        # 受保护条款配额保障：top_n 截断后若无 definition 或 behavior_penalty 条款，
        # 从 scored 中取 rerank_score 最高的受保护条款插入 top_n 末尾（替换最后一条）。
        # 场景：qwen3-rerank 对"处罚种类"等定义性条款、对"行为+量刑"条款打分偏低
        # （与具体案件语义相关性弱），但法律推理中这两类条款是论证核心：
        #   - 第十条支撑"拘留需有明确法律授权"
        #   - 第二十六条等支撑"具体行为+具体量刑标准"
        # 配额只保底1条，不破坏 rerank 结果主体顺序。
        if len(scored) > top_n:
            top = list(scored[:top_n])
            has_protected = any(
                _is_protected_role(_classify_role(c.get("metadata", {}))) for c in top
            )
            if not has_protected:
                # 从 overflow 中找分数最高的受保护条款
                overflow = scored[top_n:]
                protected_candidates = [
                    c for c in overflow
                    if _is_protected_role(_classify_role(c.get("metadata", {})))
                ]
                if protected_candidates:
                    best_protected = max(
                        protected_candidates, key=lambda c: c.get("rerank_score", 0.0)
                    )
                    top[-1] = best_protected  # 替换 top_n 最后一条
                    event(
                        "rerank.definition_quota",
                        article=best_protected.get("metadata", {}).get("article_no", ""),
                        source=best_protected.get("metadata", {}).get("source", ""),
                        rerank_score=best_protected.get("rerank_score", 0.0),
                    )
                    scored = top

        # 处罚种类定义配额保障：如果查询含概念词（如"拘留"），top_n 中必须有正文同时
        # 含概念词和"种类"的 definition 条款（即处罚种类定义条款，如第十条）。
        # 行为+量刑配额保障：top_n 中必须有正文含概念词和"扰乱"或"秩序"的
        # behavior_penalty 条款（即扰乱公共秩序条款，如第二十六条）。
        # 两个配额保障分别替换 top_n 的倒数第 2 和倒数第 1 条，避免互相冲突。
        # 被替换的条款放回 overflow，供后续配额保障使用。
        if len(scored) > top_n:
            top = list(scored[:top_n])
            concept_keywords = [kw for kw in _CONCEPT_KEYWORDS if kw in query]
            overflow = list(scored[top_n:])
            if concept_keywords:
                # 检查是否有处罚种类定义条款
                has_kind_def = any(
                    _is_protected_role(_classify_role(c.get("metadata", {})))
                    and "种类" in (c.get("text", "") or "")
                    and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                    for c in top
                )
                if not has_kind_def:
                    kind_def_candidates = [
                        c for c in overflow
                        if _is_protected_role(_classify_role(c.get("metadata", {})))
                        and "种类" in (c.get("text", "") or "")
                        and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                    ]
                    if kind_def_candidates:
                        best = max(
                            kind_def_candidates,
                            key=lambda c: c.get("rerank_score", 0.0)
                        )
                        overflow.append(top[-1])  # 被替换的条款放回 overflow
                        top[-1] = best  # 替换倒数第 1 条
                        event(
                            "rerank.kind_definition_quota",
                            article=best.get("metadata", {}).get("article_no", ""),
                            source=best.get("metadata", {}).get("source", ""),
                            rerank_score=best.get("rerank_score", 0.0),
                        )

                # 检查是否有扰乱公共秩序 behavior_penalty 条款
                has_disturb_bp = any(
                    _classify_role(c.get("metadata", {})) == "behavior_penalty"
                    and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                    and ("扰乱" in (c.get("text", "") or "") or "秩序" in (c.get("text", "") or ""))
                    for c in top
                )
                if not has_disturb_bp:
                    bp_candidates = [
                        c for c in overflow
                        if _classify_role(c.get("metadata", {})) == "behavior_penalty"
                        and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                        and ("扰乱" in (c.get("text", "") or "") or "秩序" in (c.get("text", "") or ""))
                    ]
                    if bp_candidates:
                        best = max(
                            bp_candidates,
                            key=lambda c: c.get("rerank_score", 0.0)
                        )
                        top[-2] = best  # 替换倒数第 2 条
                        event(
                            "rerank.behavior_penalty_quota",
                            article=best.get("metadata", {}).get("article_no", ""),
                            source=best.get("metadata", {}).get("source", ""),
                            rerank_score=best.get("rerank_score", 0.0),
                        )
                scored = top

        # 引用追踪：由 retrieval.py 的 REFERENCE_QUOTA 在候选池阶段统一处理，
        # rerank 阶段不再重复追踪，避免同一条被引用条款被重复加入导致 token 浪费。

        return scored[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
