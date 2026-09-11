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
from app.policy import get_policy
from app.tracing import event, span


# 默认排序任务指令（法律检索场景）：强调同时覆盖行为规定与处罚依据，
# 引导 rerank 模型不要只按表面语义相关性排序，保留定义/量刑等间接相关条款。
_DEFAULT_INSTRUCT = (
    "给定一个法律咨询问题，请检索与之相关的法律法规条文。"
    "需要同时覆盖行为规定条款、处罚种类定义条款和处罚依据条款，"
    "即使某些条款与问题的表面语义相关性较低，也请保留。"
)

# 各 rerank 模型单次请求最大文档数（API 硬约束，超过返回 HTTP 400）
# 依据：https://help.aliyun.com/zh/model-studio/developer-reference/general-text-sorting-model
# - qwen3.7-text-rerank / qwen3-rerank：500 条
# - qwen3-vl-rerank：文本 100 条（图片 40 / 视频 4）
# - gte-rerank-v2：30000 条
_RERANK_MAX_DOCS = {
    "qwen3.7-text-rerank": 500,
    "qwen3-rerank": 500,
    "qwen3-vl-rerank": 100,
    "gte-rerank-v2": 500,
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
_DEFINITION_KEYWORDS = tuple(get_policy()["rerank"]["definition_keywords"])

# 行为定义+量刑章节关键词：治安管理处罚法第三章各节
# 句式特征："有下列行为之一的，处...拘留/罚款"——具体行为+具体量刑
# 仅治安管理处罚法第三章命中（64 条），精确无歧义
# 覆盖：扰乱公共秩序、妨害公共安全、侵犯人身财产、妨害社会管理
_BEHAVIOR_PENALTY_KEYWORDS = tuple(get_policy()["rerank"]["behavior_penalty_keywords"])

# 查询概念词：用于配额保障逻辑，检查 top_n 中是否有正文含概念词的"种类定义"条款
# 与 retrieval.py 的 _DEFINITION_KEYWORDS 保持一致
# 当查询含这些词时，top_n 中应有正文同时含概念词和"种类"的 definition 条款（如第十条）
_CONCEPT_KEYWORDS = tuple(get_policy()["rerank"]["concept_keywords"])

# 程序性章节关键词：规定执行程序、救济途径、调查取证、复议诉讼等
_PROCEDURAL_KEYWORDS = tuple(get_policy()["rerank"]["procedural_keywords"])


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


def apply_role_adjustment(
    contexts: list[dict], intents: list[str] | None = None
) -> list[dict]:
    """对 rerank 后的 contexts 做角色优先级稳定排序。

        - 只有查询明确命中对应意图时，才提升定义/行为/程序角色；
            未提供意图时保持 rerank 原始相关性顺序。
    - 稳定排序：同角色内保持 rerank_score 降序（不破坏 qwen3.7-text-rerank 结果）
    - 不删除任何法条，只调整排序
    - 空 contexts 直接返回，无副作用
    - B2 方案：definition 与 behavior_penalty 同优先级，由 rerank_score 决定顺序

    合规：不替换 qwen3.7-text-rerank 在线模型的结果，只在其后追加角色排序。
    """
    if not contexts or not intents:
        return contexts
    role_priority = dict(_ROLE_PRIORITY)
    if "definition" in intents:
        role_priority["definition"] = -1
    if "penalty" in intents:
        role_priority["behavior_penalty"] = -1
    if "procedure" in intents:
        role_priority["procedural"] = -1
    sorted_ctx = sorted(
        contexts,
        key=lambda c: (
            role_priority[_classify_role(c.get("metadata", {}))],
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
                "instruct": get_policy()["rerank"].get("instruct") or _DEFAULT_INSTRUCT,
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
        rerank_policy = get_policy()["rerank"]
        MAX_PER_SOURCE = rerank_policy["max_per_source"]

        # 全量分数查找表：遍历 API 返回的全部 results，为每个候选构建
        # id → {candidate + rerank_score} 映射。该表不受 min_score / 同 source 限流 /
        # behavior_penalty 限流等过滤影响，作为配额保障的终极回退数据源。
        _all_scored: dict[str, dict] = {}
        for r in results:
            idx = r["index"]
            score = float(r.get("relevance_score", r.get("score", 0.0)))
            cid = str(candidates[idx].get("id") or idx)
            _all_scored[cid] = {**candidates[idx], "rerank_score": score}

        source_counts: dict[str, int] = {}
        seen_sections: set[tuple[str, str]] = set()
        scored: list[dict] = []
        for r in results:
            idx = r["index"]
            # 兼容两种字段名：DashScope 用 relevance_score，OpenAI 兼容可能用 score
            score = float(r.get("relevance_score", r.get("score", 0.0)))
            meta = candidates[idx].get("metadata", {})
            role = _classify_role(meta)
            # 受保护条款（definition + behavior_penalty）完全豁免 min_score：
            # 这类条款与具体行为场景语义距离远，rerank 模型常给低分（如第十条"处罚种类"），
            # 但它们是法律推理的核心论证依据，不能因低分被过滤。不再使用 0.75 倍放宽。
            if not _is_protected_role(role) and min_score is not None and score < min_score:
                continue
            source = meta.get("source", "")
            section = meta.get("section_header", "")
            # 无 metadata 的候选（如上传材料切块）不参与 source/section 去重与限流：
            # 否则所有候选的 (source, section) 均为 ("", "")，第一条之后全部被误删
            has_meta = bool(source or section)
            section_key = (source, section)
            # 同 source + 同 section 已有更高分的条款，跳过
            # 例外：受保护条款（definition + behavior_penalty）不受同 section 去重限制
            # 同一"处罚的种类和适用"章下第十条（种类定义）与第十六条（适用规则）内容不同；
            # 治安管理处罚法第三章同节内多条行为+量刑条款内容也不同
            if has_meta and not _is_protected_role(role) and section_key in seen_sections:
                continue
            # 同 source 已达上限
            # 例外：受保护条款（definition + behavior_penalty）不受同 source 限流限制
            # 原因：治安管理处罚法有 25 条 definition 条款（第一章 9 + 第二章 16），
            # MAX_PER_SOURCE=3 会只保留 3 条，第十条（处罚种类定义）被挤出 scored 列表，
            # 无法通过配额保障进入 top_n
            if has_meta and source_counts.get(source, 0) >= MAX_PER_SOURCE and not _is_protected_role(role):
                continue
            # behavior_penalty 限流：top_n 中最多 2 条 behavior_penalty 条款
            # 原因：behavior_penalty 条款（第三章 64 条）正文含"拘留"字眼，rerank 给高分，
            # 会占满 top_n 前 3 位（如第三十六条危险物质、第五十九条损毁财物、第七十六条妨害社会管理），
            # 挤出道交法相关条款（道交法实施条例第八十三条、违法行为处理程序规定等）
            bp_count = sum(1 for s in scored if _classify_role(s.get("metadata", {})) == "behavior_penalty")
            if role == "behavior_penalty" and bp_count >= rerank_policy["behavior_penalty_max"]:
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

        # ── 配额保障（无条件执行，不再依赖 len(scored) > top_n）──
        # 先取 top/overflow，维护 top 的 id 集合；各配额在 overflow 中找不到所需条款时，
        # 回退到 _all_scored 全量查找表（该表包含被 min_score / 限流过滤掉的所有候选）。
        top = list(scored[:top_n])
        overflow = list(scored[top_n:])
        top_ids = {str(c.get("id")) for c in top}

        def _find_best(predicate) -> dict | None:
            """先在 overflow 中找最高分匹配项；找不到时回退到 _all_scored 全量查找。"""
            pool = [c for c in overflow if predicate(c)]
            if not pool:
                pool = [
                    c for cid, c in _all_scored.items()
                    if cid not in top_ids and predicate(c)
                ]
            if not pool:
                return None
            return max(pool, key=lambda c: c.get("rerank_score", 0.0))

        def _insert_quota(best: dict) -> None:
            """把配额条款插入 top：优先替换最后一条非受保护条款，保持 top_n 长度；
            若 top 中无非受保护条款且未满 top_n，则追加。"""
            top_ids.add(str(best.get("id")))
            for i in range(len(top) - 1, -1, -1):
                if not _is_protected_role(_classify_role(top[i].get("metadata", {}))):
                    top[i] = best
                    return
            if len(top) < top_n:
                top.append(best)
            else:
                top[-1] = best

        # 1. 受保护条款配额：top_n 中至少要有 1 条 definition 或 behavior_penalty 条款。
        #    场景：rerank 对"处罚种类"等定义性条款、对"行为+量刑"条款打分偏低，但这些
        #    条款是法律推理核心（第十条支撑"拘留需有明确法律授权"，第二十六条支撑具体量刑）。
        has_protected = any(
            _is_protected_role(_classify_role(c.get("metadata", {}))) for c in top
        )
        if not has_protected:
            best = _find_best(
                lambda c: _is_protected_role(_classify_role(c.get("metadata", {})))
            )
            if best is not None:
                _insert_quota(best)
                event(
                    "rerank.protected_quota",
                    article=best.get("metadata", {}).get("article_no", ""),
                    source=best.get("metadata", {}).get("source", ""),
                    rerank_score=best.get("rerank_score", 0.0),
                )

        # 2. 概念词触发的配额保障：处罚种类定义条款 + 行为+量刑条款。
        #    场景：查询含"拘留"等概念词时，top_n 中必须同时具备"处罚种类定义"（如第十条）
        #    和"行为+量刑"（如第二十六条）条款，二者共同支撑"拘留是否有法律授权"的论证。
        concept_keywords = [kw for kw in _CONCEPT_KEYWORDS if kw in query]
        if concept_keywords:
            # 2a. 处罚种类定义条款：definition + 正文含"种类" + 概念词
            has_kind_def = any(
                _is_protected_role(_classify_role(c.get("metadata", {})))
                and "种类" in (c.get("text", "") or "")
                and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                for c in top
            )
            if not has_kind_def:
                best = _find_best(
                    lambda c: _is_protected_role(_classify_role(c.get("metadata", {})))
                    and "种类" in (c.get("text", "") or "")
                    and any(kw in (c.get("text", "") or "") for kw in concept_keywords)
                )
                if best is not None:
                    _insert_quota(best)
                    event(
                        "rerank.kind_definition_quota",
                        article=best.get("metadata", {}).get("article_no", ""),
                        source=best.get("metadata", {}).get("source", ""),
                        rerank_score=best.get("rerank_score", 0.0),
                    )

            # 2b. 行为+量刑条款：behavior_penalty + 正文含概念词
            has_behavior_bp = any(
                _classify_role(c.get("metadata", {})) == "behavior_penalty"
                and any(
                    kw in (c.get("text", "") or "")
                    or kw in ((c.get("metadata") or {}).get("penalty_context", ""))
                    for kw in concept_keywords
                )
                for c in top
            )
            if not has_behavior_bp:
                best = _find_best(
                    lambda c: _classify_role(c.get("metadata", {})) == "behavior_penalty"
                    and any(
                        kw in (c.get("text", "") or "")
                        or kw in ((c.get("metadata") or {}).get("penalty_context", ""))
                        for kw in concept_keywords
                    )
                )
                if best is not None:
                    _insert_quota(best)
                    event(
                        "rerank.behavior_penalty_quota",
                        article=best.get("metadata", {}).get("article_no", ""),
                        source=best.get("metadata", {}).get("source", ""),
                        rerank_score=best.get("rerank_score", 0.0),
                    )

        # 引用追踪：由 retrieval.py 的 REFERENCE_QUOTA 在候选池阶段统一处理，
        # rerank 阶段不再重复追踪，避免同一条被引用条款被重复加入导致 token 浪费。

        return top[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
