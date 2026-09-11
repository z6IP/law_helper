"""查询改写：把口语化问题扩展为更贴近法条表述的检索查询。

查询扩展规则由 config/policy.json 管理，代码只负责执行通用的触发条件和查询生成。

新增 multi_query_rewrite：LLM 多查询改写（Multi-Query Retrieval），
针对用户口语化提问与法条术语不匹配的问题。LLM 只做语言层改写，
不判断法律对错，不提供法律结论，不编造法条编号。

"""
from __future__ import annotations

import re

from app.errors import LawHelperError
from app.config import get_settings
from app.llm import get_llm
from app.prompt_loader import render_prompt
from app.policy import get_policy
from app.tracing import event

def expand_query(question: str) -> list[str]:
    """返回用于检索/重排的扩展查询列表（关键词注入）。

    本函数作为 multi_query_rewrite 的兜底补强：当用户口语化查询
    与法条术语几乎无词项交集（如"闯红灯" vs 道交法第九十条
    "机动车驾驶人违反道路通行规定 处警告 罚款"——BM25 共同词项为 0），
    LLM 改写不稳定且未必能稳定命中通用处罚条款关键词，硬编码规则
    能 100% 召回关键条款。规则命中后注入法条术语扩展，作为额外
    检索查询加入 retrieval_queries 参与 BM25+向量双路融合。

    一条用户查询可能命中多条规则扩展（如"闯红灯"同时命中道交法
    第九十条、记分办法第十条、道交法第六十二条），每条扩展独立
    精确命中一个法条，避免单条扩展包含多法条关键词时 BM25 在长正文
    条款上被稀释（如记分第十条 11 款长正文会让"一次记6分"权重降低）。

    无规则命中时返回空列表，调用方跳过不加入检索查询列表。
    """
    q = (question or "").strip()
    if not q:
        return []
    policy_rules = get_policy()["query"]["expansion_rules"]
    expansions: list[str] = []
    seen: set[str] = set()
    for rule in policy_rules:
        triggers = tuple(rule["triggers"])
        requires = tuple(rule["requires"])
        enhancement = rule["enhancement"]
        if not any(t in q for t in triggers):
            continue
        if requires and not any(r in q for r in requires):
            continue
        if enhancement in seen:
            continue
        seen.add(enhancement)
        expansions.append(enhancement)
    return expansions


def classify_query_intents(question: str) -> list[str]:
    """按法律问题表述识别轻量意图，不调用模型、不作法律判断。

    支持两类规则：
    - keywords：命中任一关键词即视为该意图；
    - patterns：命中任一正则模式即视为该意图（适合否定、例外等需要
      前缀/后缀约束的精确匹配）。
    """
    q = (question or "").strip()
    if not q:
        return []
    rules = get_policy()["query"].get("intent_rules", {})
    intents: list[str] = []
    for intent, rule in rules.items():
        if any(keyword in q for keyword in rule.get("keywords", [])):
            intents.append(intent)
            continue
        patterns = rule.get("patterns", [])
        if patterns and any(re.search(p, q) for p in patterns):
            intents.append(intent)
    return intents


def expand_query_by_intent(question: str) -> tuple[list[str], list[str]]:
    """返回 (意图列表, 意图检索扩展)，仅用于提高召回覆盖率。"""
    intents = classify_query_intents(question)
    rules = get_policy()["query"].get("intent_rules", {})
    expansions: list[str] = []
    seen: set[str] = set()
    for intent in intents:
        for enhancement in rules[intent].get("enhancements", []):
            if enhancement not in seen:
                seen.add(enhancement)
                expansions.append(enhancement)
    return intents, expansions


def expand_synonyms(question: str) -> list[str]:
    """R4：基于同义词/近义词表把口语化关键词映射为法条术语。

    与 expansion_rules 不同：这里保留原问题结构，仅替换其中的同义关键词，
    用于补充 LLM 多查询改写的盲区。所有改写仅用于检索，不进入最终回答。
    """
    q = (question or "").strip()
    if not q:
        return []
    synonym_map = get_policy()["query"].get("synonym_map", {})
    results: list[str] = []
    seen: set[str] = {q}
    for term, paraphrases in synonym_map.items():
        if term not in q:
            continue
        for para in paraphrases:
            if para in seen:
                continue
            seen.add(para)
            results.append(para)
    return results


def _multi_query_system(query_count: int | None = None) -> str:
    settings = get_settings()
    return render_prompt(
        "multi_query",
        query_count=query_count or settings.multi_query_count,
    )


def multi_query_rewrite(question: str, n: int | None = None) -> list[str]:
    """LLM 多查询改写：返回 n 条不同视角的检索查询（不含原问题）。

    合规性：LLM 只做语言层改写，不判断法律对错，不进入最终回答。
    失败时返回空列表，调用方使用原问题单路检索兜底。

    Args:
        question: 用户原始问题（已做多轮历史改写后的独立问题）
        n: 期望的改写条数，默认 3

    Returns:
        改写后的查询列表，长度 0~n；失败时为空列表
    """
    q = (question or "").strip()
    if not q:
        return []
    n = n or get_settings().multi_query_count

    user_prompt = f"用户问题：{q}\n\n请输出 {n} 条改写查询，每行一条："
    try:
        raw = get_llm().chat(
            _multi_query_system(), user_prompt,
            temperature=get_settings().retrieval_temperature,
        )
    except LawHelperError:
        event("multi_query_rewrite.failed", reason="llm_error")
        return []

    # 清洗：去空行、去编号前缀、去引号包裹、去首尾空白
    rewrites: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # 去除「1.」「1、」「-」「*」等列表前缀
        line = re.sub(r"^[\d]+[.、)\]]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        # 去引号包裹
        line = line.strip("「」“”\"‘’'").strip()
        if not line:
            continue
        if line in rewrites:
            continue  # 去重
        rewrites.append(line)
        if len(rewrites) >= n:
            break

    event(
        "multi_query_rewrite.done",
        original=question,
        rewrites=rewrites,
        count=len(rewrites),
    )
    return rewrites


