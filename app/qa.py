"""问答编排：检索 → 重排 → 生成 → 引用。"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Literal

import numpy as np

from app.config import get_settings
from app.errors import LawHelperError
from app.llm import get_llm
from app.policy import get_policy
from app.prompt_loader import render_prompt
from app.query_expansion import (
    expand_query,
    expand_query_by_intent,
    expand_synonyms,
    multi_query_rewrite,
)
from app.retrieval import _tokenize, get_retrieval_engine
from app.rerank import apply_role_adjustment, get_reranker
from app.schemas import Reference
from app.tracing import event, span

_RefusalType = Literal["no_law", "insufficient", "out_of_scope"]
from app.upload_retrieval import select_relevant_chunks


def _ensure_expansion_hits(
    expansions: list[str],
    candidates: list[dict],
    contexts: list[dict],
    engine,
) -> list[dict]:
    """扩展查询关键条款保底：对每个扩展查询取 BM25 top2 条款，
    如在 candidates 中但不在 contexts 中，强制加入 contexts。

    解决 multi-款文章（如记分第十条 11 款）rerank 分数被长正文稀释，
    关键条款未进入 top_n 的问题。最多追加 3 条，BM25 阈值 20。
    追加后重新做角色排序，保证受保护条款仍在前。
    """
    if not expansions or engine._bm25 is None:
        return contexts

    settings = get_settings()
    expansion_ensure_quota = settings.expansion_ensure_quota
    expansion_ensure_min_bm25 = settings.expansion_ensure_min_bm25
    expansion_ensure_top_per_query = get_policy()["retrieval"]["expansion_ensure_top_per_query"]
    existing_ids = {c.get("id") for c in contexts}
    ensured = 0
    for exp_q in expansions:
        if ensured >= expansion_ensure_quota:
            break
        bm25_scores = np.asarray(engine._bm25.get_scores(_tokenize(exp_q)))
        top_indices = np.argsort(-bm25_scores)[:expansion_ensure_top_per_query]
        for top_idx in top_indices:
            if ensured >= expansion_ensure_quota:
                break
            top_idx = int(top_idx)
            if float(bm25_scores[top_idx]) < expansion_ensure_min_bm25:
                continue
            did = engine._ids[top_idx]
            if did in existing_ids:
                continue
            for c in candidates:
                if c.get("id") == did:
                    contexts.append(c)
                    existing_ids.add(did)
                    ensured += 1
                    event(
                        "qa.expansion_ensure",
                        source=c.get("metadata", {}).get("source", ""),
                        article_no=c.get("metadata", {}).get("article_no", ""),
                    )
                    break
    if ensured:
        contexts = apply_role_adjustment(contexts)
    return contexts


def _ensure_parent_articles(candidates: list[dict], contexts: list[dict]) -> list[dict]:
    """子款在 contexts 中但父条不在时，从 candidates 补回父条。

    解决父条因长正文被 rerank 低分挤出 top_n，导致 LLM 看不到
    完整法条（含处罚前置句"一次记6分"等）的问题。
    补回后 _merge_references 会将父条与子款合并为单条引用。
    """
    if not candidates or not contexts:
        return contexts

    existing_ids = {c.get("id") for c in contexts}
    # contexts 中已存在的父条 (source, article_no)
    parent_in_contexts = {
        (c.get("metadata", {}).get("source"), c.get("metadata", {}).get("article_no"))
        for c in contexts
        if (c.get("metadata") or {}).get("chunk_type") == "article"
    }
    # 需要补回的父条 key
    needed_parents: set[tuple[str, str]] = set()
    for c in contexts:
        meta = c.get("metadata") or {}
        if meta.get("chunk_type") != "clause":
            continue
        parent_no = meta.get("parent_article_no")
        source = meta.get("source")
        if not parent_no or not source:
            continue
        key = (source, parent_no)
        if key not in parent_in_contexts:
            needed_parents.add(key)

    if not needed_parents:
        return contexts

    # 从 candidates 中查找父条并追加（保持 candidates 顺序）
    added = 0
    for c in candidates:
        meta = c.get("metadata") or {}
        if meta.get("chunk_type") != "article":
            continue
        key = (meta.get("source"), meta.get("article_no"))
        if key in needed_parents and c.get("id") not in existing_ids:
            contexts.append(c)
            existing_ids.add(c.get("id"))
            needed_parents.discard(key)
            added += 1

    if added:
        event(
            "qa.parent_article_ensure",
            added=added,
            context_count=len(contexts),
        )
    return contexts


def _inject_penalty_context(contexts: list[dict]) -> list[dict]:
    """对仍缺父条的子款，把 penalty_context 作为前缀拼入 text。

    兜底机制：当 _ensure_parent_articles 也找不到父条时（父条
    未进入 candidates），通过注入处罚前置句让 LLM 至少看到
    "一次记6分""处...拘留"等关键处罚信息。
    直接修改 c["text"]，_build_user_prompt 和 _merge_references
    自然看到注入后的内容。
    """
    if not contexts:
        return contexts

    # contexts 中已存在的父条 (source, article_no)
    parent_in_contexts = {
        (c.get("metadata", {}).get("source"), c.get("metadata", {}).get("article_no"))
        for c in contexts
        if (c.get("metadata") or {}).get("chunk_type") == "article"
    }

    injected = 0
    for c in contexts:
        meta = c.get("metadata") or {}
        if meta.get("chunk_type") != "clause":
            continue
        parent_no = meta.get("parent_article_no")
        source = meta.get("source")
        if not parent_no or not source:
            continue
        # 父条已在 contexts 中，无需注入（父条完整 text 已含处罚信息）
        if (source, parent_no) in parent_in_contexts:
            continue
        penalty_ctx = (meta.get("penalty_context") or "").strip()
        if not penalty_ctx:
            continue
        text = (c.get("text") or "").strip()
        # 避免重复注入
        if text.startswith(penalty_ctx):
            continue
        c["text"] = f"{penalty_ctx}\n{text}"
        injected += 1

    if injected:
        event("qa.penalty_context_injected", injected=injected)
    return contexts


def _restrict_to_primary_source(
    candidates: list[dict], law_source: str | None
) -> list[dict]:
    """只在用户明确指定法规时过滤，默认保留跨法规候选。"""
    if not law_source:
        return candidates
    return [
        candidate for candidate in candidates
        if (candidate.get("metadata") or {}).get("source") == law_source
    ]


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    """加载主法律问答模板，并注入当前语料中的法规名称。"""
    laws = get_settings().law_sources
    law_list = "".join(f"《{name}》" for name in laws)
    return render_prompt("legal_system", law_list=law_list)


@lru_cache(maxsize=None)
def _domain_prompt(domain: str) -> str:
    """渲染指定领域的领域提示词段（无占位符，纯静态文本）。"""
    prompt_name = get_policy()["domains"][domain]["prompt"]
    return render_prompt(prompt_name)


def _match_domains(contexts: list[dict]) -> tuple[str, ...]:
    """依据 policy 中 domains 的 law_keywords 与检索候选的 metadata.source
    做子串匹配，返回命中的领域名（有序、去重）。"""
    domains = get_policy().get("domains") or {}
    sources = {(c.get("metadata") or {}).get("source", "") for c in contexts}
    matched: list[str] = []
    for domain, cfg in domains.items():
        keywords = cfg.get("law_keywords") or []
        if any(kw and any(kw in src for src in sources) for kw in keywords):
            matched.append(domain)
    return tuple(matched)


def _compose_system_prompt(contexts: list[dict]) -> str:
    """基础法律 system prompt + 命中领域的领域段；contexts 为空或未命中时
    返回基础 prompt，行为与原 _system_prompt() 完全一致。
    领域模板缺失或渲染失败时降级为基础 prompt，不中断主回答流程。"""
    domains = _match_domains(contexts)
    if not domains:
        return _system_prompt()
    parts = [_system_prompt()]
    missing: list[str] = []
    for domain in domains:
        try:
            parts.append(_domain_prompt(domain))
        except (FileNotFoundError, ValueError):
            missing.append(domain)
    if missing:
        event("prompt.domain_missing", domains=missing)
        return _system_prompt()
    event("prompt.domain_injected", domains=list(domains))
    return "\n\n".join(parts)


@lru_cache(maxsize=1)
def _off_topic_prompt_template() -> str:
    """加载无相关法条时的拒答模板。"""
    return render_prompt("off_topic", question="{question}")


@lru_cache(maxsize=None)  # 按 refusal_type 缓存，3 种类型各自常驻，避免 maxsize=1 反复淘汰
def _refusal_prompt_template(refusal_type: _RefusalType = "no_law") -> str:
    """A5：加载细分拒答场景的模板；模板缺失时回退到 off_topic 兜底。"""
    template_name = f"refusal_{refusal_type}"
    try:
        return render_prompt(template_name, question="{question}")
    except FileNotFoundError:
        return _off_topic_prompt_template()


def _classify_refusal(
    candidates: list[dict],
    contexts: list[dict],
    law_source: str | None = None,
) -> _RefusalType:
    """A5：根据检索结果细分拒答场景。

    - out_of_scope：用户指定的 law_source 不在服务范围内；
    - insufficient：检索到了候选但重排后均低于相关性阈值，法条不足；
    - no_law：完全未检索到任何候选条文。
    """
    if law_source:
        valid_sources = set(get_settings().law_sources)
        if law_source not in valid_sources:
            return "out_of_scope"
    if not contexts:
        return "insufficient" if candidates else "no_law"
    return "no_law"


# 无意义输入黑名单：问候、应答、寒暄等，命中即走拒答分支，不进入检索
_TRIVIAL_TOKENS = set(get_policy()["query"]["trivial_tokens"])

# 法律相关关键词：短 query 命中任一关键词才进入 RAG，否则视为无意义输入
# 道路交通安全 + 立法法/通用法律语境关键词，确保不同法规的短问句都能进入检索
_LAW_KEYWORDS = tuple(get_policy()["query"]["law_keywords"])


def _is_trivial_query(query: str) -> bool:
    """判断是否为无意义输入：单字、纯数字、纯标点、问候寒暄、
    或长度 ≤4 且不含任何法律关键词的短 query。

    此类输入直接走拒答分支，不进入检索 / 重排流程，也不返回任何引用。
    """
    q = (query or "").strip().lower()
    if not q:
        return True
    # 单字符（单字、单数字、单标点）
    if len(q) <= 1:
        return True
    # 纯数字（含小数点）
    if re.fullmatch(r"[\d.]+", q):
        return True
    # 纯标点 / 空白 / 符号
    if re.fullmatch(r"[\s\W_]+", q):
        return True
    # 命中问候寒暄等黑名单
    if q in _TRIVIAL_TOKENS:
        return True
    # 短 query 且不含任何法律关键词
    if len(q) <= 4 and not any(k in q for k in _LAW_KEYWORDS):
        return True
    return False


# 多轮历史感知改写（condense question）：把追问 + 最近历史改写成独立完整的问题，
# 再进入检索与生成；与 GitHub 高星实践（LangChain create_history_aware_retriever）等价
@lru_cache(maxsize=1)
def _context_resolve_system() -> str:
    """加载历史问题改写 system prompt。"""
    return render_prompt("context_resolve")


_SANITIZE_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SANITIZE_RE_LABEL = re.compile(r"^(改写后的独立问题|改写后的问题|改写后|独立问题)\s*[:：]\s*")


def _sanitize_rewrite(text: str) -> str:
    """清洗改写输出：去思考标签、去标签前缀、取首行、去包裹引号。

    防止 qwen3 系列（可能夹带 <think>）或带「改写后的独立问题：」标签、
    引号包裹的输出污染检索 query。
    """
    text = _SANITIZE_RE_THINK.sub("", text or "")
    text = text.strip()
    text = _SANITIZE_RE_LABEL.sub("", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0].strip("「」“”\"‘’'").strip()


def _resolve_context(question: str, history: list[dict]) -> tuple[str, bool]:
    """历史感知改写：追问 + 最近历史 → 独立完整问题。

    返回 (resolved, ok)：ok=False 表示改写失败（LLM 异常 / 输出为空或超长），
    调用方应跳过 trivial 拒答直接进检索，由重排阈值兜底，
    避免「改写挂了 + 追问短」被双重误杀。
    仅取最近 history_max_messages 条消息（3 轮），temperature=0 保证确定性输出。
    """
    if not history:
        return question, True
    settings = get_settings()
    turns = [
        f"{'用户' if (m.get('role') == 'user') else '助手'}: {str(m.get('content', ''))}"
        for m in history[-settings.history_max_messages:]
        if m.get("content")
    ]
    if not turns:
        return question, True
    user_prompt = (
        "对话历史：\n" + "\n".join(turns)
        + f"\n\n用户最新问题：{question}\n\n改写后的独立问题："
    )
    try:
        rewritten = get_llm().chat(
            _context_resolve_system(), user_prompt,
            temperature=get_settings().retrieval_temperature,
        )
        rewritten = _sanitize_rewrite(rewritten)
        if not rewritten or len(rewritten) > settings.rewrite_max_length:  # 超长视为解释性输出，改写失败
            return question, False
        if rewritten != question:
            event("query_rewrite.changed", before=question, after=rewritten)
        return rewritten, True
    except LawHelperError:
        return question, False  # 降级：改写失败不影响主流程


# 对话元问题：询问「对话本身」而非法律内容（如「我刚刚的问题是什么」）。
# 用正则约束「时间词 + 对话行为词」组合，避免「刚才那个法条」这类法律追问被误判
_META_RE = re.compile(
    r"(刚刚|刚才|上一句|上一个问题|之前|前面)[^。？?]{0,8}(问|说|聊|回答|问题|提到)"
    r"|(问了|说了)(些|的)?(什么|啥)"
)
_META_ANSWER_SYSTEM = render_prompt("meta_answer")


def _is_conversation_meta(question: str) -> bool:
    """是否为询问对话本身的元问题（需要携带历史作答，而非检索法条）。"""
    return bool(_META_RE.search(question))


def _history_prompt(question: str, history: list[dict]) -> str:
    turns = [
        f"{'用户' if (m.get('role') == 'user') else '助手'}: {str(m.get('content', ''))}"
        for m in history
        if m.get("content")
    ]
    return "对话历史：\n" + "\n".join(turns) + f"\n\n用户的问题：{question}\n\n回答："


_CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百零]+款$")

# 中文数字（一~九十九）转整数，用于款号排序（如「十一」→ 11）
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num_to_int(text: str) -> int:
    """中文数字（一~九十九）转整数，如「十一」→ 11。"""
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        return _CN_DIGITS.get(left, 1) * 10 + (_CN_DIGITS.get(right, 0) if right else 0)
    return _CN_DIGITS.get(text, 0)


def _normalize_source_name(source: str) -> str:
    """A9：清洗法规名称——去日期后缀、`+` 转空格、去首尾空白。不添加书名号。

    书名号由展示层（前端 References 组件、上下文标题 _format_source_with_marks）统一添加，
    避免后端与前端重复包裹导致《《……》》。
    """
    if not source:
        return source
    name = source.strip()
    # 去掉末尾可能残留的日期后缀（兼容离线清洗未完全生效的场景）
    name = re.sub(r"_\d{8}$", "", name).replace("+", " ")
    return name


def _format_source_with_marks(source: str) -> str:
    """A9：返回带书名号的《法规全称》，用于上下文标题与自检匹配。"""
    name = _normalize_source_name(source)
    if not name:
        return name
    if not (name.startswith("《") and name.endswith("》")):
        name = f"《{name}》"
    return name


def _extract_clause_label(article_no: str) -> str:
    """从完整条号中提取款号标签，如 '第九十一条第一款' -> '第一款'。"""
    if not article_no:
        return ""
    match = _CLAUSE_RE.search(article_no.strip())
    return match.group(0) if match else ""


def _article_group_key(context: dict) -> tuple[str, str]:
    """A1：按 (法规全称, 条号) 分组；同条下的多款合并为一条引用。"""
    meta = context.get("metadata") or {}
    source = _normalize_source_name(meta.get("source", ""))
    # 优先使用 parent_article_no（纯条号），无则回退到 article_no
    article_no = meta.get("parent_article_no") or meta.get("article_no", "")
    # 如果 article_no 仍包含款号，去除款号保留条号
    article_no = _CLAUSE_RE.sub("", article_no).strip()
    return (source, article_no)


def _merge_references(contexts: list[dict]) -> list[Reference]:
    """A1/A9：同一法规同条号下的多款合并为单条引用，并统一法规全称格式。

    合并规则：
    - 按 (source, 条号) 分组，保留 contexts 中首次出现的顺序；
    - 组内只有一款时，直接使用该款原文；
    - 组内有多款时，按完整条号字典序排序，并在每款原文前标注款号；
    - merged_from 记录参与合并的原始 chunk id，便于溯源。
    """
    if not contexts:
        return []

    # 保持原始顺序的分组
    group_order: list[tuple[str, str]] = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in contexts:
        key = _article_group_key(c)
        if key not in groups:
            group_order.append(key)
        groups.setdefault(key, []).append(c)

    merged: list[Reference] = []
    for key in group_order:
        group = groups[key]
        source, article_no = key
        first_meta = group[0].get("metadata") or {}
        section_header = first_meta.get("section_header", "")
        ids = [str(c.get("id", "")) for c in group if c.get("id")]

        if len(group) == 1:
            text = group[0]["text"]
        else:
            # 同一法条多款：按款号数值排序并拼接（中文数字转整数，避免字典序错乱）
            def _clause_sort_key(c: dict) -> int:
                label = _extract_clause_label(c.get("metadata", {}).get("article_no", ""))
                m = re.search(r"第([零一二三四五六七八九十]+)款", label)
                return _cn_num_to_int(m.group(1)) if m else 0

            sorted_group = sorted(group, key=_clause_sort_key)
            parts: list[str] = []
            for c in sorted_group:
                clause = _extract_clause_label(
                    c.get("metadata", {}).get("article_no", "")
                )
                prefix = f"（{clause}）" if clause else ""
                parts.append(f"{prefix}\n{c['text']}".strip())
            text = "\n\n".join(parts)

        merged.append(
            Reference.model_validate({
                "source": source,
                "article_no": article_no,
                "section_header": section_header,
                "text": text,
                "merged_from": ids,
            })
        )

    event("answer.references_merged", before=len(contexts), after=len(merged))
    return merged


# A4：规则层可识别的引用格式《法规全称》第X条（款可选）
_CITATION_RE = re.compile(
    r"《([^》]+)》\s*第\s*([一二三四五六七八九十百零\d]+)\s*条"
    r"(?:\s*第\s*([一二三四五六七八九十百零\d]+)\s*款)?"
)


def _self_check_answer(
    question: str,
    answer: str,
    contexts: list[dict],
) -> tuple[bool, list[str]]:
    """A4 规则层自检：扫描答案中的法规引用，检查是否引用了检索上下文之外的法条。

    仅检查以《法规名称》第X条形式明确出现的引用。返回 (通过, 告警列表)。
    """
    if not answer or not contexts:
        return True, []

    allowed: set[tuple[str, str]] = set()
    for c in contexts:
        meta = c.get("metadata") or {}
        source = _format_source_with_marks(meta.get("source", ""))
        article_no = meta.get("parent_article_no") or meta.get("article_no", "")
        article_no = _CLAUSE_RE.sub("", article_no).strip()
        if source and article_no:
            allowed.add((source, article_no))

    warnings: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        cited_source = match.group(1).strip()
        cited_article = f"第{match.group(2)}条"

        source_hits = [
            (src, art) for src, art in allowed
            if cited_source in src or src in cited_source
        ]
        if not source_hits:
            warnings.append(f"答案引用了未检索到的法规《{cited_source}》")
            continue

        article_hit = any(art == cited_article for _, art in source_hits)
        if not article_hit:
            warnings.append(
                f"答案引用了《{cited_source}》{cited_article}，但该条未在检索上下文中出现"
            )

    if warnings:
        event("answer.self_check.warning", question=question, warnings=warnings)
        return False, warnings
    return True, []


def _build_user_prompt(
    question: str,
    contexts: list[dict],
    document_text: str | None = None,
    document_chunks: list[str] | None = None,
    law_source: str | None = None,
    article_no: str | None = None,
) -> str:
    blocks = []
    if document_chunks is not None:
        for idx, chunk in enumerate(document_chunks, 1):
            blocks.append(f"【用户上传材料·片段{idx}】\n{chunk}")
    elif document_text:
        blocks.append(f"用户上传的材料内容如下：\n{document_text}")
    # A9：上下文标题强制使用《法规全称》+ 条号，为模型输出做格式示范。
    # 按法规分组注入：同一法规的条文归在一起，法规之间用分隔线隔开，
    # 便于模型直观看到跨法规关系（如行为/通行规定条款与处罚种类/处罚依据条款分属不同法规）。
    if contexts:
        groups: dict[str, list[dict]] = {}
        group_order: list[str] = []
        for c in contexts:
            meta = c.get("metadata", {})
            source = _format_source_with_marks(meta.get("source", ""))
            if source not in groups:
                group_order.append(source)
                groups[source] = []
            groups[source].append(c)
        for i, source in enumerate(group_order):
            if i > 0:
                blocks.append("──────────")
            for c in groups[source]:
                meta = c.get("metadata", {})
                art_no = meta.get("article_no", "")
                section = meta.get("section_header", "")
                header = f"【{source}·{art_no}】" + (f"（{section}）" if section else "")
                blocks.append(f"{header}\n{c['text']}")
    context_text = "\n\n".join(blocks)
    scope = ""
    if law_source or article_no:
        scope = (
            "本次回答的法源范围已限定为："
            f"{law_source or '指定范围内'}"
            f"{('·' + article_no) if article_no else ''}。"
            "不得引用范围之外的法规或法条。\n\n"
        )
    cross_law_hint = ""
    if len({(c.get("metadata") or {}).get("source", "") for c in contexts}) > 1:
        cross_law_hint = (
            "注意：上述法条来自不同的法律法规，请分析它们之间的关联关系"
            "（如行为/通行规定条款与处罚种类/处罚依据条款分属不同法规），"
            "综合得出判断。\n\n"
        )
    return (
        f"以下是系统根据用户问题检索到的相关法律法规条文原文：\n\n"
        f"{scope}"
        f"{context_text}\n\n"
        f"{cross_law_hint}"
        f"用户问题：{question}\n\n"
        f"请严格依据上述检索到的法条原文，结合用户上传的材料回答，"
        f"回答中不要提及法条的来源（不要说「用户提供」「你提供」等）。"
    )


def _retrieve_contexts(
    resolved: str,
    document_text: str | None = None,
    law_source: str | None = None,
    article_no: str | None = None,
) -> tuple[list[dict], list[dict], list[str] | None]:
    """共享检索管线：材料检索 → 多查询改写 → 扩展 → 检索 → 重排 → 角色调整 → 保底补全。

    answer() 与 answer_stream() 共用，消除双份维护导致的流式/非流式行为分叉。
    返回 (contexts, candidates, document_chunks)。
    """
    settings = get_settings()

    # 材料检索：长材料切块取 top3，短材料返回 None 以全文注入
    document_chunks: list[str] | None = None
    if document_text:
        with span("upload_retrieval.select"):
            document_chunks = select_relevant_chunks(resolved, document_text)

    engine = get_retrieval_engine()
    reranker = get_reranker()

    # Multi-Query 改写：LLM 把口语化问题改写为多视角检索查询（仅语言层操作）。
    # 改写查询仅用于检索，不进入最终 context；失败时回退到原问题单路检索。
    with span("multi_query_rewrite"):
        rewrites = multi_query_rewrite(resolved, n=settings.multi_query_count)
    # 关键词注入兜底：对触发词命中的查询（如"闯红灯"）注入法条术语，
    # 解决 BM25 与法条正文无词项交集导致的召回失败（闯红灯→道交法第九十条）。
    # 扩展查询放在 rewrites 之前：让它优先享受 retrieval.py REWRITE_QUOTA 配额，
    # 避免 LLM 改写查询占满 3 条配额后扩展查询的关键条款被挤出候选池。
    # expand_query 可能返回多条扩展（闯红灯命中道交法90条+记分办法10条+道交法62条），
    # 每条精确命中一个法条，避免单条扩展关键词被长正文条款稀释。
    expansions = expand_query(resolved)
    intents, intent_expansions = expand_query_by_intent(resolved)
    expansions.extend(x for x in intent_expansions if x not in expansions)
    # R2 否定意图集成：生成去除否定词的辅助查询，帮助向量召回正面要件条款。
    # 例如"没戴头盔怎么处罚"去否定后得到"戴头盔怎么处罚"，可作为语义补充。
    if "negation" in intents:
        negation_free = re.sub(r"(没|未|没有|不)\s*", "", resolved).strip()
        if negation_free and negation_free != resolved and negation_free not in rewrites:
            rewrites.append(negation_free)
            event("query.negation_rewrite", original=resolved, negation_free=negation_free)
    # R4 同义词改写：把口语化关键词映射到法条术语，作为 LLM 改写的兜底补强。
    synonym_rewrites = expand_synonyms(resolved)
    for syn in synonym_rewrites:
        if syn not in rewrites:
            rewrites.append(syn)
    if synonym_rewrites:
        event("query.synonym_rewrite", original=resolved, synonyms=synonym_rewrites)
    retrieval_queries = [resolved] + expansions + rewrites
    event("query.intent", intents=intents, expansion_count=len(intent_expansions))
    # rerank 阶段仍使用用户原始独立问题，保持与用户意图对齐
    retrieval_q = resolved
    with span("retrieval", query_count=len(retrieval_queries), top_k=settings.top_k_retrieve):
        candidates = engine.multi_query_search(
            retrieval_queries,
            top_k=settings.top_k_retrieve,
            law_source=law_source,
            article_no=article_no,
        )
    candidates = _restrict_to_primary_source(candidates, law_source)
    event("retrieval.candidates", count=len(candidates))
    with span("rerank", top_n=settings.rerank_top_n, min_score=settings.rerank_min_score):
        contexts = reranker.rerank(
            retrieval_q,
            candidates,
            top_n=settings.rerank_top_n,
            min_score=settings.rerank_min_score,
        )
    event("rerank.hits", count=len(contexts))
    # 角色优先级调整：定义性 > 实体性 > 程序性
    # 规则法（基于 section_header），不调 LLM；只调整排序，不删除任何法条
    with span("role_adjustment"):
        contexts = apply_role_adjustment(contexts, intents)

    # 扩展查询关键条款保底 + 父条补全 + 处罚上下文注入
    contexts = _ensure_expansion_hits(expansions, candidates, contexts, engine)
    contexts = _ensure_parent_articles(candidates, contexts)
    contexts = _inject_penalty_context(contexts)
    return contexts, candidates, document_chunks


def answer(
    question: str,
    history: list[dict] | None = None,
    document_text: str | None = None,
    law_source: str | None = None,
    article_no: str | None = None,
) -> tuple[str, list[Reference]]:
    history = history or []

    # 多轮：先做历史感知改写，trivial 判定与检索均使用改写后的独立问题
    # （防止「那扣几分？」这类追问被 trivial 拦截误杀）；
    # 改写失败（rewrite_ok=False）时跳过 trivial 拒答直接进检索，由重排阈值兜底；
    # 无历史时 resolved 即原问题，行为与单轮完全一致
    with span("query_rewrite"):
        resolved, rewrite_ok = _resolve_context(question, history)

    # 无意义输入（单字 / 纯数字 / 问候 / 短词无法律关键词）：不进入检索，直接拒答
    # 若有上传文件内容，则跳过 trivial 拦截，允许对短问题结合材料作答
    if rewrite_ok and _is_trivial_query(resolved) and not document_text:
        event("trivial_reject", question=question)
        user_prompt = _refusal_prompt_template("out_of_scope").format(question=question)
        llm_text = get_llm().chat(_system_prompt(), user_prompt)
        return llm_text, []

    # 对话元问题（如「我刚刚的问题是什么」）：仅凭对话历史回答，不检索、不附引用
    if history and _is_conversation_meta(resolved):
        event("conversation_meta", question=resolved)
        llm_text = get_llm().chat(_META_ANSWER_SYSTEM, _history_prompt(question, history))
        return llm_text, []

    # 检索管线（材料检索 → 改写 → 扩展 → 检索 → 重排 → 角色调整 → 保底补全）
    contexts, candidates, document_chunks = _retrieve_contexts(
        resolved, document_text, law_source, article_no
    )

    # 无相关法条：不附带任何引用，由 LLM 简短拒答
    if not contexts:
        refusal_type = _classify_refusal(candidates, contexts, law_source)
        event("no_contexts", refusal_type=refusal_type)
        if document_text:
            user_prompt = _build_user_prompt(
                resolved, [], document_text, document_chunks, law_source, article_no
            )
        else:
            user_prompt = _refusal_prompt_template(refusal_type).format(question=question)
        llm_text = get_llm().chat(_system_prompt(), user_prompt)
        return llm_text, []

    user_prompt = _build_user_prompt(
        resolved, contexts, document_text, document_chunks, law_source, article_no
    )
    with span("llm_generate"):
        llm_text = get_llm().chat(_compose_system_prompt(contexts), user_prompt)

    # A4 反向约束/幻觉自检：规则层扫描答案中的法条引用是否落在检索上下文中
    if get_settings().answer_self_check_enabled:
        passed, warnings = _self_check_answer(resolved, llm_text, contexts)
        if not passed:
            event("answer.self_check.blocked", warnings=warnings)
            llm_text += (
                "\n\n（系统自检提示：回答中出现了未在检索结果中出现的法条引用，"
                "请谨慎参考：" + "；".join(warnings) + "）"
            )

    # R10 可观测性：记录最终引用来源与条号分布
    ref_sources: dict[str, int] = {}
    ref_articles: dict[str, int] = {}
    for c in contexts:
        meta = c.get("metadata") or {}
        src = meta.get("source", "unknown")
        ref_sources[src] = ref_sources.get(src, 0) + 1
        art = meta.get("article_no", "unknown")
        ref_articles[art] = ref_articles.get(art, 0) + 1
    event("answer.references", count=len(contexts), sources=ref_sources, articles=ref_articles)

    references = _merge_references(contexts)
    return llm_text, references


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    document_text: str | None = None,
    law_source: str | None = None,
    article_no: str | None = None,
):
    """流式问答：先产出引用法条事件，再逐段产出回答文本增量。

    每个产出为 dict：
      - {"type": "references", "references": [...]}  （无相关法条时为空列表）
      - {"type": "reasoning", "content": "..."}  （推理模型的思考过程，普通模型无此事件）
      - {"type": "delta", "content": "..."}

    优化：在检索前立即发送一条 reasoning 事件，让前端思考区域立刻有内容，
    消除「发送问题后等待几秒才看到思考开始」的空白期。
    每一步都会向前端 yield 进度 reasoning 事件，让用户看到实时进度。
    """
    t0 = time.perf_counter()
    history = history or []

    # 无意义输入（无历史时的单字 / 纯数字 / 问候 / 短词无法律关键词）：
    # 不发预热思考，直接走拒答分支，前端不会出现思考区域
    # 若有上传文件内容，跳过 trivial 拦截，允许结合材料作答
    if not history and _is_trivial_query(question) and not document_text:
        event("trivial_reject", question=question)
        yield {"type": "references", "references": []}
        user_prompt = _refusal_prompt_template("out_of_scope").format(question=question)
        for kind, text in get_llm().chat_stream(_system_prompt(), user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # 非 trivial 查询：先发一条进度事件（前端可选择展示或忽略）
    yield {"type": "progress", "content": "正在分析你的问题..."}

    # 多轮：历史感知改写（追问 + 最近历史 → 独立完整问题），失败自动降级原问题；
    # 改写成功但结果仍为无意义输入时拒答（不发引用）；
    # 改写失败时跳过 trivial 拒答直接进检索，由重排阈值兜底
    resolved = question
    rewrite_ok = True
    if history:
        yield {"type": "progress", "content": "正在结合上下文理解问题..."}
        with span("query_rewrite"):
            resolved, rewrite_ok = _resolve_context(question, history)
        if rewrite_ok and _is_trivial_query(resolved) and not document_text:
            event("trivial_reject", resolved=resolved)
            yield {"type": "references", "references": []}
            user_prompt = _refusal_prompt_template("out_of_scope").format(question=question)
            for kind, text in get_llm().chat_stream(_system_prompt(), user_prompt):
                if kind == "reasoning":
                    yield {"type": "reasoning", "content": text}
                else:
                    yield {"type": "delta", "content": text}
            return

    # 对话元问题（如「我刚刚的问题是什么」）：仅凭对话历史回答，不检索、不附引用
    if history and _is_conversation_meta(resolved):
        event("conversation_meta", question=resolved)
        yield {"type": "references", "references": []}
        for kind, text in get_llm().chat_stream(_META_ANSWER_SYSTEM, _history_prompt(question, history)):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # Step 1: 材料检索
    if document_text:
        yield {"type": "progress", "content": "正在定位材料相关内容..."}
    # Step 2: 法条检索
    yield {"type": "progress", "content": "正在检索相关法条..."}
    contexts, candidates, document_chunks = _retrieve_contexts(
        resolved, document_text, law_source, article_no
    )

    references = _merge_references(contexts)
    yield {"type": "references", "references": [r.model_dump() for r in references]}

    if not contexts:
        refusal_type = _classify_refusal(candidates, contexts, law_source)
        event("no_contexts", refusal_type=refusal_type)
        if document_text:
            user_prompt = _build_user_prompt(
                resolved, [], document_text, document_chunks, law_source, article_no
            )
        else:
            user_prompt = _refusal_prompt_template(refusal_type).format(question=question)
        for kind, text in get_llm().chat_stream(_system_prompt(), user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # Step 3: LLM 生成（使用改写后的独立问题，与 LangChain rephrase_question=True 一致）
    yield {"type": "progress", "content": "正在思考回答..."}
    user_prompt = _build_user_prompt(
        resolved, contexts, document_text, document_chunks, law_source, article_no
    )
    with span("llm_generate"):
        for kind, text in get_llm().chat_stream(_compose_system_prompt(contexts), user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
    event("done", total_ms=round((time.perf_counter() - t0) * 1000, 2))