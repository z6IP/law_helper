"""问答编排：检索 → 重排 → 生成 → 引用。"""
from __future__ import annotations

import re
import time

from app.config import get_settings
from app.llm import get_llm
from app.query_expansion import expand_query
from app.retrieval import get_retrieval_engine
from app.rerank import get_reranker
from app.schemas import Reference

SYSTEM_PROMPT = (
    "你是「小Z」，一名熟悉道路交通安全相关法律法规（包括《中华人民共和国道路交通安全法》"
    "《中华人民共和国道路交通安全法实施条例》《道路交通事故处理程序规定》"
    "《道路交通安全违法行为记分管理办法》）的智能法律助手。"
    "系统会为你检索并提供与用户问题相关的法条原文供你参考，"
    "这些法条并非用户提供，而是系统根据问题自动检索得到的。"
    "请严格依据提供给你的法条原文回答问题，不要编造法条内容。"
    "回答需简洁、准确，并在涉及具体条款时指出条号。"
    "如果提供的法条不足以回答，请明确说明依据不足。"
    "如果用户问题明显与道路交通安全法律法规无关，请直接说明你只能回答道路交通安全相关法律法规的问题，"
    "不要引用任何法条原文，也不要编造法条。"
    "【表述规范】回答中涉及法条来源时，"
    "请统一使用「系统检索到的法条」「根据检索到的法条」，"
    "或根据检索到的法条来源直接说「根据《中华人民共和国道路交通安全法》」"
    "「根据《中华人民共和国道路交通安全法实施条例》」"
    "「根据《道路交通事故处理程序规定》」"
    "「根据《道路交通安全违法行为记分管理办法》」，"
    "绝对不要出现「你提供的法条」「用户提供的法条」等表述。"
    "回答中用「你」指代提问的用户即可。"
    "【处罚回答规范】回答处罚类问题时请注意：罚款处罚依据《道路交通安全法》"
    "及其实施条例的相应条款，记分处罚依据《道路交通安全违法行为记分管理办法》的相应条款。"
    "机动车驾驶人的违法行为通常同时涉及罚款与记分，若两类依据都已检索到，"
    "应同时给出罚款金额与记分分值，完整回答；"
    "行人、乘车人、非机动车驾驶人不适用记分制度，仅说明罚款处罚；"
    "若某行为仅检索到罚款依据而未见记分依据（或反之），"
    "只依据已检索到的部分回答，不要编造未检索到的处罚内容。"
)


# 检索未命中相关法条时的 user prompt：不附带任何法条 context，
# 引导 LLM 简短说明职责范围，避免引用不相关法条
_OFF_TOPIC_PROMPT_TEMPLATE = (
    "用户问题：{question}\n\n"
    "系统未检索到与该问题相关的道路交通安全法律法规条文。"
    "请简短说明你只能回答与道路交通安全相关法律法规相关的问题，不要引用或编造任何法条。"
)


# 无意义输入黑名单：问候、应答、寒暄等，命中即走拒答分支，不进入检索
_TRIVIAL_TOKENS = {
    "你好", "您好", "你好啊", "在吗", "在不在", "谢谢", "感谢", "好的", "好",
    "嗯", "哦", "哈", "哈哈", "呵呵", "嗨", "hi", "hello", "hey",
    "早", "早上好", "中午好", "下午好", "晚上好", "再见", "拜拜", "88",
    "ok", "okay", "yes", "no", "666", "555", "嗯嗯", "哦哦",
    "帮助", "help", "怎么用", "你是谁", "你叫什么", "你叫啥", "名字",
    "介绍", "自我介绍", "你是", "你能做什么", "你能干嘛",
}

# 法律相关关键词：短 query 命中任一关键词才进入 RAG，否则视为无意义输入
_LAW_KEYWORDS = (
    "法", "交通", "驾驶", "车辆", "机动车", "酒驾", "酒", "事故", "违章",
    "违法", "罚款", "扣分", "驾照", "驾驶证", "行驶证", "行人", "道路",
    "高速", "红绿灯", "信号灯", "限速", "停车", "超速", "逆行", "闯",
    "保险", "责任", "行人", "非机动车", "电动车", "摩托", "头盔", "安全带",
    "调解", "复核", "管辖", "鉴定", "逃逸", "协商", "认定", "赔偿",
)


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


def _build_user_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        meta = c.get("metadata", {})
        source = meta.get("source", "")
        article_no = meta.get("article_no", "")
        section = meta.get("section_header", "")
        header = f"【{source}·{article_no}】" + (f"（{section}）" if section else "")
        blocks.append(f"{header}\n{c['text']}")
    context_text = "\n\n".join(blocks)
    return (
        f"以下是系统根据用户问题检索到的相关法律法规条文原文：\n\n"
        f"{context_text}\n\n"
        f"用户问题：{question}\n\n"
        f"请严格依据上述检索到的法条原文回答，回答中不要提及法条的来源（不要说「用户提供」「你提供」等）。"
    )


def answer(question: str, history: list[dict] | None = None) -> tuple[str, list[Reference]]:
    settings = get_settings()

    # 无意义输入（单字 / 纯数字 / 问候 / 短词无法律关键词）：不进入检索，直接拒答
    if _is_trivial_query(question):
        user_prompt = _OFF_TOPIC_PROMPT_TEMPLATE.format(question=question)
        llm_text = get_llm().chat(SYSTEM_PROMPT, user_prompt)
        return llm_text, []

    engine = get_retrieval_engine()
    reranker = get_reranker()

    retrieval_q = expand_query(question)
    candidates = engine.search(retrieval_q, top_k=settings.top_k_retrieve)
    contexts = reranker.rerank(
        retrieval_q,
        candidates,
        top_n=settings.rerank_top_n,
        min_score=settings.rerank_min_score,
    )

    # 无相关法条：不附带任何引用，由 LLM 简短拒答
    if not contexts:
        user_prompt = _OFF_TOPIC_PROMPT_TEMPLATE.format(question=question)
        llm_text = get_llm().chat(SYSTEM_PROMPT, user_prompt)
        return llm_text, []

    user_prompt = _build_user_prompt(question, contexts)
    llm_text = get_llm().chat(SYSTEM_PROMPT, user_prompt)

    references = [
        Reference(
            source=c["metadata"].get("source", ""),
            article_no=c["metadata"].get("article_no", ""),
            section_header=c["metadata"].get("section_header", ""),
            text=c["text"],
        )
        for c in contexts
    ]
    return llm_text, references


def answer_stream(question: str, history: list[dict] | None = None):
    """流式问答：先产出引用法条事件，再逐段产出回答文本增量。

    每个产出为 dict：
      - {"type": "references", "references": [...]}  （无相关法条时为空列表）
      - {"type": "reasoning", "content": "..."}  （推理模型的思考过程，普通模型无此事件）
      - {"type": "delta", "content": "..."}

    优化：在检索前立即发送一条 reasoning 事件，让前端思考区域立刻有内容，
    消除「发送问题后等待几秒才看到思考开始」的空白期。
    每一步都会向前端 yield 进度 reasoning 事件，让用户看到实时进度。
    """
    settings = get_settings()
    t0 = time.perf_counter()

    # 无意义输入（单字 / 纯数字 / 问候 / 短词无法律关键词）：不发预热思考，
    # 直接走拒答分支，前端不会出现思考区域
    if _is_trivial_query(question):
        yield {"type": "references", "references": []}
        user_prompt = _OFF_TOPIC_PROMPT_TEMPLATE.format(question=question)
        for kind, text in get_llm().chat_stream(SYSTEM_PROMPT, user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # 非 trivial 查询：先发一条进度事件（前端可选择展示或忽略）
    yield {"type": "progress", "content": "正在分析你的问题..."}

    # Step 1: 检索
    yield {"type": "progress", "content": "正在检索相关法条..."}
    engine = get_retrieval_engine()
    retrieval_q = expand_query(question)
    candidates = engine.search(retrieval_q, top_k=settings.top_k_retrieve)
    t_search = time.perf_counter()
    print(f"[Timing] 检索耗时: {t_search - t0:.2f}s, 候选数: {len(candidates)}")

    # Step 2: 重排（搜索阶段统一显示"正在搜索..."，不暴露候选数等内部细节）
    reranker = get_reranker()
    contexts = reranker.rerank(
        retrieval_q,
        candidates,
        top_n=settings.rerank_top_n,
        min_score=settings.rerank_min_score,
    )
    t_rerank = time.perf_counter()
    print(f"[Timing] 重排耗时: {t_rerank - t_search:.2f}s, 命中数: {len(contexts)}")

    references = [
        {
            "source": c["metadata"].get("source", ""),
            "article_no": c["metadata"].get("article_no", ""),
            "section_header": c["metadata"].get("section_header", ""),
            "text": c["text"],
        }
        for c in contexts
    ]
    yield {"type": "references", "references": references}

    if not contexts:
        user_prompt = _OFF_TOPIC_PROMPT_TEMPLATE.format(question=question)
        for kind, text in get_llm().chat_stream(SYSTEM_PROMPT, user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # Step 3: LLM 生成
    yield {"type": "progress", "content": "正在思考回答..."}
    user_prompt = _build_user_prompt(question, contexts)
    for kind, text in get_llm().chat_stream(SYSTEM_PROMPT, user_prompt):
        if kind == "reasoning":
            yield {"type": "reasoning", "content": text}
        else:
            yield {"type": "delta", "content": text}
    t_done = time.perf_counter()
    print(f"[Timing] 总耗时: {t_done - t0:.2f}s")