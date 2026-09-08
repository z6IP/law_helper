"""问答编排：检索 → 重排 → 生成 → 引用。"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from app.config import get_settings
from app.errors import LawHelperError
from app.llm import get_llm
from app.query_expansion import hyde_embed, multi_query_rewrite
from app.retrieval import get_retrieval_engine
from app.rerank import apply_role_adjustment, get_reranker
from app.schemas import Reference
from app.tracing import event, span
from app.upload_retrieval import select_relevant_chunks


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    """构建 system prompt，动态注入 statute/ 下所有法规名作为可回答范围。

    应用启动后 statute/ 内容固定，lru_cache 缓存结果避免重复扫描目录。
    """
    laws = get_settings().law_sources
    law_list = "".join(f"《{name}》" for name in laws)
    return (
        "你是「小Z」，一名熟悉中国法律法规的智能法律助手。"
        f"你能就以下法律法规回答问题：{law_list}。"
        "系统会为你检索并提供与用户问题相关的法条原文供你参考，"
        "这些法条并非用户提供，而是系统根据问题自动检索得到的。"
        "请严格依据提供给你的法条原文回答问题，不要编造法条内容。\n"
        "【回答结构】遵循「结论先行 + 法条支撑 + 实操建议」结构：\n"
        "1. 先用 1-2 句话直接给出结论或实质性回答（不要先堆砌法条）；\n"
        "2. 在结论中或结论后引用必要法条作为依据（指出条号），但不要整段照抄原文；\n"
        "3. 针对用户实际场景给出可操作的建议、注意事项或后续步骤（如该怎么办、需准备什么、"
        "可能的法律后果、如何维权、如何避免风险等）。\n"
        "禁止只罗列法条原文而不给结论和建议。法条原文是支撑，不是回答本身。"
        "如果提供的法条不足以回答，请明确说明依据不足，并给出合理的指引。"
        "如果用户问题明显与上述法律法规无关，请直接说明你只能回答上述法律法规相关的问题，"
        "不要引用任何法条原文，也不要编造法条。"
        "【定义性条款运用】当检索结果中包含定义性/种类性条款"
        "（如《治安管理处罚法》第十条明确将行政拘留列为法定处罚种类之一）时，"
        "应在分析处罚性质、拘留依据等问题时主动引用，"
        "用以说明该处罚属于法定种类的哪一类、是否需有具体法律授权等关键判断。"
        "不要因为定义性条款不直接描述行为就忽略它——"
        "它往往是判断处罚合法性、程序正当性的根本依据。\n"
        "【表述规范】回答中涉及法条来源时，"
        "请统一使用「系统检索到的法条」「根据检索到的法条」，"
        f"或根据检索到的法条来源直接说「根据《xxx》」等（如「根据{law_list}」），"
        "绝对不要出现「你提供的法条」「用户提供的法条」等表述。"
        "回答中用「你」指代提问的用户即可。\n"
        "【法治原则】恪守「法无禁止即可为，法无授权即禁止」，二者方向相反不可混用：\n"
        "1. 私权利（民事主体依法享有的以保障个人需求为核心的人格权、财产权等合法权益）"
        "适用「法无禁止即可为」——法律未明文禁止的行为即属可自由行使的权利；\n"
        "2. 公权力（国家机关及公职人员管理公共事务的法定职权，广义涵盖人大、政协、"
        "一府两院等所有行使国家权力的机关）适用「法无授权即禁止」"
        "——必须有法律明确授权方可为之，无授权则不得剥夺或限制公民权利。\n"
        "二者相辅相成：公民可大胆运用权利，亦可监督政府；政府须谨慎用权，并尊重公民每一项权利。\n"
        "判断时必须先分清讨论对象是私权利还是公权力，再适用对应原则，严禁倒置——"
        "不得对私权利要求「须有授权方可为」，亦不得对公权力放任「未禁止即可为」。\n"
        "据此，当检索到的法条未明文禁止某行为时，不得以「地方规定可能禁止」「现场可能有禁令」"
        "等模糊表述稀释公民权利；要禁止某行为或限制某权利，必须能指出明确的上位法依据。\n"
        "【法律分析方法论】当用户询问「某行为是否合法」「处罚是否正确」「法律依据是否错位」"
        "等定性问题时，必须按以下三步框架分析，严禁跳步或发散：\n"
        "第一步【行为定性】：先判断用户描述的行为本身在法律上是什么性质——是合法行为、"
        "违法行为还是禁止行为？依据哪部法律的哪一条？该条是明文禁止、"
        "有条件允许、还是完全未提及？如果法律完全未提及该行为，"
        "或反向承认该行为可做（如限速条款实质承认可通行），则行为本身不违法。\n"
        "第二步【处罚依据】：处罚决定援引的是哪部法律的哪一条？"
        "该条规定的处罚种类是什么（罚款、拘留、吊销等）？该条是否明确针对用户描述的行为？"
        "如果处罚条款针对的行为与用户描述的行为不匹配，则存在法律依据错位。\n"
        "第三步【错位分析】：行为定性与处罚依据是否匹配？"
        "是否出现「把A法范畴的行为用B法处罚」「把合法行为当违法行为处罚」"
        "「把此违法当彼违法处罚」等错位？错位即违法——公权力必须严格依法，"
        "法无授权即禁止，法律依据错位等于没有法律依据。\n"
        "严禁发散到用户未提及的场景。如果用户问的是「上高速被拘留」，"
        "不要发散到「事故处理」「赔偿调解」「责任认定」等用户未提及的事项；"
        "只聚焦于「上高速这个行为是否违法」和「拘留依据是否匹配」两个核心问题。\n"
        "【思考语言】你的内部思考/推理过程（reasoning）必须全程使用中文，"
        "包括语义解析、知识激活、逻辑拆解、因果推演、候选对比、安全校验等所有环节，"
        "不要使用英文，确保用户展开思考时看到的是完整中文。"
    )


@lru_cache(maxsize=1)
def _off_topic_prompt_template() -> str:
    """构建无相关法条时的拒答模板，泛化领域描述。"""
    return (
        "用户问题：{question}\n\n"
        "系统未检索到与该问题相关的法律法规条文。"
        "请简短说明你只能回答与所支持法律法规相关的问题，不要引用或编造任何法条。"
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
# 道路交通安全 + 立法法/通用法律语境关键词，确保不同法规的短问句都能进入检索
_LAW_KEYWORDS = (
    # 道路交通安全相关
    "法", "交通", "驾驶", "车辆", "机动车", "酒驾", "酒", "事故", "违章",
    "违法", "罚款", "扣分", "驾照", "驾驶证", "行驶证", "行人", "道路",
    "高速", "红绿灯", "信号灯", "限速", "停车", "超速", "逆行", "闯",
    "追尾", "醉驾", "肇事",
    "保险", "责任", "行人", "非机动车", "电动车", "摩托", "头盔", "安全带",
    "调解", "复核", "管辖", "鉴定", "逃逸", "协商", "认定", "赔偿",
    # 立法法 / 通用法律语境
    "立法", "制定", "法规", "规章", "备案", "解释", "效力", "修改", "废止",
    "授权", "条例", "行政法规", "地方性法规", "自治条例", "单行条例",
    "规范性文件", "公布", "施行", "法律案",
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


# 多轮历史感知改写（condense question）：把追问 + 最近历史改写成独立完整的问题，
# 再进入检索与生成；与 GitHub 高星实践（LangChain create_history_aware_retriever）等价
@lru_cache(maxsize=1)
def _context_resolve_system() -> str:
    """历史改写 system prompt，领域泛化为「中国法律法规」。"""
    return (
        "你是一个查询改写助手。请结合对话历史，把用户最新问题改写成一个完整、独立、"
        "无需上下文也能理解的问题。\n规则：\n"
        "1. 把「它」「这个」「那个」等指代词替换为历史中的具体对象\n"
        "2. 补全省略的主语/宾语（如「怎么修」→「XX怎么维修」）\n"
        "3. 领域为中国法律法规，补全时保留法律语境\n"
        "4. 【关键】必须从历史中提取具体场景词（如车辆类型「摩托车」、"
        "地点「高速公路」、处罚「拘留」、地名「东莞」等）补全到改写后的问题中，"
        "确保改写后的问题即使脱离上下文也包含完整场景信息\n"
        "5. 若当前问题已完整独立且包含具体场景，原样输出\n"
        "6. 只输出改写后的问题，不要任何解释，不要思考过程"
    )


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
        rewritten = get_llm().chat(_context_resolve_system(), user_prompt, temperature=0.0)
        rewritten = _sanitize_rewrite(rewritten)
        if not rewritten or len(rewritten) > 200:  # 超长视为解释性输出，改写失败
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
_META_ANSWER_SYSTEM = (
    "你是法律助手小Z。用户正在询问与本次对话本身相关的问题（例如自己刚才问了什么）。"
    "请仅根据提供的对话历史回答，不要编造历史中不存在的内容；"
    "若历史不足以回答，请如实说明。回答使用中文并保持简短。"
)


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


def _build_user_prompt(
    question: str,
    contexts: list[dict],
    document_text: str | None = None,
    document_chunks: list[str] | None = None,
) -> str:
    blocks = []
    if document_chunks is not None:
        for idx, chunk in enumerate(document_chunks, 1):
            blocks.append(f"【用户上传材料·片段{idx}】\n{chunk}")
    elif document_text:
        blocks.append(f"用户上传的材料内容如下：\n{document_text}")
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
        f"请严格依据上述检索到的法条原文，结合用户上传的材料回答，"
        f"回答中不要提及法条的来源（不要说「用户提供」「你提供」等）。"
    )


def answer(question: str, history: list[dict] | None = None, document_text: str | None = None) -> tuple[str, list[Reference]]:
    settings = get_settings()
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
        user_prompt = _off_topic_prompt_template().format(question=question)
        llm_text = get_llm().chat(_system_prompt(), user_prompt)
        return llm_text, []

    # 对话元问题（如「我刚刚的问题是什么」）：仅凭对话历史回答，不检索、不附引用
    if history and _is_conversation_meta(resolved):
        event("conversation_meta", question=resolved)
        llm_text = get_llm().chat(_META_ANSWER_SYSTEM, _history_prompt(question, history))
        return llm_text, []

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
        rewrites = multi_query_rewrite(resolved)
    retrieval_queries = [resolved] + rewrites
    # HyDE：LLM 生成假设性法律回答 → 嵌入向量 → 参与向量检索融合。
    # 合规：假设性文档文本绝不进入最终 context，仅用其向量做检索；
    # 失败时返回 None，跳过 HyDE 路径，行为退化为 P0。
    with span("hyde"):
        hyde_vec = hyde_embed(resolved)
    # rerank 阶段仍使用用户原始独立问题，保持与用户意图对齐
    retrieval_q = resolved
    with span("retrieval", query_count=len(retrieval_queries), top_k=settings.top_k_retrieve, hyde=hyde_vec is not None):
        candidates = engine.multi_query_search(
            retrieval_queries, top_k=settings.top_k_retrieve, hyde_vector=hyde_vec,
        )
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
        contexts = apply_role_adjustment(contexts)

    # 无相关法条：不附带任何引用，由 LLM 简短拒答
    if not contexts:
        event("no_contexts")
        if document_text:
            user_prompt = _build_user_prompt(resolved, [], document_text, document_chunks)
        else:
            user_prompt = _off_topic_prompt_template().format(question=question)
        llm_text = get_llm().chat(_system_prompt(), user_prompt)
        return llm_text, []

    user_prompt = _build_user_prompt(resolved, contexts, document_text, document_chunks)
    with span("llm_generate"):
        llm_text = get_llm().chat(_system_prompt(), user_prompt)

    references = [
        Reference.model_validate({
            "source": c["metadata"].get("source", ""),
            "article_no": c["metadata"].get("article_no", ""),
            "section_header": c["metadata"].get("section_header", ""),
            "text": c["text"],
        })
        for c in contexts
    ]
    return llm_text, references


def answer_stream(question: str, history: list[dict] | None = None, document_text: str | None = None):
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
    history = history or []

    # 无意义输入（无历史时的单字 / 纯数字 / 问候 / 短词无法律关键词）：
    # 不发预热思考，直接走拒答分支，前端不会出现思考区域
    # 若有上传文件内容，跳过 trivial 拦截，允许结合材料作答
    if not history and _is_trivial_query(question) and not document_text:
        event("trivial_reject", question=question)
        yield {"type": "references", "references": []}
        user_prompt = _off_topic_prompt_template().format(question=question)
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
            user_prompt = _off_topic_prompt_template().format(question=question)
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

    # Step 1: 材料检索（长材料切块取 top3，短材料返回 None 以全文注入）
    document_chunks: list[str] | None = None
    if document_text:
        yield {"type": "progress", "content": "正在定位材料相关内容..."}
        with span("upload_retrieval.select"):
            document_chunks = select_relevant_chunks(resolved, document_text)

    # Step 2: 法条检索
    yield {"type": "progress", "content": "正在检索相关法条..."}
    engine = get_retrieval_engine()
    # Multi-Query 改写：LLM 把口语化问题改写为多视角检索查询（仅语言层操作）。
    # 改写查询仅用于检索，不进入最终 context；失败时回退到原问题单路检索。
    with span("multi_query_rewrite"):
        rewrites = multi_query_rewrite(resolved)
    retrieval_queries = [resolved] + rewrites
    # HyDE：LLM 生成假设性法律回答 → 嵌入向量 → 参与向量检索融合。
    # 合规：假设性文档文本绝不进入最终 context，仅用其向量做检索；
    # 失败时返回 None，跳过 HyDE 路径，行为退化为 P0。
    with span("hyde"):
        hyde_vec = hyde_embed(resolved)
    # rerank 阶段仍使用用户原始独立问题，保持与用户意图对齐
    retrieval_q = resolved
    with span("retrieval", query_count=len(retrieval_queries), top_k=settings.top_k_retrieve, hyde=hyde_vec is not None):
        candidates = engine.multi_query_search(
            retrieval_queries, top_k=settings.top_k_retrieve, hyde_vector=hyde_vec,
        )
    event("retrieval.candidates", count=len(candidates))

    # Step 2: 重排（搜索阶段统一显示"正在搜索..."，不暴露候选数等内部细节）
    reranker = get_reranker()
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
        contexts = apply_role_adjustment(contexts)

    references = [
        Reference.model_validate({
            "source": c["metadata"].get("source", ""),
            "article_no": c["metadata"].get("article_no", ""),
            "section_header": c["metadata"].get("section_header", ""),
            "text": c["text"],
        })
        for c in contexts
    ]
    yield {"type": "references", "references": [r.model_dump() for r in references]}

    if not contexts:
        event("no_contexts")
        if document_text:
            user_prompt = _build_user_prompt(resolved, [], document_text, document_chunks)
        else:
            user_prompt = _off_topic_prompt_template().format(question=question)
        for kind, text in get_llm().chat_stream(_system_prompt(), user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
        return

    # Step 3: LLM 生成（使用改写后的独立问题，与 LangChain rephrase_question=True 一致）
    yield {"type": "progress", "content": "正在思考回答..."}
    user_prompt = _build_user_prompt(resolved, contexts, document_text, document_chunks)
    with span("llm_generate"):
        for kind, text in get_llm().chat_stream(_system_prompt(), user_prompt):
            if kind == "reasoning":
                yield {"type": "reasoning", "content": text}
            else:
                yield {"type": "delta", "content": text}
    event("done", total_ms=round((time.perf_counter() - t0) * 1000, 2))