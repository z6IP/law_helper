"""问答编排：检索 → 重排 → 生成 → 引用。"""
from __future__ import annotations

import re
import time
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.errors import LawHelperError
from app.llm import get_llm
from app.query_expansion import expand_query, hyde_embed, multi_query_rewrite
from app.retrieval import _tokenize, get_retrieval_engine
from app.rerank import apply_role_adjustment, get_reranker
from app.schemas import Reference
from app.tracing import event, span
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

    EXPANSION_ENSURE_QUOTA = 3
    EXPANSION_ENSURE_MIN_BM25 = 20.0
    existing_ids = {c.get("id") for c in contexts}
    ensured = 0
    for exp_q in expansions:
        if ensured >= EXPANSION_ENSURE_QUOTA:
            break
        bm25_scores = np.asarray(engine._bm25.get_scores(_tokenize(exp_q)))
        top_indices = np.argsort(-bm25_scores)[:2]
        for top_idx in top_indices:
            if ensured >= EXPANSION_ENSURE_QUOTA:
                break
            top_idx = int(top_idx)
            if float(bm25_scores[top_idx]) < EXPANSION_ENSURE_MIN_BM25:
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
        "系统会根据用户问题自动检索相关法条原文并提供给你，"
        "请严格依据检索到的法条回答，不要编造法条内容。\n"
        "【回答结构】遵循「结论先行 + 法条支撑 + 实操建议」：\n"
        "1. 结论：针对用户问题给出明确、中肯的判断和态度（如是否合法、可能涉及哪些情形、核心结论是什么），"
        "不是单纯罗列法条处罚。结论中应包含核心处罚内容（罚款数额、记分、拘留天数等具体后果），"
        "但处罚内容是支撑判断的依据，不能替代判断本身。不要把处罚数额挪到实操建议里；\n"
        "2. 法条支撑：只写处罚依据（条号+简述处罚内容），不写「行为定性」「匹配分析」等标签；\n"
        "3. 实操建议：给出可操作的建议或后续步骤，不要重复结论中已给出的罚款数额。\n"
        "结论必须涵盖问题的全部情形——当同一行为因主体不同（机动车/非机动车/行人）"
        "或情节不同而适用不同法条时，必须分情形分别说明，不得遗漏。"
        "但当多个主体处罚完全相同时（如行人与非机动车驾驶人同适用第八十九条），"
        "应合并为一条写，不要重复列示相同处罚。\n"
        "【依据不足】当检索到的法条不足以回答时，逐条分析每条法条与问题的关系"
        "（直接适用/间接相关/不相关及原因），再说明依据不足并指引应查询哪部法。\n"
        "【定义性条款】检索结果中的定义性/种类性条款（如处罚种类定义）"
        "是判断处罚合法性的根本依据，应在分析处罚性质时主动引用。\n"
        "【法治原则】恪守「法无禁止即可为，法无授权即禁止」："
        "对私权利（公民/驾驶人）适用「法无禁止即可为」——法律没有禁止的行为，公民有权为之，"
        "行政机关不得处罚；对公权力（公安机关）适用「法无授权即禁止」——"
        "行政机关必须有明确的法律授权才能实施处罚，尤其是行政拘留这类限制人身自由的处罚。"
        "两者相辅相成，不可倒置。"
        "当问题涉及处罚合法性判断（如「XX被拘留/罚款是否合法」「怎么看XX被处罚」）时，"
        "必须在结论中同时运用原则的两侧进行分析："
        "先从私权利侧指出「法无禁止即可为」——若检索到的法条未明确禁止该行为，"
        "则公民有权为之，行政机关不得以此为由处罚；"
        "再从公权力侧指出「法无授权即禁止」——行政机关必须有明确的法律授权才能实施处罚，"
        "尤其是行政拘留这类限制人身自由的处罚，法无明文规定不得拘留。\n"
        "【法律分析】思考时按三步推理：行为定性→处罚依据（法条含多款/项时精确到款/项）→"
        "匹配分析（行为与处罚是否对应）。"
        "当检索到同一法条的多款/项时，必须逐一评估每款/项与用户行为的关系，"
        "引用所有相关的款/项，不得只选一款而忽略其他相关款/项。"
        "引用某项时，必须核对该项描述的行为主体/对象与用户描述是否一致，不得张冠李戴："
        "例如《治安管理处罚法》第七十六条第（一）项是「偷开他人机动车」，"
        "第（二）项是「偷开他人航空器、机动船舶」，摩托车属于机动车，"
        "偷开摩托车只能适用第（一）项，不能引用第（二）项。"
        "示例：摩托车冲卡闯入高速公路收费站→可能同时适用《治安管理处罚法》第二十六条"
        "第（一）项（扰乱企业、事业单位秩序，收费站运营单位属于企业事业单位）"
        "和第（四）项（非法拦截或者强登、扒乘机动车），应分别说明。"
        "引用某法条时若其他法规对其适用有交集性限制应一并说明。"
        "检索到的法条凡与问题相关的，都应在「法条支撑」中引用，不得遗漏。"
        "但答案中只输出处罚依据，不输出「行为定性」「匹配分析」标签。"
        "只聚焦用户描述的行为和检索到的法条，严禁发散到未提及的场景。\n"
        "【思考方式】思考是推理过程，不是答案草稿。禁止以下行为：\n"
        "- 复述检索到的法条原文（你已能看到 context，不要抄）\n"
        "- 草拟答案文本（不要写「草拟结论：」「草拟法条支撑：」等）\n"
        "- 检查约束条件（不要写「检查约束：结论先行：有」等）\n"
        "- 多轮循环（不要调整→整合→检查→再检查）\n"
        "正确方式：用精炼的分析语言记录推理，每步 1-2 句话：\n"
        "1. 行为定性（法律性质）\n"
        "2. 法条匹配（哪条适用，精确到款）\n"
        "3. 分情形（主体/情节不同时分别列出）\n"
        "推理完成即可输出答案。\n"
        "【表述】法条来源用「根据检索到的法条」或「根据《xxx》」，"
        "不要出现「你提供的法条」等表述。用「你」指代用户。\n"
        "【思考语言】内部思考（reasoning）全程使用中文。"
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
    # 关键词注入兜底：对触发词命中的查询（如"闯红灯"）注入法条术语，
    # 解决 BM25 与法条正文无词项交集导致的召回失败（闯红灯→道交法第九十条）。
    # 扩展查询放在 rewrites 之前：让它优先享受 retrieval.py REWRITE_QUOTA 配额，
    # 避免 LLM 改写查询占满 3 条配额后扩展查询的关键条款被挤出候选池。
    # expand_query 可能返回多条扩展（闯红灯命中道交法90条+记分办法10条+道交法62条），
    # 每条精确命中一个法条，避免单条扩展关键词被长正文条款稀释。
    expansions = expand_query(resolved)
    retrieval_queries = [resolved] + expansions + rewrites
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

    # 扩展查询关键条款保底：见 _ensure_expansion_hits 文档字符串
    contexts = _ensure_expansion_hits(expansions, candidates, contexts, engine)

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
    # 关键词注入兜底：对触发词命中的查询（如"闯红灯"）注入法条术语，
    # 解决 BM25 与法条正文无词项交集导致的召回失败（闯红灯→道交法第九十条）。
    # 扩展查询放在 rewrites 之前：让它优先享受 retrieval.py REWRITE_QUOTA 配额，
    # 避免 LLM 改写查询占满 3 条配额后扩展查询的关键条款被挤出候选池。
    # expand_query 可能返回多条扩展（闯红灯命中道交法90条+记分办法10条+道交法62条），
    # 每条精确命中一个法条，避免单条扩展关键词被长正文条款稀释。
    expansions = expand_query(resolved)
    retrieval_queries = [resolved] + expansions + rewrites
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

    # 扩展查询关键条款保底：见 _ensure_expansion_hits 文档字符串
    contexts = _ensure_expansion_hits(expansions, candidates, contexts, engine)

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