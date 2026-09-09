"""查询改写：把口语化问题扩展为更贴近法条表述的检索查询。

历史：曾采用「规则词典 + 关键词增强」实现硬编码扩展，针对转弯让直行、
追尾、逃逸、酒驾、闯红灯、酒精换算、地方规定冲突等场景注入法条术语。

现状：多路检索（retrieval.py 按 source 分路并行检索）已从根本上解决
跨法规召回覆盖问题，每个法规都有候选进入重排阶段，重排模型基于语义
相关性打分，不再需要人工注入术语来「拉」特定法规的召回分。

因此 expand_query 现直接返回原问题，保留函数签名兼容历史调用。
原 _RULES 数据结构与规则定义保留备查，未来若多路检索效果不足仍可回退。

新增 multi_query_rewrite：LLM 多查询改写（Multi-Query Retrieval），
针对用户口语化提问与法条术语不匹配的问题。LLM 只做语言层改写，
不判断法律对错，不提供法律结论，不编造法条编号。

新增 hyde_embed：HyDE 假设性文档嵌入（Hypothetical Document Embedding），
LLM 生成假设性法律回答 → 用 qwen3.7-text-embedding 编码 → 返回向量。
合规约束：假设性文档文本绝不进入最终 context，仅用于生成检索向量；
LLM 失败或 embedding 失败时返回 None，调用方跳过 HyDE 路径。
"""
from __future__ import annotations

import re

from app.errors import LawHelperError
from app.llm import get_llm
from app.tracing import event

# 历史规则定义（保留备查，当前未启用）
# 每条规则：(触发词, 必需词, 增强片段)
_RULES: list[tuple[tuple[str, ...], tuple[str, ...], str]] = [
    # 转弯 vs 直行：责任划分类问题 → 实施条例第51/52条「转弯让直行」
    (
        ("左转", "右转", "转弯", "掉头", "转向"),
        ("直行", "相撞", "碰撞", "撞车", "撞", "责任", "让行", "先行", "谁让", "让不让"),
        "转弯的机动车让直行的车辆先行",
    ),
    # 追尾 → 法第43条「保持安全距离」
    (
        ("追尾",),
        (),
        "后车未与前车保持安全距离 责任认定",
    ),
    # 肇事逃逸 → 法第99条第(三)项
    (
        ("逃逸", "逃跑", "跑了", "逃走"),
        ("撞", "事故", "伤", "人", "死"),
        "造成交通事故后逃逸",
    ),
    # 酒驾 / 醉驾 → 法第91条 + 记分办法第八条(一) + GB 19522-2024
    (
        ("酒驾", "酒后开车", "喝酒开车", "喝完酒开车", "醉驾", "醉酒驾驶", "醉酒开车"),
        (),
        "饮酒后驾驶机动车 醉酒驾驶机动车 处罚 记分 一次记12分 饮酒后驾驶机动车 车辆驾驶人员血液酒精含量阈值 饮酒后驾车 ≥20,<80mg/100mL 醉酒驾车 ≥80mg/100mL 血液与呼气酒精含量换算 1:2200",
    ),
    # 闯红灯 → 法第90条 + 记分办法第10条(八) + 条例第38条
    # 拆为多条独立扩展：每条精确命中一个法条，避免 BM25 在长正文条款上被稀释
    (
        ("闯红灯", "闯红灯了", "红灯闯"),
        (),
        "机动车驾驶人违反道路交通安全法律、法规关于道路通行规定 处警告或者二十元以上二百元以下罚款",  # 命中道交法第九十条
    ),
    (
        ("闯红灯", "闯红灯了", "红灯闯"),
        (),
        "驾驶机动车不按交通信号灯指示通行 一次记6分",  # 命中记分管理办法第十条第八项
    ),
    (
        ("闯红灯", "闯红灯了", "红灯闯"),
        (),
        "行人通过路口或者横过道路 应当走人行横道 按交通信号灯指示通行",  # 命中道交法第六十二条行人闯红灯
    ),
    # GB 19522-2024 血液/呼气酒精含量阈值与换算
    (
        ("血液", "呼气", "吹气", "抽血", "酒精", "酒驾", "醉驾", "饮酒驾驶", "醉酒驾驶"),
        ("换算", "转换", "对应", "相当", "等于", "阈值", "标准", "mg", "2200"),
        "车辆驾驶人员血液、呼气酒精含量阈值 血液酒精含量 呼气酒精含量 1:2200 换算",
    ),
    # 地方性规定 / 地方与上位法冲突 → 道交法第67条 + 立法法第五章
    (
        ("地方", "地方政府", "地方规定", "本地", "各地", "省市", "省份", "当地"),
        ("不让", "禁止", "限制", "不能", "不许", "不准", "允许", "可以", "规定",
         "不同", "差异", "冲突", "不一致", "区别", "特殊", "例外"),
        "地方性法规 行政法规 效力 上位法 下位法 适用 备案审查 改变 撤销 高速公路 机动车 不得进入 设计最高时速",
    ),
]


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
    expansions: list[str] = []
    seen: set[str] = set()
    for triggers, requires, enhancement in _RULES:
        if not any(t in q for t in triggers):
            continue
        if requires and not any(r in q for r in requires):
            continue
        if enhancement in seen:
            continue
        seen.add(enhancement)
        expansions.append(enhancement)
    return expansions


# Multi-Query 改写 system prompt
# 合规约束（对应研究报告 0.1 / 5.1）：
#   - 只做语言层改写，不判断法律对错
#   - 不得添加法律结论（如「该行为违法」）
#   - 不得编造具体法条编号
#   - 只是把口语化表述改写为更接近法条术语的检索查询
_MULTI_QUERY_SYSTEM = (
    "你是检索查询改写助手。请把用户问题改写为 3 个不同视角的查询，"
    "用于在法律法规条文库中检索。\n"
    "【严格约束】\n"
    "1. 只做语言层改写，不判断法律对错\n"
    "2. 不得添加法律结论（如「该行为违法」「该行为合法」）\n"
    "3. 不得使用「违法」「合法」「犯罪」「无罪」等法律判断词，"
    "改用中性描述如「违反」「行为」「处罚」「责任」\n"
    "4. 不得编造具体法条编号\n"
    "5. 只是把用户口语化表述改写为更接近法条术语的检索查询\n"
    "6. 每行一条改写，不要编号、不要解释、不要引号\n"
    "【通用处罚依据扩展】当用户问题涉及「XX怎么判」「XX怎么处罚」「XX被罚款」「XX被拘留」"
    "等行为+处罚类查询时，除了字面同义替换外，至少一条改写应扩展到通用处罚条款的关键词，"
    "帮助 BM25 匹配通用处罚依据条款。通用处罚条款通常表述为：\n"
    "- 「违反道路通行规定 处警告 罚款」（道交法通用处罚条款）\n"
    "- 「记分 分值 累积记分制度」（记分条款）\n"
    "- 「行政拘留 处罚种类」（处罚种类定义条款）\n"
    "- 「扣留机动车 扣留驾驶证」（行政强制措施条款）\n"
    "改写示例：\n"
    "- 原：「闯红灯怎么判」→\n"
    "  违反交通信号灯 不按信号灯指示通行 处罚\n"
    "  违反道路通行规定 处警告 罚款 记分\n"
    "  红灯 禁止通行 信号灯 违反规定\n"
    "- 原：「东莞摩托车上高速拘留」→\n"
    "  摩托车 高速公路 行驶 拘留\n"
    "  两轮摩托车 高速公路 载人 处罚\n"
    "  扰乱秩序 冲卡 阻碍执法 拘留 处罚\n"
    "  违反道路通行规定 处警告 罚款 扣留机动车\n"
    "- 原：「离婚房产怎么分」→\n"
    "  离婚 夫妻共同财产 分割\n"
    "  离婚 婚姻关系存续期间 财产\n"
    "  离婚 房产 分割 法律规定"
)


def multi_query_rewrite(question: str, n: int = 3) -> list[str]:
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

    user_prompt = f"用户问题：{q}\n\n请输出 {n} 条改写查询，每行一条："
    try:
        raw = get_llm().chat(_MULTI_QUERY_SYSTEM, user_prompt, temperature=0.0)
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


# HyDE system prompt
# 合规约束（对应研究报告 0.2 / 5.1.2）：
#   - LLM 生成的假设性文档仅用于生成检索向量，不进入最终 context
#   - 最终回答完全不参考假设性文档内容
#   - 假设性文档可能法律错误，但 HyDE 论文证明在线 Embedding 模型会过滤错误细节
_HYDE_SYSTEM = (
    "请针对用户提出的法律问题，生成一段假设性的法律回答文档。"
    "这段文档将仅用于生成检索向量以查找真实法条，不会出现在最终回答中，"
    "因此即使你对法律细节不确定也可以输出，但请尽量贴近法律条文的表述方式。\n"
    "【约束】\n"
    "1. 输出 100-200 字的连贯文字，不要分条编号\n"
    "2. 描述与该问题可能相关的法律概念、行为类型、处罚种类，"
    "并提及相关定义性概念（如「拘留的定义」「处罚的种类」）\n"
    "3. 不要引用具体法条编号（你不确定编号是否正确）\n"
    "4. 不要使用「违法」「合法」「犯罪」等法律判断词\n"
    "5. 不要输出「该行为违法/合法」等最终法律结论\n"
    "6. 只输出假设性文档正文，不要任何前缀或解释"
)


def hyde_embed(question: str) -> list[float] | None:
    """HyDE 假设性文档嵌入：返回假设性回答的向量（不返回文本）。

    合规约束（极重要）：
    - LLM 生成的假设性文档文本绝不进入最终 context
    - 仅返回归一化后的向量供 retrieval 做向量检索
    - 失败时返回 None，调用方跳过 HyDE 路径，行为退化为 P0

    Args:
        question: 用户原始问题（已做多轮历史改写后的独立问题）

    Returns:
        归一化向量 list[float]；LLM 或 embedding 失败时为 None
    """
    q = (question or "").strip()
    if not q:
        return None

    try:
        # 延迟导入避免 query_expansion ↔ retrieval ↔ embeddings 循环依赖
        from app.embeddings import get_embedding_model

        hyde_text = get_llm().chat(_HYDE_SYSTEM, q, temperature=0.0)
    except LawHelperError:
        event("hyde.failed", reason="llm_error")
        return None

    hyde_text = (hyde_text or "").strip()
    if not hyde_text:
        event("hyde.failed", reason="empty_output")
        return None

    try:
        vec = get_embedding_model().embed_query(hyde_text)
    except Exception:  # noqa: BLE001 - embedding API 异常统一降级
        event("hyde.failed", reason="embedding_error")
        return None

    # 不记录 hyde_text 本身，避免 LLM 知识泄露到日志
    event("hyde.done", vector_dim=len(vec))
    return vec
