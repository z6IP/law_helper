from types import SimpleNamespace

from app.config import get_settings
from app.ingestion import Article, _expand_article_chunks, _parent_article_no
from app.policy import get_policy
from app.qa import _restrict_to_primary_source
from app.query_expansion import classify_query_intents, expand_query_by_intent
from app.rerank import apply_role_adjustment
from app.retrieval import _tokenize


def test_clause_chunks_keep_parent_and_add_child_chunks():
    article = Article(
        article_no="第九十条",
        section_header="第一章 总则",
        text="（一）违反规定的，处警告。\n（二）情节严重的，处以罚款。",
    )

    chunks = _expand_article_chunks([article])

    assert [chunk.article_no for chunk in chunks] == [
        "第九十条",
        "第九十条第一款",
        "第九十条第二款",
    ]
    assert chunks[0].text == article.text
    assert "处以罚款" in chunks[2].text
    assert _parent_article_no(chunks[2].article_no) == "第九十条"


def test_clause_chunks_inherit_parent_penalty_context():
    """子款继承父条处罚前置句，解决子款正文不含处罚词导致的召回失败。"""
    article = Article(
        article_no="第二十六条",
        section_header="第三章 第一节 扰乱公共秩序的行为和处罚",
        text=(
            "有下列行为之一的，处五日以上十日以下拘留，可以并处五百元以下罚款：\n"
            "（一）扰乱机关、团体、企业、事业单位秩序，致使工作不能正常进行，尚未造成严重损失的；\n"
            "（二）扰乱车站、港口、码头、机场、商场、公园、展览馆或者其他公共场所秩序的；"
        ),
    )

    chunks = _expand_article_chunks([article])

    # 父条不携带 penalty_context
    assert chunks[0].penalty_context == ""
    # 子款正文不含处罚词，但继承的 penalty_context 含处罚前置句
    assert "拘留" not in chunks[1].text
    assert "扰乱机关" in chunks[1].text
    assert "处五日以上十日以下拘留" in chunks[1].penalty_context
    # 所有子款共享同一处罚上下文
    assert chunks[2].penalty_context == chunks[1].penalty_context


def test_intent_expansion_covers_multiple_legal_intents():
    intents, expansions = expand_query_by_intent("闯红灯怎么处罚，事故责任怎么认定？")

    assert intents == ["penalty", "liability"]
    assert "处罚依据" in expansions[0]
    assert any("事故责任认定" in expansion for expansion in expansions)
    assert classify_query_intents("") == []


def test_primary_source_filter_only_applies_when_explicitly_requested():
    candidates = [
        {"id": "a", "metadata": {"source": "法规A"}},
        {"id": "b", "metadata": {"source": "法规B"}},
    ]

    assert _restrict_to_primary_source(candidates, None) == candidates
    assert _restrict_to_primary_source(candidates, "法规B") == [candidates[1]]


def test_role_adjustment_requires_matching_query_intent():
    contexts = [
        {"metadata": {"section_header": "程序"}, "rerank_score": 0.9},
        {"metadata": {"section_header": "行为和处罚"}, "rerank_score": 0.5},
    ]

    assert apply_role_adjustment(contexts) == contexts
    assert apply_role_adjustment(contexts, ["penalty"]) == [contexts[1], contexts[0]]


def test_legal_tokenizer_keeps_professional_terms():
    """R1：加载用户词典后，法律/交通专业术语应被切分为整词。"""
    tokens = _tokenize("机动车驾驶人闯红灯且未戴安全头盔")
    assert "机动车驾驶人" in tokens
    assert "闯红灯" in tokens
    assert "未戴安全头盔" in tokens


def test_policy_has_r7_retrieval_parameters():
    """R7：policy.json 中应包含新下放的多路检索参数。"""
    policy = get_policy()["retrieval"]
    required_keys = (
        "hnsw_construction_ef",
        "hnsw_search_ef",
        "hnsw_M",
        "protected_roles",
        "penalty_signal_markers",
        "concept_score_boost",
        "route_floor_min",
        "route_floor_max",
        "protected_search_multiplier",
        "protected_path_rrf_weight",
        "fusion_candidate_cap_multiplier",
        "same_law_reference_patterns",
        "cross_law_reference_pattern",
    )
    for key in required_keys:
        assert key in policy, f"{key} should be in policy.json retrieval"


def test_settings_has_r7_retrieval_parameters(monkeypatch):
    """R7：Settings 应支持通过环境变量覆盖 R7 参数。"""
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
    get_settings.cache_clear()
    settings = get_settings()
    for attr in (
        "hnsw_construction_ef",
        "hnsw_search_ef",
        "hnsw_M",
        "route_floor_min",
        "route_floor_max",
        "concept_score_boost",
        "protected_path_rrf_weight",
        "protected_search_multiplier",
        "fusion_candidate_cap_multiplier",
        "rerank_candidate_limit",
    ):
        assert hasattr(settings, attr), f"Settings should have {attr}"
    get_settings.cache_clear()


def test_policy_has_r4_r5_r9_parameters():
    """R4/R5/R9：policy.json 中应包含同义词表、层级展开配额、动态权重参数。"""
    policy = get_policy()["query"]
    assert "synonym_map" in policy, "synonym_map should be in policy.json query"
    assert "闯红灯" in policy["synonym_map"]

    retrieval = get_policy()["retrieval"]
    assert "article_hierarchy_expand_quota" in retrieval
    assert retrieval["article_hierarchy_expand_quota"] > 0
    assert "dynamic_weights" in retrieval
    assert "definition" in retrieval["dynamic_weights"]
    assert "bm25_weight" in retrieval["dynamic_weights"]["definition"]


def test_resolve_bm25_weight_by_intent():
    """R9：根据查询意图动态选择 BM25 权重。"""
    from app.retrieval import _resolve_bm25_weight

    # 概念/定义查询偏向语义，BM25 权重应降低
    assert _resolve_bm25_weight("什么是行政拘留") == 0.3
    # 处罚/责任查询偏向关键词精确命中，BM25 权重应提高
    assert _resolve_bm25_weight("闯红灯怎么处罚") == 0.7
    # 普通查询回退到 Settings 默认值
    assert _resolve_bm25_weight("你好") == get_settings().bm25_weight


def test_expand_article_hierarchy():
    """R5：命中父条时补充子款，命中子款时带回父条。"""
    from app.retrieval import RetrievalEngine

    engine = RetrievalEngine()
    engine._loaded = True
    engine._ids = ["p1", "c1", "c2", "p2"]
    engine._documents = ["父条1全文", "父条1第一款", "父条1第二款", "父条2全文"]
    engine._metadatas = [
        {"source": "A", "article_no": "第一条", "parent_article_no": "第一条", "chunk_type": "article"},
        {"source": "A", "article_no": "第一条第一款", "parent_article_no": "第一条", "chunk_type": "clause"},
        {"source": "A", "article_no": "第一条第二款", "parent_article_no": "第一条", "chunk_type": "clause"},
        {"source": "A", "article_no": "第二条", "parent_article_no": "第二条", "chunk_type": "article"},
    ]

    # 命中父条 1，应补充两个子款
    hits = [{"id": "p1", "metadata": engine._metadatas[0], "score": 1.0}]
    expanded = engine._expand_article_hierarchy(hits, quota=8)
    assert len(expanded) == 2
    assert {h["id"] for h in expanded} == {"c1", "c2"}

    # 命中子款 c1，应带回父条 p1
    hits = [{"id": "c1", "metadata": engine._metadatas[1], "score": 1.0}]
    expanded = engine._expand_article_hierarchy(hits, quota=8)
    assert len(expanded) == 1
    assert expanded[0]["id"] == "p1"

    # 配额为 0 时不展开
    assert engine._expand_article_hierarchy(hits, quota=0) == []

    # 受 article_no 过滤约束
    expanded = engine._expand_article_hierarchy(hits, quota=8, article_no="第二条")
    assert expanded == []


def test_ensure_parent_articles_adds_missing_parent():
    """子款在 contexts 中但父条不在时，从 candidates 补回父条。"""
    from app.qa import _ensure_parent_articles

    candidates = [
        {"id": "c1", "text": "（八）不按信号灯通行", "metadata": {
            "source": "记分办法", "article_no": "第十条第八款",
            "parent_article_no": "第十条", "chunk_type": "clause"}},
        {"id": "p1", "text": "一次记6分：...", "metadata": {
            "source": "记分办法", "article_no": "第十条",
            "parent_article_no": "第十条", "chunk_type": "article"}},
    ]
    contexts = [candidates[0]]  # 只有子款

    out = _ensure_parent_articles(candidates, contexts)

    assert len(out) == 2
    assert "p1" in {c["id"] for c in out}


def test_inject_penalty_context_when_parent_missing():
    """父条不在 contexts 中时，给子款注入 penalty_context。"""
    from app.qa import _inject_penalty_context

    contexts = [{
        "id": "c1",
        "text": "（八）驾驶机动车不按交通信号灯指示通行的；",
        "metadata": {
            "source": "记分办法", "article_no": "第十条第八款",
            "parent_article_no": "第十条", "chunk_type": "clause",
            "penalty_context": "机动车驾驶人有下列交通违法行为之一，一次记6分：",
        },
    }]

    out = _inject_penalty_context(contexts)

    assert "一次记6分" in out[0]["text"]
    assert out[0]["text"].startswith("机动车驾驶人有下列交通违法行为之一，一次记6分")


def test_inject_penalty_context_skips_when_parent_present():
    """父条已在 contexts 中时，不重复注入 penalty_context。"""
    from app.qa import _inject_penalty_context

    contexts = [
        {"id": "p1", "text": "机动车驾驶人有下列交通违法行为之一，一次记6分：（一）...", "metadata": {
            "source": "记分办法", "article_no": "第十条",
            "parent_article_no": "第十条", "chunk_type": "article"}},
        {"id": "c1", "text": "（八）驾驶机动车不按交通信号灯指示通行的；", "metadata": {
            "source": "记分办法", "article_no": "第十条第八款",
            "parent_article_no": "第十条", "chunk_type": "clause",
            "penalty_context": "机动车驾驶人有下列交通违法行为之一，一次记6分："}},
    ]

    out = _inject_penalty_context(contexts)

    # 子款 text 不应被注入 penalty_context（父条已在）
    assert out[1]["text"] == "（八）驾驶机动车不按交通信号灯指示通行的；"
