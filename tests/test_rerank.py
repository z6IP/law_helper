"""重排序配额保障与受保护条款豁免测试。

覆盖 min_score 豁免、配额保障 overflow 补入、_all_scored 回退、instruct 配置读取。
"""
from __future__ import annotations

import types

import app.rerank as rerank_module
from app.rerank import Reranker


def _make_reranker(monkeypatch, results):
    """构造一个已加载的 Reranker，mock requests.post 返回指定 rerank 结果。"""
    monkeypatch.setattr(
        rerank_module,
        "get_settings",
        lambda: types.SimpleNamespace(rerank_model_id="qwen3.7-text-rerank"),
    )
    resp = types.SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"output": {"results": results}},
    )
    monkeypatch.setattr(rerank_module.requests, "post", lambda *a, **k: resp)
    reranker = Reranker()
    # 直接注入 endpoint 与 key，绕过 _ensure_loaded 的 .env 校验
    reranker._endpoint = "http://test"
    reranker._api_key = "test_key"
    return reranker


def _candidate(cid, text, section_header, source="测试法"):
    return {
        "id": cid,
        "text": text,
        "metadata": {
            "source": source,
            "section_header": section_header,
            "article_no": cid,
        },
    }


def test_protected_role_exempts_min_score(monkeypatch):
    """受保护条款（definition）完全豁免 min_score，低分仍被保留。"""
    candidates = [
        _candidate("substantive", "道路通行的一般规定", "第四章 道路通行规定"),
        _candidate(
            "definition_low",
            "治安管理处罚的种类分为：警告、罚款、行政拘留、吊销许可证。",
            "第二章 处罚的种类和适用",
        ),
    ]
    results = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.01},  # 远低于 min_score=0.2
    ]
    reranker = _make_reranker(monkeypatch, results)

    out = reranker.rerank("摩托车上高速被拘留", candidates, top_n=2, min_score=0.2)

    ids = [c["id"] for c in out]
    assert "definition_low" in ids  # 受保护条款不被 min_score 过滤


def test_definition_quota_inserted_from_overflow(monkeypatch):
    """top 中缺少受保护条款时，从 overflow 中补入 definition 条款。"""
    candidates = [
        _candidate("s1", "道路通行规定", "第四章 道路通行规定", source="同一法"),
        _candidate("s2", "法律责任规定", "第七章 法律责任", source="同一法"),
        _candidate(
            "d1",
            "治安管理处罚的种类分为：警告、罚款、行政拘留。",
            "第二章 处罚的种类和适用",
            source="同一法",
        ),
    ]
    results = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.8},
        {"index": 2, "relevance_score": 0.3},  # definition 低分，落在 overflow
    ]
    reranker = _make_reranker(monkeypatch, results)

    out = reranker.rerank("摩托车上高速被拘留", candidates, top_n=2, min_score=0.2)

    assert "d1" in [c["id"] for c in out]  # 配额保障把 definition 补入 top


def test_behavior_penalty_quota_fallback_to_all_scored(monkeypatch):
    """behavior_penalty 被 behavior_penalty_max 过滤出 scored 后，通过 _all_scored 回退补入。"""
    section = "第三章 第一节 扰乱公共秩序的行为和处罚"
    candidates = [
        _candidate("bp1", "有违反治安管理行为的，处警告。", section, source="治安法"),
        _candidate("bp2", "有违反治安管理行为的，处警告。", section, source="治安法"),
        # 第 3 条被 behavior_penalty_max=2 过滤出 scored，但文本含"拘留"是关键条款
        _candidate("bp3", "扰乱公共秩序的，处拘留。", section, source="治安法"),
    ]
    results = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.8},
        {"index": 2, "relevance_score": 0.7},
    ]
    reranker = _make_reranker(monkeypatch, results)

    out = reranker.rerank("摩托车上高速被拘留", candidates, top_n=2, min_score=0.2)

    assert "bp3" in [c["id"] for c in out]  # 从 _all_scored 回退找回含"拘留"的行为条款


def test_no_concept_keyword_skips_quota(monkeypatch):
    """查询不含概念词时，不触发处罚种类定义/行为量刑配额保障。"""
    candidates = [
        _candidate("s1", "道路通行规定", "第四章 道路通行规定"),
        _candidate("s2", "法律责任规定", "第七章 法律责任"),
    ]
    results = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.8},
    ]
    reranker = _make_reranker(monkeypatch, results)

    out = reranker.rerank("摩托车如何通行", candidates, top_n=2, min_score=0.2)

    # 查询不含概念词，结果保持 rerank 原始顺序，不额外补入任何条款
    assert [c["id"] for c in out] == ["s1", "s2"]


def test_instruct_from_policy(monkeypatch):
    """rerank 请求的 instruct 使用中文法律检索指令（policy 配置化）。"""
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"output": {"results": []}},
        )

    monkeypatch.setattr(rerank_module.requests, "post", fake_post)
    monkeypatch.setattr(
        rerank_module,
        "get_settings",
        lambda: types.SimpleNamespace(rerank_model_id="qwen3.7-text-rerank"),
    )
    reranker = Reranker()
    reranker._endpoint = "http://test"
    reranker._api_key = "test_key"

    reranker.rerank(
        "拘留",
        [_candidate("x", "条款内容", "第二章 处罚的种类和适用")],
        top_n=1,
    )

    instruct = captured["payload"]["parameters"]["instruct"]
    assert "处罚种类" in instruct
    assert "行为规定" in instruct
