"""问答编排核心函数的单元测试（A1/A4/A5/A9）。"""
from __future__ import annotations

import pytest

from app.qa import (
    _article_group_key,
    _classify_refusal,
    _extract_clause_label,
    _format_source_with_marks,
    _merge_references,
    _normalize_source_name,
    _self_check_answer,
)


class TestNormalizeSourceName:
    """A9：法规名称清洗（不添加书名号）。"""

    def test_removes_date_suffix(self):
        assert _normalize_source_name("道路交通安全法_20210429") == "道路交通安全法"

    def test_converts_plus_to_space(self):
        assert _normalize_source_name("GB+19522-2024") == "GB 19522-2024"

    def test_keeps_source_unchanged(self):
        assert _normalize_source_name("中华人民共和国道路交通安全法") == "中华人民共和国道路交通安全法"


class TestFormatSourceWithMarks:
    """A9：法规引用格式统一（展示层加书名号）。"""

    def test_adds_book_title_marks(self):
        assert _format_source_with_marks("中华人民共和国道路交通安全法") == "《中华人民共和国道路交通安全法》"

    def test_keeps_existing_book_title_marks(self):
        assert _format_source_with_marks("《道路交通安全法》") == "《道路交通安全法》"

    def test_cleans_before_wrapping(self):
        assert _format_source_with_marks("道路交通安全法_20210429") == "《道路交通安全法》"


class TestExtractClauseLabel:
    def test_extracts_clause(self):
        assert _extract_clause_label("第九十一条第一款") == "第一款"

    def test_empty_when_no_clause(self):
        assert _extract_clause_label("第九十一条") == ""


class TestArticleGroupKey:
    """A1：引用去重合并的分组键。"""

    def test_groups_by_parent_article_no(self):
        ctx = {
            "metadata": {
                "source": "中华人民共和国道路交通安全法",
                "article_no": "第九十一条第一款",
                "parent_article_no": "第九十一条",
            }
        }
        assert _article_group_key(ctx) == (
            "中华人民共和国道路交通安全法",
            "第九十一条",
        )

    def test_falls_back_to_article_no(self):
        ctx = {
            "metadata": {
                "source": "道路交通安全法",
                "article_no": "第九十一条",
            }
        }
        assert _article_group_key(ctx) == ("道路交通安全法", "第九十一条")


class TestMergeReferences:
    """A1：同一法条多款合并为一条引用。"""

    def test_merges_clauses_same_article(self):
        contexts = [
            {
                "id": "c1",
                "metadata": {
                    "source": "中华人民共和国道路交通安全法",
                    "article_no": "第九十一条第一款",
                    "parent_article_no": "第九十一条",
                    "section_header": "法律责任",
                },
                "text": "第一款内容",
            },
            {
                "id": "c2",
                "metadata": {
                    "source": "中华人民共和国道路交通安全法",
                    "article_no": "第九十一条第二款",
                    "parent_article_no": "第九十一条",
                    "section_header": "法律责任",
                },
                "text": "第二款内容",
            },
        ]
        refs = _merge_references(contexts)
        assert len(refs) == 1
        assert refs[0].source == "中华人民共和国道路交通安全法"
        assert refs[0].article_no == "第九十一条"
        assert "第一款" in refs[0].text
        assert "第二款" in refs[0].text
        assert refs[0].merged_from == ["c1", "c2"]

    def test_keeps_single_clause_unchanged(self):
        contexts = [
            {
                "id": "c1",
                "metadata": {
                    "source": "道路交通安全法",
                    "article_no": "第九十条",
                    "section_header": "",
                },
                "text": "第九十条内容",
            }
        ]
        refs = _merge_references(contexts)
        assert len(refs) == 1
        assert refs[0].text == "第九十条内容"
        assert refs[0].merged_from == ["c1"]

    def test_preserves_order_by_first_appearance(self):
        contexts = [
            {
                "id": "a1",
                "metadata": {
                    "source": "法A",
                    "article_no": "第一条",
                    "parent_article_no": "第一条",
                },
                "text": "1",
            },
            {
                "id": "b1",
                "metadata": {
                    "source": "法B",
                    "article_no": "第二条",
                    "parent_article_no": "第二条",
                },
                "text": "2",
            },
            {
                "id": "a2",
                "metadata": {
                    "source": "法A",
                    "article_no": "第一条第二款",
                    "parent_article_no": "第一条",
                },
                "text": "1-2",
            },
        ]
        refs = _merge_references(contexts)
        assert [r.article_no for r in refs] == ["第一条", "第二条"]


class TestClassifyRefusal:
    """A5：拒答场景细分。"""

    def test_out_of_scope_when_source_invalid(self):
        assert _classify_refusal([], [], "不存在的法规") == "out_of_scope"

    def test_no_law_when_candidates_empty(self):
        assert _classify_refusal([], [], None) == "no_law"

    def test_insufficient_when_candidates_exist_but_contexts_empty(self):
        candidates = [{"id": "c1"}]
        assert _classify_refusal(candidates, [], None) == "insufficient"

    def test_no_law_when_contexts_exist(self):
        assert _classify_refusal([{"id": "c1"}], [{"id": "c1"}], None) == "no_law"


class TestSelfCheckAnswer:
    """A4：反向约束/幻觉自检规则层。"""

    @pytest.fixture
    def contexts(self):
        return [
            {
                "metadata": {
                    "source": "中华人民共和国道路交通安全法",
                    "article_no": "第九十一条",
                }
            }
        ]

    def test_passes_when_citation_in_contexts(self, contexts):
        answer = "根据《中华人民共和国道路交通安全法》第九十一条，酒驾将受处罚。"
        passed, warnings = _self_check_answer("酒驾怎么处罚", answer, contexts)
        assert passed is True
        assert warnings == []

    def test_fails_when_article_not_in_contexts(self, contexts):
        answer = "根据《中华人民共和国道路交通安全法》第九十二条，酒驾将受处罚。"
        passed, warnings = _self_check_answer("酒驾怎么处罚", answer, contexts)
        assert passed is False
        assert any("第九十二条" in w for w in warnings)

    def test_fails_when_source_not_in_contexts(self, contexts):
        answer = "根据《不存在的法规》第五条规定。"
        passed, warnings = _self_check_answer("问题", answer, contexts)
        assert passed is False
        assert any("不存在的法规" in w for w in warnings)

    def test_passes_when_no_citation(self, contexts):
        answer = "结论：需要进一步核实。"
        assert _self_check_answer("问题", answer, contexts) == (True, [])
