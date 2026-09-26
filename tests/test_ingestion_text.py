"""PDF 文本归一化与跨页重复行剔除的单元测试。

守护点：
- 字形错乱还原（彝文区误映射标点 / 全角转半角 / 条款号内空格）不破坏真实彝文；
- 重复行剔除按「出现过的不同页数」判定，同页合法短行不被误删。
"""
from __future__ import annotations

from app.ingestion import _drop_repeated_lines, _normalize_pdf_line


def test_normalize_restores_misplaced_punctuation_and_fullwidth_digits():
    # A3AC 错位标点 → 全角逗号；A3AE 特判 → 中文句号
    assert _normalize_pdf_line("\ua3ac\ua3ae") == "，。"
    # 全角数字 / 字母 → 半角
    assert _normalize_pdf_line("６０００ｍｍ") == "6000mm"
    # 《》 被错映射为 «»
    assert _normalize_pdf_line("«测试»") == "《测试》"


def test_normalize_restores_clause_number_space():
    assert _normalize_pdf_line("3. 1 范围") == "3.1 范围"


def test_normalize_keeps_real_yi_text():
    # 行内以 A3 区之外的真实彝文音节为主时，跳过 A3 区还原
    yi = "\ua000\ua001\ua002\ua3ac"
    assert _normalize_pdf_line(yi) == yi


def test_normalize_handles_empty_line():
    assert _normalize_pdf_line("") == ""


def test_drop_repeated_lines_only_removes_cross_page_duplicates():
    rows = [
        ("GB 19522", 1),
        ("GB 19522", 2),
        ("GB 19522", 3),
        ("（一）", 1),
        ("（一）", 1),
        ("（一）", 1),
        ("正文一", 1),
        ("正文二", 2),
    ]

    kept = _drop_repeated_lines(rows)

    assert "GB 19522" not in kept  # 跨 3 页重复 → 剔除
    assert kept.count("（一）") == 3  # 同页重复 3 次 → 保留
    assert "正文一" in kept and "正文二" in kept


def test_drop_repeated_lines_keeps_clause_number_fragments():
    rows = [("5.", 1), ("5.", 2), ("5.", 3), ("1", 1), ("1", 2), ("1", 3)]

    assert _drop_repeated_lines(rows) == ["5.", "5.", "5.", "1", "1", "1"]
