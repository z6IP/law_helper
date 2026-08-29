import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.query_expansion import expand_query
from app.ingestion import parse_docx
from app.rerank import get_reranker
from app.config import get_settings

LIBS = {
    "sh": {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx")},
    "df": {a.article_no: a.text for a in parse_docx(Path(ROOT) / "道路交通事故处理程序规定.docx")},
    "law": {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法_20210429.docx")},
}

r = get_reranker()
s = get_settings()


def sp(query, lib, no):
    out = r.rerank(query, [{"text": LIBS[lib][no], "metadata": {}}], top_n=1, min_score=None)
    return out[0]["rerank_score"]


# (query, 正样本 [(lib, 条号)], 负样本 [(lib, 条号)])
CASES = [
    (
        "酒驾怎么处罚",
        [("law", "第九十一条")],
        [("df", "第一百条"), ("law", "第四十四条")],
    ),
    (
        "开车撞人后逃跑会怎样",
        [("law", "第九十九条")],
        [("sh", "第五十一条"), ("law", "第四十四条")],
    ),
    (
        "高速公路上可以倒车吗",
        [("sh", "第八十二条")],
        [("df", "第一百条"), ("law", "第四十四条")],
    ),
    (
        "没有驾照开车被抓怎么处理",
        [("law", "第九十九条")],
        [("sh", "第五十一条"), ("df", "第一百条")],
    ),
    (
        "追尾了谁的责任",
        [("law", "第四十三条")],
        [("df", "第一百条"), ("sh", "第八十二条")],
    ),
    (
        "闯红灯扣几分",
        [("law", "第九十条")],
        [("df", "第一百条"), ("sh", "第八十二条")],
    ),
]

print("top_k =", s.top_k_retrieve, "| rerank_top_n =", s.rerank_top_n, "| min_score =", s.rerank_min_score)
print()
for q, pos, neg in CASES:
    rq = expand_query(q)
    tag = "" if rq == q else "  [已改写]"
    print(f"Q: {q}{tag}")
    for lib, no in pos:
        score = sp(rq, lib, no)
        print(f"   POS  {lib}:{no:<8} {score:.4f}  {'>0.2 OK' if score >= 0.2 else '** <0.2 误杀 **'}")
    for lib, no in neg:
        score = sp(rq, lib, no)
        print(f"   NEG  {lib}:{no:<8} {score:.4f}  {'<0.2 OK' if score < 0.2 else '** >=0.2 放行 **'}")
    print()