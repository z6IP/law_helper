import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.query_expansion import expand_query
from app.rerank import get_reranker
from app.retrieval import get_retrieval_engine
from app.config import get_settings

eng = get_retrieval_engine()
r = get_reranker()
s = get_settings()

q = "闯红灯怎么判"
rq = expand_query(q)
print("改写后 query:", rq)
print()
cands = eng.search(rq, top_k=s.top_k_retrieve)
print(f"=== 召回 top{s.top_k_retrieve} ===")
for i, c in enumerate(cands, 1):
    print(f"  {i}", c["metadata"].get("source", ""), c["metadata"].get("article_no", ""))
ranked = r.rerank(rq, cands, top_n=s.rerank_top_n, min_score=s.rerank_min_score)
print(f"=== 重排 top{s.rerank_top_n} (min={s.rerank_min_score}) ===")
for c in ranked:
    print(f"  {c['rerank_score']:.4f}", c["metadata"].get("source", ""), c["metadata"].get("article_no", ""))
print()

# 全库关键词扫描：检查三部法规里所有含「红灯」「信号灯+处罚」「记分」的条文
from app.ingestion import parse_docx

DOCS = [
    ("法", Path(ROOT) / "中华人民共和国道路交通安全法_20210429.docx"),
    ("条例", Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx"),
    ("程序", Path(ROOT) / "道路交通事故处理程序规定.docx"),
]
print("=== 全库扫描：含「红灯」的条文 ===")
for tag, p in DOCS:
    for a in parse_docx(p):
        if "红灯" in a.text:
            print(f"  [{tag}] {a.article_no}: {a.text[:80]}...")
print()
print("=== 全库扫描：含「记分」的条文 ===")
found = False
for tag, p in DOCS:
    for a in parse_docx(p):
        if "记分" in a.text or "记3分" in a.text or "记6分" in a.text:
            found = True
            print(f"  [{tag}] {a.article_no}: {a.text[:80]}...")
if not found:
    print("  （三部法规中均无「记分」相关内容）")
print()
print("=== 法第89/90条原文（通行规定处罚条款）===")
law = {a.article_no: a.text for a in parse_docx(DOCS[0][1])}
for k in ["第八十九条", "第九十条"]:
    print(f"  {k}: {law[k]}")