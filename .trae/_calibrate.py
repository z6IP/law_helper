import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.query_expansion import expand_query
from app.ingestion import parse_docx
from app.rerank import get_reranker
from app.retrieval import get_retrieval_engine

q = "\u5de6\u8f6c\u548c\u76f4\u884c\u76f8\u649e\u600e\u4e48\u5224"  # 左转和直行相撞怎么判
rq = expand_query(q)
print("raw    :", q)
print("expanded:", rq)
print()

sh = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx")}
df = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "道路交通事故处理程序规定.docx")}
law = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法_20210429.docx")}

r = get_reranker()


def sp(query, text):
    out = r.rerank(query, [{"text": text, "metadata": {}}], top_n=1, min_score=None)
    return out[0]["rerank_score"]


print("=== POS (改写后 query) ===")
print("  第51条", round(sp(rq, sh["\u7b2c\u4e94\u5341\u4e00\u6761"]), 4))
print("  第52条", round(sp(rq, sh["\u7b2c\u4e94\u5341\u4e8c\u6761"]), 4))
print("  法44  ", round(sp(rq, law["\u7b2c\u56db\u5341\u56db\u6761"]), 4))
print("  法76  ", round(sp(rq, law["\u7b2c\u4e03\u5341\u516d\u6761"]), 4))
print("=== NEG (改写后 query) ===")
print("  程100 ", round(sp(rq, df["\u7b2c\u4e00\u767e\u6761"]), 4))
print("  程35  ", round(sp(rq, df["\u7b2c\u4e09\u5341\u4e94\u6761"]), 4))
print("  天气  ", round(sp("\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837", sh["\u7b2c\u4e94\u5341\u4e00\u6761"]), 4))
print()

print("=== 完整检索链路 (改写后 query, top_k=12, rerank top5) ===")
eng = get_retrieval_engine()
cands = eng.search(rq, top_k=12)
ranked = r.rerank(rq, cands, top_n=5, min_score=None)
for c in ranked:
    print(
        round(c["rerank_score"], 4),
        c["metadata"].get("source", ""),
        c["metadata"].get("article_no", ""),
    )