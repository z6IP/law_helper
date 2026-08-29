import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)


def esc(s):
    return (s or "").encode("unicode_escape").decode("ascii")


from app.retrieval import get_retrieval_engine
from app.rerank import get_reranker
from app.ingestion import parse_docx

q = "\u5de6\u8f6c\u548c\u76f4\u884c\u76f8\u649e\u600e\u4e48\u5224"  # 左转和直行相撞怎么判
eng = get_retrieval_engine()
cands = eng.search(q, top_k=12)

r = get_reranker()
ranked = r.rerank(q, cands, top_n=12, min_score=None)
print("=== RAW RERANK (v2-m3) ===")
for c in ranked:
    print(
        "score=", round(c["rerank_score"], 4),
        "src=", esc(c["metadata"].get("source", "")),
        "art=", esc(c["metadata"].get("article_no", "")),
    )

sh = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx")}
df = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "道路交通事故处理程序规定.docx")}
law = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法_20210429.docx")}


def score_pair(query, text):
    out = get_reranker().rerank(query, [{"text": text, "metadata": {}}], top_n=1, min_score=None)
    return out[0]["rerank_score"]


print("=== CALIBRATION (v2-m3) ===")
print("POS:")
print("  51", round(score_pair(q, sh["\u7b2c\u4e94\u5341\u4e00\u6761"]), 4))
print("  52", round(score_pair(q, sh["\u7b2c\u4e94\u5341\u4e8c\u6761"]), 4))
print("  \u6cd544", round(score_pair(q, law["\u7b2c\u56db\u5341\u56db\u6761"]), 4))
print("  \u6cd576", round(score_pair(q, law["\u7b2c\u4e03\u5341\u516d\u6761"]), 4))
print("NEG:")
print("  \u7a0b100", round(score_pair(q, df["\u7b2c\u4e00\u767e\u6761"]), 4))
print("  \u7a0b35", round(score_pair(q, df["\u7b2c\u4e09\u5341\u4e94\u6761"]), 4))
print("  \u5929\u6c14", round(score_pair("\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837", sh["\u7b2c\u4e94\u5341\u4e00\u6761"]), 4))