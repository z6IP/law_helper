import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)


def esc(s):
    return (s or "").encode("unicode_escape").decode("ascii")


from app.retrieval import get_retrieval_engine
from app.rerank import get_reranker

eng = get_retrieval_engine()
r = get_reranker()

queries = [
    "\u5de6\u8f6c\u548c\u76f4\u884c\u76f8\u649e\u600e\u4e48\u5224",  # 左转和直行相撞怎么判
    "\u8f6c\u5f2f\u7684\u673a\u52a8\u8f66\u8ba9\u76f4\u884c\u7684\u8f66\u8f86\u5148\u884c",  # 转弯的机动车让直行的车辆先行
    "\u8f6c\u5f2f\u8ba9\u76f4\u884c \u8d23\u4efb\u8ba4\u5b9a",  # 转弯让直行 责任认定
    "\u5de6\u8f6c\u8f66\u4e0e\u76f4\u884c\u8f66\u76f8\u649e \u8d23\u4efb\u5212\u5206 \u8f6c\u5f2f\u8ba9\u76f4\u884c",  # 左转车与直行车相撞 责任划分 转弯让直行
]

for q in queries:
    print("===== Q:", esc(q), "=====")
    cands = eng.search(q, top_k=12)
    print("  PRE-RERANK (hybrid):")
    for i, c in enumerate(cands, 1):
        print("   ", i, esc(c["metadata"].get("source", "")), esc(c["metadata"].get("article_no", "")))
    ranked = r.rerank(q, cands, top_n=5, min_score=None)
    print("  POST-RERANK (v2-m3):")
    for c in ranked:
        print("   ", round(c["rerank_score"], 4), esc(c["metadata"].get("source", "")), esc(c["metadata"].get("article_no", "")))