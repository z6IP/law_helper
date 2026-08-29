import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.rerank import get_reranker
from app.retrieval import get_retrieval_engine
from app.query_expansion import expand_query

q = "\u5de6\u8f6c\u548c\u76f4\u884c\u76f8\u649e\u600e\u4e48\u5224"  # 左转和直行相撞怎么判

variants = {
    "A 单片段(转弯让直行)": q + " \u8f6c\u5f2f\u7684\u673a\u52a8\u8f66\u8ba9\u76f4\u884c\u7684\u8f66\u8f86\u5148\u884c",
    "B 双片段(当前规则)": q + " \u8f6c\u5f2f\u7684\u673a\u52a8\u8f66\u8ba9\u76f4\u884c\u7684\u8f66\u8f86\u5148\u884c \u76f8\u5bf9\u65b9\u5411\u884c\u9a76\u7684\u53f3\u8f6c\u5f2f\u7684\u673a\u52a8\u8f66\u8ba9\u5de6\u8f6c\u5f2f\u7684\u8f66\u8f86\u5148\u884c",
    "C 关键词(转弯让直行+责任认定)": q + " \u8f6c\u5f2f\u8ba9\u76f4\u884c \u8d23\u4efb\u8ba4\u5b9a",
}

eng = get_retrieval_engine()
r = get_reranker()

for name, rq in variants.items():
    print("=====", name, "=====")
    cands = eng.search(rq, top_k=12)
    ranked = r.rerank(rq, cands, top_n=5, min_score=None)
    for c in ranked:
        print(
            "  ",
            round(c["rerank_score"], 4),
            c["metadata"].get("source", ""),
            c["metadata"].get("article_no", ""),
        )
    print()