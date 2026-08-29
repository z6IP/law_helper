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

QUERIES = [
    "左转和直行相撞怎么判",
    "酒驾怎么处罚",
    "酒驾扣几分",
    "开车撞人后逃跑会怎样",
    "高速公路上可以倒车吗",
    "没有驾照开车被抓怎么处理",
    "追尾了谁的责任",
    "闯红灯怎么判",
    "闯红灯扣几分",
    "今天天气怎么样",
]

for q in QUERIES:
    rq = expand_query(q)
    cands = eng.search(rq, top_k=s.top_k_retrieve)
    ranked = r.rerank(rq, cands, top_n=s.rerank_top_n, min_score=s.rerank_min_score)
    print(f"Q: {q}")
    if not ranked:
        print("   （全部被过滤 → 拒答）")
    for c in ranked:
        print(
            f"   {c['rerank_score']:.4f}",
            c["metadata"].get("source", ""),
            c["metadata"].get("article_no", ""),
        )
    print()