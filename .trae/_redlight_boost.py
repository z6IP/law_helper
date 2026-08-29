import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.rerank import get_reranker
from app.retrieval import get_retrieval_engine
from app.config import get_settings

eng = get_retrieval_engine()
r = get_reranker()
s = get_settings()

q = "闯红灯怎么判"
boosts = {
    "当前规则": "不按照交通信号灯指示通行 违反道路通行规定 处罚",
    "V2 罚款表述": "红灯表示禁止通行 机动车驾驶人违反道路通行规定 处警告或者二十元以上二百元以下罚款",
    "V3 违法行为定性": "不按照交通信号灯指示通行 机动车驾驶人违反道路交通安全法律、法规关于道路通行规定 处警告或者罚款",
}

for name, boost in boosts.items():
    rq = f"{q} {boost}"
    cands = eng.search(rq, top_k=s.top_k_retrieve)
    has90 = any(c["metadata"].get("article_no") == "第九十条" for c in cands)
    ranked = r.rerank(rq, cands, top_n=s.rerank_top_n, min_score=s.rerank_min_score)
    print(f"===== {name} =====")
    print("  召回 top5（* = 法90条在召回中）:")
    for i, c in enumerate(cands, 1):
        mark = " *" if c["metadata"].get("article_no") == "第九十条" else ""
        print(f"    {i}", c["metadata"].get("source", "").replace("中华人民共和国", ""), c["metadata"].get("article_no", ""), mark)
    print("  重排 top3:")
    for c in ranked:
        print(f"    {c['rerank_score']:.4f}", c["metadata"].get("source", "").replace("中华人民共和国", ""), c["metadata"].get("article_no", ""))
    print()