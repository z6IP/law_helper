import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)

from app.ingestion import parse_docx
from app.rerank import get_reranker

sh = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx")}
df = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "道路交通事故处理程序规定.docx")}
law = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法_20210429.docx")}

r = get_reranker()


def sp(query, text):
    out = r.rerank(query, [{"text": text, "metadata": {}}], top_n=1, min_score=None)
    return out[0]["rerank_score"]


print("===== 追尾场景：不同 query 变体 vs 法43条 =====")
Z43 = law["第四十三条"]
variants_tail = [
    "追尾了谁的责任",  # 原始（对照）
    "追尾了谁的责任 同车道行驶 后车应当与前车保持足以采取紧急制动措施的安全距离",  # 当前规则
    "追尾 责任认定 未保持安全距离",
    "后车未与前车保持安全距离发生追尾事故 责任如何划分",
]
for q in variants_tail:
    print(f"  {sp(q, Z43):.4f}  {q}")

print()
print("===== 高速倒车场景：不同 query 变体 vs 实施条例82条 / 法44条 =====")
Z82 = sh["第八十二条"]
variants_hwy = [
    "高速公路上可以倒车吗",  # 原始（对照）
    "高速公路上可以倒车吗 机动车在高速公路上行驶不得倒车、逆行、穿越中央分隔带掉头",  # 当前规则
    "高速公路倒车怎么处罚",
    "在高速公路上倒车 违法行为 处罚规定",
]
for q in variants_hwy:
    print(f"  82条 {sp(q, Z82):.4f} | 44条 {sp(q, law['第四十四条']):.4f} | {q}")

print()
print("===== 法44条原文（排查 0.57 误命中）=====")
print(law["第四十四条"])