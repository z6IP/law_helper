import sys
from pathlib import Path

ROOT = r"c:\Users\19674\PycharmProjects\law_helper"
sys.path.insert(0, ROOT)
from app.ingestion import parse_docx

sh = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "中华人民共和国道路交通安全法实施条例_20171007.docx")}
df = {a.article_no: a.text for a in parse_docx(Path(ROOT) / "道路交通事故处理程序规定.docx")}

for k in ["第四十一条", "第五十一条", "第五十二条"]:
    print("=====", k, "=====")
    print(sh[k])
    print()

print("===== 程序规定 第一百条 =====")
print(df["第一百条"])