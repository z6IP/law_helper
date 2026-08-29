"""文档解析与向量化入库。

将《中华人民共和国道路交通安全法》docx 按「章 / 节 / 条」结构化解析，
每条法条作为一个 chunk 写入 ChromaDB。

工程约束：
- 每个 chunk 的 metadata 记录 section_header（章/节标题）与 article_no（条号）；
- 向量 id 基于 md5(source + section_header + article_no) 保证幂等 upsert。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.errors import DocumentNotFoundError, IngestionError

# 章 / 节 / 条 标题识别
CHAPTER_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)章\s*(.*)$")
SECTION_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)节\s*(.*)$")
ARTICLE_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)条")

COLLECTION_NAME = "road_traffic_law"


def _clean_title(title: str) -> str:
    """清理章/节标题内部空白（含 U+2002 全角空格），如「总　则」→「总则」。"""
    return re.sub(r"[\u2002\s]+", "", (title or "").strip())


def _clean_source_name(path) -> str:
    """由文档路径得到干净来源名：去 .docx 后缀 + 去末尾日期后缀。"""
    name = Path(path).stem
    name = re.sub(r"_\d{8}$", "", name)
    return name


@dataclass
class Article:
    article_no: str
    section_header: str
    text: str
    source: str = ""


@dataclass
class ParserState:
    chapter: str = ""
    section: str = ""
    articles: list[Article] = field(default_factory=list)
    current: Article | None = None

    @property
    def section_header(self) -> str:
        if self.chapter and self.section:
            return f"{self.chapter} / {self.section}"
        if self.chapter:
            return self.chapter
        return ""

    def flush_current(self) -> None:
        """把当前法条写入列表并清空，避免章/节切换时丢失最后一条。"""
        if self.current is not None and self.current.text:
            self.articles.append(self.current)
        self.current = None

    def start_article(self, article_no: str) -> None:
        self.flush_current()
        self.current = Article(
            article_no=article_no, section_header=self.section_header, text=""
        )

    def append_text(self, text: str) -> None:
        if self.current is None:
            return
        piece = text.strip()
        if not piece:
            return
        if self.current.text:
            self.current.text += "\n" + piece
        else:
            self.current.text = piece


def parse_docx(docx_path) -> list[Article]:
    """解析 docx，返回按条切分的法条列表。"""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("未安装 python-docx，请先 `pip install python-docx`") from exc

    try:
        doc = Document(str(docx_path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentNotFoundError(f"无法读取文档：{docx_path}") from exc

    state = ParserState()
    in_toc = False

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        # 目录起始标记，跳过目录内容
        if "目" == text and "录" in p.text:
            in_toc = True
            continue

        m_ch = CHAPTER_RE.match(text)
        m_sec = SECTION_RE.match(text)
        m_art = ARTICLE_RE.match(text)

        if m_ch and not in_toc:
            # 先落盘当前法条，再切换章，保证上一章最后一条不丢失
            state.flush_current()
            state.chapter = f"第{m_ch.group(1)}章 {_clean_title(m_ch.group(2))}"
            state.section = ""
            continue
        if m_sec and not in_toc:
            state.flush_current()
            state.section = f"第{m_sec.group(1)}节 {_clean_title(m_sec.group(2))}"
            continue
        if m_art:
            # 进入正文后终止目录识别
            in_toc = False
            article_no = f"第{m_art.group(1)}条"
            state.start_article(article_no)
            # 条号后可能紧跟正文（如「第一条 为了维护...」）
            rest = text[m_art.end():].strip()
            state.append_text(rest)
            continue

        # 普通文本：视为当前法条的延续段落
        if state.current is not None:
            state.append_text(text)

    if state.current is not None and state.current.text:
        state.articles.append(state.current)

    return state.articles


def _make_id(source: str, section_header: str, article_no: str) -> str:
    raw = f"{source}|{section_header}|{article_no}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def ingest() -> int:
    """解析根目录下所有 .docx 并幂等 upsert 到 ChromaDB，返回入库条文总数。

    多个文档共用同一 collection，向量 id 基于 md5(source + section_header + article_no)
    保证同一条文重复入库时覆盖而非重复。
    """
    settings = get_settings()
    paths = settings.docx_full_paths
    if not paths:
        raise DocumentNotFoundError("statute/ 目录下未找到任何 .docx 法规文档")

    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_full_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 16,
            "hnsw:M": 16,
        },
    )

    embedding_model = get_embedding_model()
    total = 0
    for path in paths:
        articles = parse_docx(path)
        if not articles:
            raise IngestionError(f"未从文档中解析到任何法条：{path.name}")

        source = _clean_source_name(path)
        documents = [a.text for a in articles]
        metadatas = [
            {"article_no": a.article_no, "section_header": a.section_header, "source": source}
            for a in articles
        ]
        ids = [_make_id(source, a.section_header, a.article_no) for a in articles]
        embeddings = embedding_model.embed_documents(documents)
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        total += len(articles)

    return total


def main() -> None:
    count = ingest()
    print(f"入库完成，共 {count} 条")


if __name__ == "__main__":
    main()