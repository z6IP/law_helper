"""文档解析与向量化入库。

将《中华人民共和国道路交通安全法》docx 按「章 / 节 / 条」结构化解析，
每条法条作为一个 chunk 写入 ChromaDB。

工程约束：
- 每个 chunk 的 metadata 记录 section_header（章/节标题）与 article_no（条号）；
- 向量 id 基于 md5(source + section_header + article_no) 保证幂等 upsert。
"""
from __future__ import annotations

import base64
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
    """由文档路径得到干净来源名：去后缀 + 去末尾日期后缀，`+` 转空格。"""
    name = Path(path).stem
    name = re.sub(r"_\d{8}$", "", name)
    name = name.replace("+", " ")  # 如 "GB+19522-2024" -> "GB 19522-2024"
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


# ── PDF 解析（国家标准：章 = "1 范围"，条款 = "3.1"/"5.2.1"，含表格）──
_CHAPTER_RE = re.compile(r"^(\d{1,3})\s+(\S.*)$")
_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\s*(.*)$")
_FRAG_RE = re.compile(r"^(\d+)\.$")       # 被换行拆开的条款号片段，如 "5."
_LEAF_RE = re.compile(r"^(\d+)(.*)$")     # 条款号末段，后面可能紧跟标题
_PAGENUM_RE = re.compile(r"^[0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]{1,3}$")
_PDF_TITLE_LINES = {"车辆驾驶人员血液、呼气酒精含量", "阈值与检验"}


def _clause_level(no: str) -> int:
    return len(no.split("."))


def _clean_table_cell(cell) -> str:
    return " ".join((cell or "").replace("\n", " ").split())


def _format_table(rows) -> str:
    lines = []
    for row in rows:
        cells = [c for c in row if c is not None]
        lines.append(" | ".join(_clean_table_cell(c) for c in cells))
    return "\n".join(lines)


def _lines_to_events(body_lines: list[str]) -> list[tuple]:
    """把文本行转换为解析事件流。

    事件格式：("heading", no, rest, level) / ("text", line)
    """
    # 从首章（"1 范围"）开始，丢弃封面/前言
    start = None
    for i, ln in enumerate(body_lines):
        if _CHAPTER_RE.match(ln):
            start = i
            break
    if start is None:
        return []
    body_lines = body_lines[start:]

    # 还原被换行拆开的条款号片段（"5.\n2.\n1 ..." -> "5.2.1"）
    events: list[tuple] = []
    pending = ""
    for ln in body_lines:
        m_frag = _FRAG_RE.match(ln)
        if m_frag:
            pending += m_frag.group(1) + "."
            continue
        if pending:
            m_leaf = _LEAF_RE.match(ln)
            if m_leaf:
                no = pending + m_leaf.group(1)
                rest = m_leaf.group(2).strip()
                pending = ""
                events.append(("heading", no, rest, _clause_level(no)))
                continue
            pending = ""  # 孤立片段，丢弃
        m_clause = _CLAUSE_RE.match(ln)
        if m_clause:
            no = m_clause.group(1)
            events.append(("heading", no, m_clause.group(2).strip(), _clause_level(no)))
            continue
        m_ch = _CHAPTER_RE.match(ln)
        if m_ch:
            no = m_ch.group(1)
            events.append(("heading", no, _clean_title(m_ch.group(2)), 1))
            continue
        if ln:
            events.append(("text", ln))
    return events


def _events_to_articles(events: list[tuple], source: str = "") -> tuple[list[Article], str]:
    """依据层级构建条目：叶子节点成块，容器节点只作 section_header。

    返回 (articles, cur_chapter)。
    """
    articles: list[Article] = []

    class _Node:
        __slots__ = ("no", "level", "section", "text", "has_child")

        def __init__(self, no, level, section):
            self.no = no
            self.level = level
            self.section = section
            self.text = ""
            self.has_child = False

    def _emit(node: "_Node") -> None:
        if not node.has_child and node.text.strip():
            articles.append(
                Article(
                    article_no=node.no,
                    section_header=node.section,
                    text=node.text,
                    source=source,
                )
            )

    stack: list[_Node] = []
    cur_chapter = ""
    for ev in events:
        if ev[0] == "heading":
            _, no, rest, level = ev
            if stack and level > stack[-1].level:
                stack[-1].has_child = True
            while stack and stack[-1].level >= level:
                _emit(stack.pop())
            if level == 1:
                cur_chapter = f"{no} {rest}"
                section, text = "", ""
            else:
                section, text = cur_chapter, rest
            node = _Node(no, level, section)
            node.text = text
            stack.append(node)
        else:  # text
            if not stack:
                continue
            piece = ev[1].strip()
            if piece:
                stack[-1].text = (
                    stack[-1].text + "\n" + piece if stack[-1].text else piece
                )
    while stack:
        _emit(stack.pop())

    return articles, cur_chapter


def _ocr_pdf(pdf_path) -> list[Article]:
    """扫描型 PDF 的 OCR 兜底：逐页转图片后用视觉模型识别文字。

    调用配置中的 ocr_model（默认 qwen3.5-ocr），复用同一 OpenAI 兼容 API。
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("未安装 pymupdf，请先 `pip install pymupdf`") from exc

    from app.llm import get_llm

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentNotFoundError(f"无法读取文档：{pdf_path}") from exc

    settings = get_settings()
    llm = get_llm()
    source = _clean_source_name(pdf_path)

    try:
        all_lines: list[str] = []
        total = len(doc)
        for page_idx, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72))
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            text = llm.ocr_images([b64])
            all_lines.extend(line.strip() for line in text.splitlines() if line.strip())
            print(f"[OCR] {source} 第 {page_idx}/{total} 页识别完成")

        events = _lines_to_events(all_lines)
        articles, _ = _events_to_articles(events, source)
        if not articles:
            raise IngestionError(f"OCR 未识别到有效条款内容：{Path(pdf_path).name}")
        return articles
    finally:
        doc.close()


def parse_pdf(pdf_path) -> list[Article]:
    """解析 PDF，返回按「章 / 条款」切分的条目列表。

    - 章标题（如「4 酒精含量值」）→ section_header；
    - 条款号（如「3.1」「5.2.1」）→ article_no；
    - 表格用 PyMuPDF find_tables 提取并转文字；
    - 无文本层时进入 _ocr_pdf 兜底（OCR 模型由用户提供）。
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("未安装 pymupdf，请先 `pip install pymupdf`") from exc

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentNotFoundError(f"无法读取文档：{pdf_path}") from exc

    with doc:
        # 1) 逐页提取：版面文本行（过滤页眉/页脚/重复标题/表格单元格）+ 表格
        body_lines: list[str] = []
        tables: list[str] = []
        for page in doc:
            page_h = page.rect.height
            table_rects = []
            page_tables = []
            for t in page.find_tables():
                table_rects.append(pymupdf.Rect(t.bbox))
                try:
                    rows = t.extract()
                except Exception:  # noqa: BLE001
                    rows = []
                if rows:
                    page_tables.append(_format_table(rows))
            tables.extend(page_tables)

            lines = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    bbox = pymupdf.Rect(line["bbox"])
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if not text:
                        continue
                    if "GB19522" in text:  # 页眉/页脚标准号
                        continue
                    if text in _PDF_TITLE_LINES:  # 正文页顶部重复的文档标题
                        continue
                    if bbox.y0 > page_h - 70 and _PAGENUM_RE.fullmatch(text):  # 页脚页码
                        continue
                    if any(bbox.intersects(r) for r in table_rects):  # 表格单元格另行规整
                        continue
                    lines.append((bbox.y0, bbox.x0, text))
            lines.sort(key=lambda t: (t[0], t[1]))
            # 版面是「条款号左槽 + 正文右缩进」，编号相对正文块垂直居中而导致 y 错位
            # （如左槽号 y=403、正文首行 y=401）。按 y 邻近归组成同一视觉行、组内按 x
            # 排序，才能还原「编号在前、正文在后」的阅读序。
            y_tol = 5.0
            row_buf: list[tuple[float, str]] = []
            row_y: float | None = None
            for y, x, t in lines:
                if row_y is None or abs(y - row_y) <= y_tol:
                    row_buf.append((x, t))
                    if row_y is None:
                        row_y = y
                else:
                    body_lines.extend(t for _, t in sorted(row_buf, key=lambda p: p[0]))
                    row_buf = [(x, t)]
                    row_y = y
            if row_buf:
                body_lines.extend(t for _, t in sorted(row_buf, key=lambda p: p[0]))

        source = _clean_source_name(pdf_path)
        events = _lines_to_events(body_lines)
        if not events:
            return _ocr_pdf(pdf_path)  # 无文本层 → OCR 兜底
        articles, cur_chapter = _events_to_articles(events, source)

        # 2) 注入表格：追加到正文中引用「表 N」的条款，并记录（表格, 所属章节）对。
        #    仅对实际被注入的表格生成独立 chunk，避免 tables 与引用顺序错位导致 section 错误。
        table_iter = iter(tables)
        injected_tables: list[tuple[str, str]] = []
        for a in articles:
            if "表" in a.text:
                try:
                    tbl = next(table_iter)
                except StopIteration:
                    break
                a.text = f"{a.text}\n{tbl}"
                injected_tables.append((tbl, a.section_header or cur_chapter or "附表"))

        # 3) 表格同时作为独立 chunk，避免注入错位导致检索丢失
        for idx, (tbl, section) in enumerate(injected_tables, 1):
            table_title = f"表{idx}"
            if "酒精" in tbl and "阈值" in tbl:
                table_title = "表1 车辆驾驶人员血液酒精含量阈值"
            articles.append(
                Article(
                    article_no=f"表{idx}",
                    section_header=section,
                    text=f"{table_title}\n{tbl}",
                    source=source,
                )
            )

        if not articles:
            raise IngestionError(f"未从文档中解析到任何内容：{Path(pdf_path).name}")
        return articles


def _make_id(source: str, section_header: str, article_no: str) -> str:
    raw = f"{source}|{section_header}|{article_no}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def ingest() -> int:
    """解析 statute/ 下所有 .docx 与 .pdf 并幂等 upsert 到 ChromaDB，返回入库条目总数。

    多个文档共用同一 collection，向量 id 基于 md5(source + section_header + article_no)
    保证同一条文重复入库时覆盖而非重复。
    """
    settings = get_settings()
    paths = settings.docx_full_paths + settings.pdf_full_paths
    if not paths:
        raise DocumentNotFoundError("statute/ 目录下未找到任何 .docx / .pdf 文档")

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
        suffix = Path(path).suffix.lower()
        if suffix == ".docx":
            articles = parse_docx(path)
        elif suffix == ".pdf":
            articles = parse_pdf(path)
        else:
            continue
        if not articles:
            raise IngestionError(f"未从文档中解析到任何内容：{path.name}")

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