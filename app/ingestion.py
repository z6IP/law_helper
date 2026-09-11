"""文档解析与向量化入库。

将《中华人民共和国道路交通安全法》docx 按「章 / 节 / 条」结构化解析，
每条法条作为父 chunk 写入 ChromaDB；含多个分款的法条同时生成款项子 chunk。

工程约束：
- 每个 chunk 的 metadata 记录 section_header（章/节标题）与 article_no（条号）；
- 向量 id 基于 sha256(source + section_header + article_no) 保证幂等 upsert。
"""
from __future__ import annotations

import base64
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.errors import DocumentNotFoundError, IngestionError
from app.rerank import _classify_role
from app.policy import get_policy
from app.tracing import event, span

# 章 / 节 / 条 标题识别
CHAPTER_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)章\s*(.*)$")
SECTION_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)节\s*(.*)$")
# ARTICLE_RE 同时识别「第X条」与「第X条之Y」修正案条款（如刑法第一百二十条之一）
# group(1): 条号主体，group(2): 之Y 后缀（可空）
ARTICLE_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)条(之[零一二三四五六七八九十百千]+)?")
# 常见中文分款格式：（一）、（二）、(一)、一、
CLAUSE_RE = re.compile(
    r"(?m)^(?:（([一二三四五六七八九十百千万]+)）|"
    r"\(([一二三四五六七八九十百千万]+)\)|"
    r"([一二三四五六七八九十百千万]+)、)\s*"
)

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


def _category_from_path(path: Path) -> str:
    """从文档路径取所在子文件夹名作为分类（statute/基础法律依据/xxx.docx → 基础法律依据）。

    statute/ 根目录下的旧文件（无子文件夹）返回空字符串。
    """
    parent = path.parent
    # statute/子文件夹/文件.docx → parent.name 即子文件夹名
    # 兼容 statute/根目录/文件.docx 的情况，返回空字符串
    if parent.name == "statute":
        return ""
    return parent.name


# 上位法 / 下位法 / 补充 规则
_HIERARCHY_RULES = get_policy()["retrieval"]["category_hierarchy"]
_SUPPLEMENT_SOURCES = set(get_policy()["retrieval"]["supplement_sources"])


def _hierarchy_from_path(path: Path, source: str) -> str:
    """按子文件夹 + 文件名规则判定层级。

    - 基础法律依据 → 上位法
    - 事故处理赔偿 / 行政处罚程序 → 下位法
    - 中华人民共和国道路交通安全法实施条例 → 补充（即便位于基础法律依据中也作为补充）
    """
    if source in _SUPPLEMENT_SOURCES:
        return "补充"
    category = _category_from_path(path)
    return _HIERARCHY_RULES.get(category, "")


@dataclass
class Article:
    article_no: str
    section_header: str
    text: str
    source: str = ""
    penalty_context: str = ""


def _split_clauses(article: Article) -> list[Article]:
    """保留完整父条文，并为至少两款的条文生成款项子块。"""
    matches = list(CLAUSE_RE.finditer(article.text))
    if len(matches) < 2:
        return [article]

    # 提取父条中第一个款项标记前的处罚前置句（如"有下列行为之一的，处五日以上
    # 十日以下拘留……："），作为所有子款共享的处罚上下文。子款正文往往只含行为
    # 描述、不含处罚词（如"拘留"），导致下游受保护召回按处罚词过滤时误删正确子款；
    # 通过 penalty_context 让子款携带父条处罚信息，保证被正确召回与配额保障。
    penalty_context = article.text[:matches[0].start()].strip()

    chunks = [article]
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(article.text)
        text = article.text[start:end].strip()
        if not text:
            continue
        clause_no = next(group for group in match.groups() if group is not None)
        chunks.append(
            Article(
                article_no=f"{article.article_no}第{clause_no}款",
                section_header=article.section_header,
                text=text,
                source=article.source,
                penalty_context=penalty_context,
            )
        )
    return chunks


def _expand_article_chunks(articles: list[Article]) -> list[Article]:
    """构建父条文 + 款项子块索引，降低长条文整体向量的信息稀释。"""
    return [chunk for article in articles for chunk in _split_clauses(article)]


def _parent_article_no(article_no: str) -> str:
    """从款项子块条号中提取父条号。"""
    match = re.match(
        r"^(第[零一二三四五六七八九十百千]+条(?:之[零一二三四五六七八九十百千]+)?)",
        article_no,
    )
    return match.group(1) if match else article_no


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
            # 包含修正案后缀（如「第一百二十条之一」），避免同一条号被识别为多条
            article_no = f"第{m_art.group(1)}条{m_art.group(2) or ''}"
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
            event("ingest.ocr_page", source=source, page=page_idx, total=total)

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


MANIFEST_FILENAME = "ingestion_manifest.db"

# metadata schema 版本：每次扩展 ChromaDB metadata 字段或切换嵌入模型时 bump 此值，
# _manifest_conn 检测到旧表 schema 不一致时会清空 manifest，强制全量重新导入，
# 保证旧记录也能更新到新的 metadata 字段或新嵌入模型向量（基于 idempotent upsert，不会产生重复向量）。
# v4: 切换嵌入模型 qwen3.7-text-embedding(1024维) → BGE-base-zh-v1.5(768维)
# v6: 增加法条父块/款项子块 metadata
# v7: 子款继承父条处罚上下文 penalty_context
METADATA_SCHEMA_VERSION = 7


# file_hash 不设 UNIQUE：允许同一文档内容出现在多个分类子文件夹（不同 source）。
# 注意：旧库已带 UNIQUE 约束，升级时需删除 manifest 文件触发全量重建
# （幂等 upsert，不会产生重复向量）。
_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_manifest (
    source                  TEXT PRIMARY KEY,
    file_hash               TEXT NOT NULL,
    article_count           INTEGER NOT NULL DEFAULT 0,
    metadata_schema_version INTEGER NOT NULL DEFAULT 1
)
"""


@dataclass
class IngestResult:
    """增量导入结果统计。"""

    total: int = 0          # 当前库中该 source 的条文总数（新增+更新后）
    added: int = 0          # 新增 source 的条文数
    updated: int = 0        # 内容变化后重新写入的条文数
    removed: int = 0        # 因 source 被删除而从库中移除的条文数
    skipped: int = 0        # 文件未变化、跳过的 source 条文数
    message: str = ""       # 人类可读摘要


def _make_id(source: str, section_header: str, article_no: str) -> str:
    raw = f"{source}|{section_header}|{article_no}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _make_unique_ids(articles: list[Article], source: str) -> list[str]:
    """为解析结果生成唯一稳定 ID，兼容偶发的重复条号。"""
    ids: list[str] = []
    seen: set[str] = set()
    for article in articles:
        base_id = _make_id(source, article.section_header, article.article_no)
        article_id = base_id
        if article_id in seen:
            text_digest = hashlib.sha256(article.text.encode("utf-8")).hexdigest()[:12]
            article_id = _make_id(
                source,
                article.section_header,
                f"{article.article_no}|{text_digest}",
            )
        while article_id in seen:
            article_id = hashlib.sha256(f"{article_id}|{len(ids)}".encode("utf-8")).hexdigest()
        seen.add(article_id)
        ids.append(article_id)
    return ids


def _compute_file_hash(path: Path) -> str:
    """计算文件内容的 sha256，用于检测文件是否被修改。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(chroma_dir: Path) -> Path:
    return chroma_dir / MANIFEST_FILENAME


def _manifest_conn(chroma_dir: Path):
    """打开（必要时创建）manifest SQLite 库并确保表结构存在。

    若检测到旧表缺少 metadata_schema_version 列，则 DROP 整张表并重建，
    令 manifest 清空、所有 source 下次被识别为「新增」并重新 upsert，
    从而把新的 metadata 字段（category / 层级 等）刷写到已有向量上。
    幂等 upsert 保证不会产生重复向量。
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(chroma_dir / MANIFEST_FILENAME)
    try:
        cols = conn.execute("PRAGMA table_info(ingestion_manifest)").fetchall()
        col_names = {row[1] for row in cols}
        if col_names and "metadata_schema_version" not in col_names:
            conn.execute("DROP TABLE ingestion_manifest")
        elif col_names:
            # 旧 schema 的 file_hash 带 UNIQUE 约束：同一内容多副本会触发 IntegrityError。
            # CREATE TABLE IF NOT EXISTS 不会改动已存在的表，需显式检测并重建
            # （manifest 仅为缓存，重建后触发全量重导，幂等 upsert 不会产生重复向量）。
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='ingestion_manifest'"
            ).fetchone()
            if row and "file_hash" in (row[0] or "") and "UNIQUE" in (row[0] or ""):
                conn.execute("DROP TABLE ingestion_manifest")
    except sqlite3.OperationalError:
        pass
    conn.execute(_MANIFEST_SCHEMA)
    conn.commit()
    return conn


def _load_manifest(chroma_dir: Path) -> dict:
    """读取 manifest 为 {source: {"hash": ..., "articles": ..., "schema": ...}}；空表返回空字典。"""
    conn = _manifest_conn(chroma_dir)
    try:
        rows = conn.execute(
            "SELECT source, file_hash, article_count, metadata_schema_version "
            "FROM ingestion_manifest"
        ).fetchall()
    finally:
        conn.close()
    return {
        source: {
            "hash": file_hash,
            "articles": article_count,
            "schema": schema_version,
        }
        for source, file_hash, article_count, schema_version in rows
    }


def _save_manifest(chroma_dir: Path, manifest: dict) -> None:
    """全量落盘 manifest：先清空再插入，等价于旧 JSON 的整体覆盖语义。"""
    conn = _manifest_conn(chroma_dir)
    try:
        conn.execute("DELETE FROM ingestion_manifest")
        conn.executemany(
            "INSERT INTO ingestion_manifest "
            "(source, file_hash, article_count, metadata_schema_version) "
            "VALUES (?, ?, ?, ?)",
            [
                (source, m["hash"], m.get("articles", 0), METADATA_SCHEMA_VERSION)
                for source, m in manifest.items()
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _ingest_single_file(
    path: Path,
    collection,
    embedding_model,
) -> tuple[str, int]:
    """解析单个文件并 upsert 到 collection，返回 (source, article_count)。"""
    source = _clean_source_name(path)
    category = _category_from_path(path)
    hierarchy = _hierarchy_from_path(path, source)
    with span("ingest.file", source=source, file=path.name, category=category, hierarchy=hierarchy):
        suffix = path.suffix.lower()
        if suffix == ".docx":
            articles = parse_docx(path)
        elif suffix == ".pdf":
            articles = parse_pdf(path)
        else:
            raise IngestionError(f"不支持的文件类型：{path.name}")
        if not articles:
            raise IngestionError(f"未从文档中解析到任何内容：{path.name}")

        articles = _expand_article_chunks(articles)
        documents = [a.text for a in articles]
        metadatas = [
            {
                "article_no": a.article_no,
                "parent_article_no": _parent_article_no(a.article_no),
                "chunk_type": "clause" if a.article_no != _parent_article_no(a.article_no) else "article",
                "section_header": a.section_header,
                "source": source,
                "category": category,
                "层级": hierarchy,
                "penalty_context": a.penalty_context,
            }
            for a in articles
        ]
        # role 基于完整 metadata 判定，与 rerank.py 中 _classify_role 调用方式一致
        for meta in metadatas:
            meta["role"] = _classify_role(meta)
        ids = _make_unique_ids(articles, source)
        embeddings = embedding_model.embed_documents(documents)
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
    return source, len(articles)


def ingest() -> IngestResult:
    """增量导入 statute/ 下所有 .docx 与 .pdf 到 ChromaDB。

    - 通过文件内容 hash 检测新增/修改的文档，只重新嵌入变化的文档；
    - 通过 source 名称检测已被移除的文档，从库中删除；
    - 向量 id 基于 sha256(source + section_header + article_no) 保证幂等 upsert。
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
            "hnsw:construction_ef": settings.hnsw_construction_ef,
            "hnsw:search_ef": settings.hnsw_search_ef,
            "hnsw:M": settings.hnsw_M,
        },
    )

    # 维度检测：已有向量维度与当前模型不匹配时，删除旧集合重建。
    # 场景：切换嵌入模型（如 qwen3.7-text-embedding 1024维 → BGE-base-zh-v1.5 768维）。
    # 直接运行 python -m app.ingestion 时 main.py 的 preload 重建逻辑不会触发，
    # 因此在此处显式检测，保证 CLI 直接导入也能自动处理维度变更。
    existing = collection.get(include=["embeddings"])
    existing_emb = existing.get("embeddings")
    if existing_emb is not None and len(existing_emb) > 0:
        actual_dim = len(existing_emb[0])
        expected_dim = settings.embedding_dimensions
        if actual_dim != expected_dim:
            event(
                "ingest.dim_mismatch",
                stored_dim=actual_dim,
                expected_dim=expected_dim,
            )
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": settings.hnsw_construction_ef,
                    "hnsw:search_ef": settings.hnsw_search_ef,
                    "hnsw:M": settings.hnsw_M,
                },
            )
            # 维度变更后必须清空 manifest，否则 manifest 记录的 schema 哈希
            # 与当前文件一致，会误判为「无变化」而跳过重新嵌入，
            # 导致旧维度向量残留、新模型未真正生效。
            manifest_path = _manifest_path(settings.chroma_full_dir)
            try:
                manifest_path.unlink()
            except FileNotFoundError:
                pass
    elif stale_manifest := _load_manifest(settings.chroma_full_dir):
        # collection 已空但 manifest 仍记录历史 source 时，清空 manifest 强制全量重嵌。
        # 场景：collection 被外部清空（手动删除目录 / 重建中途失败残留），
        # 但 manifest 仍记录历史 source，会被识别为「未变化」而跳过，导致永远不重建。
        event("ingest.stale_manifest_after_empty", sources=len(stale_manifest))
        manifest_path = _manifest_path(settings.chroma_full_dir)
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass

    manifest = _load_manifest(settings.chroma_full_dir)
    current_sources: dict[str, dict] = {}
    changed_paths: list[Path] = []

    for path in paths:
        source = _clean_source_name(path)
        file_hash = _compute_file_hash(path)
        current_sources[source] = {"hash": file_hash, "path": str(path.name)}
        old = manifest.get(source)
        # 内容变化、新增、或 metadata schema 版本不一致（如新增了 category/层级字段）
        # 都触发重新 upsert，把新 metadata 刷到已有向量上（幂等，不会重复）
        if (
            old is None
            or old.get("hash") != file_hash
            or old.get("schema") != METADATA_SCHEMA_VERSION
        ):
            changed_paths.append(path)

    # 检测已删除的 source：manifest 中有记录但当前 statute/ 中不存在
    removed_sources = [s for s in manifest if s not in current_sources]

    result = IngestResult()
    embedding_model = get_embedding_model()

    for source in removed_sources:
        old_count = manifest.get(source, {}).get("articles", 0)
        existing = collection.get(where={"source": source}, include=[])
        ids_to_remove = existing.get("ids", [])
        if ids_to_remove:
            collection.delete(ids=ids_to_remove)
        result.removed += len(ids_to_remove)
        event("ingest.remove", source=source, count=len(ids_to_remove))

    for path in changed_paths:
        source, count = _ingest_single_file(path, collection, embedding_model)
        current_sources[source]["articles"] = count
        if manifest.get(source):
            result.updated += count
            event("ingest.update", source=source, count=count)
        else:
            result.added += count
            event("ingest.add", source=source, count=count)

    # 未变化的 source 计入 skipped
    for source in manifest:
        if source in current_sources and source not in removed_sources and source not in [
            _clean_source_name(p) for p in changed_paths
        ]:
            result.skipped += manifest[source].get("articles", 0)

    # 同步所有当前 source 的文章数到 manifest
    for source in current_sources:
        if "articles" not in current_sources[source]:
            # 该 source 未变化，保留原数量
            current_sources[source]["articles"] = manifest.get(source, {}).get("articles", 0)

    _save_manifest(settings.chroma_full_dir, current_sources)

    result.total = collection.count()
    if result.added or result.updated or result.removed:
        result.message = (
            f"新增 {result.added} 条，更新 {result.updated} 条，"
            f"移除 {result.removed} 条，跳过 {result.skipped} 条，当前共 {result.total} 条"
        )
    else:
        result.message = f"所有 {len(paths)} 个文档均未变化，跳过导入，当前共 {result.total} 条"
    return result


def main() -> None:
    result = ingest()
    print(result.message)


if __name__ == "__main__":
    main()