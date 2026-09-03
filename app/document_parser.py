"""用户上传文件解析：docx / pdf / 图片 OCR。

按后缀路由：
- .docx -> python-docx
- .pdf  -> pymupdf 文本提取；文本层太薄时走 OCR 兜底
- 图片  -> OCR 模型
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from app.errors import IngestionError


# PDF 文本层过少时触发 OCR 兜底
_MIN_PDF_TEXT_LEN = 20
# PDF 页数上限，避免大文件渲染/OCR 时内存溢出
_MAX_PDF_PAGES = 50


def _parse_docx(file_bytes: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("无法解析 Word 文档") from exc

    texts: list[str] = []
    for p in doc.paragraphs:
        piece = p.text.strip()
        if piece:
            texts.append(piece)
    return "\n".join(texts)


def _parse_pdf(file_bytes: bytes) -> str:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("未安装 pymupdf，请先 `pip install pymupdf`") from exc

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("无法解析 PDF 文档") from exc

    try:
        if len(doc) > _MAX_PDF_PAGES:
            raise IngestionError(f"PDF 页数超过 {_MAX_PDF_PAGES} 页限制")
        parts: list[str] = []
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text)
        full_text = "\n".join(parts)

        if len(full_text.strip()) < _MIN_PDF_TEXT_LEN:
            # 文本层过薄，按页渲染为图片后走 OCR
            return _ocr_pdf_pages(file_bytes)
        return full_text
    finally:
        doc.close()


def _ocr_pdf_pages(file_bytes: bytes) -> str:
    """将 PDF 每一页渲染为图片后进行 OCR，返回合并文本。"""
    import pymupdf

    from app.config import get_settings
    from app.llm import get_llm

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        images_b64: list[str] = []
        settings = get_settings()
        for page in doc:
            matrix = pymupdf.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            images_b64.append(b64)
        if not images_b64:
            return ""
        return get_llm().ocr_images(images_b64)
    finally:
        doc.close()


def _parse_image(file_bytes: bytes) -> str:
    from app.llm import get_llm

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    try:
        return get_llm().ocr_images([b64])
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("图片 OCR 识别失败") from exc


def parse_document(file_bytes: bytes, filename: str) -> str:
    """解析用户上传文件，返回纯文本内容。

    Args:
        file_bytes: 文件二进制内容。
        filename: 原始文件名，用于判断文件类型。

    Returns:
        提取/识别后的文本。

    Raises:
        IngestionError: 不支持的文件类型或解析失败。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        return _parse_docx(file_bytes)
    if suffix == ".pdf":
        return _parse_pdf(file_bytes)
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return _parse_image(file_bytes)
    raise IngestionError(f"不支持的文件类型：{suffix}")
