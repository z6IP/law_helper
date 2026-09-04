"""附件物理文件管理：安全删除上传文件。"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import BASE_DIR

logger = logging.getLogger(__name__)

UPLOADS_DIR = BASE_DIR / "data" / "uploads"


def _resolve_path(stored_name: str) -> Path:
    """返回安全的文件路径，防止目录穿越。"""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(stored_name).name
    return (UPLOADS_DIR / safe_name).resolve()


def delete_attachments(stored_names: list[str]) -> None:
    """批量删除 uploads 目录中的物理文件；忽略不存在或删除失败的文件。

    删除失败不影响会话删除流程，仅记录日志。
    """
    for name in stored_names:
        path = _resolve_path(name)
        try:
            if path.exists() and path.is_file():
                path.unlink()
                logger.info("已删除附件: %s", path)
        except OSError:
            logger.exception("删除附件失败: %s", path)


def delete_upload(stored_name: str) -> bool:
    """删除单个上传文件，返回是否成功。"""
    path = _resolve_path(stored_name)
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except OSError:
        logger.exception("删除附件失败: %s", path)
    return False
