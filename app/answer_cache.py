"""问题 → 回答缓存（已永久禁用）。

原为 SQLite 持久化缓存，命中则跳过检索→重排→生成链路。
现永久禁用，避免旧缓存命中导致新代码回答不生效。

所有函数保留签名但 no-op：
- get() 永远返回 None（不命中）
- put() 空操作（不写入）
- clear() 空操作
- 模块加载不再创建 DB 文件
"""
from __future__ import annotations


def get(question: str) -> dict | None:
    """已禁用：永远返回 None（不命中）。"""
    return None


def put(question: str, answer: str, references: list[dict]) -> None:
    """已禁用：空操作（不写入）。"""
    return


def clear() -> None:
    """已禁用：空操作。"""
    return
