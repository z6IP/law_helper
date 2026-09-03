"""上传材料检索：内存精简版。

不对用户上传材料建 Chroma collection，而是在内存中切块后，
用本地 embedding 模型做粗排、reranker 做精排，取 top3 相关片段注入 prompt。
"""
from __future__ import annotations

import re

import numpy as np

from app.embeddings import get_embedding_model
from app.rerank import get_reranker
from app.tracing import event, span

# 材料短于此字数直接全文注入，避免切块/嵌入开销
_DOC_RAG_MIN_LEN = 1500
# 切块目标/最大长度
_CHUNK_TARGET = 400
_CHUNK_MAX = 600
# 粗排 top_k 与精排 top_n
_COARSE_K = 8
DOC_TOP_N = 3


def split_chunks(text: str) -> list[str]:
    """按段落/句界将材料切分为 chunk。

    - 按换行拆段落；
    - 相邻短段落合并至接近 _CHUNK_TARGET 字；
    - 单个段落超过 _CHUNK_MAX 时按句界硬切。
    """
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs: list[str] = []
    for p in raw_paragraphs:
        # 单段过长时按句界拆分
        if len(p) > _CHUNK_MAX:
            pieces = _split_by_sentence(p)
            paragraphs.extend(pieces)
        else:
            paragraphs.append(p)

    chunks: list[str] = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) < _CHUNK_TARGET:
            buffer = buffer + "\n" + p if buffer else p
            continue
        # 当前 buffer 已接近目标，先落盘
        if buffer:
            chunks.append(buffer)
            buffer = p
        else:
            buffer = p
    if buffer:
        chunks.append(buffer)

    # 合并后若仍有超长，再按句界切（保护性处理）
    final: list[str] = []
    for c in chunks:
        if len(c) > _CHUNK_MAX:
            final.extend(_split_by_sentence(c))
        else:
            final.append(c)
    return final


def _split_by_sentence(text: str) -> list[str]:
    """按中文句界切分并保留完整 chunk，返回列表。"""

    # 在句末标点后拆分，保留标点
    parts = re.split(r"([。！？；])", text)
    sentences: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        # 已积累到一个完整句子，或缓冲过长
        if len(buf) >= _CHUNK_TARGET or (buf and buf[-1] in "。！？；"):
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences


def select_relevant_chunks(question: str, document_text: str) -> list[str] | None:
    """从用户上传材料中检索与问题最相关的 top3 片段。

    - 材料短于 _DOC_RAG_MIN_LEN 时返回 None，调用方直接全文注入；
    - 否则切块 → embedding 粗排 → reranker 精排；
    - 返回 chunk 文本列表。
    """
    if not document_text:
        return None
    if len(document_text) < _DOC_RAG_MIN_LEN:
        return None

    chunks = split_chunks(document_text)
    if not chunks:
        return None

    with span("upload_retrieval", chunks=len(chunks)):
        # 1) 向量粗排：归一化 embedding 点积即余弦
        embedding_model = get_embedding_model()
        chunk_embeddings = np.asarray(embedding_model.embed_documents(chunks))
        q_vec = np.asarray(embedding_model.embed_query(question))
        coarse_scores = chunk_embeddings @ q_vec
        coarse_indices = np.argsort(-coarse_scores)[:_COARSE_K]
        coarse_candidates = [chunks[i] for i in coarse_indices]

        # 2) CrossEncoder 精排，不设 min_score
        candidates = [{"text": c} for c in coarse_candidates]
        ranked = get_reranker().rerank(
            question, candidates, top_n=DOC_TOP_N, min_score=None
        )
        hits = [r["text"] for r in ranked]
        event("upload_retrieval.hits", count=len(hits))
        return hits
