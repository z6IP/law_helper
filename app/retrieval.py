"""混合检索：BM25 + 向量语义 + RRF 融合，支持多路检索。

工程约束：
- 使用 RRF 融合，BM25_WEIGHT=0.5、RRF_LAMBDA=60；
- 向量检索与 BM25 关键词召回双路，结果用于后续重排序。
- 动态 embedding 维度检测：如果 ChromaDB 中存储的向量维度
  与当前 embedding 模型输出维度不匹配，会在加载时自动检测并标记，
  由上层（main.py startup）负责重建索引。
- 多路检索：按 source 分路并行检索 + 全局路合并候选，
  保证跨法规召回覆盖（如治安管理处罚法不被道交法挤占）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.errors import RetrievalError
from app.ingestion import COLLECTION_NAME
from app.tracing import event, span


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut(text) if t.strip()]


def _rrf(rank: int) -> float:
    settings = get_settings()
    return 1.0 / (settings.rrf_lambda + rank)


def _get_embedding_dim() -> int:
    """探测当前 embedding 模型的输出维度。"""
    test_vec = get_embedding_model().embed_query("维度检测")
    return len(test_vec)


def _rank_from_scores(desc_scores: np.ndarray) -> np.ndarray:
    """将（降序优先的）得分数组转换为每个元素的 rank（0 起）。"""
    order = np.argsort(desc_scores)
    rank = np.empty_like(order, dtype=int)
    rank[order] = np.arange(len(order))
    return rank


class RetrievalEngine:
    """BM25 + 向量双路召回（RRF 融合），支持按 source 多路检索。"""

    def __init__(self) -> None:
        self._loaded = False
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._dim_mismatch: bool = False  # 维度不匹配标记
        self._source_groups: dict[str, list[int]] | None = None  # source → 全库索引

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_full_dir))
        # HNSW 索引配置：ef_construction 影响构建速度，ef_search 影响查询速度
        # 小数据集（72 条法条）用较小参数即可，加速构建和查询
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 16,
                "hnsw:M": 16,
            },
        )
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        self._ids = result["ids"]
        self._documents = result["documents"] or []
        self._metadatas = result["metadatas"] or []
        emb = result.get("embeddings")
        self._embeddings = np.asarray(emb) if emb is not None and len(emb) > 0 else None

        # 维度检测：如果已有向量维度与当前模型不匹配，标记
        if self._embeddings is not None and len(self._embeddings) > 0:
            expected_dim = _get_embedding_dim()
            actual_dim = self._embeddings.shape[1]
            if actual_dim != expected_dim:
                event(
                    "retrieval.dim_mismatch",
                    stored_dim=actual_dim,
                    expected_dim=expected_dim,
                )
                self._dim_mismatch = True

        if self._documents and not self._dim_mismatch:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._documents])
        self._loaded = True

    @property
    def needs_rebuild(self) -> bool:
        """索引是否需要重建（维度不匹配）。"""
        return self._dim_mismatch

    def _group_by_source(self) -> dict[str, list[int]]:
        """构建 source → 全库索引列表的映射（首次调用时缓存）。"""
        if self._source_groups is None:
            groups: dict[str, list[int]] = {}
            for i, m in enumerate(self._metadatas):
                src = m.get("source", "")
                groups.setdefault(src, []).append(i)
            self._source_groups = groups
        return self._source_groups

    def _fuse_and_topk(
        self,
        query: str,
        indices: list[int] | np.ndarray,
        top_k: int,
        bm25_scores_all: np.ndarray | None = None,
        q_vec: np.ndarray | None = None,
    ) -> list[dict]:
        """对指定索引子集做 BM25+向量 RRF 融合，返回 top_k。

        为避免重复计算，bm25_scores_all 与 q_vec 可由调用方预计算后传入；
        未传入时内部计算（多路场景下预计算一次复用）。
        """
        indices = list(indices)
        if not indices:
            return []

        if bm25_scores_all is None:
            bm25_scores_all = np.asarray(self._bm25.get_scores(_tokenize(query)))
        if q_vec is None:
            q_vec = np.asarray(get_embedding_model().embed_query(query))

        bm25_scores = bm25_scores_all[indices]
        if self._embeddings is not None:
            vec_scores_all = self._embeddings @ q_vec
            vec_scores = vec_scores_all[indices]
        else:
            vec_scores = np.zeros(len(indices))

        bm25_rank = _rank_from_scores(-bm25_scores)
        vec_rank = _rank_from_scores(-vec_scores)

        settings = get_settings()
        w_bm25 = settings.bm25_weight
        w_vec = 1.0 - settings.bm25_weight

        scores = np.asarray(
            [w_bm25 * _rrf(bm25_rank[i]) + w_vec * _rrf(vec_rank[i]) for i in range(len(indices))]
        )
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "id": self._ids[indices[i]],
                "text": self._documents[indices[i]],
                "metadata": self._metadatas[indices[i]],
                "score": float(scores[i]),
            }
            for i in order
        ]

    @staticmethod
    def _merge_dedupe(global_hits: list[dict], per_source_hits: list[dict]) -> list[dict]:
        """合并去重：同 id 保留全局路得分（全局路优先），分路结果仅补充不重复的法条。"""
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for h in global_hits:
            if h["id"] not in seen_ids:
                seen_ids.add(h["id"])
                merged.append(h)
        for h in per_source_hits:
            if h["id"] not in seen_ids:
                seen_ids.add(h["id"])
                merged.append(h)
        return merged

    def _single_route_search(self, query: str, top_k: int) -> list[dict]:
        """单路检索（原逻辑，multi_route_enabled=False 时使用）。"""
        n = len(self._documents)

        # 1) BM25 得分
        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))

        # 2) 向量余弦相似度
        q_vec = np.asarray(get_embedding_model().embed_query(query))
        if self._embeddings is None:
            vec_scores = np.zeros(n)
        else:
            vec_scores = self._embeddings @ q_vec  # 已归一化，点积即余弦

        # 3) 各自排序（rank 从 0 开始 -> RRF 用 +1）
        bm25_rank = _rank_from_scores(-bm25_scores)
        vec_rank = _rank_from_scores(-vec_scores)

        settings = get_settings()
        w_bm25 = settings.bm25_weight
        w_vec = 1.0 - settings.bm25_weight

        scores = np.asarray(
            [w_bm25 * _rrf(bm25_rank[i]) + w_vec * _rrf(vec_rank[i]) for i in range(n)]
        )
        order = np.argsort(-scores)[:top_k]

        return [
            {
                "id": self._ids[i],
                "text": self._documents[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in order
        ]

    def search(self, query: str, top_k: int) -> list[dict]:
        """返回融合排序后的 top_k 结果。

        每项结构：{"id", "text", "metadata", "score"}

        多路检索：全局路（无过滤 top_k_global 条）+ 按 source 分路（每路 top_k_per_source 条）
        合并去重后返回，保证跨法规召回覆盖。
        """
        self._ensure_loaded()
        if not self._documents or self._dim_mismatch:
            # 维度不匹配时回退到仅 BM25 搜索
            if not self._documents or self._bm25 is None:
                return []
            n = len(self._documents)
            bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
            order = np.argsort(-bm25_scores)[:top_k]
            return [
                {
                    "id": self._ids[i],
                    "text": self._documents[i],
                    "metadata": self._metadatas[i],
                    "score": float(bm25_scores[i]),
                }
                for i in order
            ]

        settings = get_settings()
        if not settings.multi_route_enabled:
            return self._single_route_search(query, top_k)

        # 多路检索：预计算 BM25 全库得分和查询向量，分路复用，避免重复 API 调用
        bm25_scores_all = np.asarray(self._bm25.get_scores(_tokenize(query)))
        q_vec = np.asarray(get_embedding_model().embed_query(query))

        # 路径 1：全局路（无过滤，整体最相关法条）
        all_indices = list(range(len(self._documents)))
        global_hits = self._fuse_and_topk(
            query, all_indices, settings.top_k_global,
            bm25_scores_all=bm25_scores_all, q_vec=q_vec,
        )

        # 路径 2：按 source 分路（保证每个法规都有候选进入重排）
        source_groups = self._group_by_source()
        per_source_hits: list[dict] = []
        for source, indices in source_groups.items():
            hits = self._fuse_and_topk(
                query, indices, settings.top_k_per_source,
                bm25_scores_all=bm25_scores_all, q_vec=q_vec,
            )
            per_source_hits.extend(hits)

        # 合并去重
        merged = self._merge_dedupe(global_hits, per_source_hits)

        event(
            "retrieval.multi_route",
            global_count=len(global_hits),
            per_source_count=len(per_source_hits),
            merged_count=len(merged),
        )

        # 截断到调用方要求的 top_k（实际由 rerank 再次排序，此处保留全部合并结果）
        # 注：不截断，让 rerank 在更大候选集上选 top_n，跨法规覆盖更好
        return merged

    def _definition_boost_search(self, query: str, top_k: int) -> list[dict]:
        """定义性条款召回路径：在 role=definition 子集上按 source 分组检索。

        当查询含定义性概念词（拘留/处罚/种类/定义等）时，在定义性条款子集上
        做 BM25+向量 RRF 融合，召回定义性条款。解决"行为查询 vs 定义条款"
        语义鸿沟问题（如"拘留"对应治安管理处罚法第十条定义"行政拘留"）。

        关键设计：
        1. 按 source 分组检索，保证每个法规的定义性条款都有候选进入
           （否则道交法定义性条款因语义更接近查询而占满 top_k，
           治安管理处罚法第十条永远无法被召回）
        2. 不截断：每个 source 返回全部定义性条款（通常 < 30 条），
           合并后也不截断，让 rerank 模型在完整候选集上排序
           （避免低 RRF 分数的第十条被 cap 截断）
        """
        self._ensure_loaded()
        if not self._documents or self._bm25 is None:
            return []

        # 预计算全库 BM25 得分和查询向量，分路复用
        bm25_scores_all = np.asarray(self._bm25.get_scores(_tokenize(query)))
        q_vec = np.asarray(get_embedding_model().embed_query(query))

        # 按 source 分组，在定义性子集上分路检索（不截断，返回全部）
        source_groups = self._group_by_source()
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for source, all_indices in source_groups.items():
            # 该 source 内的定义性索引
            def_indices = [
                i for i in all_indices
                if (self._metadatas[i] or {}).get("role") == "definition"
            ]
            if not def_indices:
                continue
            # top_k 设为 len(def_indices)，返回该 source 全部定义性条款
            hits = self._fuse_and_topk(
                query, def_indices, len(def_indices),
                bm25_scores_all=bm25_scores_all, q_vec=q_vec,
            )
            for h in hits:
                if h["id"] not in seen_ids:
                    seen_ids.add(h["id"])
                    merged.append(h)

        return merged

    # 定义性概念关键词：查询命中时触发定义性召回路径
    _DEFINITION_KEYWORDS = (
        "拘留", "处罚", "种类", "定义", "什么是", "概念",
        "罚款", "警告", "吊销", "暂扣", "驱逐",
    )

    def _has_definition_intent(self, query: str) -> bool:
        """检测查询是否含定义性概念词。"""
        return any(kw in query for kw in self._DEFINITION_KEYWORDS)

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int,
        hyde_vector: list[float] | None = None,
    ) -> list[dict]:
        """多查询并行检索 + 跨查询 RRF 融合，可选 HyDE 向量路。

        对每条查询调用 search() 得到候选及该查询内的排名，按 RRF
        （1/(lambda+rank)）跨查询累加得分，去重后按总分降序返回。

        - 无 HyDE 向量且单查询时直接退化为 search()，避免无谓融合开销
        - 多查询或传入 HyDE 向量时跨路 RRF 是 scale-free，无需归一化各路得分
        - 改写查询与 HyDE 向量仅用于检索，最终回答仍来自 rerank 选出的法条原文
        - HyDE 向量路只参与排名融合，不引入任何文本（合规：假设性文档不进 context）
        - 末尾按总分降序截断到 top_k*3，与改造前候选规模一致
        """
        if not queries and hyde_vector is None:
            return []
        if not queries:
            # 仅 HyDE 向量路：等价于纯向量检索
            return self._hyde_only_search(hyde_vector, top_k)
        if len(queries) == 1 and hyde_vector is None:
            # 单查询无 HyDE：若查询含定义性概念词，仍走定义性召回路径，
            # 否则定义性条款（如治安管理处罚法第十条）永远不会进入候选。
            if not self._has_definition_intent(queries[0]):
                return self.search(queries[0], top_k)
            # 含定义性意图：走融合路径（定义性召回加权 ×4）
            queries = list(queries)  # 避免修改入参

        # 每条查询的 ranked 列表（按 score 降序，rank 0 起）
        per_query_ranked: list[list[dict]] = []
        for q in queries:
            hits = self.search(q, top_k)
            # search() 已按 score 降序返回；保险起见再排一次
            hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
            per_query_ranked.append(hits)

        # id → (累加 RRF 分, doc)
        fused: dict[str, dict] = {}
        scores: dict[str, float] = {}
        for ranked in per_query_ranked:
            for rank, h in enumerate(ranked):
                did = h["id"]
                scores[did] = scores.get(did, 0.0) + _rrf(rank)
                if did not in fused:
                    fused[did] = h

        # 定义性召回路径：查询含定义性概念词时，在 role=definition 子集上检索
        # 解决"行为查询 vs 定义条款"语义鸿沟（如"拘留"对应第十条定义"行政拘留"）
        # 定义性路径 RRF 分数加权（×4），相当于多条查询命中，弥补单路劣势
        # 配额保障：定义性 top_k 条候选强制纳入融合候选池（不被主路径高分挤掉）
        def_intent = any(self._has_definition_intent(q) for q in queries)
        def_hits: list[dict] = []
        if def_intent:
            def_hits = self._definition_boost_search(queries[0], max(top_k * 3, top_k))
            def_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
            for rank, h in enumerate(def_hits):
                did = h["id"]
                scores[did] = scores.get(did, 0.0) + _rrf(rank) * 4
                if did not in fused:
                    fused[did] = h

        # HyDE 向量路：取 top_k*3 向量相似度候选参与 RRF 融合
        # 不遍历全库（库可能上千条，会触发 rerank API 文档数上限）
        if hyde_vector is not None and self._embeddings is not None:
            hyde_vec = np.asarray(hyde_vector, dtype=self._embeddings.dtype)
            hyde_scores = self._embeddings @ hyde_vec
            hyde_top_n = max(top_k * 3, top_k)
            hyde_order = np.argsort(-hyde_scores)[:hyde_top_n]
            for rank, i in enumerate(hyde_order):
                did = self._ids[i]
                scores[did] = scores.get(did, 0.0) + _rrf(rank)
                if did not in fused:
                    fused[did] = {
                        "id": did,
                        "text": self._documents[i],
                        "metadata": self._metadatas[i],
                        "score": float(hyde_scores[i]),
                    }

        # 按总分降序排序并截断到 top_k*3，与改造前候选规模一致
        cap = max(top_k * 3, top_k)
        merged = [
            {**fused[did], "score": scores[did]}
            for did in sorted(scores, key=lambda d: -scores[d])
        ][:cap]

        # 定义性配额保障：定义性候选若因 RRF 分数低被挤出 cap，
        # 追加 top 20 条定义性候选到 merged 末尾，确保定义性条款（如治安管理处罚法第十条）
        # 有机会进入 rerank 候选池，由 rerank 模型最终决定排序。
        # def_hits 已按 score 降序排序，前 20 条即最高分的定义性条款。
        # 上限 20 条控制 rerank 候选规模（约 30+20=50 条），降低 API 调用成本。
        DEF_QUOTA = 20
        if def_intent and def_hits:
            existing_ids = {h["id"] for h in merged}
            added = 0
            for h in def_hits:
                if added >= DEF_QUOTA:
                    break
                if h["id"] not in existing_ids:
                    merged.append({**h, "score": scores.get(h["id"], 0.0)})
                    existing_ids.add(h["id"])
                    added += 1
            event(
                "retrieval.definition_quota",
                quota_added=added,
                merged_count=len(merged),
                def_hits_count=len(def_hits),
                quota_limit=DEF_QUOTA,
            )

        event(
            "retrieval.multi_query",
            query_count=len(queries),
            hyde_enabled=hyde_vector is not None,
            definition_boost=def_intent,
            merged_count=len(merged),
        )
        return merged

    def _hyde_only_search(self, hyde_vector: list[float], top_k: int) -> list[dict]:
        """仅 HyDE 向量检索路（无文本查询时使用）。"""
        self._ensure_loaded()
        if self._embeddings is None or not self._documents:
            return []
        hyde_vec = np.asarray(hyde_vector, dtype=self._embeddings.dtype)
        scores = self._embeddings @ hyde_vec
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "id": self._ids[i],
                "text": self._documents[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in order
        ]


@lru_cache
def get_retrieval_engine() -> RetrievalEngine:
    engine = RetrievalEngine()
    engine._ensure_loaded()
    return engine
