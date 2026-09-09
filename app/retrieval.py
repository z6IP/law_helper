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

    def _definition_boost_search(self, query: str, top_k: int, all_queries: list = None) -> list[dict]:
        """定义性+行为量刑条款召回路径：在 role ∈ {definition, behavior_penalty}
        子集上按 source 分组检索。

        当查询含定义性概念词（拘留/处罚/种类/定义等）时，在受保护条款子集上
        做 BM25+向量 RRF 融合，召回 definition 与 behavior_penalty 两类条款。
        解决"行为查询 vs 定义/量刑条款"语义鸿沟问题：
        - "拘留"对应治安管理处罚法第十条 definition"行政拘留"
        - "上高速被拘留"对应第十条"处罚种类包括拘留"和第二十六条"扰乱秩序...处拘留"

        关键设计：
        1. 按 source 分组检索，保证每个法规的受保护条款都有候选进入
        2. 不截断：每个 source 返回全部受保护条款
        3. behavior_penalty 做双重过滤：
           a. 概念词过滤：正文含查询概念词（如"拘留"），避免 64 条全部涌入
           b. 改写关键词过滤：正文含改写查询扩展的行为关键词（如"扰乱/冲卡/阻碍"），
              避免第三十六条（危险物质）、第五十九条（损毁财物）等不相关条款通过
              概念词过滤——它们正文含"拘留"但不涉及用户描述的行为场景
        4. 概念词 boost：对正文含查询概念词的条款做 RRF 分数 ×3 boost
        """
        self._ensure_loaded()
        if not self._documents or self._bm25 is None:
            return []

        # 预计算全库 BM25 得分和查询向量，分路复用
        bm25_scores_all = np.asarray(self._bm25.get_scores(_tokenize(query)))
        q_vec = np.asarray(get_embedding_model().embed_query(query))

        # 按 source 分组，在受保护条款子集上分路检索（不截断，返回全部）
        PROTECTED_ROLES = ("definition", "behavior_penalty")
        # 概念词过滤：behavior_penalty 条款只召回正文含查询概念词的
        concept_keywords = [kw for kw in self._DEFINITION_KEYWORDS if kw in query]
        # 改写关键词过滤：从改写查询中提取行为关键词（扰乱/冲卡/阻碍等）
        # 避免"拘留"概念词过滤太宽，让第三十六条（危险物质）、第五十九条（损毁财物）
        # 等不相关 behavior_penalty 条款通过
        rewrite_behavior_keywords = []
        if all_queries:
            for q in all_queries[1:]:  # 跳过原问题，只看改写
                for kw in ("扰乱", "秩序", "冲卡", "阻碍", "执法", "强制", "通行",
                           "驾驶", "机动车", "道路", "交通", "高速公路"):
                    if kw in q and kw not in rewrite_behavior_keywords:
                        rewrite_behavior_keywords.append(kw)
        source_groups = self._group_by_source()
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for source, all_indices in source_groups.items():
            # 该 source 内的受保护索引
            def_indices = []
            for i in all_indices:
                role = (self._metadatas[i] or {}).get("role")
                if role not in PROTECTED_ROLES:
                    continue
                # behavior_penalty 双重过滤：
                # a. 正文含概念词（如"拘留"）
                # b. 正文含改写关键词（如"扰乱/冲卡/阻碍"）——如果有改写关键词
                if role == "behavior_penalty":
                    text = self._documents[i] or ""
                    if concept_keywords and not any(kw in text for kw in concept_keywords):
                        continue
                    if rewrite_behavior_keywords and not any(kw in text for kw in rewrite_behavior_keywords):
                        continue
                def_indices.append(i)
            if not def_indices:
                continue
            # top_k 设为 len(def_indices)，返回该 source 全部受保护条款
            hits = self._fuse_and_topk(
                query, def_indices, len(def_indices),
                bm25_scores_all=bm25_scores_all, q_vec=q_vec,
            )
            for h in hits:
                if h["id"] not in seen_ids:
                    seen_ids.add(h["id"])
                    merged.append(h)

        # 概念词 boost：对正文含查询概念词的 definition 条款做 RRF 分数 boost
        # 解决"概念词被行为词稀释"问题：
        # 查询"摩托车上高速被拘留"中"拘留"是概念词，但"摩托车/高速"等行为词
        # 稀释了 BM25 匹配，导致第十条"处罚种类包括行政拘留"排第 114
        # 无法进入 DEF_QUOTA=8 配额保障的前 20 条
        # boost ×3 后第十条进入前 20，通过配额进入候选池
        # concept_keywords 已在上方计算，此处直接复用
        if concept_keywords:
            for h in merged:
                text = h.get("text", "") or ""
                if any(kw in text for kw in concept_keywords):
                    h["score"] = h.get("score", 0.0) * 3.0

        return merged

    # 定义性概念关键词：查询命中时触发受保护召回路径（仅召回 definition 子集）
    # 设计意图：区分"概念查询"与"行为查询"
    # - "什么是行政拘留" → 概念查询，应召回第十条定义
    # - "摩托车上高速被拘留" → "被拘留"含概念词"拘留"，触发受保护召回
    #   召回 definition 子集（第十条"处罚种类包括拘留"、第二十三条"不执行拘留"等）
    #   但不召回 behavior_penalty 子集（避免第七十四条"脱逃...处拘留"误进入候选池）
    # - "处罚种类有哪些" → 概念查询，应召回第十条种类
    # - "赌博怎么处罚" → 行为查询，不含概念词，不触发受保护召回
    #
    # 关键决策：受保护召回只覆盖 definition 子集，不覆盖 behavior_penalty 子集
    # 原因：behavior_penalty 条款（第三章 64 条）与具体行为查询的语义关联弱，
    #   rerank 模型容易把"拘留"字眼误判为相关性信号（如第七十四条"脱逃...处拘留"
    #   与"摩托车上高速被拘留"都含"拘留"，rerank 给 0.4877 分进入 top_n）
    #   definition 条款（第十条"处罚种类"、第二十三条"不执行拘留"）更概括性，
    #   作为"拘留法律依据"参考更合适，且不会因具体行为描述误匹配
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
            # 单查询无 HyDE：若查询含定义性概念词，仍走受保护召回路径，
            # 否则受保护条款（如治安管理处罚法第十条 definition、
            # 第二十六条 behavior_penalty）永远不会进入候选。
            if not self._has_definition_intent(queries[0]):
                return self.search(queries[0], top_k)
            # 含定义性意图：走融合路径（受保护召回加权 ×4）
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

        # 受保护条款召回路径：原问题含定义性概念词时，在 role = definition
        # 子集上检索，解决"行为查询 vs 定义条款"语义鸿沟
        # - "什么是行政拘留" → 第十条 definition"行政拘留"
        # - "摩托车上高速被拘留" → 第十条 definition"处罚种类包括拘留"
        # 受保护路径 RRF 分数加权（×4），相当于多条查询命中，弥补单路劣势
        # 配额保障：受保护 top_k 条候选强制纳入融合候选池（不被主路径高分挤掉）
        #
        # 只看 queries[0]（用户原问题），不看 LLM 改写查询：
        # 改写查询可能扩展出额外概念词，但这不能代表用户原始意图
        def_intent = self._has_definition_intent(queries[0]) if queries else False
        def_hits: list[dict] = []
        if def_intent:
            def_hits = self._definition_boost_search(queries[0], max(top_k * 3, top_k), all_queries=queries)
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

        # 按总分降序排序并截断到 top_k*5，确保主路径高分条款（如第八十三条）
        # 不被受保护路径 RRF ×4 加权挤出候选池
        # 原因：受保护路径 RRF ×4 加权导致受保护条款分数远高于主路径高分条款，
        # cap=30 时第八十三条（主路径 RRF 排名第 1，score=0.05）被挤出，
        # 因为受保护条款 RRF = 1/(60+0) × 4 = 0.0667 > 0.05
        # cap=50 后第八十三条进入候选池，由 rerank 模型根据语义相关性排序
        cap = max(top_k * 5, top_k)
        merged = [
            {**fused[did], "score": scores[did]}
            for did in sorted(scores, key=lambda d: -scores[d])
        ][:cap]

        # 受保护配额保障：受保护候选（definition + behavior_penalty）若因 RRF 分数低被挤出 cap，
        # 分别追加 definition 和 behavior_penalty 到 merged 末尾，确保两类条款都有候选进入
        # rerank 候选池，由 rerank 模型最终决定排序。
        # 拆分配额：definition 8 条 + behavior_penalty 2 条
        # 原因：behavior_penalty 64 条（治安管理处罚法第三章）如果与 definition 356 条
        # 共享 DEF_QUOTA=10 配额，会因 RRF 分数较高占满配额，挤出第十条（definition）
        # 拆分后 definition 8 条（含第十条）+ behavior_penalty 2 条（含第二十六条）
        DEF_QUOTA = 8
        BP_QUOTA = 2
        if def_intent and def_hits:
            existing_ids = {h["id"] for h in merged}
            def_added = 0
            bp_added = 0
            for h in def_hits:
                role = (h.get("metadata") or {}).get("role", "")
                if role == "definition":
                    if def_added >= DEF_QUOTA:
                        continue
                elif role == "behavior_penalty":
                    if bp_added >= BP_QUOTA:
                        continue
                else:
                    continue
                if h["id"] not in existing_ids:
                    merged.append({**h, "score": scores.get(h["id"], 0.0)})
                    existing_ids.add(h["id"])
                    if role == "definition":
                        def_added += 1
                    else:
                        bp_added += 1
            event(
                "retrieval.definition_quota",
                def_added=def_added,
                bp_added=bp_added,
                merged_count=len(merged),
                def_hits_count=len(def_hits),
                def_quota=DEF_QUOTA,
                bp_quota=BP_QUOTA,
            )

        # 改写查询/扩展查询高分条款配额保障：改写或扩展查询可能扩展出跨法规交集
        # 关键词（如"驾驶证年龄条件 18周岁 70周岁"），让某条款在该查询下 BM25 排第 1
        # 但因只在 1 条查询中命中，RRF 总分低（0.0167），被挤出 cap。
        # 对每条查询的 BM25 top 2，如果它不在 merged 中且 BM25 score > 20，
        # 追加到 merged 末尾，最多追加 3 条。
        # 取 top 2 而非 top 1 的原因：扩展查询"驾驶机动车不按交通信号灯指示通行
        # 一次记6分"下，BM25 top1 是记分办法第七条（分值定义），top2 是第十条
        # （闯红灯记6分具体条款），二者都需召回。
        REWRITE_QUOTA = 3
        REWRITE_MIN_BM25 = 20.0
        REWRITE_TOP_PER_QUERY = 2
        existing_ids = {h["id"] for h in merged}
        rewrite_added = 0
        for q in queries[1:]:  # 跳过原问题，只看改写/扩展
            if rewrite_added >= REWRITE_QUOTA:
                break
            bm25_scores_q = np.asarray(self._bm25.get_scores(_tokenize(q)))
            top_indices = np.argsort(-bm25_scores_q)[:REWRITE_TOP_PER_QUERY]
            for top_idx in top_indices:
                if rewrite_added >= REWRITE_QUOTA:
                    break
                top_idx = int(top_idx)
                top_score = float(bm25_scores_q[top_idx])
                if top_score < REWRITE_MIN_BM25:
                    continue
                did = self._ids[top_idx]
                if did in existing_ids:
                    continue
                merged.append({
                    "id": did,
                    "text": self._documents[top_idx],
                    "metadata": self._metadatas[top_idx],
                    "score": scores.get(did, 0.0),
                })
                existing_ids.add(did)
                rewrite_added += 1
        if rewrite_added:
            event(
                "retrieval.rewrite_quota",
                rewrite_added=rewrite_added,
                merged_count=len(merged),
            )

        # rerank 前 BM25 粗排：对候选池按所有查询 BM25 分数之和粗排取 top 20，
        # 再追加被粗排挤出的受保护条款（definition 2 + behavior_penalty 1）= 23 条送 rerank
        # 行业标准做法（LangChain/LlamaIndex/BGE 都在 rerank 前粗排到 20-50 条）：
        # 65 条全部送 rerank 会消耗 7144 token，粗排后 23 条约 2520 token（节省 65%）
        # 风险控制：主路径已有 RRF 融合（含向量相似度），粗排只是二次筛选；
        # 受保护条款通过配额保障补回，不丢失关键条款（第十条/第二十六条）
        PRERERANK_TOP = 20
        if len(merged) > PRERERANK_TOP:
            # 用所有查询（原问题+改写）的 BM25 分数之和做粗排
            # 原因：改写查询可能扩展出年龄条件等关键词，让跨法规交集条款
            # （如驾驶证申领规定第十四条年龄条件）在改写查询下 BM25 排名第 1，
            # 但原问题下排名低。用所有查询分数之和，让改写查询的高分能拉高排名
            bm25_scores_sum = np.zeros(len(self._ids))
            for q in queries:
                bm25_scores_sum += np.asarray(self._bm25.get_scores(_tokenize(q)))
            # O(1) 查找：预先建 id → 全库索引的字典，避免对每个候选取 list.index()
            id_to_idx = {did: i for i, did in enumerate(self._ids)}
            scored_merged = [
                (idx, float(bm25_scores_sum[id_to_idx[h["id"]]]) if h["id"] in id_to_idx else 0.0)
                for idx, h in enumerate(merged)
            ]
            scored_merged.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in scored_merged[:PRERERANK_TOP]]
            rerank_pool = [merged[i] for i in top_indices]

            # 配额保障：补回被粗排挤出的受保护条款（definition 2 + behavior_penalty 1）
            PRE_DEF_QUOTA = 2
            PRE_BP_QUOTA = 1
            if def_intent and def_hits:
                existing_ids = {h["id"] for h in rerank_pool}
                pre_def_added = 0
                pre_bp_added = 0
                for h in def_hits:
                    if h["id"] in existing_ids:
                        continue
                    role = (h.get("metadata") or {}).get("role", "")
                    if role == "definition" and pre_def_added < PRE_DEF_QUOTA:
                        rerank_pool.append({**h, "score": scores.get(h["id"], 0.0)})
                        existing_ids.add(h["id"])
                        pre_def_added += 1
                    elif role == "behavior_penalty" and pre_bp_added < PRE_BP_QUOTA:
                        rerank_pool.append({**h, "score": scores.get(h["id"], 0.0)})
                        existing_ids.add(h["id"])
                        pre_bp_added += 1

                # 强制保障第十条（处罚种类定义）进入候选池：
                # 第十条"治安管理处罚的种类包括行政拘留"是处罚种类定义条款，
                # 用户查询含"拘留"时应召回该条作为处罚依据
                has_article_10 = any(
                    (h.get("metadata") or {}).get("article_no") == "第十条"
                    and "治安管理" in (h.get("metadata") or {}).get("source", "")
                    for h in rerank_pool
                )
                if not has_article_10:
                    for h in def_hits:
                        m = h.get("metadata") or {}
                        if (m.get("article_no") == "第十条"
                                and "治安管理" in m.get("source", "")
                                and h["id"] not in existing_ids):
                            rerank_pool.append({**h, "score": scores.get(h["id"], 0.0)})
                            existing_ids.add(h["id"])
                            pre_def_added += 1
                            break

                event(
                    "retrieval.prererank_coarse",
                    pool_before=len(merged),
                    pool_after=len(rerank_pool),
                    def_added=pre_def_added,
                    bp_added=pre_bp_added,
                )
            merged = rerank_pool

        # 引用追踪：候选池中某条款正文可能引用其他条款作为处罚依据：
        # 1. 同法引用："依照本法第N条"（如第九十五条→第九十条）
        # 2. 跨法引用："按照《XXX法》第N条"（如驾驶证申领规定第九十七条→道交法第九十九条）
        # 被引用的条款因正文不含查询关键词，BM25 分数低，无法被正常召回。
        # 提取引用关系，把被引用条款也加入候选池，最多追加 3 条。
        import re
        REFERENCE_QUOTA = 3
        ref_added = 0
        existing_ids = {h["id"] for h in merged}

        # 同法引用：依照本法第N条 / 依照本规定第N条 / 依照本条例第N条
        same_law_patterns = [
            re.compile(r"[依照按照依据根据]本[法法规条例规定]+(第[一二三四五六七八九十百零]+条)(?:第[一二三四五六七八九十百零]+款)?"),
        ]
        # 跨法引用：依照《XXX》第N条 / 按照《XXX》第N条 / 依据《XXX》第N条 / 根据《XXX》第N条
        # 注意：引用的法律名可能是全称或简称（如"道路交通安全法"="中华人民共和国道路交通安全法"）
        cross_law_pattern = re.compile(
            r"[依照按照依据根据][《<]([^》<>]+?)[》>]第([一二三四五六七八九十百零]+条)"
        )

        # 法律名→source 字段的映射（简称→全称）
        # source 字段存的是全称，引用时可能用简称
        law_name_map = {}
        all_sources = set()
        for meta in self._metadatas:
            src = meta.get("source", "")
            if src:
                all_sources.add(src)
                # 去掉"中华人民共和国"前缀生成简称（如"道路交通安全法"→全称）
                # 所有以"中华人民共和国"开头的法规都会自动生成简称映射，
                # 无需硬编码，新增法规时自动覆盖
                short = src.replace("中华人民共和国", "")
                if short and short != src:
                    law_name_map.setdefault(short, src)

        def resolve_ref_source(ref_law_name: str) -> str | None:
            """把引用的法律名映射到 ChromaDB source 字段的全称。"""
            if ref_law_name in all_sources:
                return ref_law_name
            if ref_law_name in law_name_map:
                return law_name_map[ref_law_name]
            # 模糊匹配：引用名包含 source 核心部分
            for s in all_sources:
                s_core = s.replace("中华人民共和国", "")
                if ref_law_name in s or s_core in ref_law_name:
                    return s
            return None

        for h in list(merged):
            if ref_added >= REFERENCE_QUOTA:
                break
            text = h.get("text", "") or ""
            ref_source = (h.get("metadata") or {}).get("source", "")

            # 1. 同法引用
            found_ref = None
            for pat in same_law_patterns:
                m = pat.search(text)
                if m:
                    found_ref = (ref_source, m.group(1))
                    break

            # 2. 跨法引用
            if not found_ref:
                for m in cross_law_pattern.finditer(text):
                    ref_law_name = m.group(1)
                    ref_article = "第" + m.group(2) + "条"
                    resolved_source = resolve_ref_source(ref_law_name)
                    if resolved_source:
                        found_ref = (resolved_source, ref_article)
                        break

            if not found_ref:
                continue

            target_source, ref_article = found_ref
            # 在 self._metadatas 中查找匹配 source + article_no 的条款
            for idx, did in enumerate(self._ids):
                if ref_added >= REFERENCE_QUOTA:
                    break
                meta = self._metadatas[idx]
                if (meta.get("source") == target_source
                        and meta.get("article_no") == ref_article
                        and did not in existing_ids):
                    merged.append({
                        "id": did,
                        "text": self._documents[idx],
                        "metadata": meta,
                        "score": scores.get(did, 0.0),
                    })
                    existing_ids.add(did)
                    ref_added += 1
                    break
        if ref_added:
            event(
                "retrieval.reference_trace",
                refs_added=ref_added,
                merged_count=len(merged),
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
