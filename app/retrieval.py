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
from app.policy import get_policy
from app.tracing import event, span


def _ensure_legal_dict_loaded() -> None:
    """按需加载法律/交通领域用户词典。

    词典路径：项目根目录 / config / legal_dict.txt。
    文件缺失或加载失败时降级为默认分词，不影响主流程。

    除了 jieba.load_userdict 外，对词典中的每个词调用 add_word 强制加入，
    确保像"机动车驾驶人"、"安全头盔"等默认模型容易切分的专业术语被保留为整词。
    """
    if getattr(_ensure_legal_dict_loaded, "_loaded", False):
        return
    import jieba
    from app.config import BASE_DIR

    dict_path = BASE_DIR / "config" / "legal_dict.txt"
    if dict_path.is_file():
        try:
            jieba.load_userdict(str(dict_path))
            added = 0
            for line in dict_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                word = parts[0]
                freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                jieba.add_word(word, freq=freq)
                added += 1
            event("retrieval.legal_dict_loaded", path=str(dict_path), words=added)
        except Exception as exc:  # noqa: BLE001
            event("retrieval.legal_dict_failed", error=str(exc))
    _ensure_legal_dict_loaded._loaded = True


def _tokenize(text: str) -> list[str]:
    import jieba

    _ensure_legal_dict_loaded()
    return [t for t in jieba.cut(text) if t.strip()]


def _rrf(rank: int) -> float:
    settings = get_settings()
    return 1.0 / (settings.rrf_lambda + rank)


def _resolve_bm25_weight(query: str) -> float:
    """R9：根据查询意图动态选择 BM25 权重，未命中意图时使用 Settings 默认值。

    设计思路：
    - 概念/定义/效力类查询语义性强，降低 BM25 权重，提高向量语义权重；
    - 处罚/否定/责任类查询依赖关键词精确命中，提高 BM25 权重；
    - 其他场景保持默认平衡。
    """
    settings = get_settings()
    default_weight = settings.bm25_weight
    try:
        from app.query_expansion import classify_query_intents
    except ImportError:  # pragma: no cover
        return default_weight
    intents = classify_query_intents(query)
    if not intents:
        return default_weight
    dynamic = get_policy()["retrieval"].get("dynamic_weights", {})
    for intent in intents:
        if intent in dynamic and "bm25_weight" in dynamic[intent]:
            return float(dynamic[intent]["bm25_weight"])
    return default_weight


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
                "hnsw:construction_ef": settings.hnsw_construction_ef,
                "hnsw:search_ef": settings.hnsw_search_ef,
                "hnsw:M": settings.hnsw_M,
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
            self._bm25 = BM25Okapi([_tokenize(self._lexical_text(i)) for i in range(len(self._documents))])
        self._loaded = True

    def _lexical_text(self, index: int) -> str:
        """返回用于 BM25 的上下文文本，补充法规、章节和条号 metadata。

        penalty_context 让子款在 BM25 词项上与处罚词（如"拘留"）建立匹配，
        解决子款正文仅含行为描述、不含处罚词导致的召回失败。
        """
        metadata = self._metadatas[index] or {}
        return " ".join(
            str(value)
            for value in (
                metadata.get("source", ""),
                metadata.get("section_header", ""),
                metadata.get("article_no", ""),
                metadata.get("penalty_context", ""),
                self._documents[index],
            )
            if value
        )

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

        w_bm25 = _resolve_bm25_weight(query)
        w_vec = 1.0 - w_bm25

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

        w_bm25 = _resolve_bm25_weight(query)
        w_vec = 1.0 - w_bm25

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

    def search(
        self,
        query: str,
        top_k: int,
        law_source: str | None = None,
        article_no: str | None = None,
    ) -> list[dict]:
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
            bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
            scoped_indices = self._scope_indices(law_source, article_no)
            order = sorted(scoped_indices, key=lambda i: -bm25_scores[i])[:top_k]
            hits = [
                {
                    "id": self._ids[i],
                    "text": self._documents[i],
                    "metadata": self._metadatas[i],
                    "score": float(bm25_scores[i]),
                }
                for i in order
            ]
            return self._filter_scope(hits, law_source, article_no)

        settings = get_settings()
        if not settings.multi_route_enabled:
            scoped_indices = self._scope_indices(law_source, article_no)
            bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
            q_vec = np.asarray(get_embedding_model().embed_query(query))
            return self._fuse_and_topk(
                query,
                scoped_indices,
                top_k,
                bm25_scores_all=bm25_scores,
                q_vec=q_vec,
            )

        # 多路检索：预计算 BM25 全库得分和查询向量，分路复用，避免重复 API 调用
        bm25_scores_all = np.asarray(self._bm25.get_scores(_tokenize(query)))
        q_vec = np.asarray(get_embedding_model().embed_query(query))

        # 路径 1：全局路（无过滤，整体最相关法条）
        all_indices = self._scope_indices(law_source, article_no)
        global_hits = self._fuse_and_topk(
            query, all_indices, settings.top_k_global,
            bm25_scores_all=bm25_scores_all, q_vec=q_vec,
        )

        # 路径 2：按 source 分路（保证每个法规都有候选进入重排）
        source_groups = self._group_by_source()
        if law_source or article_no:
            source_groups = {
                source: [
                    i for i in indices
                    if i in all_indices
                ]
                for source, indices in source_groups.items()
            }
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

    def _scope_indices(
        self, law_source: str | None, article_no: str | None
    ) -> list[int]:
        return [
            i for i, metadata in enumerate(self._metadatas)
            if (not law_source or metadata.get("source") == law_source)
            and (not article_no or metadata.get("article_no") == article_no)
        ]

    @staticmethod
    def _filter_scope(
        hits: list[dict], law_source: str | None, article_no: str | None
    ) -> list[dict]:
        if not law_source and not article_no:
            return hits
        return [
            h for h in hits
            if (not law_source or (h.get("metadata") or {}).get("source") == law_source)
            and (not article_no or (h.get("metadata") or {}).get("article_no") == article_no)
        ]

    def _expand_article_hierarchy(
        self,
        hits: list[dict],
        quota: int,
        law_source: str | None = None,
        article_no: str | None = None,
    ) -> list[dict]:
        """R5：法条条-款项层级展开。

        命中父条时补充其款项子块，命中子款时带回父条上下文。
        用于解决长父条向量稀释、子款具体要件无法被单独召回的问题。
        """
        if quota <= 0 or not hits or not self._documents:
            return []

        # 构建 (source, article_no/parent_article_no) -> indices 的映射
        children_map: dict[tuple[str, str], list[int]] = {}
        parent_map: dict[tuple[str, str], int] = {}
        for idx, meta in enumerate(self._metadatas):
            src = meta.get("source", "")
            if law_source and src != law_source:
                continue
            parent_no = meta.get("parent_article_no", "")
            chunk_type = meta.get("chunk_type", "")
            art_no = meta.get("article_no", "")
            if chunk_type == "clause" and parent_no:
                children_map.setdefault((src, parent_no), []).append(idx)
            if chunk_type == "article" and art_no:
                parent_map[(src, art_no)] = idx

        added: list[dict] = []
        seen_ids: set[str] = {h["id"] for h in hits}
        for h in hits:
            meta = h.get("metadata") or {}
            src = meta.get("source", "")
            art_no = meta.get("article_no", "")
            parent_no = meta.get("parent_article_no", "")
            chunk_type = meta.get("chunk_type", "")
            if article_no and art_no != article_no and parent_no != article_no:
                continue
            # 父条命中：补充所有子款
            if chunk_type == "article":
                for idx in children_map.get((src, art_no), []):
                    did = self._ids[idx]
                    if did in seen_ids:
                        continue
                    added.append({
                        "id": did,
                        "text": self._documents[idx],
                        "metadata": self._metadatas[idx],
                        "score": h.get("score", 0.0) * 0.95,
                    })
                    seen_ids.add(did)
                    if len(added) >= quota:
                        return added
            # 子款命中：带回父条上下文
            elif chunk_type == "clause" and parent_no:
                idx = parent_map.get((src, parent_no))
                if idx is None:
                    continue
                did = self._ids[idx]
                if did in seen_ids:
                    continue
                added.append({
                    "id": did,
                    "text": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "score": h.get("score", 0.0) * 0.95,
                })
                seen_ids.add(did)
                if len(added) >= quota:
                    return added
        return added

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

        # 按 source 分组，在受保护条款子集上分路检索（不截断，返回全部）。
        # 具体行为相关性由 BM25、向量检索和 rerank 综合判断，不使用固定行为词过滤。
        retrieval_policy = get_policy()["retrieval"]
        PROTECTED_ROLES = tuple(retrieval_policy.get("protected_roles", ["definition", "behavior_penalty"]))
        concept_keywords = [kw for kw in self._DEFINITION_KEYWORDS if kw in query]
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
                # 仅保留与当前概念词有文本联系的行为处罚条款，具体相关性
                # 交给后续 BM25、向量检索和 rerank 判断。
                if role == "behavior_penalty":
                    text = self._documents[i] or ""
                    # 子款正文可能不含处罚词，合并父条处罚上下文后再判断，
                    # 避免第二十六条第一款第一项等正确子款被误删。
                    penalty_ctx = (self._metadatas[i] or {}).get("penalty_context", "")
                    combined_text = f"{text} {penalty_ctx}"
                    def has_penalty_signal(keyword: str) -> bool:
                        if keyword == "处罚":
                            markers = retrieval_policy.get(
                                "penalty_signal_markers", ["拘留", "罚款", "警告", "吊销", "暂扣"]
                            )
                            return any(marker in combined_text for marker in markers)
                        return keyword in combined_text

                    if concept_keywords and not any(
                        has_penalty_signal(kw) for kw in concept_keywords
                    ):
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
            concept_boost = get_settings().concept_score_boost
            for h in merged:
                text = h.get("text", "") or ""
                if any(kw in text for kw in concept_keywords):
                    h["score"] = h.get("score", 0.0) * concept_boost

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
    _DEFINITION_KEYWORDS = tuple(get_policy()["rerank"]["concept_keywords"])

    def _has_definition_intent(self, query: str) -> bool:
        """检测查询是否含定义性概念词。"""
        return any(kw in query for kw in self._DEFINITION_KEYWORDS)

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int,
        law_source: str | None = None,
        article_no: str | None = None,
    ) -> list[dict]:
        """多查询并行检索 + 跨查询 RRF 融合。

        对每条查询调用 search() 得到候选及该查询内的排名，按 RRF
        （1/(lambda+rank)）跨查询累加得分，去重后按总分降序返回。

        - 多查询跨路 RRF 是 scale-free，无需归一化各路得分
        - 候选融合后直接交给 rerank，不再进行 BM25 二次粗排
        """
        if not queries:
            return []
        if len(queries) == 1 and not (law_source or article_no):
            # 单查询若含定义性概念词，仍走受保护召回路径，
            # 否则受保护条款（如治安管理处罚法第十条 definition、
            # 第二十六条 behavior_penalty）永远不会进入候选。
            if not self._has_definition_intent(queries[0]):
                return self.search(queries[0], top_k)
            # 含定义性意图：走融合路径（受保护召回加权 ×4）
            queries = list(queries)  # 避免修改入参

        # 每条查询的 ranked 列表（按 score 降序，rank 0 起）
        per_query_ranked: list[list[dict]] = []
        for q in queries:
            hits = self.search(q, top_k, law_source=law_source, article_no=article_no)
            # search() 已按 score 降序返回；保险起见再排一次
            hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
            per_query_ranked.append(hits)

        # 保留每条检索路线的头部候选，防止后续受保护条款扩展把直接命中条款挤出候选池。
        # 例如原问题已经命中“高速公路上的两轮摩托车”条款，不能因为“拘留”触发了
        # 大量行为处罚条款保护召回，就在 RRF 截断阶段丢失该直接命中。
        settings = get_settings()
        route_floor = max(settings.route_floor_min, min(settings.route_floor_max, top_k // 3))
        route_floor_hits = {
            hit["id"]: hit
            for ranked in per_query_ranked
            for hit in ranked[:route_floor]
        }

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
            # 各改写查询可能表达不同的法律关系：原问题保留“拘留”语义，
            # 行为改写则可能命中“扰乱公共秩序”等具体构成要件。
            # 每条查询独立召回后合并，避免只用原问题导致行为条款排名过低。
            protected_by_id: dict[str, dict] = {}
            multiplier = max(settings.protected_search_multiplier, 1)
            for protected_query in queries:
                for hit in self._definition_boost_search(
                    protected_query, max(top_k * multiplier, top_k)
                ):
                    old = protected_by_id.get(hit["id"])
                    if old is None or hit.get("score", 0.0) > old.get("score", 0.0):
                        protected_by_id[hit["id"]] = hit
            def_hits = sorted(
                protected_by_id.values(),
                key=lambda h: h.get("score", 0.0),
                reverse=True,
            )
            for rank, h in enumerate(def_hits):
                did = h["id"]
                scores[did] = scores.get(did, 0.0) + _rrf(rank) * settings.protected_path_rrf_weight
                if did not in fused:
                    fused[did] = h

        # 按总分降序排序并截断到 top_k * fusion_candidate_cap_multiplier，确保主路径高分条款（如第八十三条）
        # 不被受保护路径加权挤出候选池。
        # 原因：受保护路径加权导致受保护条款分数远高于主路径高分条款，
        # cap=30 时第八十三条（主路径 RRF 排名第 1，score=0.05）被挤出，
        # 因为受保护条款 RRF = 1/(60+0) × 4 = 0.0667 > 0.05
        # cap 放大后第八十三条进入候选池，由 rerank 模型根据语义相关性排序
        cap = max(top_k * settings.fusion_candidate_cap_multiplier, top_k)
        merged = [
            {**fused[did], "score": scores[did]}
            for did in sorted(scores, key=lambda d: -scores[d])
        ][:cap]
        for did, hit in route_floor_hits.items():
            if did not in {item["id"] for item in merged}:
                merged.append({**hit, "score": scores.get(did, hit.get("score", 0.0))})

        # 受保护配额保障：受保护候选（definition + behavior_penalty）若因 RRF 分数低被挤出 cap，
        # 分别追加 definition 和 behavior_penalty 到 merged 末尾，确保两类条款都有候选进入
        # rerank 候选池，由 rerank 模型最终决定排序。
        # 拆分配额：definition 8 条 + behavior_penalty 2 条
        # 原因：behavior_penalty 64 条（治安管理处罚法第三章）如果与 definition 356 条
        # 共享 DEF_QUOTA=10 配额，会因 RRF 分数较高占满配额，挤出第十条（definition）
        # 拆分后 definition 8 条（含第十条）+ behavior_penalty 2 条（含第二十六条）
        retrieval_policy = get_policy()["retrieval"]
        DEF_QUOTA = retrieval_policy["definition_quota"]
        BP_QUOTA = retrieval_policy["behavior_penalty_quota"]
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
        REWRITE_QUOTA = retrieval_policy["rewrite_quota"]
        REWRITE_MIN_BM25 = retrieval_policy["rewrite_min_bm25"]
        REWRITE_TOP_PER_QUERY = retrieval_policy["rewrite_top_per_query"]
        existing_ids = {h["id"] for h in merged}
        rewrite_added = 0
        for q in queries[1:]:  # 跳过原问题，只看改写/扩展
            if rewrite_added >= REWRITE_QUOTA or self._bm25 is None:
                # 空库/维度不匹配时 BM25 未构建，跳过配额保障，避免 AttributeError
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

        # 直接将融合后的候选送入 rerank；保留各路线头部候选，避免保护路径覆盖直接命中。
        existing_ids = {h["id"] for h in merged}
        for did, hit in route_floor_hits.items():
            if did not in existing_ids:
                merged.append({**hit, "score": scores.get(did, hit.get("score", 0.0))})
                existing_ids.add(did)

        # 引用追踪：候选池中某条款正文可能引用其他条款作为处罚依据：
        # 1. 同法引用："依照本法第N条"（如第九十五条→第九十条）
        # 2. 跨法引用："按照《XXX法》第N条"（如驾驶证申领规定第九十七条→道交法第九十九条）
        # 被引用的条款因正文不含查询关键词，BM25 分数低，无法被正常召回。
        # 提取引用关系，把被引用条款也加入候选池，最多追加 3 条。
        import re
        REFERENCE_QUOTA = retrieval_policy["reference_quota"]
        ref_added = 0
        existing_ids = {h["id"] for h in merged}

        # 同法引用：依照本法第N条 / 依照本规定第N条 / 依照本条例第N条
        same_law_patterns = [
            re.compile(p)
            for p in retrieval_policy.get("same_law_reference_patterns", [
                r"[依照按照依据根据]本[法法规条例规定]+(第[一二三四五六七八九十百零]+条)(?:第[一二三四五六七八九十百零]+款)?",
            ])
        ]
        # 跨法引用：依照《XXX》第N条 / 按照《XXX》第N条 / 依据《XXX》第N条 / 根据《XXX》第N条
        # 注意：引用的法律名可能是全称或简称（如"道路交通安全法"="中华人民共和国道路交通安全法"）
        cross_law_pattern = re.compile(
            retrieval_policy.get(
                "cross_law_reference_pattern",
                r"[依照按照依据根据][《<]([^》<>]+?)[》>]第([一二三四五六七八九十百零]+条)",
            )
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

        # R5 条-款项层级展开：父条命中时补充其款项子块，子款命中时带回父条上下文
        hierarchy_quota = retrieval_policy.get("article_hierarchy_expand_quota", 8)
        if hierarchy_quota > 0:
            expanded = self._expand_article_hierarchy(
                merged,
                hierarchy_quota,
                law_source=law_source,
                article_no=article_no,
            )
            if expanded:
                existing_ids = {h["id"] for h in merged}
                for h in expanded:
                    if h["id"] not in existing_ids:
                        merged.append(h)
                        existing_ids.add(h["id"])
                event(
                    "retrieval.hierarchy_expand",
                    added=len(expanded),
                    quota=hierarchy_quota,
                    merged_count=len(merged),
                )

        # RRF 候选直接进入 rerank，不再做 BM25 粗排；这里仅按 RRF 融合分数
        # 控制 rerank 输入预算。每条检索路线的头部候选已在 route_floor_hits 中
        # 保留，避免预算截断时丢失原问题或改写查询的直接命中。
        candidate_limit = get_settings().rerank_candidate_limit
        if len(merged) > candidate_limit:
            selected: list[dict] = []
            selected_ids: set[str] = set()
            for did in route_floor_hits:
                for hit in merged:
                    if hit["id"] == did:
                        selected.append(hit)
                        selected_ids.add(did)
                        break
            ranked_remaining = sorted(
                (hit for hit in merged if hit["id"] not in selected_ids),
                key=lambda hit: hit.get("score", 0.0),
                reverse=True,
            )
            merged = (selected + ranked_remaining)[:candidate_limit]
            event(
                "retrieval.rerank_budget",
                before=len(selected) + len(ranked_remaining),
                after=len(merged),
                limit=candidate_limit,
            )

        # R10 检索可观测性汇总：来源分布与候选得分统计
        source_distribution: dict[str, int] = {}
        score_list: list[float] = []
        for h in merged:
            src = (h.get("metadata") or {}).get("source", "unknown")
            source_distribution[src] = source_distribution.get(src, 0) + 1
            score_list.append(float(h.get("score", 0.0)))
        event(
            "retrieval.observability",
            query_count=len(queries),
            candidate_count=len(merged),
            source_distribution=source_distribution,
            score_max=max(score_list) if score_list else 0.0,
            score_min=min(score_list) if score_list else 0.0,
            score_avg=round(sum(score_list) / len(score_list), 6) if score_list else 0.0,
        )

        event(
            "retrieval.multi_query",
            query_count=len(queries),
            definition_boost=def_intent,
            merged_count=len(merged),
        )
        return merged


@lru_cache
def get_retrieval_engine() -> RetrievalEngine:
    engine = RetrievalEngine()
    engine._ensure_loaded()
    return engine
