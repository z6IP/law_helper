"""FastAPI 后端入口，提供 /api/v1/... 接口。"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import ingestion
from app.config import get_settings
from app.errors import LawHelperError
from app.qa import answer, answer_stream
from app.schemas import ChatRequest, ChatResponse, IngestResponse

app = FastAPI(title="小Z - 道路交通安全法 AI 助手", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_preload():
    """启动时预加载模型 + 验证索引正确性，消除首次请求冷启动延迟。"""
    import chromadb
    from app.ingestion import COLLECTION_NAME

    settings = get_settings()
    chroma_path = str(settings.chroma_full_dir)
    client = chromadb.PersistentClient(path=chroma_path)

    # Step 1: 加载 Embedding 模型（local_files_only=True，跳过网络检查）
    print("[预加载] 正在加载 Embedding 模型...")
    from app.embeddings import get_embedding_model

    emb_model = get_embedding_model()
    # warmup: 跑一次推理，确保权重完全加载
    _ = emb_model.embed_query("小ZAI助手启动预热")
    print(f"[预加载] Embedding 模型就绪，输出维度: {len(_)}")

    # Step 2: 加载 Reranker 模型（local_files_only=True，跳过网络检查）
    print("[预加载] 正在加载 Reranker 模型...")
    from app.rerank import get_reranker

    reranker = get_reranker()
    # 触发模型加载（懒加载 → 实际加载）
    reranker._ensure_loaded()
    # warmup: 跑一次最小推理，确保 CrossEncoder 权重全部加载
    import torch
    with torch.inference_mode():
        _ = reranker._model.predict([("预热查询", "预热文档")])
    print("[预加载] Reranker 模型就绪")

    # Step 3: 校验索引（维度检测优先，条件重建）
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 16,
            "hnsw:M": 16,
        },
    )
    count = collection.count()
    print(f"[预加载] 向量库当前有 {count} 条记录")

    if count == 0:
        print("[预加载] 向量库为空，开始入库...")
        n = ingestion.ingest()
        print(f"[预加载] 入库完成，共 {n} 条")
        from app.retrieval import get_retrieval_engine as _gre
        _gre.cache_clear()
    else:
        from app.retrieval import get_retrieval_engine
        engine = get_retrieval_engine()
        if engine.needs_rebuild:
            print("[预加载] 检测到维度不匹配，正在重建索引...")
            import os
            import shutil
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path, ignore_errors=True)
            from app.retrieval import get_retrieval_engine as _gre
            _gre.cache_clear()
            n = ingestion.ingest()
            print(f"[预加载] 重建索引完成，共 {n} 条")

    # Step 4: 加载 / 刷新检索引擎（BM25 等组件就绪）
    print("[预加载] 正在加载检索引擎...")
    from app.retrieval import get_retrieval_engine as _gre
    _gre.cache_clear()
    get_retrieval_engine()
    print("[预加载] 检索引擎就绪")


@app.exception_handler(LawHelperError)
async def law_error_handler(request, exc: LawHelperError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text, references = answer(req.question, req.history)
    return ChatResponse(answer=text, references=references)


@app.post("/api/v1/chat/stream")
def chat_stream(req: ChatRequest):
    """流式返回：先 references 事件，再逐段 delta 文本（NDJSON）。"""

    def gen():
        for payload in answer_stream(req.question, req.history):
            yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest():
    count = ingestion.ingest()
    return IngestResponse(status="ok", articles=count, message="入库完成")


@app.get("/api/v1/settings/llm_model")
def llm_model():
    return {"llm_model": get_settings().llm_model}