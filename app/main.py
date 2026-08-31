"""FastAPI 后端入口，提供 /api/v1/... 接口。"""
from __future__ import annotations

import json
import os
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi_throttle import RateLimiter
from pydantic import ValidationError
from starsessions import CookieStore, SessionAutoloadMiddleware, SessionMiddleware

from app import ingestion, session_store
from app.config import get_settings
from app.errors import LawHelperError
from app.qa import answer, answer_stream
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SessionData,
    SessionSaveRequest,
)

settings = get_settings()

app = FastAPI(title="小Z - 道路交通安全法 AI 助手", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# starsessions：基于 Cookie 的服务端会话，自动加载 /api/v1 路径的 session
# 注意：SessionAutoloadMiddleware 必须在 SessionMiddleware 之前 add，这样运行时 SessionMiddleware 先初始化 session_handler
app.add_middleware(SessionAutoloadMiddleware, paths=["/api/v1"])
app.add_middleware(
    SessionMiddleware,
    store=CookieStore(secret_key=settings.session_secret_key),
    lifetime=settings.session_lifetime_seconds,
    cookie_https_only=False,
    cookie_same_site="lax",
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


def _browser_session_key(req: Request) -> str:
    """为当前浏览器会话生成/复用一个稳定的 key，用于限流。"""
    key = req.session.get("browser_session_id")
    if not key:
        key = uuid.uuid4().hex
        req.session["browser_session_id"] = key
    return key


def _ensure_session_access(req: Request, session_id: str | None) -> None:
    """校验当前浏览器会话有权访问指定的 conversation session_id。

    对于尚未持久化的新会话，自动注册到当前浏览器会话，避免新建空会话无法发送第一条消息。
    """
    if session_id is None:
        return
    allowed = set(req.session.get("session_ids", []))
    if session_id in allowed:
        return
    try:
        path = session_store._session_path(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="非法的会话 ID")
    if not os.path.exists(path):
        _register_session_id(req, session_id)
        return
    raise HTTPException(status_code=403, detail="无权访问该会话")


def _register_session_id(req: Request, session_id: str) -> None:
    """将 conversation session_id 注册到当前浏览器会话的允许列表中。"""
    allowed = set(req.session.get("session_ids", []))
    allowed.add(session_id)
    req.session["session_ids"] = list(allowed)


# 按浏览器会话限流：每个浏览器会话在 60 秒内最多请求 10 次聊天接口
chat_limiter = RateLimiter(times=10, seconds=60, key_func=_browser_session_key)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(chat_limiter)])
def chat(req: ChatRequest, request: Request):
    _ensure_session_access(request, req.session_id)
    text, references = answer(req.question, req.history)
    return ChatResponse(answer=text, references=references)


@app.post("/api/v1/chat/stream", dependencies=[Depends(chat_limiter)])
def chat_stream(req: ChatRequest, request: Request):
    """流式返回：先 references 事件，再逐段 delta 文本（NDJSON）。"""
    _ensure_session_access(request, req.session_id)

    def gen():
        try:
            for payload in answer_stream(req.question, req.history):
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        except LawHelperError as exc:
            # 生成器中途异常：以 error 事件透传给前端，避免流被静默截断
            yield json.dumps(
                {"type": "error", "content": exc.message}, ensure_ascii=False
            ) + "\n"
        except ValidationError:
            # 引用法条等数据校验失败：同样透传，避免流中途 500 截断
            yield json.dumps(
                {"type": "error", "content": "引用法条数据校验失败"}, ensure_ascii=False
            ) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest():
    count = ingestion.ingest()
    return IngestResponse(status="ok", articles=count, message="入库完成")


@app.get("/api/v1/settings/llm_model")
def llm_model():
    return {"llm_model": get_settings().llm_model}


@app.get("/api/v1/sessions", response_model=list[SessionData])
def list_sessions(request: Request):
    """当前浏览器会话有权访问的会话列表（按 updated_at 降序）。"""
    allowed = set(request.session.get("session_ids", []))
    all_sessions = session_store.list_all()
    return [SessionData.model_validate(s) for s in all_sessions if s["id"] in allowed]


@app.put("/api/v1/sessions/{session_id}", response_model=SessionData)
def save_session(session_id: str, req: SessionSaveRequest, request: Request):
    """upsert 会话（空会话不应调用此接口）。"""
    _ensure_session_access(request, session_id)
    _register_session_id(request, session_id)
    updated_at = session_store.save(
        session_id, req.title, [m.model_dump() for m in req.messages]
    )
    return SessionData(
        id=session_id, title=req.title, updated_at=updated_at, messages=req.messages
    )


@app.delete("/api/v1/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    _ensure_session_access(request, session_id)
    ok = session_store.delete(session_id)
    if ok:
        allowed = set(request.session.get("session_ids", []))
        allowed.discard(session_id)
        request.session["session_ids"] = list(allowed)
    return {"status": "ok" if ok else "not_found"}
