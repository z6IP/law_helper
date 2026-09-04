"""FastAPI 后端入口，提供 /api/v1/... 接口。"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi_throttle import RateLimiter
from starsessions import CookieStore, SessionAutoloadMiddleware, SessionMiddleware

from app import answer_cache, ingestion, jobs, session_db, session_store
from app.config import get_settings
from app.document_parser import parse_document
from app.errors import LawHelperError
from app.llm import get_llm
from app.qa import answer
from app.tracing import event, finish_trace, query_traces, span, start_trace
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    Reference,
    RunningJobsResponse,
    SessionData,
    SessionSaveRequest,
    SummarizeTitleRequest,
    SummarizeTitleResponse,
    TracesResponse,
)

settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_HTML_PATH = BASE_DIR / "dashboard.html"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"


def _local_only(request: Request) -> None:
    """仅允许回环地址访问（可观测面板为本地运维工具，不对外提供服务）。"""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="仅限本地访问")


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


# 预热状态：startup 立即返回，预热在后台线程进行
# - ready: 预热是否完成（完成后 chat 接口才允许调用）
# - error: 预热异常信息（None 表示无异常）
# - stage: 当前阶段描述，便于排障
_PRELOAD_STATE = {"ready": False, "error": None, "stage": "pending"}


def _run_preload() -> None:
    """后台线程：预加载模型 + 验证索引正确性，消除首次请求冷启动延迟。

    将原本同步阻塞 startup 的预热逻辑迁到后台线程，使 health 等接口在
    uvicorn 监听端口后即可响应，run.py 不必等模型加载完成才检测到端口连通。
    """
    # 预热作为独立 trace（与请求链路区分），便于排查冷启动问题
    start_trace(kind="preload")
    try:
        import chromadb
        from app.ingestion import COLLECTION_NAME

        settings = get_settings()
        chroma_path = str(settings.chroma_full_dir)
        client = chromadb.PersistentClient(path=chroma_path)

        # Step 1: 加载 Embedding 模型（local_files_only=True，跳过网络检查）
        _PRELOAD_STATE["stage"] = "embedding"
        event("preload.stage", stage="embedding")
        from app.embeddings import get_embedding_model

        with span("preload.embedding"):
            emb_model = get_embedding_model()
            # warmup: 跑一次推理，确保权重完全加载
            _ = emb_model.embed_query("小ZAI助手启动预热")
            event("preload.embedding.done", dim=len(_))

        # Step 2: 加载 Reranker 模型（local_files_only=True，跳过网络检查）
        _PRELOAD_STATE["stage"] = "reranker"
        event("preload.stage", stage="reranker")
        from app.rerank import get_reranker

        with span("preload.reranker"):
            reranker = get_reranker()
            # 触发模型加载（懒加载 → 实际加载）
            reranker._ensure_loaded()
            # warmup: 跑一次最小推理，确保 CrossEncoder 权重全部加载
            import torch
            with torch.inference_mode():
                _ = reranker._model.predict([("预热查询", "预热文档")])

        # Step 3: 校验索引（维度检测优先，条件重建；否则启动增量导入）
        _PRELOAD_STATE["stage"] = "index"
        event("preload.stage", stage="index")
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
        event("preload.collection.count", count=count)

        from app.retrieval import get_retrieval_engine

        engine = get_retrieval_engine()
        if engine.needs_rebuild:
            event("preload.rebuild", reason="维度不匹配")
            import shutil

            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path, ignore_errors=True)
            from app.retrieval import get_retrieval_engine as _gre

            _gre.cache_clear()
            result = ingestion.ingest()
            event("preload.rebuild.done", message=result.message)
        else:
            result = ingestion.ingest()
            event("preload.ingest.done", message=result.message)

        # 法条语料发生变化时清空答案缓存，避免返回基于旧语料的缓存回答
        if result.added or result.updated or result.removed:
            answer_cache.clear()

        # Step 4: 加载 / 刷新检索引擎（BM25 等组件就绪）
        _PRELOAD_STATE["stage"] = "retrieval"
        event("preload.stage", stage="retrieval")
        from app.retrieval import get_retrieval_engine as _gre

        with span("preload.retrieval"):
            _gre.cache_clear()
            get_retrieval_engine()

        _PRELOAD_STATE["ready"] = True
        _PRELOAD_STATE["stage"] = "done"
    except Exception as exc:  # noqa: BLE001
        _PRELOAD_STATE["error"] = f"{type(exc).__name__}: {exc}"
        _PRELOAD_STATE["stage"] = "error"
        event("preload.failed", error=_PRELOAD_STATE["error"])
    finally:
        finish_trace(status="error" if _PRELOAD_STATE["error"] else "ok")


@app.on_event("startup")
async def startup_preload():
    """启动时在后台线程预加载模型，避免阻塞 HTTP 接口响应。

    启动前先把历史 JSON 会话文件迁移到 SQLite，保证历史数据不丢失。
    """
    try:
        migrated = session_store.migrate_from_json()
        if migrated:
            logger = logging.getLogger(__name__)
            logger.info("已从 JSON 迁移 %d 条历史会话到 SQLite", migrated)
    except Exception:  # noqa: BLE001
        # 迁移失败不应阻塞启动
        logging.getLogger(__name__).exception("历史会话迁移失败")
    threading.Thread(target=_run_preload, name="backend-preload", daemon=True).start()


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
    if not session_id or not all(ch.isalnum() or ch == "-" for ch in session_id):
        raise HTTPException(status_code=400, detail="非法的会话 ID")
    # 数据库中不存在该会话视为新会话，自动注册；否则必须已被当前浏览器允许
    if not session_store.exists(session_id):
        _register_session_id(req, session_id)
        return
    raise HTTPException(status_code=403, detail="无权访问该会话")


def _register_session_id(req: Request, session_id: str) -> None:
    """将 conversation session_id 注册到当前浏览器会话的允许列表中。"""
    allowed = set(req.session.get("session_ids", []))
    allowed.add(session_id)
    req.session["session_ids"] = list(allowed)


def _effective_question(question: str, document_text: str | None) -> str:
    """当用户只上传文件未输入问题时，使用默认提示词，避免 LLM 无问题可答。"""
    if not question.strip() and document_text:
        return "请结合上传的材料进行回答"
    return question


# 按浏览器会话限流：每个浏览器会话在 60 秒内最多请求 10 次聊天接口
chat_limiter = RateLimiter(times=10, seconds=60, key_func=_browser_session_key)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/ready")
def ready():
    """返回预热状态：ready=True 时 chat 接口才可调用。"""
    return {
        "ready": _PRELOAD_STATE["ready"],
        "stage": _PRELOAD_STATE["stage"],
        "error": _PRELOAD_STATE["error"],
    }


def _ensure_preload_ready() -> None:
    """chat 接口前置检查：预热未完成或失败时返回 503，避免冷启动超时。"""
    if _PRELOAD_STATE["error"] is not None:
        raise HTTPException(
            status_code=503,
            detail=f"预热失败，请查看后端日志：{_PRELOAD_STATE['error']}",
        )
    if not _PRELOAD_STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"模型加载中（阶段：{_PRELOAD_STATE['stage']}），请稍后再试",
        )


def _user_input_title(content: str, max_length: int = 18) -> str:
    """极短或无法回答的输入，生成固定格式标题：用户输入了<内容>。"""
    prefix = "用户输入了"
    limit = max_length - len(prefix)
    return f"{prefix}{content.strip()[:max(1, limit)]}"


def _summarize_title_from_messages(messages: list[dict]) -> str:
    """根据会话消息生成总结性标题。"""
    # 只有一条用户消息且内容为空、只带了文件时，按文件类型生成固定描述
    if len(messages) == 1:
        msg = messages[0]
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            file_names = msg.get("fileNames") or msg.get("file_names") or []
            if not content and file_names:
                image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                image_count = sum(1 for f in file_names if str(f).lower().endswith(tuple(image_exts)))
                file_count = len(file_names)
                if image_count == file_count:
                    return f"用户发了{file_count}张图片"
                return f"用户发了{file_count}份文件"
    # 若第一条用户消息极短，直接固定格式概括，避免 LLM 根据助手回复过度发挥
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user:
        content = (first_user.get("content") or "").strip()
        if content and len(content) <= 2:
            return _user_input_title(content)
    # 否则使用 LLM 总结对话内容
    turns: list[str] = []
    first_user_text = ""
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        file_names = m.get("fileNames") or m.get("file_names") or []
        if role == "user":
            prefix = "用户"
            if not first_user_text and content:
                first_user_text = content
        elif role == "assistant":
            prefix = "助手"
        else:
            continue
        if file_names:
            content = (content + " " if content else "") + f"[附件：{', '.join(file_names)}]"
        if content:
            turns.append(f"{prefix}：{content}")
    # 若 LLM 无法总结，使用第一条用户内容兜底，避免标题为「新对话」
    fallback_title = _user_input_title(first_user_text) if first_user_text else "新对话"
    prompt = "请根据以下对话内容，用不超过 10 个字生成一个会话标题。只输出标题，不要解释。\n\n" + "\n".join(turns)
    system = "你是摘要助手，请用不超过 10 个字总结对话内容作为标题，不要加引号或解释。"
    return get_llm().chat(system, prompt).strip()[:18] or fallback_title


@app.post(
    "/api/v1/sessions/{session_id}/summarize",
    response_model=SummarizeTitleResponse,
    dependencies=[Depends(_ensure_preload_ready)],
)
def summarize_session(session_id: str, req: SummarizeTitleRequest, request: Request):
    """根据会话消息生成总结性标题并持久化。"""
    _ensure_session_access(request, session_id)
    messages = req.messages or []
    title = _summarize_title_from_messages(messages)
    # 将总结后的标题持久化，刷新页面后仍能保持最新标题
    try:
        session_store.save(session_id, title, messages)
    except Exception:  # noqa: BLE001
        pass
    return SummarizeTitleResponse(title=title)


# 上传文件解析：限制大小与类型，与前端保持一致
_MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
_ALLOWED_UPLOAD_EXTS = {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _save_upload(content: bytes, filename: str | None) -> tuple[str, str]:
    """保存上传文件到本地，返回 (url, saved_name)。"""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "unknown").suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOADS_DIR / saved_name
    with open(dest, "wb") as f:
        f.write(content)
    return f"/api/v1/uploads/{saved_name}", saved_name


@app.post("/api/v1/chat/upload", dependencies=[Depends(chat_limiter)])
def chat_upload(request: Request, file: UploadFile = File(...)):
    """上传并解析文件，返回提取的文本内容（仅作为一次性上下文）以及文件 URL。"""
    try:
        suffix = Path(file.filename or "unknown").suffix.lower()
        if suffix not in _ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=415, detail=f"不支持的文件类型：{suffix}")
        # 只读取到限制大小 + 1 字节，超限立即拒绝，避免大文件占用内存
        content = file.file.read(_MAX_UPLOAD_SIZE + 1)
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="文件大小超过 5MB 限制")
        text = parse_document(content, file.filename or "unknown")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    url, _ = _save_upload(content, file.filename)

    return {"text": text, "url": url, "name": file.filename}


@app.get("/api/v1/uploads/{filename}")
def get_upload(filename: str, request: Request):
    """获取上传的文件，仅限当前浏览器会话中可访问的会话附件。"""
    # 会话鉴权：文件必须属于当前浏览器已授权会话的附件
    allowed = set(request.session.get("session_ids", []))
    if not session_db.is_attachment_accessible(filename, allowed):
        raise HTTPException(status_code=403, detail="无权访问该文件")

    # 简单安全校验：防止目录穿越
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dest = (UPLOADS_DIR / filename).resolve()
        dest.relative_to(UPLOADS_DIR.resolve())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=403, detail="非法路径") from exc

    if not dest.exists() or not dest.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(dest)


@app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(_ensure_preload_ready), Depends(chat_limiter)])
def chat(req: ChatRequest, request: Request):
    _ensure_session_access(request, req.session_id)
    question = _effective_question(req.question, req.document_text)
    # 单轮无历史时尝试命中答案缓存，跳过检索→重排→生成
    if not req.history:
        cached = answer_cache.get(question)
        if cached:
            start_trace(
                kind="chat",
                question=question,
                session_id=req.session_id,
                cache_hit=True,
            )
            references = [Reference.model_validate(r) for r in cached["references"]]
            finish_trace()
            return ChatResponse(answer=cached["answer"], references=references)

    start_trace(kind="chat", question=question, session_id=req.session_id)
    text, references = answer(question, req.history, req.document_text)
    # 仅缓存有明确法条引用的回答（拒答/无关问题不缓存）
    if not req.history and references:
        answer_cache.put(question, text, [r.model_dump() for r in references])
    finish_trace()
    return ChatResponse(answer=text, references=references)


@app.post("/api/v1/chat/stream", dependencies=[Depends(_ensure_preload_ready), Depends(chat_limiter)])
def chat_stream(req: ChatRequest, request: Request):
    """提交后台问答任务，并以 NDJSON 流式回放事件（references / reasoning / delta / progress）。

    生成在独立后台线程执行，与 HTTP 连接解耦：客户端断开（切会话 / 刷新 / 关闭）不会中断生成，
    完成后自动写入会话存储。
    """
    _ensure_session_access(request, req.session_id)
    question = _effective_question(req.question, req.document_text)
    if not req.session_id:
        raise HTTPException(status_code=400, detail="缺少会话 ID")
    title = req.title or (question.strip()[:18] or "新对话")
    try:
        jobs.submit_chat_job(
            req.session_id,
            question,
            req.history,
            title,
            req.document_text,
            file_names=req.file_names,
            attachments=[a.model_dump() for a in (req.attachments or [])],
        )
    except jobs.JobConflictError as exc:
        raise HTTPException(status_code=409, detail=exc.message)

    return StreamingResponse(_job_event_stream(req.session_id), media_type="application/x-ndjson")


def _job_event_stream(session_id: str):
    """订阅某个后台任务的完整事件流：从头回放，未结束时阻塞等待新事件。"""
    job = jobs.get_job(session_id)
    if job is None:
        yield json.dumps(
            {"type": "error", "content": "会话不存在或已结束"}, ensure_ascii=False
        ) + "\n"
        return
    cursor = 0
    while True:
        events, status = job.read_from(cursor)
        for ev in events:
            yield json.dumps(ev, ensure_ascii=False) + "\n"
        cursor += len(events)
        if status != "running":
            break
        job.wait_for(cursor)


@app.get("/api/v1/chat/jobs/{session_id}/stream")
def job_stream(session_id: str, request: Request):
    """订阅某个会话的后台生成事件（用于刷新/关闭标签页后恢复展示进行中的回答）。"""
    _ensure_session_access(request, session_id)
    return StreamingResponse(_job_event_stream(session_id), media_type="application/x-ndjson")


@app.get("/api/v1/chat/jobs/running", response_model=RunningJobsResponse)
def running_jobs(request: Request):
    """返回当前浏览器会话可见的、正在生成回答的会话 ID 列表。"""
    allowed = set(request.session.get("session_ids", []))
    return RunningJobsResponse(
        sessions=[sid for sid in jobs.list_running_session_ids() if sid in allowed]
    )


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest():
    _ensure_preload_ready()
    start_trace(kind="ingest")
    result = ingestion.ingest()
    from app.retrieval import get_retrieval_engine as _gre

    _gre.cache_clear()
    answer_cache.clear()  # 法条语料更新后，旧答案缓存全部失效
    finish_trace()
    return IngestResponse(status="ok", articles=result.total, message=result.message)


@app.get("/api/v1/settings/llm_model")
def llm_model():
    return {"llm_model": get_settings().llm_model}


@app.get("/api/v1/traces", response_model=TracesResponse, dependencies=[Depends(_local_only)])
def traces(limit: int = 200, offset: int = 0):
    """观测面板只读接口：返回 trace 摘要列表与汇总统计（按开始时间倒序）。仅本地可访问。"""
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    return TracesResponse(**query_traces(limit=limit, offset=offset))


@app.get("/dashboard", include_in_schema=False, dependencies=[Depends(_local_only)])
def dashboard_page():
    """本地运维可观测面板（单文件 HTML，仅回环地址可访问）。"""
    return HTMLResponse(content=_read_dashboard_html())


def _read_dashboard_html() -> str:
    try:
        return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        return "<html><body><h1>可观测性面板</h1><p>dashboard.html 缺失</p></body></html>"


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
