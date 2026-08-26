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