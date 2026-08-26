"""请求 / 响应数据模型。"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户提问")
    history: list[dict] = Field(default_factory=list, description="可选的历史对话")


class Reference(BaseModel):
    article_no: str = Field(..., description="条号，如“第九十一条”")
    section_header: str = Field("", description="所属章/节标题")
    text: str = Field(..., description="法条原文")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="生成的回答")
    references: list[Reference] = Field(default_factory=list, description="引用法条")


class IngestResponse(BaseModel):
    status: str = Field(..., description="结果状态")
    articles: int = Field(0, description="入库法条数量")
    message: str = Field("", description="补充说明")