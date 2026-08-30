"""请求 / 响应数据模型。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BaseSchema(BaseModel):
    """共享模型配置：去除字符串首尾空白、忽略未知字段、校验默认值。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
        validate_default=True,
    )


class ChatRequest(_BaseSchema):
    question: str = Field(..., min_length=1, description="用户提问")
    history: list[dict] = Field(
        default_factory=list,
        max_length=50,
        description="可选的历史对话（最多 50 条）",
    )


class Reference(_BaseSchema):
    source: str = Field("", description="法条来源文档名称")
    article_no: str = Field(..., min_length=1, description="条号，如“第九十一条”")
    section_header: str = Field("", description="所属章/节标题")
    text: str = Field(..., min_length=1, description="法条原文")

    @field_validator("article_no", "text", mode="after")
    @classmethod
    def _ensure_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("该字段不能为空或仅包含空白字符")
        return v


class ChatResponse(_BaseSchema):
    answer: str = Field(..., min_length=1, description="生成的回答")
    references: list[Reference] = Field(default_factory=list, description="引用法条")


class IngestResponse(_BaseSchema):
    status: str = Field(..., min_length=1, description="结果状态")
    articles: int = Field(0, ge=0, description="入库法条数量")
    message: str = Field("", description="补充说明")


class SessionMessage(_BaseSchema):
    role: Literal["user", "assistant"] = Field(..., description="消息角色：user / assistant")
    content: str = Field(..., min_length=1, description="消息文本")
    references: list[Reference] = Field(default_factory=list, description="引用法条")
    reasoning: str | None = Field(None, description="思考过程文本")


class SessionSaveRequest(_BaseSchema):
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="会话标题",
    )
    messages: list[SessionMessage] = Field(default_factory=list, description="会话消息")


class SessionData(_BaseSchema):
    id: str = Field(..., min_length=1, description="会话 ID")
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="会话标题",
    )
    updated_at: str = Field("", description="最后更新时间（ISO）")
    messages: list[SessionMessage] = Field(default_factory=list, description="会话消息")
