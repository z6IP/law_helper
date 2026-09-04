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


class SummarizeTitleRequest(_BaseSchema):
    messages: list[dict] = Field(..., description="会话消息列表")


class SummarizeTitleResponse(_BaseSchema):
    title: str = Field(..., min_length=1, description="总结后的会话标题")


class Attachment(_BaseSchema):
    name: str = Field(..., description="原始文件名")
    type: Literal["image", "document"] = Field("document", description="附件类型")
    url: str = Field(..., description="可访问 URL")


class ChatRequest(_BaseSchema):
    question: str = Field(..., description="用户提问")
    session_id: str | None = Field(
        None,
        description="当前会话 ID，用于权限校验和按会话限流",
    )
    history: list[dict] = Field(
        default_factory=list,
        max_length=50,
        description="可选的历史对话（最多 50 条）",
    )
    title: str | None = Field(
        None,
        max_length=200,
        description="会话标题；后端后台落库时使用（缺省时按问题前 18 字生成）",
    )
    document_text: str | None = Field(
        None,
        description="用户上传文件解析后的文本，仅作为本次问答的一次性上下文",
    )
    file_names: list[str] | None = Field(
        default=None,
        description="当前用户消息携带的附件文件名列表（兼容字段）",
    )
    attachments: list[Attachment] | None = Field(
        default=None,
        description="当前用户消息携带的附件元信息列表",
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


class RunningJobsResponse(_BaseSchema):
    sessions: list[str] = Field(default_factory=list, description="正在生成回答的会话 ID 列表")


class IngestResponse(_BaseSchema):
    status: str = Field(..., min_length=1, description="结果状态")
    articles: int = Field(0, ge=0, description="入库法条数量")
    message: str = Field("", description="补充说明")


class SessionMessage(_BaseSchema):
    role: Literal["user", "assistant"] = Field(..., description="消息角色：user / assistant")
    content: str = Field(..., description="消息文本；允许空字符串，以支持仅上传文件的消息")
    references: list[Reference] = Field(default_factory=list, description="引用法条")
    reasoning: str | None = Field(None, description="思考过程文本")
    fileNames: list[str] = Field(default_factory=list, description="附件文件名列表（兼容字段）")
    attachments: list[Attachment] = Field(default_factory=list, description="附件元信息列表")


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


class TraceItem(_BaseSchema):
    """Dashboard 展示的一条 trace 摘要。"""

    trace_id: str = Field(..., description="trace 唯一标识")
    kind: str = Field(..., description="trace 类型：chat/chat_stream/preload/ingest")
    question: str | None = Field(None, description="用户问题（仅问答类有）")
    session_id: str | None = Field(None, description="会话 ID")
    cache_hit: bool = Field(False, description="是否命中答案缓存")
    status: str = Field("ok", description="trace 状态：ok/error")
    started_at: str = Field("", description="开始时间（ISO）")
    duration_ms: float = Field(0, description="总耗时（毫秒）")
    prompt_tokens: int = Field(0, description="累计输入 token")
    completion_tokens: int = Field(0, description="累计输出 token")
    total_tokens: int = Field(0, description="累计总 token")


class TracesResponse(_BaseSchema):
    """trace 列表 + 汇总统计。"""

    traces: list[TraceItem] = Field(default_factory=list, description="trace 摘要列表")
    total_count: int = Field(0, description="trace 总数")
    total_tokens: int = Field(0, description="全部 trace 累计 token")
    cache_hit_count: int = Field(0, description="命中缓存的 trace 数")
    avg_duration_ms: float = Field(0, description="平均耗时（毫秒）")
