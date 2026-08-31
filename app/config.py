"""应用配置：从 .env 读取运行参数。

工程约定：LLM 配置（OPENAI_API_BASE / OPENAI_API_KEY / LLM_MODEL）必须写在 .env 中，
通过 pydantic-settings 统一加载。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 阿里云百炼 LLM（OpenAI 兼容接口）
    openai_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    llm_model: str = "qwen-plus"

    # OCR 模型（扫描型 PDF 兜底，默认复用同一 OpenAI 兼容接口）
    ocr_model: str = "qwen3.5-ocr"
    ocr_dpi: int = 200

    # 数据与存储
    chroma_dir: str = "chroma"
    # 会话持久化目录（本地文件存储，每会话一个 JSON）
    sessions_dir: str = "data/sessions"

    # Embedding / Rerank 模型（轻量版，CPU 友好）
    embedding_model_id: str = "BAAI/bge-small-zh-v1.5"
    rerank_model_id: str = "BAAI/bge-reranker-v2-m3"
    modelscope_cache_dir: str = "models"

    # 多轮对话：参与历史改写的最大消息条数（3 轮 = 6 条）
    history_max_messages: int = Field(6, ge=0, le=20)

    # 检索参数
    top_k_retrieve: int = Field(10, ge=1, le=50)
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0)
    rrf_lambda: int = Field(60, ge=1)
    rerank_top_n: int = Field(3, ge=1, le=10)
    # 重排相关性阈值：低于该分的候选视为不相关并丢弃，
    # 全部丢弃时由 LLM 简短拒答，不引用任何法条
    # 实测（bge-reranker-v2-m3 + 查询改写，8 场景端到端标定）：
    #   相关条文 top1 ≈ 0.49~0.99（转弯让直行 0.91 / 酒驾 0.99 / 追尾 0.49 / 闯红灯 0.83 等）
    #   无关 query / 弱相关候选普遍 <0.11
    #   0.2 落在两侧之间；注意：相关条文分数依赖查询改写弥合口语→法条术语的鸿沟，
    #   新增改写规则后需回归此阈值
    rerank_min_score: float = Field(0.2, ge=0.0, le=1.0)

    # 服务
    backend_url: str = "http://127.0.0.1:8000"

    # 会话 Cookie（starsessions）
    session_secret_key: str = ""
    session_lifetime_seconds: int = 3600 * 24 * 14

    # CORS：开发时前端在 http://localhost:5173；生产通过逗号分隔配置多个来源
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("session_secret_key")
    @classmethod
    def _validate_session_secret_key(cls, v: str) -> str:
        if not v:
            raise ValueError("SESSION_SECRET_KEY 必须在 .env 中配置")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

    @property
    def docx_full_paths(self) -> list[Path]:
        """statute/ 目录下所有 .docx 法规文档（语料数据源，自动纳入新法规）。"""
        return sorted((BASE_DIR / "statute").glob("*.docx"))

    @property
    def pdf_full_paths(self) -> list[Path]:
        """statute/ 目录下所有 .pdf 文档（语料数据源，自动纳入新标准）。"""
        return sorted((BASE_DIR / "statute").glob("*.pdf"))

    @property
    def chroma_full_dir(self) -> Path:
        p = Path(self.chroma_dir)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    @property
    def sessions_full_dir(self) -> Path:
        p = Path(self.sessions_dir)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    @property
    def modelscope_full_dir(self) -> Path:
        p = Path(self.modelscope_cache_dir)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()