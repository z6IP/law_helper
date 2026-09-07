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

# 用户数据根目录：~/.law_helper（首次使用时自动创建）。
# 所有运行时数据（会话、附件、SQLite 数据库等）默认存储在此，
# 与项目代码分离，便于升级与备份。
USER_DATA_DIR = Path.home() / ".law_helper"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    # 会话持久化目录（默认 ~/.law_helper/session_history，存放历史 JSON 迁移源）
    sessions_dir: str = "session_history"

    # 单用户模式：默认开启，所有会话自动归属当前浏览器，无需严格鉴权。
    # 若部署到多用户/网络环境，应设为 False，启用 session_ids 严格隔离。
    single_user_mode: bool = True

    # Embedding / Rerank 模型（阿里云百炼 API）
    embedding_model_id: str = "qwen3.7-text-embedding"
    rerank_model_id: str = "qwen3.7-text-rerank"
    # Embedding 向量维度（qwen3.7-text-embedding 支持 2560/2048/1536/1024/768/512/256，默认 1024）
    embedding_dimensions: int = 1024
    # DashScope API base（rerank 接口使用，区别于 OpenAI 兼容接口）
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/api/v1"

    # 多轮对话：参与历史改写的最大消息条数（3 轮 = 6 条）
    history_max_messages: int = Field(6, ge=0, le=20)

    # 检索参数
    top_k_retrieve: int = Field(10, ge=1, le=50)
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0)
    rrf_lambda: int = Field(60, ge=1)
    rerank_top_n: int = Field(4, ge=1, le=10)
    # 重排相关性阈值：低于该分的候选视为不相关并丢弃，
    # 全部丢弃时由 LLM 简短拒答，不引用任何法条
    # 注意：此阈值基于 bge-reranker-v2-m3 标定，切换为 qwen3.7-text-rerank 后
    # 需根据实际得分分布重新标定（qwen3.7-text-rerank 的 relevance_score 范围为 0~1）
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
        """statute/ 子文件夹下所有 .docx 法规文档（语料数据源，自动纳入新法规）。

        递归扫描子文件夹（基础法律依据 / 事故处理赔偿 / 行政处罚程序），
        子文件夹名作为分类写入向量库 metadata。
        """
        return sorted((BASE_DIR / "statute").rglob("*.docx"))

    @property
    def pdf_full_paths(self) -> list[Path]:
        """statute/ 子文件夹下所有 .pdf 文档（语料数据源，自动纳入新标准）。

        递归扫描子文件夹，子文件夹名作为分类写入向量库 metadata。
        """
        return sorted((BASE_DIR / "statute").rglob("*.pdf"))

    @property
    def law_sources(self) -> list[str]:
        """statute/ 目录下所有 docx/pdf 的干净法规名列表，
        作为 SYSTEM_PROMPT 中助手可回答的法规清单，避免在提示词中硬编码法规名。
        复用 ingestion._clean_source_name 保证与向量库 source 字段一致。
        """
        # 延迟导入避免 config ↔ ingestion 循环依赖（ingestion 在 module level 导入 config）
        from app.ingestion import _clean_source_name

        paths = self.docx_full_paths + self.pdf_full_paths
        return [_clean_source_name(p) for p in paths]

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
            p = USER_DATA_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()