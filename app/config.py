"""应用配置：从 .env 读取运行参数。

工程约定：LLM 配置（OPENAI_API_BASE / OPENAI_API_KEY / LLM_MODEL）必须写在 .env 中，
通过 pydantic-settings 统一加载。
"""
from functools import lru_cache
from pathlib import Path

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

    # 数据与存储
    docx_path: str = "中华人民共和国道路交通安全法_20210429.docx"
    chroma_dir: str = "chroma"

    # Embedding / Rerank 模型（轻量版，CPU 友好）
    embedding_model_id: str = "BAAI/bge-small-zh-v1.5"
    rerank_model_id: str = "BAAI/bge-reranker-base"
    modelscope_cache_dir: str = "models"

    # 检索参数
    top_k_retrieve: int = 5
    bm25_weight: float = 0.5
    rrf_lambda: int = 60
    rerank_top_n: int = 3
    # 重排相关性阈值：低于该分的候选视为不相关并丢弃，
    # 全部丢弃时由 LLM 简短拒答，不引用任何法条
    # 实测分布：无关 query top1≈0~0.04，相关 query top1>0.5，0.2 落在两者间留有缓冲
    rerank_min_score: float = 0.2

    # 服务
    backend_url: str = "http://127.0.0.1:8000"

    @property
    def docx_full_path(self) -> Path:
        p = Path(self.docx_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    @property
    def chroma_full_dir(self) -> Path:
        p = Path(self.chroma_dir)
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