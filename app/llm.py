"""阿里云百炼大模型调用封装（OpenAI 兼容接口）。"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.errors import LLMError
from app.prompt_loader import render_prompt
from app.tracing import event, span


class BailianClient:
    """阿里云百炼对话封装（懒加载单例）。"""

    def __init__(self) -> None:
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
            raise LLMError("请在 .env 中配置有效的 OPENAI_API_KEY")
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float | None = None, enable_thinking: bool | None = None) -> str:
        """调用百炼生成回答，返回文本内容；未显式传入时使用 Settings 策略。

        非流式调用（含正式答案生成）默认关闭思考：思考过程只通过流式 chat_stream()
        实时产出供前端展示；非流式路径无 reasoning 事件通道，开启思考会白白消耗
        token 却无法让用户看到。如某场景确需思考，由调用方传 enable_thinking=True。
        """
        self._ensure_loaded()
        settings = get_settings()
        temperature = settings.answer_temperature if temperature is None else temperature
        enable_thinking = False if enable_thinking is None else enable_thinking
        try:
            with span("llm.chat", model=settings.llm_model, temperature=temperature):
                resp = self._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    extra_body={"enable_thinking": enable_thinking},
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    event(
                        "llm.tokens",
                        model=settings.llm_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                    )
                return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - 统一转为领域异常
            raise LLMError(f"大模型调用失败：{exc}") from exc

    def chat_stream(self, system_prompt: str, user_prompt: str):
        """流式调用百炼生成回答，逐段产出 (kind, text) 元组。

        kind ∈ {"reasoning", "content"}：
        - reasoning：推理模型的思考过程（来自 delta.reasoning_content，普通模型为 None）
        - content：正文（来自 delta.content）

        检索链构建方式：retrieval（BM25+向量 RRF 融合）→ rerank（CrossEncoder）→
        角色调整后的法条原文作为 user_prompt 上下文注入；本方法只负责「调用模型思考过程
        + 基于上下文生成答案」环节。思考开关和预算由 Settings 控制，
        思考过程通过 reasoning_content 流式产出供前端实时展示。
        """
        self._ensure_loaded()
        settings = get_settings()
        try:
            resp = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.answer_temperature,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={
                    "enable_thinking": settings.thinking_enabled,
                    "thinking_budget": settings.thinking_budget,
                },
            )
            usage = None
            with span("llm.chat_stream", model=settings.llm_model):
                for chunk in resp:
                    # 末尾 usage chunk：choices 可能为空，需先取用量再跳过
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                        continue
                    if not (chunk.choices and chunk.choices[0].delta):
                        continue
                    delta = chunk.choices[0].delta
                    # 思考过程（推理模型才有，普通模型该字段为 None）
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        yield ("reasoning", rc)
                    # 正文
                    if delta.content:
                        yield ("content", delta.content)
            if usage is not None:
                event(
                    "llm.tokens",
                    model=settings.llm_model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
        except Exception as exc:  # noqa: BLE001 - 统一转为领域异常
            raise LLMError(f"大模型调用失败：{exc}") from exc

    def ocr_images(self, base64_images: list[str], prompt: str | None = None) -> str:
        """调用视觉/OCR 模型识别图片中的文字。

        Args:
            base64_images: PNG 图片的 base64 编码列表（不含 data URI 前缀）。
            prompt: 用户提示词，为空时使用默认 OCR 提示。
        """
        self._ensure_loaded()
        settings = get_settings()
        if not base64_images:
            return ""
        default_prompt = render_prompt("ocr_default")
        user_content: list[dict] = [{"type": "text", "text": prompt or default_prompt}]
        for b64 in base64_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        try:
            with span("llm.ocr", model=settings.ocr_model, pages=len(base64_images)):
                resp = self._client.chat.completions.create(
                    model=settings.ocr_model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=settings.ocr_temperature,
                    extra_body={"enable_thinking": False},
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    event(
                        "llm.tokens",
                        model=settings.ocr_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                    )
                return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OCR 模型调用失败：{exc}") from exc


@lru_cache
def get_llm() -> BailianClient:
    return BailianClient()