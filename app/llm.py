"""阿里云百炼大模型调用封装（OpenAI 兼容接口）。"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.errors import LLMError


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

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """调用百炼生成回答，返回文本内容。temperature 可覆盖默认值（如查询改写用 0.0）。"""
        self._ensure_loaded()
        settings = get_settings()
        try:
            resp = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                # qwen3 系列关闭思考输出（非流式调用不需要思考）；其他模型忽略该参数
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - 统一转为领域异常
            raise LLMError(f"大模型调用失败：{exc}") from exc

    def chat_stream(self, system_prompt: str, user_prompt: str):
        """流式调用百炼生成回答，逐段产出 (kind, text) 元组。

        kind ∈ {"reasoning", "content"}：
        - reasoning：推理模型的思考过程（来自 delta.reasoning_content，普通模型为 None）
        - content：正文（来自 delta.content）
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
                temperature=0.2,
                stream=True,
            )
            for chunk in resp:
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
        default_prompt = (
            "请准确识别图片中的文字，按原始排版输出。"
            "保留章节标题（如'1 范围'）和条款编号（如'3.1'、'5.2.1'）的格式，每个逻辑行单独一行。"
            "表格请转换为'列名1 | 列名2 | ...'的文本格式，不要添加多余解释。"
        )
        user_content: list[dict] = [{"type": "text", "text": prompt or default_prompt}]
        for b64 in base64_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        try:
            resp = self._client.chat.completions.create(
                model=settings.ocr_model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OCR 模型调用失败：{exc}") from exc


@lru_cache
def get_llm() -> BailianClient:
    return BailianClient()