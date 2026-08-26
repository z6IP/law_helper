"""统一异常层级（LawHelperError）。

工程约定：错误处理统一走 LawHelperError 层级，便于 API 层集中捕获并返回语义化错误。
"""


class LawHelperError(Exception):
    """应用异常基类。"""

    status_code = 500
    message = "内部错误"

    def __init__(self, message: str | None = None):
        if message is not None:
            self.message = message
        super().__init__(self.message)


class ConfigError(LawHelperError):
    status_code = 500
    message = "配置错误"


class IngestionError(LawHelperError):
    status_code = 500
    message = "文档入库失败"


class RetrievalError(LawHelperError):
    status_code = 500
    message = "检索失败"


class LLMError(LawHelperError):
    status_code = 502
    message = "大模型调用失败"


class DocumentNotFoundError(LawHelperError):
    status_code = 404
    message = "文档不存在"