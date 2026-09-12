"""测试配置：将项目根目录加入模块搜索路径，并提供必填环境变量兜底。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 方案B：embedding 后端 / 模型 / 维度改为必填（无代码默认值），由 .env 提供。
# 测试环境可能没有 .env，这里用 setdefault 提供兜底值，避免 Settings 实例化失败；
# 若已存在环境变量或 .env 中的配置，不会被覆盖。
os.environ.setdefault("SESSION_SECRET_KEY", "test-secret")
os.environ.setdefault("EMBEDDING_BACKEND", "api")
os.environ.setdefault("EMBEDDING_MODEL_ID", "qwen3-vl-embedding")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "1024")
