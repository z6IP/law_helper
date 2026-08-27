"""一键启动前后端服务。

后端：uvicorn app.main:app（http://127.0.0.1:8000）
前端：streamlit run ui/streamlit_app.py（http://localhost:8501）

使用 law_helper_env conda 环境的 Python 解释器。
Ctrl+C 一次性退出两个子进程。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from time import sleep

# law_helper_env conda 环境的 Python 解释器
_PYTHON = r"D:\miniconda3\envs\law_helper_env\python.exe"
_BACKEND_URL = "http://127.0.0.1:8000/api/v1/health"


def _wait_backend_ready(timeout: float = 120.0, interval: float = 2.0) -> bool:
    """轮询后端 health 端点，直到就绪或超时。

    后端 startup_preload 会加载 Embedding / Reranker 模型 + 重建索引，
    这通常需要 30~90 秒，固定 sleep 不可靠。
    """
    elapsed = 0.0
    while elapsed < timeout:
        try:
            with urllib.request.urlopen(_BACKEND_URL, timeout=3) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    if body.get("status") == "ok":
                        return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        except Exception:
            pass
        print(f"[启动] 等待后端就绪... ({elapsed:.0f}s / {timeout:.0f}s)")
        sleep(interval)
        elapsed += interval
    return False


def _stream(prefix: str, pipe) -> None:
    """子进程输出加前缀后透传到当前控制台。"""
    for raw in pipe:
        line = raw.decode("utf-8", errors="replace")
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()


def main() -> None:
    # 强制子进程使用 UTF-8 输出，避免 Windows 控制台 GBK 编码导致的中文乱码
    _env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # 使用 law_helper_env 的 Python 启动 uvicorn
    backend = subprocess.Popen(
        [
            _PYTHON, "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=_env,
    )

    # 立即启动日志透传线程，让后端预加载日志实时可见
    threading.Thread(target=_stream, args=("backend", backend.stdout), daemon=True).start()

    # 轮询等待后端就绪（最多 120 秒），避免前端首请求落空（WinError 10061）
    if not _wait_backend_ready(timeout=120.0, interval=2.0):
        print("[启动] 后端在超时时间内未就绪，退出。")
        try:
            backend.terminate()
            sleep(0.5)
            backend.kill()
        except Exception:
            pass
        sys.exit(1)

    print("[启动] 后端就绪，启动前端...")
    frontend = subprocess.Popen(
        [_PYTHON, "-m", "streamlit", "run", "ui/streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=_env,
    )
    threading.Thread(target=_stream, args=("frontend", frontend.stdout), daemon=True).start()

    print("=" * 60)
    print("后端: http://127.0.0.1:8000  (uvicorn)")
    print("前端: http://localhost:8501  (streamlit)")
    print("按 Ctrl+C 退出两个服务")
    print("=" * 60)

    def _terminate(*_):
        for p in (backend, frontend):
            try:
                p.terminate()
            except Exception:
                pass
        # 给子进程一点时间清理
        sleep(0.5)
        for p in (backend, frontend):
            try:
                p.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # 阻塞主线程：任一子进程退出则全部退出
    while True:
        if backend.poll() is not None or frontend.poll() is not None:
            _terminate()
        sleep(1)


if __name__ == "__main__":
    main()
