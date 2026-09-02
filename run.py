"""一键启动前后端服务。

后端：uvicorn app.main:app（http://127.0.0.1:8000）
前端：React + Vite 开发服务器（http://localhost:5173）

前后端日志直接输出到当前控制台，不再落盘 .log 文件；
使用 law_helper_env conda 环境的 Python 解释器。
Ctrl+C 一次性退出两个子进程。
"""
from __future__ import annotations

import http.client
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from time import sleep, time as _time

# 尝试设置 Windows 控制台代码页为 UTF-8；同时启用行缓冲避免输出延迟
subprocess.run("chcp 65001", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True
    )

# law_helper_env conda 环境的 Python 解释器
_PYTHON = r"D:\miniconda3\envs\law_helper_env\python.exe"

# 子进程输出仅保留报错信息；启动/就绪信息由 run.py 自身的汇总提示负责
_ERROR_RE = re.compile(r"error|exception|traceback|critical|fatal|warning|warn", re.IGNORECASE)
# traceback 续行（缩进的 File/line/raise/at 行）
_TRACEBACK_RE = re.compile(r'^\s*(File "|line \d+\b|raise |at )')
# uvicorn 自带的 INFO:/DEBUG: 框架前缀，去除以免与 logger 级别重复
_FRAMING_RE = re.compile(r"^\s*(?:INFO|DEBUG)\s*:\s*")


def _should_emit(line: str) -> bool:
    """判断子进程输出是否值得展示（仅报错信息），其余噪音过滤。"""
    return bool(_ERROR_RE.search(line) or _TRACEBACK_RE.search(line))


def _setup_logging() -> None:
    """配置控制台日志格式：时间 + 级别 + 来源，只输出到控制台，不落盘。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-8s %(message)s", "%H:%M:%S")
    )
    for name in ("backend", "frontend"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.propagate = False


def _stream(prefix: str, pipe) -> None:
    """子进程输出经 logger 精简后打到控制台：仅报错信息。"""
    logger = logging.getLogger(prefix)
    for raw in pipe:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.strip() or not _should_emit(line):
            continue
        logger.info(_FRAMING_RE.sub("", line))


def _wait_for_backend(timeout: float = 300.0, interval: float = 1.0) -> bool:
    """等待后端预热完成（/api/v1/ready 返回 ready=True）。

    原实现直接轮询 /api/v1/health，但预热在 startup 事件中同步执行，
    FastAPI 在 startup 完成前不会响应任何 HTTP 请求，导致每次 conn.request
    都因 2 秒 socket 超时而伪失败，循环重试到 deadline 仍未通过。
    现在预热已迁到后台线程，startup 立即返回，health 与 ready 接口可即时响应；
    这里改为轮询 /ready，并在等待期间打印预热阶段进度。
    """
    deadline = _time() + timeout
    last_stage = None
    while _time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
            conn.request("GET", "/api/v1/ready")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status == 200:
                data = json.loads(body.decode("utf-8"))
                if data.get("ready"):
                    return True
                if data.get("error"):
                    print(f"\n[错误] 后端预热失败：{data['error']}")
                    return False
                stage = data.get("stage") or "unknown"
                if stage != last_stage:
                    print(f"[启动] 后端预热中... 阶段：{stage}")
                    last_stage = stage
        except Exception:
            pass
        sleep(interval)
    return False


def main() -> None:
    # 强制子进程使用 UTF-8 输出，避免 Windows 控制台 GBK 编码导致的中文乱码
    _env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # 日志仅输出到控制台，不落盘 .log 文件
    _setup_logging()

    # 先声明进程句柄，使 _terminate 在任意时刻调用都能安全引用
    backend = None
    frontend = None

    def _terminate(*_):
        print("\n[退出] 正在终止前后端服务...")
        for p in (backend, frontend):
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
        # 给子进程一点时间清理
        sleep(0.5)
        for p in (backend, frontend):
            if p is not None:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(0)

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
    threading.Thread(target=_stream, args=("backend", backend.stdout), daemon=True).start()

    # 等待后端预热完成（模型加载、索引校验）后再启动前端，避免前端代理报错
    print("[启动] 等待后端预热完成（最长 5 分钟，重启电脑后首次启动可能较慢）...")
    if not _wait_for_backend():
        print("\n[错误] 后端预热超时，请查看上方输出的后端日志")
        _terminate()
    print("[启动] 后端预热完成，启动前端...")

    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    # 直接调用 node 运行 vite.js，避免 npm/.cmd 中间层产生孤儿进程
    frontend = subprocess.Popen(
        ["node", "./node_modules/vite/bin/vite.js"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=frontend_dir,
        env=_env,
    )
    threading.Thread(target=_stream, args=("frontend", frontend.stdout), daemon=True).start()

    print("=" * 60)
    print("后端: http://127.0.0.1:8000  (uvicorn)")
    print("前端: http://localhost:5173  (vite)")
    print("按 Ctrl+C 退出两个服务")
    print("=" * 60)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # 阻塞主线程：任一子进程异常退出则打印对应日志尾部并整体退出
    while True:
        if backend.poll() is not None:
            print("\n[退出] 后端进程已退出，前端随之停止。")
            _terminate()
        if frontend.poll() is not None:
            print("\n[退出] 前端进程已退出，后端随之停止。")
            _terminate()
        sleep(1)


if __name__ == "__main__":
    main()
