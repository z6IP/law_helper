"""一键启动前后端服务。

后端：uvicorn app.main:app（http://127.0.0.1:8000）
前端：React + Vite 开发服务器（http://localhost:5173）

前后端日志分别写入 logs/backend.log 与 logs/frontend.log，控制台互不干扰；
使用 law_helper_env conda 环境的 Python 解释器。
Ctrl+C 一次性退出两个子进程。
"""
from __future__ import annotations

import http.client
import io
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import sleep, time as _time

# 尝试设置 Windows 控制台代码页为 UTF-8；同时启用行缓冲避免输出延迟
subprocess.run("chcp 65001", shell=True, check=False)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True
    )

# law_helper_env conda 环境的 Python 解释器
_PYTHON = r"D:\miniconda3\envs\law_helper_env\python.exe"


def _log_dir() -> Path:
    """日志目录：项目根目录下的 logs/。"""
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    directory = base / "logs"
    directory.mkdir(exist_ok=True)
    return directory


def _open_log(name: str):
    """以追加模式打开日志文件，返回文件对象。"""
    path = _log_dir() / name
    return open(path, "a", encoding="utf-8", buffering=1)


def _write_banner(fobj) -> None:
    """每次启动在日志开头写入时间分隔线。"""
    fobj.write(f"\n{'=' * 60}\n")
    fobj.write(f"启动时间: {datetime.now().isoformat()}\n")
    fobj.write(f"{'=' * 60}\n")


def _stream(prefix: str, pipe, fobj) -> None:
    """子进程输出写入日志文件。"""
    for raw in pipe:
        line = raw.decode("utf-8", errors="replace")
        fobj.write(line)
        fobj.flush()


def _log_tail(path: Path, lines: int = 15) -> str:
    """读取日志文件最后 N 行，用于异常退出时提示用户。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception:
        return ""


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

    backend_log = _open_log("backend.log")
    frontend_log = _open_log("frontend.log")
    _write_banner(backend_log)
    _write_banner(frontend_log)

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
        backend_log.close()
        frontend_log.close()
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
    threading.Thread(target=_stream, args=("backend", backend.stdout, backend_log), daemon=True).start()

    # 等待后端预热完成（模型加载、索引校验）后再启动前端，避免前端代理报错
    print("[启动] 等待后端预热完成（最长 5 分钟，重启电脑后首次启动可能较慢）...")
    if not _wait_for_backend():
        print("\n[错误] 后端预热超时，请查看 logs/backend.log")
        print(_log_tail(_log_dir() / "backend.log"))
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
    threading.Thread(target=_stream, args=("frontend", frontend.stdout, frontend_log), daemon=True).start()

    print("=" * 60)
    print("后端: http://127.0.0.1:8000  (uvicorn)")
    print("前端: http://localhost:5173  (vite)")
    print("日志: logs/backend.log")
    print("      logs/frontend.log")
    print("按 Ctrl+C 退出两个服务")
    print("=" * 60)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # 阻塞主线程：任一子进程异常退出则打印对应日志尾部并整体退出
    while True:
        if backend.poll() is not None:
            print("\n[退出] 后端进程已退出，前端随之停止。")
            print("[退出] 后端日志尾部：")
            print(_log_tail(_log_dir() / "backend.log"))
            _terminate()
        if frontend.poll() is not None:
            print("\n[退出] 前端进程已退出，后端随之停止。")
            print("[退出] 前端日志尾部：")
            print(_log_tail(_log_dir() / "frontend.log"))
            _terminate()
        sleep(1)


if __name__ == "__main__":
    main()
