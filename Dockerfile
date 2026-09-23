# 构建参数：镜像源可切换。
# 默认走 Debian / PyPI 官方源——构建已迁移到 GitHub Actions（境外 runner），
# 官方源最快。如需在国内构建，用 --build-arg 传回阿里云源：
#   --build-arg APT_MIRROR=mirrors.aliyun.com \
#   --build-arg APT_SECURITY_MIRROR=mirrors.aliyun.com \
#   --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG APT_MIRROR=deb.debian.org
ARG APT_SECURITY_MIRROR=security.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple

# ===== Stage 1: builder（装 Python 依赖）=====
FROM python:3.13-slim AS builder

# ARG 在 FROM 之后需重新声明，才能被本阶段的 RUN 使用
ARG APT_MIRROR
ARG APT_SECURITY_MIRROR
ARG PIP_INDEX_URL

WORKDIR /app

# 系统依赖：编译工具 + OpenMP（pymupdf/chromadb 可能需要）
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
    sed -i "s|security.debian.org|${APT_SECURITY_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements 加速缓存
COPY requirements.txt .
# BuildKit 缓存 pip 下载目录：跨构建复用已下载的 wheel，
# requirements 变化导致该层缓存失效时，未变动的包无需重新下载，大幅缩短重装时间。
# 注意：使用缓存挂载时不能再传 --no-cache-dir（否则禁写缓存、缓存挂载失效）。
# 装完清理 __pycache__/*.pyc：它们对运行无用，但会被 COPY --from=builder 一并搬进
# 运行时镜像，属于可直接省掉的体积。
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip -i "${PIP_INDEX_URL}" && \
    /opt/venv/bin/pip install -r requirements.txt -i "${PIP_INDEX_URL}" && \
    find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.pyc' -delete

# ===== Stage 2: runner（运行时镜像）=====
FROM python:3.13-slim AS runner

# 同上：ARG 需在本阶段重新声明
ARG APT_MIRROR
ARG APT_SECURITY_MIRROR

WORKDIR /app

# 运行时系统依赖（仅保留 libgomp1）
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
    sed -i "s|security.debian.org|${APT_SECURITY_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# 从 builder 复制 venv
COPY --from=builder /opt/venv /opt/venv

# 设置环境变量：使用 venv 的 Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    HOME=/root

# 复制项目代码（.dockerignore 已排除 .env/node_modules/__pycache__ 等）
COPY app/ ./app/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY statute/ ./statute/
COPY requirements.txt .

# 数据目录（运行时由 volume 挂载，这里先建占位）
RUN mkdir -p /app/chroma /root/.law_helper

# 暴露 8000（仅容器内网，docker-compose 不映射到宿主机公网）
EXPOSE 8000

# uvicorn 单 worker（2G 内存服务器建议 1，避免多进程占用过高）
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
