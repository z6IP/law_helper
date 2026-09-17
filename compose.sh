#!/usr/bin/env bash
# 服务器侧 docker compose 包装脚本：补齐 ${BACKEND_IMAGE} 后转发命令。
#
# 背景：docker-compose.yml 中 backend.image 为 ${BACKEND_IMAGE:?...}，
# 任何 compose 子命令（含 stop / ps / logs）在解析阶段都会校验该变量；
# 而 BACKEND_IMAGE 并不写在 deploy.env 里，是由 ACR_REGISTRY_HOST / ACR_NAMESPACE /
# BACKEND_IMAGE_REPO 拼装出来的。deploy.sh 自己会 source deploy.env 并导出，但
# sync_index.ps1 等外部脚本直接跑 docker compose 会报「BACKEND_IMAGE 未设置」。
# 此脚本把这段环境加载集中到一处，供外部调用。
#
# 用法（在服务器 /opt/law_helper 下，或用绝对路径调用）：
#   bash compose.sh stop backend
#   bash compose.sh up -d
#   IMAGE_TAG=<完整40位sha> bash compose.sh up -d    # 需要指定历史镜像时
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f deploy.env ]; then
  echo "[错误] 缺少 deploy.env（部署期配置），请先 cp deploy.env.example deploy.env 并填写" >&2
  exit 1
fi

# set -a 使 deploy.env 中的变量自动导出，供 docker compose 做 ${BACKEND_IMAGE} 插值
set -a
# shellcheck disable=SC1091
. ./deploy.env
set +a

: "${ACR_REGISTRY_HOST:?deploy.env 缺少 ACR_REGISTRY_HOST}"
: "${ACR_NAMESPACE:?deploy.env 缺少 ACR_NAMESPACE}"
# 未显式指定 IMAGE_TAG 时，沿用当前运行容器所用的标签，绝不回退 :latest：
# 回退 latest 会让 sync_index.ps1 的 `compose.sh up -d` 把后端换成服务器上
# 遗留的旧镜像（或直接找不到镜像），与 docker-compose.yml 的设计意图冲突。
if [ -z "${IMAGE_TAG:-}" ]; then
  # 末尾的 || true：容器不存在时 docker inspect 返回非 0，set -o pipefail + set -e
  # 会让赋值语句直接终止脚本并吞掉下面的可读报错，这里先兜住退出码。
  IMAGE_TAG="$(docker inspect -f '{{.Config.Image}}' law-helper-backend 2>/dev/null | sed 's/.*://' || true)"
fi
: "${IMAGE_TAG:?无法确定镜像标签：请显式传 IMAGE_TAG=<完整 sha>，或先执行 bash deploy.sh}"
# 与 deploy.sh 中 BACKEND_IMAGE 的拼装保持一致，修改此处需同步 deploy.sh
export BACKEND_IMAGE="${ACR_REGISTRY_HOST}/${ACR_NAMESPACE}/${BACKEND_IMAGE_REPO:-law-helper-backend}:${IMAGE_TAG}"

exec docker compose "$@"
