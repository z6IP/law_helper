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
# 与 deploy.sh 中 BACKEND_IMAGE 的拼装保持一致，修改此处需同步 deploy.sh
export BACKEND_IMAGE="${ACR_REGISTRY_HOST}/${ACR_NAMESPACE}/${BACKEND_IMAGE_REPO:-law-helper-backend}:${IMAGE_TAG:-latest}"

exec docker compose "$@"
