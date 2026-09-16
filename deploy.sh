#!/usr/bin/env bash
# 部署脚本：在服务器 /opt/law_helper/ 下执行
# 用法: bash deploy.sh
#       IMAGE_TAG=<git-sha> bash deploy.sh   # 回滚/切换到指定镜像版本
#
# ===== 后端镜像不在本机构建 =====
#   构建链路：GitHub Actions 构建 → 推送阿里云 ACR → 本脚本只负责拉取与重启。
#   原因：2G 内存机器上构建 2GB 级镜像会与运行中的后端容器争抢内存（后端加载
#   embedding/rerank 模型常驻约 1.4G），触发内存颠簸导致整机卡死。构建搬离服务器后，
#   这类事故在结构上不再可能发生。
#
# ===== 一次性准备（新机器或首次启用时执行一次）=====
#   1) 阿里云 ACR 控制台：创建个人版实例、命名空间、镜像仓库 law-helper-backend
#   2) GitHub 仓库 Settings → Secrets and variables → Actions，新增：
#        ACR_REGISTRY / ACR_NAMESPACE / ACR_USERNAME / ACR_PASSWORD
#   3) 服务器登录 ACR（凭证写入 ~/.docker/config.json）：
#        docker login <ACR_REGISTRY> -u <用户名> -p <密码>
#      注意：ACR 的「免密拉取」只面向 ACK/ACS/ECI 等 K8s 产品，普通 ECS 不支持，
#      因此必须在服务器保存凭证——这是本方案明确的安全权衡，
#      请确认该文件权限为 600（chmod 600 ~/.docker/config.json）。
#   4) 复制配置样例并填写：
#        cp deploy.env.example deploy.env && vi deploy.env
#
# ===== 前端仍采用「本地构建上传」模式 =====
#   2G 内存服务器跑 vite build 会导致 OOM 整机卡死，切勿在服务器上构建前端。
#   正确流程：本地 cd frontend && npm run build
#            然后 scp -r dist/* root@<服务器IP>:/opt/law_helper/frontend/dist/
#            最后在服务器上运行 bash deploy.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "===== 0. 部署配置 ====="
if [ ! -f deploy.env ]; then
  echo "[错误] 缺少 deploy.env（部署期配置，已被 .gitignore 忽略）"
  echo ""
  echo "请先复制模板并填写："
  echo "  cp deploy.env.example deploy.env"
  echo "  vi deploy.env"
  echo ""
  echo "至少需要填 ACR_REGISTRY_HOST 与 ACR_NAMESPACE，例如："
  echo "  ACR_REGISTRY_HOST=registry.cn-hangzhou.aliyuncs.com"
  echo "  ACR_NAMESPACE=your-namespace"
  exit 1
fi

# 先记住命令行传入的 IMAGE_TAG，避免被 deploy.env 中的同名变量覆盖
_CLI_IMAGE_TAG="${IMAGE_TAG:-}"

# set -a 使 deploy.env 中的变量自动导出，供 docker compose 做 ${BACKEND_IMAGE} 插值
set -a
# shellcheck disable=SC1091
. ./deploy.env
set +a

: "${ACR_REGISTRY_HOST:?deploy.env 缺少 ACR_REGISTRY_HOST}"
: "${ACR_NAMESPACE:?deploy.env 缺少 ACR_NAMESPACE}"
BACKEND_IMAGE_REPO="${BACKEND_IMAGE_REPO:-law-helper-backend}"
IMAGE_TAG="${_CLI_IMAGE_TAG:-latest}"
export BACKEND_IMAGE="${ACR_REGISTRY_HOST}/${ACR_NAMESPACE}/${BACKEND_IMAGE_REPO}:${IMAGE_TAG}"
echo "目标镜像: ${BACKEND_IMAGE}"

if [ -n "$_CLI_IMAGE_TAG" ]; then
  echo "[提示] 已显式指定 IMAGE_TAG=${_CLI_IMAGE_TAG}（回滚模式）：将拉取该历史镜像。"
  echo "       注意本脚本第 1 步仍会把仓库代码更新到最新 main，"
  echo "       即最终状态为「旧镜像内的旧应用代码 + 最新的部署配置」。"
fi

echo ""
echo "===== 1. 拉取最新代码 ====="
# 服务器代码应完全镜像 GitHub，用 fetch + reset --hard 避免合并冲突/凭证提示
git fetch origin main
git reset --hard origin/main
# 清理未跟踪文件；deploy.env 与 frontend/dist/ 均被 .gitignore 忽略，不会被误删
git clean -fd

echo ""
echo "===== 1.5 安全前置：TLS 证书 ====="
# 域名（可覆盖为真实值）
DOMAIN="${DOMAIN:-lawhelper.xyz}"

# 1) TLS 证书：优先 certbot 正式证书 → 其次手动上传的证书 → 最后自签名占位
CERT_DIR="/etc/nginx/certs"
LIVE_DIR="/etc/letsencrypt/live/${DOMAIN}"
sudo mkdir -p "$CERT_DIR"
if [ -f "$LIVE_DIR/fullchain.pem" ] && [ -f "$LIVE_DIR/privkey.pem" ]; then
  # certbot 真证书已就绪：软链接到固定路径（续期后无需改 nginx 配置）
  sudo ln -sf "$LIVE_DIR/fullchain.pem" "$CERT_DIR/fullchain.pem"
  sudo ln -sf "$LIVE_DIR/privkey.pem" "$CERT_DIR/privkey.pem"
  echo "已链接 certbot 正式证书：$LIVE_DIR"
elif [ -f "$CERT_DIR/fullchain.pem" ] && [ -f "$CERT_DIR/privkey.pem" ]; then
  echo "已存在证书文件（手动上传 / 自签名占位），直接使用"
else
  echo "未检测到任何证书，生成自签名占位证书（域名就绪后请用 certbot 或手动上传替换）..."
  sudo openssl req -x509 -nodes -days 90 -newkey rsa:2048 \
    -keyout "$CERT_DIR/privkey.pem" -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=${DOMAIN}" 2>/dev/null
fi

echo ""
echo "===== 2. 前端产物检查（本地构建上传模式，不在服务器构建）====="
# 前端必须在本地构建后上传产物，服务器只做存在性校验，避免 vite build 打爆内存
if [ ! -f frontend/dist/index.html ]; then
  echo "[错误] 未检测到前端构建产物 frontend/dist/index.html"
  echo ""
  echo "请先在本地电脑执行："
  echo "  cd frontend"
  echo "  npm run build"
  echo "然后将产物上传到服务器："
  echo "  scp -r dist/* root@<服务器IP>:/opt/law_helper/frontend/dist/"
  echo ""
  echo "上传完成后重新运行: bash deploy.sh"
  exit 1
fi
echo "前端构建产物已就绪（frontend/dist/index.html），跳过服务器端构建"

echo ""
echo "===== 3. 拉取后端镜像（不在服务器构建）====="
# 拉取失败必须立即中止：nginx 依赖 backend 的健康状态（depends_on: service_healthy），
# 若镜像不可用还继续 up -d，会让 nginx 一直等待、服务直接起不来。
if ! docker compose pull backend; then
  echo ""
  echo "[错误] 镜像拉取失败，已中止部署（未重启服务，线上保持原状）"
  echo ""
  echo "请依次排查："
  echo "  1) 服务器是否已登录 ACR：docker login ${ACR_REGISTRY_HOST}"
  echo "  2) deploy.env 中的 ACR_REGISTRY_HOST / ACR_NAMESPACE / BACKEND_IMAGE_REPO 是否正确"
  echo "  3) 该标签是否已由 GitHub Actions 推送到 ACR：${BACKEND_IMAGE}"
  echo "  4) 网络连通性（若填的是 VPC 内网地址，需与 ACR 同地域）"
  exit 1
fi

# 双保险：确认镜像确实落到本地，避免标签拼写错误等造成"拉取成功但镜像不存在"
if ! docker image inspect "${BACKEND_IMAGE}" >/dev/null 2>&1; then
  echo "[错误] 镜像不存在：${BACKEND_IMAGE}（已中止部署，未重启服务）"
  exit 1
fi
echo "镜像就绪：${BACKEND_IMAGE}"

echo ""
echo "===== 4. 设置前端文件权限 ====="
chmod -R a+rX frontend/dist

echo ""
echo "===== 5. 确保 swap（运行期内存峰值兜底）====="
# swap 防止后端瞬时内存峰值（预热 / 检索高峰）把 2G 机器逼到 OOM 整机卡死。
# 构建已移出服务器，此处仅服务运行期。
has_swap=false
if swapon --show 2>/dev/null | grep -q . || grep -q '^/' /proc/swaps 2>/dev/null; then
  has_swap=true
fi
if [ "$has_swap" = false ]; then
  echo "未检测到 swap，创建 2G swapfile..."
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 644 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile || true
  # 持久化到 fstab，重启后仍生效
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "swap 已存在，跳过"
fi

echo ""
echo "===== 6. 重启服务 ====="
docker compose up -d

echo ""
echo "===== 7. 等待后端健康检查 ====="
echo "后端预热中（embedding/rerank warmup + 索引校验），约需 1-2 分钟..."
timeout 180 bash -c 'until docker compose ps backend | grep -q "healthy"; do sleep 5; echo "  等待中..."; done' || echo "  超时，请手动检查: docker compose logs backend"

echo ""
echo "===== 8. 清理悬空镜像（安全）====="
# 只清悬空镜像（untagged），不影响正在使用的镜像，也不影响带标签的旧版本。
# 绝不要使用 docker system prune --volumes：那会删除 law_helper_data 命名卷，
# 其中存放 SQLite 业务库与用户上传附件，删除后不可恢复。
docker image prune -f

echo ""
echo "===== 9. 验证新镜像已生效 ====="
# /settings/security 的 turnstile_script_srcs 字段只有本次改造后的后端才会返回，
# 用它确认容器确实跑在新拉取的镜像上，而不是旧容器残留。
_SEC=$(docker compose exec -T backend python -c \
  "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/api/v1/settings/security',timeout=5).read().decode())" \
  2>/dev/null || true)
case "$_SEC" in
  *turnstile_script_srcs*)
    echo "新镜像已生效：$_SEC"
    ;;
  *)
    echo "[提示] 未能确认新镜像字段，请手动检查："
    echo "  curl -s localhost:8000/api/v1/settings/security"
    ;;
esac

echo ""
echo "===== 10. 查看服务状态 ====="
docker compose ps

echo ""
echo "===== 部署完成 ====="
echo "当前镜像: ${BACKEND_IMAGE}"
echo "回滚方式: IMAGE_TAG=<git-sha> bash deploy.sh"
