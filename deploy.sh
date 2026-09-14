#!/usr/bin/env bash
# 部署脚本：在服务器 /opt/law_helper/ 下执行
# 用法: bash deploy.sh
#
# 前端采用「本地构建上传」模式：
#   2G 内存服务器跑 vite build 会导致 OOM 整机卡死，切勿在服务器上构建前端。
#   正确流程：本地 cd frontend && npm run build
#            然后 scp -r dist/* root@<服务器IP>:/opt/law_helper/frontend/dist/
#            最后在服务器上运行 bash deploy.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "===== 1. 拉取最新代码 ====="
# 服务器代码应完全镜像 GitHub，用 fetch + reset --hard 避免合并冲突/凭证提示
git fetch origin main
git reset --hard origin/main
git clean -fd  # 清理未跟踪文件（frontend/dist/ 已被 .gitignore 忽略，不会误删上传的产物）

echo ""
echo "===== 1.5 安全前置：TLS 证书与访问口令 ====="
# 域名与 Basic Auth 用户名（可覆盖为真实值）
DOMAIN="${DOMAIN:-your-domain.com}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-admin}"

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

# 2) Basic Auth 口令文件（首次生成；已存在则跳过）
HTPASSWD_FILE="nginx/.htpasswd"
if [ ! -f "$HTPASSWD_FILE" ]; then
  if [ -z "${BASIC_AUTH_PASS:-}" ]; then
    echo "首次部署需设置访问口令（用户：${BASIC_AUTH_USER}）："
    read -rsp "请输入访问口令: " BASIC_AUTH_PASS || true
    echo
  fi
  if [ -z "$BASIC_AUTH_PASS" ]; then
    echo "口令不能为空。请设置 BASIC_AUTH_PASS 环境变量后重试，或手动生成："
    echo "  printf '${BASIC_AUTH_USER}:' > nginx/.htpasswd && openssl passwd -apr1 >> nginx/.htpasswd"
    exit 1
  fi
  printf '%s:%s\n' "$BASIC_AUTH_USER" "$(openssl passwd -apr1 "$BASIC_AUTH_PASS")" > "$HTPASSWD_FILE"
  chmod 644 "$HTPASSWD_FILE"
  echo "已生成 $HTPASSWD_FILE"
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
echo "===== 3. 重建后端 Docker 镜像 ====="
docker compose build backend

echo ""
echo "===== 4. 设置前端文件权限 ====="
chmod -R a+rX frontend/dist

echo ""
echo "===== 5. 确保 swap（2G 内存运行期兜底）====="
# swap 防止后端瞬时内存峰值（预热 / 检索高峰）把 2G 机器逼到 OOM 整机卡死
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
echo "===== 8. 查看服务状态 ====="
docker compose ps

echo ""
echo "===== 部署完成 ====="
