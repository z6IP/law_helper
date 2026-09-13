#!/usr/bin/env bash
# 部署脚本：在服务器 /opt/law_helper/ 下执行
# 用法: bash deploy.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "===== 1. 拉取最新代码 ====="
# 服务器代码应完全镜像 GitHub，用 fetch + reset --hard 避免合并冲突/凭证提示
git fetch origin main
git reset --hard origin/main
git clean -fd  # 清理未跟踪的文件（如构建产物）

echo ""
echo "===== 1.5 安全前置：TLS 证书与访问口令 ====="
# 域名与 Basic Auth 用户名（可覆盖为真实值）
DOMAIN="${DOMAIN:-your-domain.com}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-admin}"

# 1) TLS 证书：域名就绪前用自签名占位保证 nginx 能启动；域名就绪后 certbot 申请真证书
CERT_DIR="/etc/nginx/certs"
LIVE_DIR="/etc/letsencrypt/live/${DOMAIN}"
sudo mkdir -p "$CERT_DIR"
if [ -f "$LIVE_DIR/fullchain.pem" ] && [ -f "$LIVE_DIR/privkey.pem" ]; then
  # 真证书已就绪：软链接到固定路径（certbot 续期后无需改 nginx 配置）
  sudo ln -sf "$LIVE_DIR/fullchain.pem" "$CERT_DIR/fullchain.pem"
  sudo ln -sf "$LIVE_DIR/privkey.pem" "$CERT_DIR/privkey.pem"
  echo "已链接正式证书：$LIVE_DIR"
elif [ ! -f "$CERT_DIR/fullchain.pem" ] || [ ! -f "$CERT_DIR/privkey.pem" ]; then
  echo "未检测到正式证书，生成自签名占位证书（域名就绪后请用 certbot 替换）..."
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
  chmod 600 "$HTPASSWD_FILE"
  echo "已生成 $HTPASSWD_FILE"
fi

echo ""
echo "===== 2. 重建后端 Docker 镜像 ====="
docker compose build backend

echo ""
echo "===== 3. 构建前端 ====="
# 小内存机器（<=2G）构建会卡死，自动加 swap 兜底
# 注意：set -euo pipefail 下管道中任一命令失败都会退出脚本，
# 因此用 || true 兜底 swapon --show 不支持的情况，再检测 /proc/swaps
has_swap=false
if swapon --show 2>/dev/null | grep -q . || grep -q '^/' /proc/swaps 2>/dev/null; then
  has_swap=true
fi
if [ "$has_swap" = false ]; then
  echo "未检测到 swap，创建 2G swapfile 防止 OOM 卡死..."
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile || true
  # 持久化到 fstab，重启后仍生效
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

cd frontend
# 清理旧产物，确保新 hash 文件名
rm -rf dist
npm install  # 确保依赖完整（首次或 package.json 变动时）
# 给 Node 加内存上限，避免 Vite 转译 1217 模块时堆溢出
NODE_OPTIONS="--max-old-space-size=2048" npm run build
cd ..

echo ""
echo "===== 4. 设置前端文件权限 ====="
chmod -R a+rX frontend/dist
# 重启 nginx 让权限立即生效（避免 403）
docker compose restart nginx 2>/dev/null || true

echo ""
echo "===== 5. 重启服务 ====="
docker compose up -d

echo ""
echo "===== 6. 等待后端健康检查 ====="
echo "后端预热中（embedding/rerank warmup + 索引校验），约需 1-2 分钟..."
timeout 180 bash -c 'until docker compose ps backend | grep -q "healthy"; do sleep 5; echo "  等待中..."; done' || echo "  超时，请手动检查: docker compose logs backend"

echo ""
echo "===== 7. 查看服务状态 ====="
docker compose ps

echo ""
echo "===== 部署完成 ====="
echo "域名就绪前：http://<服务器IP>（自签名证书会触发浏览器告警，属正常过渡期现象）"
echo "域名就绪后（需先完成下方证书申请）：https://<你的域名>"
echo ""
echo "===== 附：域名就绪后申请正式 HTTPS 证书 ====="
echo "1. 将 nginx/nginx.conf 中的两处 your-domain.com 替换为真实域名"
echo "2. 安装 certbot 并申请证书（webroot 模式，无需停 nginx）："
echo "   sudo certbot certonly --webroot -w /var/www/certbot -d 你的域名"
echo "3. 重新运行: bash deploy.sh（自动链接正式证书到固定路径并 reload）"
echo "4. 证书续期（certbot timer 默认启用，续期后自动 reload nginx）："
echo "   sudo certbot renew --deploy-hook 'docker exec law-helper-nginx nginx -s reload'"
echo "5. 在 .env 中设置：COOKIE_HTTPS_ONLY=true、CORS_ORIGINS=https://你的域名"
