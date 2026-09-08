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
echo "访问 http://<服务器IP> 验证"
