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
cd frontend
# 清理旧产物，确保新 hash 文件名
rm -rf dist
npm install  # 确保依赖完整（首次或 package.json 变动时）
npm run build
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
