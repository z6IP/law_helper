# 上线安全检查清单

域名公网上线前的安全加固说明。本文档配合 `deploy.sh` 使用，按顺序执行即可。

## 1. 前置条件（域名与备案）

- **ICP 备案**：使用国内服务器时，域名必须完成 ICP 备案，否则 80/443 端口会被云厂商/运营商拦截。备案在云厂商控制台提交，通常需 1~2 周，**建议域名到手后第一时间提交**。
- 域名解析：将域名 A 记录指向服务器公网 IP。

## 2. 云服务器安全组（防火墙）

只放行以下端口，其余全部拒绝：

| 端口 | 用途 | 建议 |
|---|---|---|
| 22 | SSH | 改为仅允许你的办公/家庭 IP 访问 |
| 80 | HTTP（ACME + 跳转） | 全放行 |
| 443 | HTTPS | 全放行 |

SSH 加固：
- 改用密钥登录，禁用密码登录：`/etc/ssh/sshd_config` 中设 `PasswordAuthentication no`、`PubkeyAuthentication yes`。
- 修改默认 SSH 端口（如 2222）可进一步减少扫描。

## 3. fail2ban（防爆破）

```bash
sudo apt update && sudo apt install -y fail2ban
sudo systemctl enable --now fail2ban
```

默认 jail 即可拦截 SSH 爆破。如需更细配置，编辑 `/etc/fail2ban/jail.local`。

## 4. 项目文件权限

```bash
# 敏感配置仅 root 可读（.env 含 LLM API Key、会话密钥）
sudo chmod 600 /opt/law_helper/.env
# Basic Auth 口令文件
sudo chmod 600 /opt/law_helper/nginx/.htpasswd
# 证书私钥由 certbot 自动管理，权限默认 600，勿改动
```

## 5. 首次部署（域名就绪前，过渡期）

```bash
cd /opt/law_helper
# 生成自签名占位证书 + Basic Auth 口令（脚本会交互式询问口令）
bash deploy.sh
```

过渡期注意：
- `deploy.sh` 会生成自签名证书，保证 nginx 的 443 能启动。
- 用 `http://<服务器IP>` 访问时会被 301 到 https，浏览器会提示证书不受信任 —— 属正常现象，域名就绪并申请正式证书后消失。
- 所有请求（含页面与 `/api/`）都需要 Basic Auth 口令，防止 LLM 额度被公网盗刷。

## 6. 域名就绪后：申请正式证书

1. 将 `nginx/nginx.conf` 中两处 `your-domain.com` 替换为真实域名。
2. 安装 certbot：
   ```bash
   sudo apt update && sudo apt install -y certbot
   ```
3. 申请证书（webroot 模式，无需停 nginx）：
   ```bash
   sudo certbot certonly --webroot -w /var/www/certbot -d 你的域名
   ```
4. 重新运行 `bash deploy.sh`，脚本会自动把 `live/` 证书软链接到 `/etc/nginx/certs/` 固定路径并 reload。
5. 配置续期（certbot 默认已启用 timer；续期后 reload nginx）：
   ```bash
   sudo certbot renew --deploy-hook 'docker exec law-helper-nginx nginx -s reload'
   ```

> 证书路径设计：`nginx.conf` 固定引用 `/etc/nginx/certs/fullchain.pem`，`deploy.sh` 负责把 certbot 的 `live/` 证书软链接到这里（或生成自签名占位）。因此**证书更换无需再改 nginx 配置**。

## 7. 上线 .env 配置

申请完证书后，编辑 `/opt/law_helper/.env`：

```bash
# Cookie 仅通过 HTTPS 传输（上线必开）
COOKIE_HTTPS_ONLY=true

# 允许的前端来源改为正式域名
CORS_ORIGINS=https://你的域名

# 单用户模式：
#   true  = 所有会话归属当前浏览器（个人使用，配合 Basic Auth 已足够）
#   false = 严格按浏览器会话隔离（多用户场景需改造鉴权，当前不建议）
SINGLE_USER_MODE=true
```

修改后重启后端：`docker compose up -d backend`

## 8. 运维接口说明

- `/api/v1/ingest`（重建索引）已限制为**仅内网/本机**调用。运维时通过容器内触发：
  ```bash
  docker exec law-helper-backend curl -X POST http://localhost:8000/api/v1/ingest
  ```
- `/api/v1/health` 无鉴权，供负载均衡/监控探活使用。
- 可观测面板 `/dashboard` 与 `/api/v1/traces` 仅限回环地址访问，公网不可达。

## 9. 双层限流说明

- **nginx 层（IP 级）**：`/api/` 10 req/s（突发 20），`/api/v1/chat` 2 req/s（突发 5）。
- **后端层（浏览器会话级）**：chat 接口 10 次/60 秒。
- 二者叠加，既防单 IP 高频刷取，也防清 cookie 绕过。

## 10. 上线前最终核对清单

- [ ] 域名已备案、解析生效
- [ ] 安全组仅开放 80/443/SSH，SSH 改密钥登录
- [ ] fail2ban 已运行
- [ ] `.env` 与 `.htpasswd` 权限 600
- [ ] 正式证书已申请、续期钩子已配置
- [ ] `.env` 中 `COOKIE_HTTPS_ONLY=true`、`CORS_ORIGINS=https://域名`
- [ ] 浏览器访问 `https://域名` 无证书告警，聊天流式输出正常（SSE 未被限流/缓冲破坏）
