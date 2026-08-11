# dev_gz 服务器部署文档

本文档记录把当前项目部署到 `dev_gz:/root/crypto_trade` 的可重复流程。部署默认只启动纸上交易，不会真实下单。

## 目标

- 代码目录：`/root/crypto_trade`
- 运行方式：Docker Compose
- 默认运行配置：`config/production.yaml`
- 默认启动模式：`scripts/run_production.py --dry-run`
- 容器名：`autocrypto`

## 安全约束

- 不同步本地 `.env`、`.venv`、`outputs`、`data`。
- 默认服务使用 `--dry-run`，只看行情和模拟状态，不下单。
- `config/production.yaml` 默认 `portfolio.allow_spot_trading=false`、`portfolio.allow_swap_trading=false`。
- 非 dry-run 启动会自动执行 preflight，未通过会拒绝启动。

## 首次部署

在本地执行：

```bash
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude .env \
  --exclude outputs \
  --exclude data \
  ./ dev_gz:/root/crypto_trade/
```

在服务器执行：

```bash
ssh dev_gz
cd /root/crypto_trade
mkdir -p outputs
touch .env
chmod 600 .env
docker compose build
```

## 配置 OKX API

纸上交易可以暂不配置 API。进入 readonly 或 testnet/live 前必须配置：

```bash
cat > /root/crypto_trade/.env << 'EOF'
OKX_API_KEY=...
OKX_SECRET=...
OKX_PASSPHRASE=...
EOF
chmod 600 /root/crypto_trade/.env
```

API 权限只允许 Read/Trade，禁止 Withdraw，并建议绑定服务器 IP。

## preflight

纸上配置检查：

```bash
cd /root/crypto_trade
docker compose run --rm bot uv run python scripts/okx_preflight.py --config config/production.yaml --price 100000
```

OKX 模拟盘配置检查：

```bash
cd /root/crypto_trade
docker compose run --rm bot uv run python scripts/okx_preflight.py --config config/testnet_live.yaml
```

`config/testnet_live.yaml` 默认会失败在 `live_mode_switches`，这是安全预期：需要人工确认后才打开 `allow_swap_trading`。

## 启动 Docker 服务

默认 `docker-compose.yml` 使用 `--dry-run`，不会真实下单：

```bash
cd /root/crypto_trade
docker compose up -d
```

查看状态和日志：

```bash
docker compose ps
docker compose logs -f bot
tail -f /root/crypto_trade/outputs/production.log
```

停止：

```bash
docker compose down
```

## 本次部署结果

已在 `dev_gz:/root/crypto_trade` 完成：

- 代码同步。
- Docker 镜像构建：`crypto_trade-bot:latest`。
- 容器内单元测试通过。
- `config/production.yaml` preflight 通过。

但当前服务器访问 OKX 超时：

```bash
curl -I https://www.okx.com/api/v5/public/time
```

宿主机和容器内都会超时，因此服务已停止，避免持续重启刷日志。网络恢复后启动：

```bash
cd /root/crypto_trade
docker compose up -d
docker compose logs -f bot
```

如果仍超时，需要先处理服务器到 OKX 的出口网络、DNS、防火墙或代理。

## 升级部署

本地同步：

```bash
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude .env \
  --exclude outputs \
  --exclude data \
  ./ dev_gz:/root/crypto_trade/
```

服务器更新依赖并重启：

```bash
cd /root/crypto_trade
docker compose build
docker compose run --rm bot uv run python -m unittest discover -s tests
docker compose up -d
docker compose ps
```

## 切换到 OKX 模拟盘

1. 配置 Demo API 到 `/root/crypto_trade/.env`。
2. 按 `docs/okx-testnet-runbook.md` 执行。
3. 修改 `config/testnet_live.yaml`，只打开：

```yaml
portfolio:
  allow_spot_trading: false
  allow_swap_trading: true
```

4. 运行 preflight，必须 `ok: true`。
5. 修改 `docker-compose.yml` 的 `command` 为：

```yaml
command: ["uv", "run", "python", "scripts/run_production.py", "--config", "config/testnet_live.yaml"]
```

6. `docker compose up -d`

## venv 备选

如果不使用 Docker，服务器需要先安装 `python3.12-venv`：

```bash
apt-get update
apt-get install -y python3.12-venv python3-pip
cd /root/crypto_trade
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## 不建议直接实盘

当前版本只适合 paper/readonly/testnet_live 阶段。真实资金前还缺：

- 真实现货核心仓执行。
- 资金费和更完整成交明细落账。
- 外部 webhook/短信/邮件告警。
- 连续 1-2 周 OKX 模拟盘稳定运行记录。
