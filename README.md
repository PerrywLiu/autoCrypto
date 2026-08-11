# autoCrypto

加密货币趋势交易研究与 OKX 交易机器人项目。当前项目已经从单纯回测框架，推进到包含历史回测、实时 paper trading、OKX 模拟盘订单生命周期验证、生产部署、preflight 检查和组合级风控的交易系统。

> 风险提示：本项目用于量化策略研发和交易系统验证，不构成投资建议。真实资金运行前必须经过 paper、readonly、testnet_live 和小资金灰度验证。

## 当前状态

已具备：

- 基于 `backtesting.py` 的历史回测框架。
- OKX 行情接入和历史 K 线下载。
- Triple EMA 4h 生产策略机器人，默认纸上交易。
- BTC 现货核心仓 + BTC-USDT 永续合约卫星仓的组合风控模型。
- 非 dry-run 启动前 preflight 检查：配置、API key、账户权益、挂单、合约仓位、本地状态。
- OKX readonly 账户快照解析。
- OKX testnet/live 合约下单执行器：入场、成交确认、TP/SL 保护单确认、reduceOnly 平仓、撤保护单。
- 本地状态恢复、状态 schema 校验、连续 API 错误熔断、本地 JSONL 告警。
- Docker / docker compose 部署入口。

仍需谨慎对待：

- 真实现货核心仓下单默认未开放。
- 资金费和完整成交明细落账仍不完整。
- 跨重启订单生命周期恢复和外部告警仍在完善中。
- 当前默认配置适合长期 `--dry-run` 和 testnet 验证，不适合直接真实资金自动交易。

## 技术栈

- Python 3.12
- uv
- ccxt
- backtesting.py
- pandas / numpy / pandas-ta
- plotly
- pyyaml / python-dotenv
- loguru
- Docker

## 项目结构

```text
autoCrypto/
├── config/
│   ├── production.yaml          # 生产机器人默认配置，默认 dry-run
│   ├── testnet_live.yaml        # OKX 模拟盘订单验证配置
│   ├── live.yaml
│   └── strategies/              # 回测策略参数
├── docs/
│   ├── okx-live-integration.md  # OKX 实盘接入设计与当前状态
│   └── okx-testnet-runbook.md   # testnet_live 运行手册
├── scripts/
│   ├── run_production.py        # 生产机器人入口
│   ├── okx_preflight.py         # 非 dry-run 启动前检查
│   ├── live.py                  # 旧版双策略 live bot 入口
│   ├── download.py              # 历史行情下载
│   ├── backtest.py              # 标准回测入口
│   ├── backtest_online.py       # 在线/离线回测辅助
│   └── backtest_portfolio.py    # 组合策略回测
├── src/
│   ├── data/                    # OKX 数据下载、清洗、资金费模型
│   ├── strategies/              # 回测策略
│   ├── backtest/                # 回测引擎和绩效指标
│   ├── live/                    # 生产机器人、风控、账户、订单执行、告警
│   └── visualization/           # 可视化图表
├── tests/
│   └── test_live_risk.py        # 风控、preflight、订单和状态恢复测试
├── outputs/                     # 日志、交易记录、状态文件
├── Dockerfile
└── docker-compose.yml
```

## 安装

```bash
uv sync
```

如果不使用 `uv`，也可以用 `requirements.txt` 创建普通 venv，但项目当前主要以 `uv run ...` 作为运行方式。

## 环境变量

OKX 相关功能需要 `.env`：

```bash
OKX_API_KEY=your_api_key
OKX_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase
```

建议：

- API Key 只开读取和交易权限。
- 永远不要开启提现权限。
- 实盘 API Key 绑定服务器 IP 白名单。
- testnet_live 使用 OKX Demo Trading API Key，不要混用真实账户 API Key。

## 运行模式

### backtest

历史回测，不连接 OKX 账户，使用本地历史数据。

```bash
uv run python scripts/backtest.py --strategy ema_rsi_atr --symbol BTC/USDT
uv run python scripts/backtest.py --compare --symbol BTC/USDT
uv run python scripts/backtest_portfolio.py
```

### paper

实时行情 + 本地模拟账户，不发送真实订单。生产配置默认就是 paper/dry-run。

```bash
uv run python scripts/run_production.py --dry-run
```

### readonly

读取 OKX 真实账户快照，只做对账和计划输出，不下单。通过配置 `portfolio.mode: readonly` 并保持真实交易硬开关关闭。

### testnet_live

OKX 模拟盘真实订单生命周期验证。按 [docs/okx-testnet-runbook.md](docs/okx-testnet-runbook.md) 执行：

```bash
uv run python scripts/okx_preflight.py --config config/testnet_live.yaml
uv run python scripts/run_production.py --config config/testnet_live.yaml
```

preflight 不通过时，`run_production.py` 会拒绝启动非 dry-run。

### live_small / live_scaled

真实资金阶段。必须先完成 paper、readonly、testnet_live 和小资金灰度复盘，再考虑开启。

```bash
uv run python scripts/run_production.py --live
```

真实下单还受配置硬开关控制：

```yaml
portfolio:
  allow_spot_trading: false
  allow_swap_trading: false
```

默认均为 `false`，即使 `dry_run=false`，未打开对应开关也不会真实下单。

## 核心配置

生产配置位于 [config/production.yaml](config/production.yaml)。

关键字段：

```yaml
symbol: "BTC/USDT"
timeframe: "4h"
market_type: "swap"

strategy:
  name: "triple_ema"
  ema_fast: 10
  ema_mid: 20
  ema_slow: 50
  ema_trend: 200
  adx_threshold: 20
  sl_atr_mult: 1.5
  tp_atr_mult: 3.0

risk:
  initial_capital: 500
  leverage: 2
  risk_per_trade: 0.02
  max_notional_pct: 0.95
  max_daily_loss_pct: 0.05
  max_consecutive_losses: 5
  loss_pause_bars: 18

portfolio:
  enabled: true
  mode: "paper"
  max_net_exposure: 0.90
  max_gross_exposure: 1.20
  max_short_exposure: 0.30
  max_order_notional_usdt: 20
  allow_spot_trading: false
  allow_swap_trading: false

mode:
  dry_run: true
  testnet: true
```

## 下载历史数据

```bash
uv run python scripts/download.py --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2025-06-01
uv run python scripts/download.py --symbols BTC/USDT ETH/USDT --tfs 1h 4h --start 2024-01-01
```

历史数据默认写入 `data/`，输出报告和运行状态默认写入 `outputs/`。

## Docker 部署

默认 Docker 启动也是安全的纸上交易：

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

`docker-compose.yml` 默认命令：

```yaml
command: ["uv", "run", "python", "scripts/run_production.py", "--dry-run"]
```

模拟盘或实盘部署前请先阅读：

- [DEPLOY_PRODUCTION.md](DEPLOY_PRODUCTION.md)
- [docs/okx-live-integration.md](docs/okx-live-integration.md)
- [docs/okx-testnet-runbook.md](docs/okx-testnet-runbook.md)

## 验证

当前测试使用标准库 `unittest` 可直接运行：

```bash
uv run python -m unittest tests.test_live_risk
```

已覆盖的重点：

- 仓位大小计算和杠杆上限。
- 连续亏损暂停。
- production/testnet 配置安全边界。
- preflight 配置和账户检查。
- OKX 账户快照解析。
- 组合核心仓/卫星仓曝光计算。
- 订单计划、保护单、状态恢复和暂停条件。

## 推荐上线流程

1. `backtest`：确认策略参数和回测区间。
2. `paper`：至少连续运行 2 周，观察信号、状态恢复和日志。
3. `readonly`：接真实账户只读快照，确认账户解析和组合计划正确。
4. `testnet_live`：用 OKX Demo API 验证下单、TP/SL、平仓、撤单和状态恢复。
5. `live_small`：真实小资金灰度，保持单笔和总敞口硬上限。
6. `live_scaled`：只有在稳定性指标满足后逐步增加资金。

任何非 dry-run 阶段都不要跳过 preflight，除非已经人工确认账户仓位、挂单、本地状态和配置一致。
