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

纸面交易看板监听 `http://<服务器 IP>:8765/`，展示 OKX 行情、初始资金、现货核心仓、策略合约仓、设定杠杆，以及当前持仓和策略已平仓交易。当前权益按模拟现金、现货市值和策略累计盈亏计算；看板每分钟将实时估值保存到 `outputs/dashboard/equity.sqlite3`，可查看近 1 天、7 天、30 天或自定义起止时间的权益曲线。历史从首次采样时开始，无法从当前持仓准确反推此前的权益。看板只读取交易配置和 `outputs/` 中的账户文件，当前仅在 `paper` 模式提供数据。旧版现货调仓只写入日志，缺少可准确还原的逐笔成交明细，因此账户活动中会以当前持仓快照展示现货仓位。

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
uv run python -m unittest discover -s tests
```

已覆盖的重点：

- 仓位大小计算和杠杆上限。
- 连续亏损暂停。
- production/testnet 配置安全边界。
- preflight 配置和账户检查。
- OKX 账户快照解析。
- 组合核心仓/卫星仓曝光计算。
- 订单计划、保护单、状态恢复和暂停条件。
- 行情故障恢复、重启冷却兼容迁移、独立心跳与现货调仓台账。

2026-10-10 恢复逻辑修复已发布至 jp，仍为纸面交易；启动和行情轮询已验证，下一根自然收盘 K 线尚待观察。行为、验证范围和发布步骤见 [运行恢复说明](docs/production-recovery.md)。

## 推荐上线流程

源码交付遵循本地修复、测试与提交，push 到 GitHub 后，再由服务器拉取同一提交。服务器存在未提交内容时先比较并保全，不能用直接上传文件代替 Git 同步。具体步骤见 [Git 发布流程](docs/git-release-workflow.md)。

1. `backtest`：确认策略参数和回测区间。
2. `paper`：至少连续运行 2 周，观察信号、状态恢复和日志。
3. `readonly`：接真实账户只读快照，确认账户解析和组合计划正确。
4. `testnet_live`：用 OKX Demo API 验证下单、TP/SL、平仓、撤单和状态恢复。
5. `live_small`：真实小资金灰度，保持单笔和总敞口硬上限。
6. `live_scaled`：只有在稳定性指标满足后逐步增加资金。

任何非 dry-run 阶段都不要跳过 preflight，除非已经人工确认账户仓位、挂单、本地状态和配置一致。

### 整体收益与逐笔原因（2026-10-13，已发布至 jp）

交易记录上方显示累计账户收益、北京时间今日首条权益采样至最新估值的变化、合约已实现记账盈亏和已记录现货手续费。整体收益含剩余现货价格变化，已扣现货费用不重复扣减；历史手续费不完整时仅展示有记录部分。缺少今日采样或实时估值时显示未知。

逐笔「查看原因与收益」默认收起，刷新保留展开状态。后续纸面现货成交保存市场分类、仓位目标、紧急触发依据及成交后账户快照；合约保存入场指标、退出原因和平仓后账户快照。现有历史记录缺少的指标、成本和快照不补造，合约记录盈亏不冒充已扣完整费用与资金费的净收益。交易策略、风控参数和真实下单开关未修改。

验证：`python -m pytest tests -q` 共 97 项通过；本地浏览器验证默认折叠、展开/收起、刷新保留展开、历史缺失提示和手机宽度。预览使用固定接口快照及明确标识的测试样例成交，不能作为服务器发布或自然成交验收。

发布验收：镜像 `autocrypto-bot:analysis-20261013t123600z`；发布前账户、持仓、现货及合约交易记录均保留。备份 23281 条权益采样逐条核对，零缺失或改动；策略、配置、密钥和锁文件哈希不变。独立离线镜像 97 项测试通过。公网 Chrome 已验证收益摘要、逐笔展开、止损原因、旧数据缺失提示及刷新保留展开状态。机器人持续 paper，无暂停和错误；尚未观察本次发布后的新自然成交快照。发布备份及验证文件在 `/root/work/autocrypto-releases/20261013T123600Z-analysis`。

### 4小时＋1%买回缓冲影子并跑（2026-10-14，已发布服务器独立服务）

用户确认实施后新增独立 `scripts/run_shadow.py`；候选仅限制卖出后的现货加仓，必须高于4小时EMA200的1.01倍。原减仓、合约、加仓步长保持现状，不添加确认期或分批。基线与候选从同一paper账户和行情起步，分别记账，按自然收盘对比启动后收益、最大收盘回撤、调仓次数与现货费用，并记录24小时内买回、高于卖价买回和上涨时少持BTC的毛收益差。

服务器独立容器以60秒间隔只读源行情/状态，数据库与持续报告在 `jp:/root/work/autocrypto-shadow/data`。实验起点保留2026-10-13 21:43（北京时间）、权益488.25333463 USDT，已累计5根自然收盘；基线与线上账户核对通过。原bot、dashboard、策略和配置未改、未重启。原本机目录 `output/shadow-buffer-1pct` 为迁移前数据保留；本次服务器回读快照在 `output/shadow-server-20261014`。尚无策略分歧。详细口径、续跑和停止方式见 `docs/shadow-buffer-runbook.md`。

验证：服务器离线镜像105项测试通过，包含阻断/恢复、减仓不变、重启和去重、事务回滚、无生产文件写入及禁用订单操作。

2026-10-14发布更新：独立容器`autocrypto-shadow`已在jp持续并跑，无密钥和订单权限，原生产容器及账户未变。迁移原实验而非重新起算，原样本无缺失，已累计5根自然收盘并通过基线核对。服务器报告每分钟更新于`/root/work/autocrypto-shadow/data`。105项离线镜像测试通过，尚无新的调仓分歧。详见影子runbook。
