# OKX 实盘接入技术文档

## 目标

将当前 BTC 组合策略从历史回测和 paper trading，推进到可接入 OKX 的实盘交易系统。

当前策略形态：

- BTC 现货核心仓：按 regime 动态调整目标权重。
- BTC-USDT 永续合约卫星仓：使用 Triple EMA 4h 信号，2 倍杠杆。
- 组合风控：限制净 BTC 敞口、总名义敞口、空头敞口，并按 regime 限制多空方向。

最终实盘目标不是直接全自动重仓，而是分阶段达到：

1. `paper`：实时行情 + 本地模拟账户。
2. `readonly`：实时行情 + OKX 真实账户只读快照。
3. `testnet_live`：OKX 模拟盘真实下单。
4. `live_small`：OKX 实盘小资金灰度。
5. `live_scaled`：满足稳定性指标后逐步加资金。

## 当前状态

已完成：

- 组合规则已抽成共享模块：`src/live/portfolio.py`。
- OKX readonly 账户快照解析已实现：`src/live/okx_account.py`。
- 回测和线上机器人共用 regime、核心仓目标、调仓计划、合约仓位上限和组合曝光计算。
- 新增真实组合回测：`scripts/backtest_portfolio.py`。
- `ProductionBot` 已支持组合 paper 模式：
  - 输出组合计划。
  - paper 核心仓调仓。
  - paper 合约仓状态保存。
  - paper 状态跨重启恢复。
  - 输出账户快照。
- `ProductionBot` 已支持非 dry-run 的 OKX readonly 账户快照。
- 启动/运行前已支持基础账户对账，发现仓位或挂单冲突会进入 paused，禁止新开仓。
- 订单执行已抽象为 `src/live/order_executor.py`，合约入场、止盈、止损通过统一执行器提交。
- 订单成交轮询和超时撤单已完成基础实现；部分成交会被识别并随失败结果返回。
- TP/SL 保护单提交后会通过查单确认，未确认会返回失败并暂停。
- 合约开仓成功后会把真实成交价、成交数量、入场单 ID、TP/SL 保护单 ID 写入本地状态文件。
- 非 dry-run 启动时会读取本地状态，并用 OKX 仓位和保护挂单做对账；保护单缺失会暂停。
- live 平仓已走 OKX `reduceOnly` 订单，等待成交后撤销剩余保护单，并确认交易所合约仓位归零。
- 入场和平仓订单会解析交易所返回的手续费，并写入本地持仓状态和交易记录。
- 连续 API 错误会计数，达到 `live.max_api_errors` 后进入 `api_errors_exceeded` 暂停状态。
- 已新增实盘前检查脚本 `scripts/okx_preflight.py`，用于检查配置、API key、账户权益、挂单和合约仓位冲突。
- `scripts/run_production.py` 在非 dry-run 启动前默认执行 preflight；未通过会拒绝启动。
- live 状态恢复已增加 schema 校验，缺少入场单 ID、保护单 ID、仓位大小等关键字段会拒绝恢复并暂停。
- 已新增本地 JSONL 告警 `outputs/alerts.jsonl`，preflight 失败和 API 熔断会写入告警事件。
- 现货核心仓会生成 `spot_core_rebalance` 订单计划，但默认只记录不执行。
- 已增加真实交易硬开关：
  - `portfolio.allow_spot_trading: false` 默认禁止现货真实下单。
  - `portfolio.allow_swap_trading: false` 默认禁止合约真实下单。
- 当前 `portfolio.mode: paper` 时，如果 `dry_run=false`，机器人会拒绝真实下单。

尚未完成：

- 真实现货核心仓下单。
- 资金费和更完整成交明细落账。
- 更完整的挂单/止盈止损单生命周期管理。
- 更完整的跨重启订单生命周期恢复。
- 外部 webhook/短信/邮件告警。

因此当前状态只适合长期 `--dry-run` 验证，不适合直接真实资金自动交易。

## 模式定义

### backtest

用途：历史验证。

特点：

- 不连接 OKX。
- 使用本地历史 K 线。
- 模拟现货核心仓、合约卫星仓、手续费、资金费。

### paper

用途：实时行情纸上验证。

特点：

- 可读取 OKX 行情。
- 不读取真实账户作为资金来源。
- 不发送真实订单。
- 使用本地 `outputs/production_state.json` 保存 paper 状态。

### readonly

用途：真实账户只读对账。

特点：

- 读取 OKX 账户余额、现货、合约仓、挂单。
- 不发送任何下单、撤单、改单请求。
- 输出真实账户组合快照。
- 输出“应该调仓多少”，但只记录，不执行。

### testnet_live

用途：OKX 模拟盘真实订单生命周期验证。

特点：

- 使用 OKX testnet。
- 允许发送现货和合约订单。
- 必须完成成交确认、撤单、止盈止损确认。

### live_small

用途：真实小资金灰度。

特点：

- 使用 OKX 实盘。
- 单笔下单、总权益和组合敞口必须有硬上限。
- 默认只允许极小资金，例如 50-100 USDT。

## OKX 账户快照设计

新增模块建议：`src/live/okx_account.py`。

核心数据结构：

```python
@dataclass(frozen=True)
class AssetBalance:
    currency: str
    free: float
    used: float
    total: float

@dataclass(frozen=True)
class SwapPosition:
    symbol: str
    side: str
    contracts: float
    notional: float
    entry_price: float
    unrealized_pnl: float
    margin: float

@dataclass(frozen=True)
class AccountSnapshot:
    usdt: AssetBalance
    btc: AssetBalance
    swap_position: SwapPosition | None
    open_orders: list[dict]
    equity_usdt: float
    core_btc: float
    core_notional: float
    swing_notional: float
    net_exposure: float
    gross_exposure: float
```

快照计算规则：

- `equity_usdt = USDT total + BTC total * BTC/USDT price + swap unrealized_pnl`
- `core_btc = BTC spot total`
- `core_notional = core_btc * price`
- `swing_notional = abs(swap_position.notional)`
- long 合约记为正敞口，short 合约记为负敞口
- `net_exposure = (core_notional + signed_swing_notional) / equity_usdt`
- `gross_exposure = (abs(core_notional) + abs(swing_notional)) / equity_usdt`

## 启动对账流程

机器人启动时必须按顺序执行：

1. 加载历史 K 线并计算指标。
2. 读取 OKX 账户快照。
3. 读取本地状态文件。
4. 比较本地状态和 OKX 状态。
5. 发现不一致时进入 `paused`，不允许新开仓。
6. 没有冲突时生成组合计划。
7. 根据模式决定是否只记录、模拟执行或真实执行。

必须暂停的情况：

- OKX 有合约仓，但本地状态没有记录。
- 本地状态有合约仓，但 OKX 无仓位。
- OKX 有未完成挂单。
- 止盈止损单缺失。
- BTC 现货数量异常偏离本地记录。
- 账户权益小于配置下限。
- 行情时间戳落后。
- 连续 API 错误超过阈值。

## 现货核心仓执行设计

新增模块建议：`src/live/order_executor.py`。

核心行为：

- 根据组合计划生成现货买卖订单。
- 使用 `BTC/USDT` spot 市场。
- 订单金额来自 `target_notional - current_notional`。
- 下单前检查：
  - 单笔最大金额。
  - 最小下单金额。
  - 可用 USDT 或 BTC 是否足够。
  - 精度是否合法。
  - 当前是否存在未完成订单。
- 下单后必须确认成交。
- 部分成交必须记录实际成交数量和剩余数量。
- 超时未成交必须撤单。

建议第一阶段只支持限价 IOC 或小额市价单，避免挂单长期残留。

## 合约卫星仓执行设计

当前 `ProductionBot.execute_signal` 已有基础合约下单，但实盘前需要补齐：

- 下单后轮询订单状态。
- 确认成交均价。
- 写入真实 entry price。
- 止盈止损单下单后必须确认成功。
- 如果止盈止损单失败，应立即平仓或暂停。
- 平仓后必须确认合约仓位归零。
- 真实手续费、资金费从交易所读取或在成交记录中保存。

## 订单生命周期

统一订单状态：

- `planned`
- `submitted`
- `partially_filled`
- `filled`
- `cancel_requested`
- `cancelled`
- `failed`
- `rejected`
- `reconciled`

任何订单不能只因为 `create_order` 返回成功就视为完成，必须通过交易所订单状态或成交明细确认。

## 风控熔断

实盘必须增加硬保护：

- `max_live_equity_usdt`：本阶段允许管理的最大资金。
- `max_order_notional_usdt`：单笔最大下单金额。
- `max_daily_loss_pct`：日亏损熔断。
- `max_net_exposure`：组合净 BTC 敞口上限。
- `max_gross_exposure`：组合总名义敞口上限。
- `max_short_exposure`：净空头上限。
- `max_api_errors`：连续 API 错误上限。
- `stale_candle_seconds`：行情过期暂停。
- `require_stop_orders`：合约仓必须有止盈止损保护。

触发熔断后：

- 停止新开仓。
- 停止核心仓加仓。
- 允许减仓和平仓。
- 写入状态文件。
- 输出告警。

## 配置建议

`config/production.yaml` 中建议扩展：

```yaml
portfolio:
  enabled: true
  mode: "readonly"  # paper | readonly | testnet_live | live_small | live_scaled
  max_order_notional_usdt: 20
  min_order_notional_usdt: 5
  allow_spot_trading: false
  allow_swap_trading: false

live:
  max_live_equity_usdt: 100
  max_api_errors: 3
  stale_candle_seconds: 900
  require_stop_orders: true
```

默认必须保持 `portfolio.allow_spot_trading: false` 和 `portfolio.allow_swap_trading: false`，直到完成对应阶段验收后再手动切换。

## 实施顺序

### P1：readonly 账户快照

状态：已完成基础实现。

目标：

- 接入 OKX 账户读取。
- 输出真实账户组合快照。
- 不允许下单。

交付：

- `src/live/okx_account.py`
- `AccountSnapshot`
- `ProductionBot._account_snapshot()` 在非 dry-run 下使用真实账户快照。
- 单测覆盖快照解析和曝光计算。

### P2：启动对账和 paused 状态

状态：已完成基础实现。

目标：

- 启动时检查本地状态和 OKX 状态一致性。
- 不一致时暂停新交易。

交付：

- `ReconcileResult`
- `ProductionBot._paused_reason`
- 单测覆盖本地有仓/OKX 无仓、OKX 有仓/本地无仓、存在挂单等场景。

### P3：订单执行抽象

状态：已完成基础实现；已支持 entry 成交轮询、超时撤单、部分成交状态返回和保护单确认。

目标：

- 把现货/合约下单、查单、撤单、成交确认统一封装。

交付：

- `src/live/order_executor.py`
- `OrderPlan`
- `OrderResult`
- `wait_for_fill`
- `cancel_order`

### P4：testnet_live 小额模拟盘

状态：已完成合约开仓、入场成交等待、TP/SL 保护单确认、live 状态保存、状态 schema 校验、启动对账、平仓确认、手续费记录、API 连续错误熔断、本地告警和 preflight 检查的基础闭环。

前置状态：现货核心仓订单计划已完成；真实现货执行仍未开放；合约真实下单必须显式打开 `portfolio.allow_swap_trading`。

目标：

- 在 OKX 模拟盘执行真实订单生命周期。

要求：

- 单笔金额极小。
- 必须确认成交。
- 必须确认止盈止损。
- 异常自动暂停。

剩余工作：

- 将资金费和更完整成交明细写入交易记录。
- 更细粒度地区分“保护单已触发导致仓位归零”和“机器人主动趋势反转平仓”。
- 接入外部 webhook/短信/邮件告警。

运行 preflight：

```bash
.venv/bin/python scripts/okx_preflight.py --config config/production.yaml
```

模拟盘专项配置和步骤见 `docs/okx-testnet-runbook.md`，示例配置为 `config/testnet_live.yaml`。

如果本地状态已经记录了可恢复的 live 仓位，可显式允许交易所存在仓位：

```bash
.venv/bin/python scripts/okx_preflight.py --config config/production.yaml --allow-existing-state
```

启动非 dry-run 时，入口脚本会自动执行 preflight：

```bash
.venv/bin/python scripts/run_production.py --config config/production.yaml
```

只有明确确认风险时才允许跳过：

```bash
.venv/bin/python scripts/run_production.py --config config/production.yaml --skip-preflight
```

### P5：live_small 实盘灰度

目标：

- 真实账户小额运行。

要求：

- 初始最大管理资金不超过配置上限。
- 每日人工复盘订单和账户快照。
- 连续 30 天无状态错乱、无裸仓、无异常挂单后再考虑扩大资金。

## 验收标准

进入 `readonly` 的验收：

- 能读取 OKX 账户余额、BTC 现货、BTC-USDT-SWAP 合约仓位。
- 能生成真实组合快照。
- 能生成目标核心仓和调仓建议。
- 不会发送任何订单请求。
- API 错误时进入暂停状态。

进入 `testnet_live` 的验收：

- 能完成现货买卖成交确认。
- 能完成合约开仓、止盈止损挂单确认、平仓确认。
- 本地状态与 OKX 仓位一致。
- 断点重启后能恢复。

进入 `live_small` 的验收：

- `readonly` 至少稳定运行 1-2 周。
- `testnet_live` 至少稳定运行 1-2 周。
- 所有订单生命周期有日志。
- 所有异常有暂停保护。
- 最大资金和单笔金额有硬上限。

## 当前下一步

下一步应实现 P1：`readonly` 账户快照。

该步骤完成后，系统仍不自动下单，但可以用真实 OKX 账户验证：

- 当前真实核心仓是多少。
- 当前真实合约仓是多少。
- 组合净敞口是否符合预期。
- 策略当前建议加仓、减仓还是保持。
