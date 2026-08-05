# OKX 模拟盘运行手册

本文档用于把当前策略推进到 `testnet_live` 阶段。该阶段只用于 OKX 模拟盘验证订单生命周期，不用于真实资金。

## 目标

- 验证 OKX Demo API 能读取账户、挂单、合约仓位。
- 验证合约开仓、入场成交确认、TP/SL 保护单确认、平仓、撤保护单、状态保存。
- 验证异常时能暂停，并写入 `outputs/testnet_live_alerts.jsonl`。

## 前置条件

1. 使用 OKX Demo Trading API Key，不使用真实账户 API Key。
2. 本地 `.env` 已配置：

```bash
OKX_API_KEY=...
OKX_SECRET=...
OKX_PASSPHRASE=...
```

3. OKX 模拟盘账户没有未知 BTC 现货、BTC-USDT-SWAP 仓位或未完成挂单。
4. 初次运行前保留 `config/testnet_live.yaml` 中的硬开关：

```yaml
portfolio:
  allow_spot_trading: false
  allow_swap_trading: false
```

## 第一步：只做 preflight

```bash
.venv/bin/python scripts/okx_preflight.py --config config/testnet_live.yaml
```

必须看到：

- `ok: false` 且失败原因只有 `live_mode_switches`：说明静态配置基本正确，但真实交易硬开关仍关闭。
- 如果失败原因包含 API key 或模式不匹配，先处理配置，不要启动机器人。

## 第二步：打开合约模拟盘开关

只在确认上一步没有账户冲突后，把 `config/testnet_live.yaml` 改成：

```yaml
portfolio:
  allow_spot_trading: false
  allow_swap_trading: true
```

现货核心仓真实执行仍未开放，`allow_spot_trading` 必须保持 `false`。

## 第三步：再次 preflight

```bash
.venv/bin/python scripts/okx_preflight.py --config config/testnet_live.yaml
```

这一步会读取 OKX 模拟盘账户。必须看到 `ok: true` 才能启动。
如果失败原因包含账户快照、未完成挂单或已有合约仓位，先处理 OKX 模拟盘账户，不要启动机器人。

## 第四步：启动 testnet_live

```bash
.venv/bin/python scripts/run_production.py --config config/testnet_live.yaml
```

入口脚本会自动执行 preflight。preflight 不通过时会拒绝启动。

## 运行期间观察

重点检查这些文件：

- `outputs/testnet_live.log`
- `outputs/testnet_live_state.json`
- `outputs/testnet_live_trades.json`
- `outputs/testnet_live_alerts.jsonl`
- `outputs/production_account_snapshot.json`

每次产生信号后必须人工确认：

- 入场单是否成交。
- TP/SL 保护单是否存在。
- 本地 state 中是否记录了 `entry_order_id` 和 `protective_orders`。
- 本地 state 中的 live 仓位必须包含 `side`、`entry`、`notional`、`size`、`entry_order_id`、`protective_orders.take_profit`、`protective_orders.stop_loss`。
- 平仓后 OKX 模拟盘仓位是否归零。
- 平仓后剩余保护单是否被撤销。

## 暂停和恢复

出现以下暂停原因时，不要直接重启：

- `exchange_position_without_local_state`
- `local_position_without_exchange_position`
- `protective_orders_missing`
- `close_position_not_flat`
- `api_errors_exceeded`

处理顺序：

1. 在 OKX 模拟盘检查仓位和挂单。
2. 对照 `outputs/testnet_live_state.json`。
3. 手动处理裸仓或异常挂单。
4. 再运行 preflight。
5. preflight 通过后再启动。

如果本地状态已确认可恢复，且交易所存在对应仓位，可运行：

```bash
.venv/bin/python scripts/okx_preflight.py --config config/testnet_live.yaml --allow-existing-state
```

`--allow-existing-state` 仍会校验本地 state schema；缺少保护单 ID 或入场单 ID 时不会放行。

## 禁止事项

- 不要在 `testnet_live` 阶段打开 `allow_spot_trading`。
- 不要用 `--skip-preflight` 启动，除非已经人工确认所有账户状态。
- 不要把真实 OKX API Key 放进 testnet 配置。
- 不要在未复盘订单日志前进入 `live_small`。
