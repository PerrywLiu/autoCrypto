# TripleEma 4h 2x 杠杆 — 部署指南

> 基于 2023.07~2026.07 三年回测验证，全时段收益 +135.5%，月均 +2.4%，夏普 1.80

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 操作系统 | Linux / macOS | Ubuntu 22.04+ |
| Python | 3.12+ | 3.12 |
| 内存 | 256MB | 512MB |
| 磁盘 | 100MB | 500MB |
| 网络 | 稳定互联网 | 延迟 <100ms |

### 1.2 安装

```bash
# 克隆项目
git clone <repo-url> autoCrypto && cd autoCrypto

# 切换到生产分支
git checkout btc-daily-3x

# 安装 Python 依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
uv sync
```

---

## 2. OKX API 配置

### 2.1 创建 API Key

1. 登录 [OKX](https://www.okx.com) → 头像 → API
2. 创建 API Key，权限选：
   - ✅ **读取** (Read)
   - ✅ **交易** (Trade)
   - ❌ **提现** (Withdraw) — 永远不开
3. IP 白名单：绑定服务器 IP（强烈建议）
4. 模拟盘 API：[OKX Demo Trading](https://www.okx.com/zh-hant/trade-swap-demo/btc-usdt-swap) 同样操作

### 2.2 配置密钥

```bash
cat > .env << 'EOF'
OKX_API_KEY=你的API_Key
OKX_SECRET=你的Secret_Key
OKX_PASSPHRASE=你的Passphrase
EOF

# 限制权限
chmod 600 .env
```

---

## 3. 策略参数配置

文件：`config/production.yaml`（已是最优参数）

```yaml
# ═══════════════════════════════════
# 交易标的
# ═══════════════════════════════════
symbol: "BTC/USDT"          # 交易对
timeframe: "4h"              # K线周期
market_type: "swap"          # 永续合约

# ═══════════════════════════════════
# 策略参数（回测最优）
# ═══════════════════════════════════
strategy:
  name: "triple_ema"
  ema_fast: 10               # EMA10 快线
  ema_mid: 20                # EMA20 中线（入场参考）
  ema_slow: 50               # EMA50 慢线
  ema_trend: 200             # EMA200 长期趋势（牛熊分界）
  adx_period: 14             # ADX 计算周期
  adx_threshold: 20          # 最低趋势强度（<20不交易）
  atr_period: 14             # ATR 计算周期
  sl_atr_mult: 1.5           # 止损 = 入场价 ± 1.5×ATR
  tp_atr_mult: 3.0           # 止盈 = 入场价 ± 3.0×ATR（1:2盈亏比）
  cooldown_base: 6            # 基础冷却期（每笔交易后最少等6根K线=24h）
  cooldown_extra_per_loss: 3  # 每次亏损额外冷却+3根K线

# ═══════════════════════════════════
# 风控参数
# ═══════════════════════════════════
risk:
  initial_capital: 1000       # 初始保证金 USDT（建议1000起）
  leverage: 2                 # 杠杆倍数（1x现货 / 2x推荐 / 3x激进）
  risk_per_trade: 0.02        # 单笔风险 2%（保证金占比）
  max_daily_loss_pct: 0.05    # 日熔断 5%
  max_consecutive_losses: 5   # 连续亏损 5 次暂停交易

# ═══════════════════════════════════
# 运行模式
# ═══════════════════════════════════
mode:
  dry_run: true               # 纸上交易（只看不下单，初始值）
  testnet: true               # 模拟盘环境
  log_level: "INFO"           # 日志级别
  heartbeat_minutes: 15       # 巡检间隔（分钟）

# ═══════════════════════════════════
# 输出
# ═══════════════════════════════════
output:
  trade_log: "outputs/production_trades.json"
  log_file: "outputs/production.log"
```

### 3.1 参数调优指南

| 参数 | 当前值 | 调优方向 |
|------|--------|---------|
| `leverage` | 2 | 保守→1, 激进→3（回撤将相应翻倍） |
| `initial_capital` | 1000 | 按实际资金调整，最小500U |
| `risk_per_trade` | 0.02 | 保守→0.01, 激进→0.03 |
| `adx_threshold` | 20 | 提高→25（减少交易提高质量），降低→15（增加机会） |
| `sl_atr_mult` / `tp_atr_mult` | 1.5 / 3.0 | 保持1:2盈亏比，可同步扩大（如2.0/4.0） |

### 3.2 不同杠杆的回测对比

| 杠杆 | 全时段收益 | 月均 | 夏普 | 保证金回撤 | 推荐场景 |
|------|----------|------|------|----------|---------|
| 1x | +39.0% | +0.9% | 1.34 | **-14.7%** | 极度保守 |
| **2x** | **+135.5%** | **+2.4%** | **1.80** | **-27.2%** | 🔥 推荐 |
| 3x | +237.0% | +3.4% | 1.91 | -42.4% | 能承受大回撤 |

---

## 4. 部署步骤

### 第一步：纸上交易（必须，至少 2 周）

```bash
uv run python scripts/run_production.py --dry-run
```

观察指标：
- 信号频率：约 1~2 个/天
- 信号方向是否与日线 EMA200 一致
- 止损止盈价格是否合理

### 第二步：模拟盘（1~4 周）

```bash
# 修改 config/production.yaml:
#   mode.dry_run: false
#   mode.testnet: true

uv run python scripts/run_production.py
```

验证：
- 下单是否成功
- 止盈止损是否正确挂单
- 资金费率结算是否正确
- 交易记录是否保存

### 第三步：实盘小资金（建议 2~4 周）

```bash
# 修改 config/production.yaml:
#   mode.dry_run: false
#   mode.testnet: false
#   risk.initial_capital: 500  ← 小资金起步

uv run python scripts/run_production.py --live
```

### 第四步：正式运行

确认前三步都正常后，逐步增加资金。

---

## 5. 后台运行

### 5.1 systemd（推荐生产环境）

```bash
sudo cat > /etc/systemd/system/autocrypto.service << 'EOF'
[Unit]
Description=autoCrypto TripleEma 4h Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/autoCrypto
Environment="PATH=/home/your-username/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/your-username/.cargo/bin/uv run python scripts/run_production.py
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable autocrypto
sudo systemctl start autocrypto

# 查看状态
sudo systemctl status autocrypto

# 实时日志
sudo journalctl -u autocrypto -f

# 重启
sudo systemctl restart autocrypto

# 停止
sudo systemctl stop autocrypto
```

### 5.2 nohup（简单场景）

```bash
# 启动
nohup uv run python scripts/run_production.py > outputs/bot.log 2>&1 &
echo $! > bot.pid

# 查看日志
tail -f outputs/bot.log

# 停止
kill $(cat bot.pid)
```

---

## 6. 监控与维护

### 6.1 日常检查

```bash
# 机器人是否在运行
ps aux | grep run_production

# 最近日志
tail -30 outputs/production.log

# 最近交易
cat outputs/production_trades.json | python -m json.tool | tail -30

# 最新K线时间（应该每4h更新）
grep "新K线" outputs/production.log | tail -3
```

### 6.2 收益查看

```bash
# 查看回测收益
uv run python scripts/backtest_prod.py --lev 2

# 实盘收益在日志中
grep "平仓" outputs/production.log
```

### 6.3 告警信号（需要人工介入）

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| 连续亏损 >5 次 | 市场剧烈变化 | 暂停，重新回测 |
| 长时间无信号 (>3天) | ADX太低或冷却中 | 正常现象 |
| 日志停止更新 | 进程崩溃 | `systemctl restart` |
| 余额不匹配 | 资金费率扣款 | 查 OKX 账单 |

---

## 7. 风险提示

```
⚠️ 加密货币交易风险极高，请务必注意：

1. 只用亏得起的钱交易
2. 策略回测表现不代表未来收益
3. 2x杠杆下，保证金回撤可达 -27%
4. 极端行情下止损可能滑点
5. 交易所宕机/API故障可能导致无法平仓
6. 建议先用500U小资金跑1个月，稳定后再加仓
7. 永远不要开API提现权限
```

---

## 8. 常见问题

**Q: 为什么信号很少？**
A: 策略需要同时满足 5 个条件（三线排列+近EMA20+ADX>20+牛熊方向+冷却期），信号少但质量高。

**Q: 亏损了怎么办？**
A: 连续亏损 5 次会自动暂停。冷静分析原因：是市场变了还是参数需要调整？

**Q: 如何确认策略在正常工作？**
A: `grep "新K线" outputs/production.log` 每 4 小时应有一条；`grep "平仓"` 看最近交易盈亏。

**Q: 能同时跑多个币种吗？**
A: 当前版本仅支持 BTC。多币种需要修改配置后启动多个实例。

**Q: 服务器重启后会自动启动吗？**
A: 使用 systemd 部署会自动重启。使用 nohup 需要手动重启。
