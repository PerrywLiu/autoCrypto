# autoCrypto - 加密货币趋势交易回测系统

基于 `backtesting.py` 的 OKX 趋势交易策略回测框架，支持多时间框架策略开发、回测和绩效分析。

## 快速开始

### 安装

```bash
uv sync
```

### 下载数据

```bash
# 下载 BTC/USDT 1H 数据
uv run python scripts/download.py --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2025-06-01

# 批量下载多时间框架
uv run python scripts/download.py --symbols BTC/USDT ETH/USDT --tfs 1h 4h --start 2024-01-01
```

### 运行回测

```bash
# 单策略回测
uv run python scripts/backtest.py --strategy ema_rsi_atr --symbol BTC/USDT

# 多策略对比
uv run python scripts/backtest.py --compare --symbol BTC/USDT
```

## 可用策略

| 策略 | 说明 |
|------|------|
| `ema_rsi_atr` | EMA12/26 双时间框架交叉 + RSI14 过滤 + ATR14 动态止损 |
| `donchian_breakout` | 20期 Donchian 通道突破 + 4H EMA 趋势过滤器 |
| `multi_timeframe_trend` | EMA12/26/50 三重排列 + 趋势强度 + RSI 区间过滤 |

## 项目结构

```
autoCrypto/
├── config/                  # 策略参数配置 (YAML)
│   ├── settings.yaml
│   └── strategies/
├── src/
│   ├── data/               # 数据层 (ccxt 下载 + 预处理)
│   ├── strategies/          # 策略层 (基类 + 3个趋势策略)
│   ├── backtest/           # 回测层 (引擎 + 绩效指标)
│   └── visualization/      # 可视化 (Plotly 图表)
├── scripts/                # CLI 工具
│   ├── download.py
│   └── backtest.py
├── data/                   # 历史数据 (CSV, .gitignored)
└── outputs/                # 回测报告 (.gitignored)
```

## 回测验证结果

EMA-RSI-ATR 策略 on BTC/USDT 1H (2024-12-01 ~ 2025-02-28):
- 收益率: -58.37% (买入持有: -14.00%)
- 最大回撤: -62.81%
- 胜率: 39.62%
- 交易次数: 419

> ⚠️ 纯 1H 模式交易过于频繁。建议使用 4H 趋势过滤降低交易频率。
