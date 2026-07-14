#!/usr/bin/env python3
"""
多币种轮动回测

将资金等分到 N 个币种，每个币种独立运行策略，合并收益曲线。

用法:
    uv run python scripts/rotation.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from backtesting import lib
from loguru import logger

from src.data.downloader import OKXDataDownloader
from src.strategies.adaptive_hybrid import AdaptiveHybridStrategy


# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════

COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]

TIMEFRAME = "4h"
SINCE = "2023-07-01"
UNTIL = "2025-06-30"
TRAIN_RATIO = 0.7
TOTAL_CASH = 20000
COMMISSION = 0.0005


def main():
    print("╔════════════════════════════════════════════════╗")
    print("║     多币种轮动 — 自适应混合 × 8 币            ║")
    print("╚════════════════════════════════════════════════╝")

    d = OKXDataDownloader(data_dir="data/raw", market_type="swap")
    since = datetime.strptime(SINCE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until = datetime.strptime(UNTIL, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # ── Step 1: 下载数据 ──
    print(f"\n📡 下载 {len(COINS)} 个币种 {TIMEFRAME} 数据...")
    data = {}
    for sym in COINS:
        df = d.fetch_ohlcv(sym, TIMEFRAME, since, until)
        data[sym] = df
        print(f"  {sym}: {len(df)} 根K线")

    # ── Step 2: 逐币种回测 ──
    per_coin_cash = TOTAL_CASH / len(COINS)
    cash_per_coin = per_coin_cash
    months = (until - since).days / 30.44

    print(f"\n📊 逐币种回测 (每币种 {cash_per_coin:.0f} USDT, 训练{TRAIN_RATIO:.0%}/测试{1-TRAIN_RATIO:.0%})")
    print(f"{'币种':<10} {'测试收益':>8} {'回撤':>7} {'胜率':>6} {'交易':>5} {'月均':>8} {'币种盈亏':>10}")
    print("-" * 65)

    results = []
    equity_curves = {}
    test_trades_all = []

    for sym in COINS:
        df = data[sym]
        split = int(len(df) * TRAIN_RATIO)
        test_df = df.iloc[split:]

        bt = lib.FractionalBacktest(
            data=test_df.set_index("timestamp"),
            strategy=AdaptiveHybridStrategy,
            cash=cash_per_coin,
            commission=COMMISSION,
        )
        stats = bt.run()

        trades = stats["_trades"]
        equity = stats["_equity_curve"]["Equity"]

        # 月度收益
        test_months = (len(test_df) * 4) / (30.44 * 24)  # 4H → 小时 → 天 → 月
        monthly = (1 + stats["Return [%]"] / 100) ** (1 / test_months) - 1
        pnl_usdt = stats["Return [%]"] / 100 * cash_per_coin

        results.append({
            "symbol": sym,
            "return": stats["Return [%]"],
            "dd": stats["Max. Drawdown [%]"],
            "win_rate": stats.get("Win Rate [%]", 0),
            "trades": stats["# Trades"],
            "monthly": monthly,
            "pnl": pnl_usdt,
        })
        equity_curves[sym] = equity
        test_trades_all.extend(trades.to_dict("records") if len(trades) > 0 else [])

        print(f"{sym:<10} {stats['Return [%]']:>7.2f}% {stats['Max. Drawdown [%]']:>6.2f}% "
              f"{stats.get('Win Rate [%]', 0):>5.0f}% {stats['# Trades']:>5} "
              f"{monthly*100:>7.2f}% {pnl_usdt:>9.0f}")

    # ── Step 3: 聚合 ──
    print(f"\n📋 聚合结果")

    total_return_pct = sum(r["return"] for r in results) / len(results)
    avg_dd = np.mean([r["dd"] for r in results])
    avg_win = np.mean([r["win_rate"] for r in results])
    total_trades = sum(r["trades"] for r in results)
    total_pnl = sum(r["pnl"] for r in results)
    months_test = (len(test_df) * 4) / (30.44 * 24)
    monthly_agg = (1 + total_return_pct / 100) ** (1 / months_test) - 1

    print(f"  等权组合收益: {total_return_pct:.2f}%")
    print(f"  总盈利: {total_pnl:.0f} USDT (本金 {TOTAL_CASH})")
    print(f"  平均回撤: {avg_dd:.2f}%")
    print(f"  平均胜率: {avg_win:.0f}%")
    print(f"  总交易: {total_trades} 笔")
    print(f"  月均收益: {monthly_agg*100:.2f}%")
    print(f"  月均交易: {total_trades/months_test:.0f} 笔")

    # ── 盈亏币种统计 ──
    pos = sum(1 for r in results if r["return"] > 0)
    neg = sum(1 for r in results if r["return"] <= 0)
    print(f"  盈利币种: {pos}/{len(COINS)}, 亏损币种: {neg}/{len(COINS)}")

    # ── 汇总表 ──
    print(f"\n  总月交易: {total_trades/months_test:.0f} 笔/月 × {avg_win:.0f}% 胜率 = 月{monthly_agg*100:.2f}%")


if __name__ == "__main__":
    main()
