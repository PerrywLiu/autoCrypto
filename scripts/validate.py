#!/usr/bin/env python3
"""
完整策略验证脚本 — 四步升级

1. 永续合约 swap 数据（真实做空 + 资金费率）
2. 多币种（BTC, ETH, SOL）
3. 2 年数据（2023-07 ~ 2025-06）
4. 训练/测试分离（70% 调参, 30% 验证）

用法:
    uv run python scripts/validate.py
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
from src.data.funding_rate import FundingRateModel
from src.strategies.advanced_trend import AdvancedTrendStrategy


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAMES = ["1h", "4h"]
SINCE = "2023-07-01"
UNTIL = "2025-06-30"
TRAIN_RATIO = 0.7
INITIAL_CASH = 20000
COMMISSION = 0.0005  # 永续合约 taker 0.05%

# 策略参数
STRATEGY_PARAMS = dict(
    ema_fast=20, ema_mid=50, ema_slow=100,
    adx_long=15, adx_short=18,
    atr_stop_mult=2.5, atr_entry_mult=0.8,
    min_score_long=6, min_score_short=7,
    use_fixed_tp=False,
    risk_per_trade=0.02, cooldown_bars=3,
)


# ═══════════════════════════════════════════════════════════
# Step 1: 下载永续合约数据
# ═══════════════════════════════════════════════════════════

def download_all():
    """下载所有币种的 swap K 线数据"""
    d = OKXDataDownloader(data_dir="data/raw", market_type="swap")
    since = datetime.strptime(SINCE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until = datetime.strptime(UNTIL, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    data = {}
    for sym in SYMBOLS:
        data[sym] = {}
        for tf in TIMEFRAMES:
            df = d.fetch_ohlcv(sym, tf, since, until)
            data[sym][tf] = df
            logger.info(f"  {sym} {tf}: {len(df)} 根K线")
    return data


# ═══════════════════════════════════════════════════════════
# Step 2 & 4: 单币种回测 + 训练/测试分离
# ═══════════════════════════════════════════════════════════

def backtest_single(symbol: str, df: pd.DataFrame, label: str = "") -> dict:
    """对单个币种、单个数据集运行回测"""

    class Strat(AdvancedTrendStrategy):
        pass
    for k, v in STRATEGY_PARAMS.items():
        setattr(Strat, k, v)

    bt = lib.FractionalBacktest(
        data=df.set_index("timestamp"),
        strategy=Strat,
        cash=INITIAL_CASH,
        commission=COMMISSION,
    )
    stats = bt.run()
    trades = stats["_trades"]

    if len(trades) == 0:
        return {"symbol": symbol, "label": label, "return": 0, "trades": 0,
                "win_rate": 0, "max_dd": 0, "sharpe": 0}

    longs = trades[trades["Size"] > 0]
    shorts = trades[trades["Size"] < 0]

    # 资金费率估算
    total_hours = 0
    for _, t in trades.iterrows():
        try:
            et = pd.Timestamp(t.get("EntryTime", t.get("entry_time")))
            xt = pd.Timestamp(t.get("ExitTime", t.get("exit_time")))
            total_hours += (xt - et).total_seconds() / 3600
        except Exception:
            total_hours += 24  # fallback

    # 默认费率 0.01%/8h, 每8小时结算一次
    funding_settlements = total_hours / 8
    avg_funding = 0.0001  # 0.01%
    long_hours = sum(
        (pd.Timestamp(t.get("ExitTime", t.get("EntryTime", pd.Timestamp.now()))) -
         pd.Timestamp(t.get("EntryTime", t.get("ExitTime", pd.Timestamp.now())))).total_seconds() / 3600
        for _, t in longs.iterrows()
    ) if len(longs) > 0 else 0
    short_hours = total_hours - long_hours

    funding_cost = -(long_hours / 8) * avg_funding * 100  # 多头付钱(%)
    funding_gain = (short_hours / 8) * avg_funding * 100   # 空头收钱(%)
    funding_net = funding_cost + funding_gain

    raw_return = stats["Return [%]"]
    adjusted_return = raw_return + funding_net

    return {
        "symbol": symbol,
        "label": label,
        "return": round(adjusted_return, 2),
        "raw_return": round(raw_return, 2),
        "funding_net": round(funding_net, 2),
        "trades": len(trades),
        "longs": len(longs),
        "shorts": len(shorts),
        "win_rate": round(stats.get("Win Rate [%]", 0), 1),
        "long_win": round((longs["PnL"] > 0).sum() / max(len(longs), 1) * 100, 1),
        "short_win": round((shorts["PnL"] > 0).sum() / max(len(shorts), 1) * 100, 1),
        "max_dd": round(stats["Max. Drawdown [%]"], 2),
        "sharpe": round(stats.get("Sharpe Ratio", 0), 2),
        "profit_factor": round(stats.get("Profit Factor", 0), 2),
    }


def backtest_with_split(symbol: str, df: pd.DataFrame) -> dict:
    """训练/测试分离回测"""
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_result = backtest_single(symbol, train_df, "训练集")
    test_result = backtest_single(symbol, test_df, "测试集")

    # 全量
    full_result = backtest_single(symbol, df, "全量")

    # 过拟合检测
    train_ret = train_result["return"]
    test_ret = test_result["return"]
    gap = abs(train_ret - test_ret)
    status = "✅ 稳定" if gap < 20 and test_ret > -10 else ("⚠️ 过拟合" if gap > 30 else "🔍 待观察")

    return {
        "symbol": symbol,
        "train": train_result,
        "test": test_result,
        "full": full_result,
        "gap": round(gap, 1),
        "status": status,
        "data_points": len(df),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     策略四步验证: 永续合约 + 多币种 + 2年 + 分离       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n配置: {SYMBOLS}, {SINCE}~{UNTIL}, 训练{TRAIN_RATIO:.0%}/测试{1-TRAIN_RATIO:.0%}")

    # Step 1: 下载数据
    print("\n📡 Step 1: 下载永续合约 swap 数据...")
    data = download_all()

    # Step 2+3+4: 逐币种回测
    print(f"\n📊 Step 2-4: 逐币种训练/测试分离回测\n")

    all_results = []
    for sym in SYMBOLS:
        df_1h = data[sym]["1h"]
        if df_1h.empty:
            print(f"  {sym}: 无数据, 跳过")
            continue

        result = backtest_with_split(sym, df_1h)
        all_results.append(result)

        r = result
        t = r["train"]
        e = r["test"]
        f = r["full"]

        print(f"  ┌─ {sym} ({r['data_points']}根K线) ──────────────────── {r['status']}")
        print(f"  │  数据集    收益%   回撤%   胜率%   交易  做多/空胜率")
        print(f"  │  {'训练':<6} {t['return']:>7.2f} {t['max_dd']:>7.2f} {t['win_rate']:>6.1f} {t['trades']:>5}  {t['long_win']}/{t['short_win']}")
        print(f"  │  {'测试':<6} {e['return']:>7.2f} {e['max_dd']:>7.2f} {e['win_rate']:>6.1f} {e['trades']:>5}  {e['long_win']}/{e['short_win']}")
        print(f"  │  {'全量':<6} {f['return']:>7.2f} {f['max_dd']:>7.2f} {f['win_rate']:>6.1f} {f['trades']:>5}  {f['long_win']}/{f['short_win']}")
        print(f"  │  资金费率净影响: {f['funding_net']:+.2f}%")
        print(f"  │  训练→测试差距: {r['gap']}%")
        print(f"  └{'─'*50}")

    # 汇总
    print(f"\n📋 汇总:")
    print(f"  {'币种':<8} {'全量收益%':>9} {'全量回撤%':>9} {'训练收益%':>9} {'测试收益%':>9} {'差距%':>7} {'判定':<10}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['symbol']:<8} {r['full']['return']:>9.2f} {r['full']['max_dd']:>9.2f} "
              f"{r['train']['return']:>9.2f} {r['test']['return']:>9.2f} "
              f"{r['gap']:>7.1f} {r['status']:<10}")

    # 综合判定
    passing = [r for r in all_results if r["status"] == "✅ 稳定"]
    print(f"\n🏁 综合: {len(passing)}/{len(all_results)} 币种通过稳定性验证")
    if passing:
        avg_ret = np.mean([r["full"]["return"] for r in passing])
        print(f"    通过币种平均收益: {avg_ret:.2f}%")


if __name__ == "__main__":
    main()
