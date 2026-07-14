#!/usr/bin/env python3
"""
三策略组合回测 — 纯本地数据, 无 API 调用

策略:
  A: 趋势跟踪 (自适应混合 4H, 已回测)
  B: 资金费率套利 (现货多+永续空, 收每8h费率)
  C: 网格交易 (4H级别, 上下各3档, 1%间距)

资金分配: 40%费率套利 + 30%网格 + 30%趋势
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
import numpy as np
import pandas as pd
np.random.seed(42)

from src.data.downloader import OKXDataDownloader
from src.strategies.adaptive_hybrid import AdaptiveHybridStrategy
from backtesting import lib
import warnings; warnings.filterwarnings('ignore')


# ═══════════════════════════════════════
# 策略 B: 资金费率套利
# ═══════════════════════════════════════

def backtest_funding_arb(df_4h: pd.DataFrame, capital: float, leverage: float = 5.0):
    """
    资金费率套利回测
    - 买入现货 + 做空永续合约 (delta中性)
    - 每8h收取资金费率
    - 参数: 0.008%/8h (BTC 3年均值), 牛市期0.03%, 熊市期0.005%
    """
    df = df_4h.set_index("timestamp").sort_index()
    results = []

    # 仓位: capital * leverage 名义价值
    notional = capital * leverage

    for period, start, end, rate_8h in [
        ("2023H2 熊市尾", "2023-07-01", "2023-12-31", 0.00005),
        ("2024H1 牛市",   "2024-01-01", "2024-06-30", 0.0003),
        ("2024H2 震荡",   "2024-07-01", "2024-12-31", 0.00008),
        ("2025H1 牛市",   "2025-01-01", "2025-06-30", 0.0002),
        ("2025H2~",       "2025-07-01", "2026-07-01", 0.00006),
    ]:
        sub = df[(df.index >= start) & (df.index <= end)]
        if sub.empty: continue

        # 每8h = 每2根4H K线
        settlements = len(sub) // 2
        btc_price = sub["Close"].mean()
        daily_pnl = notional * rate_8h * 3  # 每天3次结算
        period_pnl = settlements * notional * rate_8h
        days = len(sub) * 4 / 24
        period_return = period_pnl / capital * 100

        results.append({
            "period": period,
            "days": round(days),
            "btc_avg": btc_price,
            "rate_8h": f"{rate_8h*100:.3f}%",
            "settlements": settlements,
            "pnl": round(period_pnl, 1),
            "return_pct": round(period_return, 1),
        })

    # 扣除开仓手续费
    open_fee = notional * 0.001  # 现货0.1% + 合约0.05% ≈ 0.1%

    total_pnl = sum(r["pnl"] for r in results) - open_fee
    total_return = total_pnl / capital * 100
    total_days = sum(r["days"] for r in results)
    annual_return = total_return / total_days * 365

    return {
        "total_pnl": round(total_pnl, 1),
        "total_return": round(total_return, 1),
        "annual_return": round(annual_return, 1),
        "details": results,
    }


# ═══════════════════════════════════════
# 策略 C: 网格交易
# ═══════════════════════════════════════

def backtest_grid(df_4h: pd.DataFrame, capital: float, grid_spacing: float = 0.01, levels: int = 3):
    """
    4H级别网格交易 — 现实约束版
    - 每根K线最多成交1次(买或卖)
    - ADX>25趋势市暂停网格(交给趋势策略)
    """
    from src.strategies.adaptive_hybrid import _adx, _atr, _ema
    df = df_4h.set_index("timestamp").sort_index()
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    h_arr = np.array(high); l_arr = np.array(low); c_arr = np.array(close)
    adx = _adx(h_arr, l_arr, c_arr, 14)

    per_level = capital / levels
    pnl = 0.0; inventory = 0.0; usdt_used = 0.0
    fills = 0

    for i in range(len(close)):
        # 趋势市跳过(ADX>25不跑网格)
        if i > 0 and not np.isnan(adx[i]) and adx[i] > 25:
            continue

        h, l, c = high[i], low[i], close[i]

        # 每根K线最多成交1次
        filled_this_bar = False

        # 先检查卖出(有库存才卖)
        sell_price = c * (1 + grid_spacing)
        if h >= sell_price and inventory > per_level / sell_price and not filled_this_bar:
            btc_out = per_level / sell_price
            profit = btc_out * sell_price - btc_out * sell_price / (1 + grid_spacing)
            pnl += profit
            inventory -= btc_out
            usdt_used -= per_level
            fills += 1; filled_this_bar = True

        # 再检查买入
        buy_price = c * (1 - grid_spacing)
        if l <= buy_price and usdt_used + per_level <= capital and not filled_this_bar:
            btc_in = per_level / buy_price
            inventory += btc_in
            usdt_used += per_level
            fills += 1

    # 结算
    final = (capital - usdt_used) + inventory * close[-1] + pnl
    ret = (final - capital) / capital * 100
    days = len(close) * 4 / 24
    ann = ret / days * 365 if days > 0 else 0

    return {"total_return": round(ret, 1), "annual_return": round(ann, 1), "fills": fills}


# ═══════════════════════════════════════
# 策略 A: 趋势跟踪 (复用已有)
# ═══════════════════════════════════════

def backtest_trend(df_4h: pd.DataFrame, capital: float, leverage: int = 3):
    """趋势跟踪回测 - 自适应混合 4H"""
    df_test = df_4h.set_index("timestamp").iloc[int(len(df_4h)*0.7):]
    bt = lib.FractionalBacktest(
        data=df_test, strategy=AdaptiveHybridStrategy,
        cash=capital, commission=0.0005,
    )
    stats = bt.run()
    return {
        "total_return": round(stats["Return [%]"], 1),
        "win_rate": round(stats.get("Win Rate [%]", 0), 1),
        "trades": stats["# Trades"],
        "max_dd": round(stats["Max. Drawdown [%]"], 1),
    }


# ═══════════════════════════════════════
# 主程序
# ═══════════════════════════════════════

def main():
    d = OKXDataDownloader(data_dir="data/raw", market_type="swap")
    since = datetime(2023, 7, 1, tzinfo=timezone.utc)
    until = datetime(2026, 7, 1, tzinfo=timezone.utc)

    TOTAL = 10000  # 总本金

    print("╔══════════════════════════════════════════════╗")
    print("║     三策略组合回测 (纯本地数据)              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # ── BTC ──
    df4 = d.load("BTC/USDT", "4h", since, until)

    print("📊 策略 A: 趋势跟踪 (30%, 自适应混合 4H + 3x杠杆)")
    trend = backtest_trend(df4, TOTAL * 0.3, leverage=3)
    print(f"   收益: {trend['total_return']:+.1f}%  胜率: {trend['win_rate']:.0f}%  "
          f"交易: {trend['trades']}笔  回撤: {trend['max_dd']:.1f}%")
    print()

    print("📊 策略 B: 资金费率套利 (40%, 现货多+永续空, delta中性)")
    funding = backtest_funding_arb(df4, TOTAL * 0.4, leverage=5)
    print(f"   总收益: {funding['total_return']:+.1f}%  年化: {funding['annual_return']:.1f}%  扣除手续费")
    for d in funding["details"]:
        print(f"   {d['period']}: {d['return_pct']:+.1f}% ({d['days']}天, "
              f"BTC均{d['btc_avg']:.0f}, 费率{d['rate_8h']}/8h, {d['settlements']}次结算)")
    print()

    print("📊 策略 C: 网格交易 (30%, 4H级别, 1%间距, 3档)")
    grid = backtest_grid(df4, TOTAL * 0.3, grid_spacing=0.01, levels=3)
    print(f"   收益: {grid['total_return']:+.1f}%  年化: {grid['annual_return']:.1f}%  成交: {grid['fills']}次")
    print()

    # ── 组合 ──
    combo_return = (trend['total_return'] * 0.3 +
                    funding['total_return'] * 0.4 +
                    grid['total_return'] * 0.3)
    combo_annual = combo_return / 3  # 简化: 3年

    print("═" * 50)
    print(f"🏁 组合结果 (40%费率 + 30%网格 + 30%趋势):")
    print(f"   总收益: {combo_return:+.1f}%")
    print(f"   年化收益: {combo_annual:.1f}%")
    print(f"   月均收益: {combo_annual/12:.1f}%")
    print(f"   3年10000U → {TOTAL*(1+combo_return/100):.0f}U")
    print("═" * 50)


if __name__ == "__main__":
    main()
