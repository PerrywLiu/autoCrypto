"""
15m 策略回测 v3 — 基于49策略研究结果优化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from backtesting import Backtest
from datetime import timezone

from src.data.downloader import OKXDataDownloader
from src.backtest.performance import compute_metrics
from src.strategies.fifteen_min_v3 import STRATEGIES_V3

SYMBOL = "BTC/USDT"
INITIAL_CASH = 100_000
COMMISSION = 0.0004

# 测试周期
PERIODS = [
    ("2023-07-01", "2024-07-01", "Y1"),
    ("2024-01-01", "2025-01-01", "牛市"),
    ("2024-07-01", "2025-07-01", "Y2"),
    ("2025-07-01", "2026-07-01", "Y3"),
    ("2023-07-01", "2026-07-01", "全时段"),
]


def load(since, until, mtf=False):
    d = OKXDataDownloader("data/raw", "swap")
    s_dt = pd.Timestamp(since, tz="utc") if since else None
    u_dt = pd.Timestamp(until, tz="utc") if until else None
    df = d.load(SYMBOL, "15m", s_dt, u_dt)
    if df.empty: return None
    df = df.set_index("timestamp").sort_index()

    if mtf:
        for tf in ["1h", "4h"]:
            try:
                ht = d.load(SYMBOL, tf, s_dt, u_dt)
                if ht.empty: continue
                ht = ht.set_index("timestamp").sort_index()
                al = ht.reindex(df.index, method="ffill").shift(1)
                al = al.rename(columns={c: f"{tf}_{c}" for c in al.columns})
                df = pd.concat([df, al], axis=1)
            except: pass
    return df.dropna()


def run(name, cls, data, period):
    try:
        bt = Backtest(data, cls, cash=INITIAL_CASH, commission=COMMISSION,
                       exclusive_orders=True)
        stats = bt.run()
        equity = stats["_equity_curve"]["Equity"]
        trades = stats["_trades"]
        m = compute_metrics(stats, equity, trades)

        buyhold = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)

        n_months = len(data) / (30 * 24 * 4)
        monthly = (1 + m["total_return_pct"]) ** (1 / max(n_months, 1)) - 1

        return {
            "策略": name, "时段": period,
            "收益率": m["total_return_pct"],
            "月均": monthly,
            "年化": m["cagr_pct"],
            "夏普": m["sharpe_ratio"],
            "回撤": m["max_drawdown_pct"],
            "胜率": m["win_rate_pct"],
            "盈亏比": m["profit_factor"],
            "交易数": m["total_trades"],
            "买入持有": buyhold,
        }
    except Exception as e:
        logger.error(f"{name} @ {period}: {str(e)[:80]}")
        return None


def main():
    print(f"\n{'='*100}")
    print(f"  15m 趋势跟踪策略回测 | 资金:{INITIAL_CASH:,.0f}U | 费率:{COMMISSION:.2%}")
    print(f"{'='*100}")

    all_r = []
    for since, until, label in PERIODS:
        print(f"\n--- {label}: {since} ~ {until} ---")
        d15 = load(since, until, mtf=False)
        dmtf = load(since, until, mtf=True)
        if d15 is None: continue
        print(f"  数据: 15m={len(d15)}行, MTF={len(dmtf) if dmtf is not None else 0}行")

        for name, cls in STRATEGIES_V3.items():
            data = d15  # 默认用纯15m
            r = run(name, cls, data, label)
            if r:
                all_r.append(r)
                print(f"  {name:25s} | 收益:{r['收益率']:>8.1%} "
                      f"月均:{r['月均']:>7.1%} 夏普:{r['夏普']:>6.2f} "
                      f"回撤:{r['回撤']:>7.1%} 胜率:{r['胜率']:>5.0%} "
                      f"交易:{r['交易数']:>4d} | BH:{r['买入持有']:>7.1%}")

    df = pd.DataFrame(all_r)
    if df.empty: return

    # 全时段排名
    full = df[df["时段"] == "全时段"].sort_values("收益率", ascending=False)
    if not full.empty:
        print(f"\n{'='*100}")
        print("  📊 全时段排名 (2023.07~2026.07)")
        print(f"{'='*100}")
        print(f"{'策略':30s} {'收益':>8s} {'月均':>7s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'交易':>5s} {'BH':>7s}")
        print("-"*87)
        for _, r in full.iterrows():
            print(f"{r['策略']:30s} {r['收益率']:>7.1%} {r['月均']:>6.1%} "
                  f"{r['年化']:>6.1%} {r['夏普']:>5.2f} {r['回撤']:>6.1%} "
                  f"{r['胜率']:>5.0%} {r['交易数']:>4d} {r['买入持有']:>6.1%}")

    # 各策略分年
    print(f"\n{'='*100}")
    print("  📊 各策略分年稳定性")
    print(f"{'='*100}")
    for name in STRATEGIES_V3:
        sub = df[(df["策略"]==name) & (df["时段"]!="全时段")]
        if len(sub) < 2: continue
        rets = sub["收益率"].values
        pos = sum(1 for r in rets if r > 0)
        avg_m = np.mean([r for r in sub["月均"]])
        avg_sh = np.mean(sub["夏普"].values)
        total_t = sub["交易数"].sum()
        print(f"  {name:30s} | 平均月收益:{avg_m:>6.1%} 夏普:{avg_sh:>6.2f} "
              f"盈利期:{pos}/{len(rets)} 总交易:{total_t}")

    # 保存
    out = Path("outputs/15m_v3_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ 结果已保存: {out}")


if __name__ == "__main__":
    main()
