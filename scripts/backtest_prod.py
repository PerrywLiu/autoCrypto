#!/usr/bin/env python
"""
生产级回测: TripleEma 4h + 可配置杠杆

用法:
    uv run python scripts/backtest_prod.py                 # 默认 2x
    uv run python scripts/backtest_prod.py --lev 1         # 1x 现货
    uv run python scripts/backtest_prod.py --lev 3         # 3x 杠杆
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import timezone
from loguru import logger
from backtesting import Backtest

from src.data.downloader import OKXDataDownloader
from src.backtest.performance import compute_metrics
from src.strategies.production import TripleEmaStrategy

# ── 配置 ──
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
MARGIN = 100_000           # 保证金
COMMISSION = 0.0004        # 0.04%
FUNDING_RATE = 0.0001      # 0.01%/8h (默认)

PERIODS = [
    ("2023-07-01", "2024-07-01", "Y1 (23.07-24.07)"),
    ("2024-01-01", "2025-01-01", "牛市 (2024)"),
    ("2024-07-01", "2025-07-01", "Y2 (24.07-25.07)"),
    ("2025-07-01", "2026-07-01", "熊市 (25.07-26.07)"),
    ("2023-07-01", "2026-07-01", "全时段 (3年)"),
]


def load_data(since: str, until: str) -> pd.DataFrame:
    d = OKXDataDownloader("data/raw", "swap")
    s = pd.Timestamp(since, tz="utc")
    u = pd.Timestamp(until, tz="utc")
    df = d.load(SYMBOL, TIMEFRAME, s, u)
    return df.set_index("timestamp").sort_index()


def calc_funding(trades, margin) -> float:
    total = 0.0
    for _, t in trades.iterrows():
        e = t.get("EntryTime"); x = t.get("ExitTime")
        sz = t.get("Size", 0); p = t.get("EntryPrice", 0)
        if pd.isna(e) or pd.isna(x) or sz == 0 or p <= 0:
            continue
        pos = abs(sz) * p
        hrs = (x - e).total_seconds() / 3600
        if hrs <= 0: continue
        stl = max(1, int(np.ceil(hrs / 8)))
        total += (pos * FUNDING_RATE * stl) if sz < 0 \
            else (-pos * FUNDING_RATE * stl)
    return total / margin


def run_one(data, lev, period_label):
    cash = MARGIN * lev
    bt = Backtest(data, TripleEmaStrategy, cash=cash,
                  commission=COMMISSION, exclusive_orders=True)
    stats = bt.run()
    equity = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]
    m = compute_metrics(stats, equity, trades)

    fund_pct = calc_funding(trades, MARGIN)
    raw_ret = (equity.iloc[-1] - cash) / MARGIN
    adj_ret = raw_ret + fund_pct

    days = len(data) / 6  # 4h → 每天6根
    n_months = days / 30.44
    monthly = (1 + adj_ret) ** (1 / max(n_months, 1)) - 1

    bh = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1

    return {
        "label": period_label,
        "资金": f"{cash/1000:.0f}k",
        "权益": equity.iloc[-1],
        "总收益": adj_ret,
        "月均": monthly,
        "夏普": m["sharpe_ratio"],
        "回撤保证金": m["max_drawdown_pct"] * lev,
        "胜率": m["win_rate_pct"],
        "交易数": m["total_trades"],
        "资金费": fund_pct,
        "BH收益": bh,
        "超额": adj_ret - bh,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lev", type=float, default=2, help="杠杆倍数")
    args = parser.parse_args()
    lev = args.lev

    logger.info(f"🚀 TripleEma {TIMEFRAME} | 保证金:{MARGIN:,}U | "
                f"{lev}x杠杆 | 手续费:{COMMISSION:.2%}")

    all_r = []
    for since, until, label in PERIODS:
        data = load_data(since, until)
        logger.info(f"  {label}: {len(data)} 根K线")
        r = run_one(data, lev, label)
        all_r.append(r)
        logger.info(
            f"    收益:{r['总收益']:>8.1%} 月:{r['月均']:>6.1%} "
            f"夏普:{r['夏普']:>6.2f} 回撤:{r['回撤保证金']:>7.1%} "
            f"胜率:{r['胜率']:>5.0%} 交易:{r['交易数']:>3d} "
            f"费:{r['资金费']:>+5.1%} | BH:{r['BH收益']:>7.1%}"
        )

    print(f"\n{'='*90}")
    print(f"  📊 TripleEma 4h {lev}x杠杆 回测汇总")
    print(f"{'='*90}")
    print(f"{'周期':20s} {'收益':>8s} {'月均':>7s} {'夏普':>6s} "
          f"{'回撤':>7s} {'胜率':>5s} {'交易':>4s} {'资金费':>7s}")
    print("-" * 74)
    for r in all_r:
        print(f"{r['label']:20s} {r['总收益']:>7.1%} {r['月均']:>6.1%} "
              f"{r['夏普']:>5.2f} {r['回撤保证金']:>6.1%} {r['胜率']:>4.0%} "
              f"{r['交易数']:>4d} {r['资金费']:>+6.1%}")

    # 保存
    out = Path(f"outputs/prod_4h_{lev}x.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_r).to_csv(out, index=False)
    logger.info(f"✅ 已保存: {out}")


if __name__ == "__main__":
    main()
