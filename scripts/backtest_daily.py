#!/usr/bin/env python
"""
BTC 日线/4h 趋势策略回测

用法:
    uv run python scripts/backtest_daily.py --tf 1d
    uv run python scripts/backtest_daily.py --tf 4h
    uv run python scripts/backtest_daily.py --tf 1d --period 2024-01-01 2025-01-01
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger
from backtesting import Backtest

from src.data.downloader import OKXDataDownloader
from src.backtest.performance import compute_metrics
from src.strategies.btc_daily import DAILY_STRATEGIES

SYMBOL = "BTC/USDT"
INITIAL_CASH = 100_000
COMMISSION = 0.0004  # 0.04% — OKX永续taker

DEFAULT_PERIODS = [
    ("2023-07-01", "2024-07-01", "Y1 (23.07-24.07)"),
    ("2024-01-01", "2025-01-01", "牛市 (2024)"),
    ("2024-07-01", "2025-07-01", "Y2 (24.07-25.07)"),
    ("2025-07-01", "2026-07-01", "熊市 (25.07-26.07)"),
    ("2023-07-01", "2026-07-01", "全时段 (3年)"),
]


def load_data(timeframe: str, since: str, until: str) -> pd.DataFrame:
    """加载K线数据"""
    downloader = OKXDataDownloader(data_dir="data/raw", market_type="swap")
    since_dt = pd.Timestamp(since, tz="utc") if since else None
    until_dt = pd.Timestamp(until, tz="utc") if until else None
    df = downloader.load(SYMBOL, timeframe, since_dt, until_dt)
    if df.empty:
        raise ValueError(f"无数据: {SYMBOL} {timeframe}")
    return df.set_index("timestamp").sort_index()


def run_backtest(strategy_class, data, name, period, cash, comm):
    """运行单次回测，返回结果字典"""
    bt = Backtest(data, strategy_class, cash=cash, commission=comm,
                   exclusive_orders=True)
    stats = bt.run()
    equity = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]
    m = compute_metrics(stats, equity, trades)

    buyhold = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
    n_months = len(data) / (365 / 12 * (24 if len(data) > 5000 else 1))  # 粗略估算月数
    days = len(data)
    n_months = days / 30.44 if days > 200 else days / (365.25 / 12)
    monthly = (1 + m["total_return_pct"]) ** (1 / max(n_months, 1)) - 1

    return {
        "策略": name, "周期": period,
        "总收益": m["total_return_pct"],
        "月均": monthly,
        "年化": m["cagr_pct"],
        "夏普": m["sharpe_ratio"],
        "最大回撤": m["max_drawdown_pct"],
        "胜率": m["win_rate_pct"],
        "盈亏比": m["profit_factor"],
        "交易数": m["total_trades"],
        "买入持有": buyhold,
        "超额收益": m["total_return_pct"] - buyhold,
    }


def print_table(results, title):
    """打印结果表格"""
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    header = f"{'策略':20s} {'总收益':>8s} {'月均':>7s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'交易':>5s} {'BH收益':>8s} {'超额':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['策略']:20s} {r['总收益']:>7.1%} {r['月均']:>6.1%} "
              f"{r['年化']:>6.1%} {r['夏普']:>5.2f} {r['最大回撤']:>6.1%} "
              f"{r['胜率']:>5.0%} {r['交易数']:>4d} {r['买入持有']:>7.1%} {r['超额收益']:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description="BTC日线/4h策略回测")
    parser.add_argument("--tf", type=str, default="1d", choices=["1d", "4h"],
                        help="时间框架 (默认: 1d)")
    parser.add_argument("--period", type=str, nargs=2, metavar=("SINCE", "UNTIL"),
                        help="自定义回测周期 YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=INITIAL_CASH,
                        help=f"初始资金 (默认: {INITIAL_CASH})")
    args = parser.parse_args()

    tf = args.tf
    periods = [(args.period[0], args.period[1], f"自定义")
               ] if args.period else DEFAULT_PERIODS

    logger.info(f"🚀 BTC {tf} 日线/4h趋势策略回测 | 资金:{args.cash:,.0f}U | 费率:{COMMISSION:.2%}")

    all_results = []

    for since, until, label in periods:
        logger.info(f"\n📅 {label}: {since} → {until}")

        try:
            data = load_data(tf, since, until)
            logger.info(f"  数据: {len(data)} 根K线")
        except Exception as e:
            logger.error(f"  数据加载失败: {e}")
            continue

        for name, cls in DAILY_STRATEGIES.items():
            try:
                r = run_backtest(cls, data, name, label, args.cash, COMMISSION)
                all_results.append(r)
                logger.info(
                    f"  {name:15s} | 收益:{r['总收益']:>7.1%} 月均:{r['月均']:>6.1%} "
                    f"夏普:{r['夏普']:>5.2f} 回撤:{r['最大回撤']:>6.1%} "
                    f"胜率:{r['胜率']:>5.0%} 交易:{r['交易数']:>3d} "
                    f"BH:{r['买入持有']:>7.1%} α:{r['超额收益']:>+6.1%}"
                )
            except Exception as e:
                logger.error(f"  {name:15s} | ❌ {str(e)[:60]}")

    df = pd.DataFrame(all_results)
    if df.empty:
        logger.error("无回测结果!")
        return

    # ── 全时段排名 ──
    full = df[df["周期"].str.contains("全时段")]
    if not full.empty:
        full_sorted = full.sort_values("总收益", ascending=False)
        print_table(full_sorted.to_dict("records"),
                    f"{tf} 全时段排名 (2023.07~2026.07)")

    # ── 各策略分年表现 ──
    print(f"\n{'='*110}")
    print(f"  📊 各策略分年稳定性 ({tf})")
    print(f"{'='*110}")
    for name in DAILY_STRATEGIES:
        sub = df[(df["策略"] == name) & (~df["周期"].str.contains("全时段"))]
        if len(sub) == 0: continue
        rets = sub["总收益"].values
        sharpes = sub["夏普"].values
        pos_years = sum(1 for r in rets if r > 0)
        avg_m = np.mean(sub["月均"].values)
        total_t = sub["交易数"].sum()
        print(f"  {name:15s} | 平均月收益:{avg_m:>6.1%} 夏普:{np.mean(sharpes):>6.2f} "
              f"盈利期:{pos_years}/{len(rets)} 总交易:{total_t}")

    # ── 保存 ──
    out = Path(f"outputs/daily_{tf}_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"\n✅ 结果已保存: {out}")


if __name__ == "__main__":
    main()
