#!/usr/bin/env python
"""
BTC 日线/4h 趋势策略回测 (含资金费率)

用法:
    uv run python scripts/backtest_daily.py --tf 1d
    uv run python scripts/backtest_daily.py --tf 4h
    uv run python scripts/backtest_daily.py --tf 1d --funding-rate 0.0003
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
COMMISSION = 0.0004           # 0.04% — OKX永续taker
FUNDING_RATE = 0.0001         # 默认 0.01%/8h (牛市均值~0.03%, 熊市~0.005%)
FUNDING_INTERVAL_HOURS = 8    # OKX 每8小时结算一次

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


def calculate_funding_impact(trades, initial_equity, funding_rate):
    """
    计算资金费率对回测结果的影响

    逐笔交易计算，按 8 小时结算周期累加：
    - 做多 (Size > 0): 支付资金费率 → 负收益
    - 做空 (Size < 0): 收取资金费率 → 正收益

    Args:
        trades: backtesting.py 输出的 _trades DataFrame
        initial_equity: 初始资金
        funding_rate: 每 8 小时的资金费率

    Returns:
        {"total_funding": float, "trades_count": int, "impact_pct": float}
    """
    if trades is None or len(trades) == 0:
        return {"total_funding": 0.0, "trades_count": 0, "impact_pct": 0.0}

    total_funding = 0.0
    trades_with_funding = 0

    for _, t in trades.iterrows():
        entry = t.get("EntryTime")
        exit_ = t.get("ExitTime")
        size = t.get("Size", 0)
        price = t.get("EntryPrice", 0)

        # 检查数据有效性（NaT 不能用 `not` 检查，要用 pd.isna）
        if pd.isna(entry) or pd.isna(exit_) or size == 0 or price <= 0:
            continue

        # 名义价值 = 合约张数 × 入场价
        pos_value = abs(size) * price

        # 持仓时长（EntryTime/ExitTime 已是 tz-aware Timestamp）
        hold_hours = (exit_ - entry).total_seconds() / 3600
        if hold_hours <= 0:
            continue

        # 结算次数 = 持仓时间 / 8 小时（向上取整）
        settlements = max(1, int(np.ceil(hold_hours / FUNDING_INTERVAL_HOURS)))

        # 做多 → 付钱，做空 → 收钱
        if size > 0:  # long
            funding = -pos_value * funding_rate * settlements
        else:          # short
            funding = pos_value * funding_rate * settlements

        total_funding += funding
        if funding != 0:
            trades_with_funding += 1

    impact_pct = (total_funding / initial_equity) if initial_equity > 0 else 0.0

    return {
        "total_funding": round(total_funding, 2),
        "trades_count": trades_with_funding,
        "impact_pct": impact_pct,
    }


def run_backtest(strategy_class, data, name, period, cash, comm, funding_rate):
    """运行单次回测，返回结果字典（含资金费率调整）"""
    bt = Backtest(data, strategy_class, cash=cash, commission=comm,
                   exclusive_orders=True)
    stats = bt.run()
    equity = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]
    m = compute_metrics(stats, equity, trades)

    # ── 资金费率调整 ──
    funding = calculate_funding_impact(trades, cash, funding_rate)

    raw_return = m["total_return_pct"]
    funding_impact = funding["impact_pct"]
    adjusted_return = raw_return + funding_impact

    buyhold = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
    days = len(data)
    n_months = days / 30.44 if days > 200 else days / (365.25 / 12)
    monthly = (1 + raw_return) ** (1 / max(n_months, 1)) - 1
    adj_monthly = (1 + adjusted_return) ** (1 / max(n_months, 1)) - 1

    return {
        "策略": name, "周期": period,
        "原始收益": raw_return,
        "资金费影响": funding_impact,
        "调整后收益": adjusted_return,
        "月均(调整)": adj_monthly,
        "月均(原始)": monthly,
        "年化": m["cagr_pct"],
        "夏普": m["sharpe_ratio"],
        "最大回撤": m["max_drawdown_pct"],
        "胜率": m["win_rate_pct"],
        "盈亏比": m["profit_factor"],
        "交易数": m["total_trades"],
        "买入持有": buyhold,
        "超额(调整)": adjusted_return - buyhold,
        "多空比": f"{funding_impact:+.1%}",
    }


def print_table(results, title):
    """打印结果表格"""
    print(f"\n{'='*125}")
    print(f"  {title}")
    print(f"{'='*125}")
    header = (f"{'策略':15s} {'原始收益':>8s} {'资金费':>8s} {'调整后':>8s} "
              f"{'月均':>7s} {'夏普':>6s} {'回撤':>7s} {'胜率':>5s} "
              f"{'交易':>4s} {'BH收益':>8s} {'超额':>8s}")
    print(header)
    print("-" * 125)
    for r in results:
        print(f"{r['策略']:15s} {r['原始收益']:>7.1%} {r['资金费影响']:>7.1%} "
              f"{r['调整后收益']:>7.1%} {r['月均(调整)']:>6.1%} {r['夏普']:>5.2f} "
              f"{r['最大回撤']:>6.1%} {r['胜率']:>5.0%} {r['交易数']:>4d} "
              f"{r['买入持有']:>7.1%} {r['超额(调整)']:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description="BTC日线/4h策略回测 (含资金费率)")
    parser.add_argument("--tf", type=str, default="1d", choices=["1d", "4h"],
                        help="时间框架 (默认: 1d)")
    parser.add_argument("--period", type=str, nargs=2, metavar=("SINCE", "UNTIL"),
                        help="自定义回测周期 YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=INITIAL_CASH,
                        help=f"初始资金 (默认: {INITIAL_CASH})")
    parser.add_argument("--funding-rate", type=float, default=FUNDING_RATE,
                        help=f"每8h资金费率 (默认: {FUNDING_RATE:.4f} = {FUNDING_RATE*100:.2f}%%)")
    args = parser.parse_args()

    tf = args.tf
    funding_rate = args.funding_rate
    periods = [(args.period[0], args.period[1], f"自定义")
               ] if args.period else DEFAULT_PERIODS

    logger.info(f"🚀 BTC {tf} 回测 | 资金:{args.cash:,.0f}U | "
                f"手续费:{COMMISSION:.2%} | 资金费率:{funding_rate:.2%}/8h")

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
                r = run_backtest(cls, data, name, label, args.cash,
                                 COMMISSION, funding_rate)
                all_results.append(r)
                logger.info(
                    f"  {name:15s} | 原始:{r['原始收益']:>7.1%} 资金费:{r['资金费影响']:>+6.1%} "
                    f"→ 调整后:{r['调整后收益']:>7.1%} 月均:{r['月均(调整)']:>6.1%} "
                    f"夏普:{r['夏普']:>5.2f} 回撤:{r['最大回撤']:>6.1%} "
                    f"胜率:{r['胜率']:>5.0%} 交易:{r['交易数']:>3d} "
                    f"BH:{r['买入持有']:>7.1%}"
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
        full_sorted = full.sort_values("调整后收益", ascending=False)
        print_table(full_sorted.to_dict("records"),
                    f"{tf} 全时段排名 (含资金费率={funding_rate:.2%}/8h)")

    # ── 各策略分年表现 ──
    print(f"\n{'='*100}")
    print(f"  📊 各策略分年稳定性 ({tf}, 含资金费率)")
    print(f"{'='*100}")
    for name in DAILY_STRATEGIES:
        sub = df[(df["策略"] == name) & (~df["周期"].str.contains("全时段"))]
        if len(sub) == 0: continue
        adj_rets = sub["调整后收益"].values
        pos_adj = sum(1 for r in adj_rets if r > 0)
        avg_m = np.mean(sub["月均(调整)"].values)
        avg_sh = np.mean(sub["夏普"].values)
        total_t = sub["交易数"].sum()
        avg_funding = np.mean([float(str(x).rstrip("%"))/100
                               for x in sub["资金费影响"]])
        print(f"  {name:15s} | 调整月均:{avg_m:>6.1%} 夏普:{avg_sh:>6.2f} "
              f"资金费:{avg_funding:>+6.1%} 盈利期:{pos_adj}/{len(sub)} 总交易:{total_t}")

    # ── 资金费率影响分析 ──
    print(f"\n{'='*100}")
    print(f"  💰 资金费率影响分析 (费率={funding_rate:.2%}/8h)")
    print(f"{'='*100}")
    for name in DAILY_STRATEGIES:
        sub = df[df["策略"] == name]
        if len(sub) == 0: continue
        total_funding = np.mean([r for r in sub["资金费影响"]])
        adj_vs_raw = np.mean(sub["调整后收益"].values - sub["原始收益"].values)
        print(f"  {name:15s} | 平均资金费率影响: {total_funding:>+7.1%}")

    # ── 保存 ──
    out = Path(f"outputs/daily_{tf}_funding_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"\n✅ 结果已保存: {out}")


if __name__ == "__main__":
    main()
