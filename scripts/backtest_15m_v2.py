"""
15分钟策略批量回测 v2 — 简洁版

8 个策略 × 3 年分段测试 + 全时段
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from backtesting import Backtest
# from tabulate import tabulate

from src.data.downloader import OKXDataDownloader
from src.backtest.performance import compute_metrics
from src.strategies.fifteen_min_v2 import ALL_STRATEGIES

SYMBOL = "BTC/USDT"
INITIAL_CASH = 100_000  # BTC价格高，需要足够资金
COMMISSION = 0.0004     # OKX永续taker费率

PERIODS = [
    ("2023-07-01", "2024-07-01", "Y1(23-24)"),
    ("2024-07-01", "2025-07-01", "Y2(24-25)"),
    ("2025-07-01", "2026-07-01", "Y3(25-26)"),
    ("2023-07-01", "2026-07-01", "全时段"),
]


def load_data(since, until):
    """加载15m纯数据"""
    d = OKXDataDownloader("data/raw", "swap")
    df = d.load(SYMBOL, "15m",
                pd.Timestamp(since, tz="utc") if since else None,
                pd.Timestamp(until, tz="utc") if until else None)
    if df.empty:
        raise ValueError("无数据")
    return df.set_index("timestamp").sort_index()


def load_mtf(since, until):
    """加载15m+1h+4h合并数据"""
    d = OKXDataDownloader("data/raw", "swap")
    since_dt = pd.Timestamp(since, tz="utc") if since else None
    until_dt = pd.Timestamp(until, tz="utc") if until else None

    df = d.load(SYMBOL, "15m", since_dt, until_dt)
    if df.empty: return None
    df = df.set_index("timestamp").sort_index()

    for tf in ["1h", "4h"]:
        try:
            ht = d.load(SYMBOL, tf, since_dt, until_dt)
            if ht.empty: continue
            ht = ht.set_index("timestamp").sort_index()
            aligned = ht.reindex(df.index, method="ffill").shift(1)
            prefix = tf
            aligned = aligned.rename(columns={
                "Open": f"{prefix}_Open", "High": f"{prefix}_High",
                "Low": f"{prefix}_Low", "Close": f"{prefix}_Close",
                "Volume": f"{prefix}_Volume"
            })
            df = pd.concat([df, aligned], axis=1)
        except Exception as e:
            logger.warning(f"合并{tf}失败: {e}")

    return df.dropna()


def run_one(strat_cls, data, name, period):
    """运行单次回测"""
    try:
        bt = Backtest(data, strat_cls, cash=INITIAL_CASH, commission=COMMISSION,
                       exclusive_orders=True)
        stats = bt.run()
        equity = stats["_equity_curve"]["Equity"]
        trades = stats["_trades"]
        m = compute_metrics(stats, equity, trades)

        # 买入持有基准
        buyhold_ret = (data.index[-1] and
                       (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1))

        return {
            "策略": name, "时段": period,
            "收益率": m["total_return_pct"],
            "年化": m["cagr_pct"],
            "夏普": m["sharpe_ratio"],
            "最大回撤": m["max_drawdown_pct"],
            "胜率": m["win_rate_pct"],
            "盈亏比": m["profit_factor"],
            "交易数": m["total_trades"],
            "持有收益": buyhold_ret if buyhold_ret else 0,
        }
    except Exception as e:
        logger.error(f"{name} @ {period} 失败: {e}")
        return None


def main():
    logger.info(f"🚀 15m策略回测 | 资金:{INITIAL_CASH}U | 费率:{COMMISSION:.2%}")

    all_results = []

    for since, until, label in PERIODS:
        logger.info(f"\n📅 {label}: {since} → {until}")

        try:
            data_15m = load_data(since, until)
            data_mtf = load_mtf(since, until)
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            continue

        logger.info(f"  15m数据: {len(data_15m)}行 | MTF数据: {len(data_mtf) if data_mtf is not None else 0}行")

        for name, cls in ALL_STRATEGIES.items():
            use_mtf = (name == "6_mtf_trend")
            data = data_mtf if use_mtf else data_15m

            if use_mtf and (data_mtf is None or len(data_mtf) < 500):
                logger.warning(f"  跳过 {name}: MTF数据不足")
                continue
            if len(data) < 500:
                continue

            r = run_one(cls, data, name, label)
            if r:
                all_results.append(r)
                logger.info(
                    f"  {name:30s} | 收益:{r['收益率']:>8.1%} | "
                    f"年化:{r['年化']:>7.1%} | 夏普:{r['夏普']:>6.2f} | "
                    f"回撤:{r['最大回撤']:>7.1%} | 胜率:{r['胜率']:>6.1%} | "
                    f"交易:{r['交易数']:>4d} | 持有:{r['持有收益']:>7.1%}"
                )

    # ═══════════════════════════════════
    # 汇总表
    # ═══════════════════════════════════
    df = pd.DataFrame(all_results)
    if df.empty:
        print("无回测结果！")
        return

    print("\n" + "=" * 120)
    print("📊 全时段汇总 (按收益率排序)")
    print("=" * 120)

    # 全时段结果
    full = df[df["时段"] == "全时段"].sort_values("收益率", ascending=False)
    if not full.empty:
        table = []
        for _, r in full.iterrows():
            table.append([
                r["策略"], f"{r['收益率']:.1%}", f"{r['年化']:.1%}",
                f"{r['夏普']:.2f}", f"{r['最大回撤']:.1%}",
                f"{r['胜率']:.1%}", f"{r['盈亏比']:.2f}",
                r["交易数"], f"{r['持有收益']:.1%}"
            ])
        headers = ["策略", "收益", "年化", "夏普", "最大回撤", "胜率", "盈亏比", "交易数", "持有收益"]
        col_widths = [32, 8, 8, 6, 8, 6, 6, 6, 8]
        header_line = "  ".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in table:
            print("  ".join(f"{str(c):^{w}}" for c, w in zip(row, col_widths)))

    # 各年度对比
    print("\n" + "=" * 120)
    print("📊 各策略分年表现")
    print("=" * 120)

    for name in ALL_STRATEGIES:
        subset = df[(df["策略"] == name) & (df["时段"] != "全时段")]
        if len(subset) < 2: continue
        rets = subset["收益率"].values
        sharpes = subset["夏普"].values
        trades = subset["交易数"].sum()
        pos_years = sum(1 for r in rets if r > 0)
        print(f"  {name:30s} | 平均收益:{np.mean(rets):>7.1%} | "
              f"平均夏普:{np.mean(sharpes):>6.2f} | "
              f"盈利年:{pos_years}/{len(rets)} | 总交易:{trades}")

    # 保存
    output = Path("outputs/15m_v2_results.csv")
    output.parent.mkdir(exist_ok=True)
    df.to_csv(output, index=False)
    logger.info(f"\n✅ 结果已保存到 {output}")


if __name__ == "__main__":
    main()
