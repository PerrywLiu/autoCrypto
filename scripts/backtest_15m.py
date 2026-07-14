"""
15分钟策略批量回测脚本

对 7 个 15 分钟专用策略进行多时段回测，评估稳健性。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger
from backtesting import Backtest

from src.data.downloader import OKXDataDownloader
from src.backtest.performance import compute_metrics
from src.strategies.fifteen_min import FIFTEEN_MIN_STRATEGIES

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════

SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
INITIAL_CASH = 10000.0
COMMISSION = 0.001  # 0.1%

# 测试周期（避免单一周期过拟合）
TEST_PERIODS = [
    ("2023-07-01", "2024-07-01", "第一年 (2023.07-2024.07)"),
    ("2024-07-01", "2025-07-01", "第二年 (2024.07-2025.07)"),
    ("2025-07-01", "2026-07-01", "第三年 (2025.07-2026.07)"),
    ("2024-01-01", "2025-01-01", "牛市期 (2024全年)"),
    ("2023-07-01", "2026-07-01", "全时段 (2023.07-2026.07)"),
]


def load_15m_data(since: str = None, until: str = None) -> pd.DataFrame:
    """加载 15 分钟 K 线数据"""
    downloader = OKXDataDownloader(data_dir="data/raw", market_type="swap")

    since_dt = pd.Timestamp(since, tz="utc") if since else None
    until_dt = pd.Timestamp(until, tz="utc") if until else None

    df = downloader.load(SYMBOL, TIMEFRAME, since_dt, until_dt)

    if df.empty:
        raise ValueError(f"无数据: {SYMBOL} {TIMEFRAME}")

    # 设置索引
    df = df.set_index("timestamp").sort_index()
    logger.info(f"数据加载完成: {len(df)} 行, {df.index[0]} ~ {df.index[-1]}")
    return df


def load_mtf_data(since: str = None, until: str = None) -> pd.DataFrame:
    """加载并合并多时间框架数据 (15m + 1h + 4h)"""
    downloader = OKXDataDownloader(data_dir="data/raw", market_type="swap")

    since_dt = pd.Timestamp(since, tz="utc") if since else None
    until_dt = pd.Timestamp(until, tz="utc") if until else None

    # 加载 15m 主数据
    df_15m = downloader.load(SYMBOL, "15m", since_dt, until_dt)
    if df_15m.empty:
        raise ValueError(f"无 15m 数据")

    df_15m = df_15m.set_index("timestamp").sort_index()
    result = df_15m

    # 合并 1h 数据
    try:
        df_1h = downloader.load(SYMBOL, "1h", since_dt, until_dt)
        if not df_1h.empty:
            df_1h = df_1h.set_index("timestamp").sort_index()
            # 前向填充对齐到 15m
            aligned = df_1h.reindex(result.index, method="ffill").shift(1)
            aligned = aligned.rename(columns={
                "Open": "1h_Open", "High": "1h_High",
                "Low": "1h_Low", "Close": "1h_Close", "Volume": "1h_Volume"
            })
            result = pd.concat([result, aligned], axis=1)
            logger.info(f"已合并 1h 数据")
    except Exception as e:
        logger.warning(f"合并 1h 数据失败: {e}")

    # 合并 4h 数据
    try:
        df_4h = downloader.load(SYMBOL, "4h", since_dt, until_dt)
        if not df_4h.empty:
            df_4h = df_4h.set_index("timestamp").sort_index()
            aligned = df_4h.reindex(result.index, method="ffill").shift(1)
            aligned = aligned.rename(columns={
                "Open": "4h_Open", "High": "4h_High",
                "Low": "4h_Low", "Close": "4h_Close", "Volume": "4h_Volume"
            })
            result = pd.concat([result, aligned], axis=1)
            logger.info(f"已合并 4h 数据")
    except Exception as e:
        logger.warning(f"合并 4h 数据失败: {e}")

    result = result.dropna()
    logger.info(f"MTF 数据准备完成: {len(result)} 行")
    return result


def run_single_backtest(strategy_class, data, name, period_label, **kwargs):
    """运行单次回测"""
    try:
        bt = Backtest(
            data=data,
            strategy=strategy_class,
            cash=INITIAL_CASH,
            commission=COMMISSION,
            exclusive_orders=True,
        )
        stats = bt.run(**kwargs)

        # 计算指标
        equity = stats["_equity_curve"]["Equity"]
        trades = stats["_trades"]
        metrics = compute_metrics(stats, equity, trades)

        return {
            "策略": name,
            "时段": period_label,
            "总收益率": f"{metrics['total_return_pct']:.2%}",
            "年化收益": f"{metrics['cagr_pct']:.2%}",
            "夏普比率": f"{metrics['sharpe_ratio']:.2f}",
            "最大回撤": f"{metrics['max_drawdown_pct']:.2%}",
            "胜率": f"{metrics['win_rate_pct']:.2%}",
            "盈亏比": f"{metrics['profit_factor']:.2f}",
            "交易次数": metrics["total_trades"],
            "月均收益": f"{(metrics['total_return_pct']) / max(len(equity) / (30*24*4), 1) * 100:.2f}%",
        }
    except Exception as e:
        logger.error(f"{name} 回测失败: {e}")
        return {
            "策略": name, "时段": period_label,
            "总收益率": f"ERROR: {e}",
        }


def main():
    logger.info("=" * 60)
    logger.info("15分钟策略批量回测")
    logger.info(f"资金: {INITIAL_CASH}U | 手续费: {COMMISSION:.2%}")
    logger.info("=" * 60)

    all_results = []

    for since, until, label in TEST_PERIODS:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试周期: {label}")
        logger.info(f"{'='*60}")

        # 加载 15m 纯数据
        try:
            data_15m = load_15m_data(since, until)
        except Exception as e:
            logger.error(f"跳过 {label}: {e}")
            continue

        # 加载多时间框架数据
        try:
            data_mtf = load_mtf_data(since, until)
        except Exception as e:
            logger.warning(f"MTF 数据加载失败: {e}，跳过 MultiTf 策略")
            data_mtf = None

        # 测试每个策略
        for name, strat_class in FIFTEEN_MIN_STRATEGIES.items():
            # 多时间框架策略需要用 merged 数据
            if name == "multi_tf_confirm":
                if data_mtf is None or len(data_mtf) < 500:
                    logger.warning(f"跳过 {name}: MTF 数据不足")
                    continue
                result = run_single_backtest(strat_class, data_mtf, name, label)
            else:
                if len(data_15m) < 500:
                    continue
                result = run_single_backtest(strat_class, data_15m, name, label)

            all_results.append(result)
            if "ERROR" not in str(result.get("总收益率", "")):
                logger.info(
                    f"  {name:25s} | "
                    f"收益={result['总收益率']:>8s} | "
                    f"夏普={result['夏普比率']:>6s} | "
                    f"回撤={result['最大回撤']:>7s} | "
                    f"胜率={result['胜率']:>6s} | "
                    f"交易={result['交易次数']}"
                )
            else:
                logger.error(f"  {name:25s} | ❌ {result['总收益率']}")

    # ═══════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════
    print("\n")
    print("=" * 100)
    print("                               📊 全 部 回 测 结 果 汇 总")
    print("=" * 100)

    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))

    # 按策略汇总
    print("\n")
    print("=" * 100)
    print("                               📈 按 策 略 平 均 表 现")
    print("=" * 100)

    for name in FIFTEEN_MIN_STRATEGIES:
        subset = df_results[df_results["策略"] == name]
        if subset.empty:
            continue

        # 提取数值
        returns = []
        sharpes = []
        drawdowns = []
        for _, row in subset.iterrows():
            try:
                ret_str = row["总收益率"]
                if "ERROR" not in str(ret_str):
                    returns.append(float(ret_str.rstrip("%")) / 100)
                    sharpes.append(float(row["夏普比率"]))
                    drawdowns.append(float(row["最大回撤"].rstrip("%")) / 100)
            except (ValueError, AttributeError):
                continue

        if returns:
            print(f"\n{name}:")
            print(f"  平均收益率: {np.mean(returns):.2%}")
            print(f"  平均夏普:   {np.mean(sharpes):.2f}")
            print(f"  平均回撤:   {np.mean(drawdowns):.2%}")
            print(f"  有效测试:   {len(returns)}/{len(subset)}")

    # 保存结果到 CSV
    output_path = Path("outputs/15m_backtest_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
