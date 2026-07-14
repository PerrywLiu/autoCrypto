#!/usr/bin/env python3
"""
趋势策略回测脚本

用法:
    # 单策略回测
    python scripts/backtest.py --strategy ema_rsi_atr --symbol BTC/USDT --start 2024-01-01

    # 多策略对比
    python scripts/backtest.py --compare --symbol BTC/USDT

    # 指定输出
    python scripts/backtest.py --strategy donchian_breakout --output outputs/reports/

依赖:
    uv add backtesting plotly    # 回测 + 可视化
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies import STRATEGY_REGISTRY, get_strategy
from src.backtest.engine import BacktestRunner
from src.visualization.charts import plot_backtest_report
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="趋势交易策略回测工具"
    )
    parser.add_argument(
        "--strategy", default="ema_rsi_atr",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="策略名称（默认: ema_rsi_atr）"
    )
    parser.add_argument(
        "--symbol", default="BTC/USDT",
        help="交易对（默认: BTC/USDT）"
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="起始日期 YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", default="2025-06-01",
        help="结束日期 YYYY-MM-DD"
    )
    parser.add_argument(
        "--cash", type=float, default=10000,
        help="初始资金（默认: 10000 USDT）"
    )
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="手续费率（默认: 0.1%%）"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="数据目录"
    )
    parser.add_argument(
        "--output", default="outputs/reports",
        help="报告输出目录"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="运行所有策略对比回测"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="不生成可视化图表"
    )

    args = parser.parse_args()

    # 初始化回测引擎
    runner = BacktestRunner(
        data_dir=args.data_dir,
        initial_cash=args.cash,
        commission=args.commission,
    )

    if args.compare:
        # 多策略对比模式
        logger.info("运行多策略对比回测...")
        df = runner.run_comparison(
            symbol=args.symbol,
            since=args.start,
            until=args.end,
        )
        print("\n" + "=" * 80)
        print("                   策略对比结果")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # 保存为 CSV
        csv_path = f"{args.output}/comparison_{args.symbol.replace('/', '_')}.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"对比结果已保存: {csv_path}")
    else:
        # 单策略回测
        strategy_class = get_strategy(args.strategy)
        logger.info(f"运行策略: {args.strategy} on {args.symbol}")

        result = runner.run(
            symbol=args.symbol,
            strategy_class=strategy_class,
            since=args.start,
            until=args.end,
        )

        # 打印摘要
        runner.print_summary(result)

        # 生成可视化
        if not args.no_plot and result.equity_curve is not None:
            plot_backtest_report(
                equity_curve=result.equity_curve,
                trades=result.trades_df,
                metrics={
                    "sharpe": result.sharpe_ratio,
                    "max_dd": result.max_drawdown_pct,
                },
                title=f"{args.strategy} - {args.symbol} 回测",
                save_dir=args.output,
            )

        # 打印交易明细
        if result.trades_df is not None and len(result.trades_df) > 0:
            print("\n最近 10 笔交易:")
            cols = ["EntryTime", "ExitTime", "Size", "EntryPrice",
                    "ExitPrice", "PnL", "ReturnPct"]
            display_cols = [c for c in cols if c in result.trades_df.columns]
            if "ReturnPct" in result.trades_df.columns:
                result.trades_df["ReturnPct"] = result.trades_df["ReturnPct"].apply(
                    lambda x: f"{x:.2%}" if x == x else ""
                )
            print(result.trades_df[display_cols].tail(10).to_string(index=True))

    logger.info("回测完成！")


if __name__ == "__main__":
    main()
