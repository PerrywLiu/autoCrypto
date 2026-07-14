"""
回测引擎模块

负责：
- 加载数据 + 合并多时间框架
- 初始化 Backtest 实例
- 运行回测并收集结果
- 支持单策略和多策略对比
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

import pandas as pd
from backtesting import Backtest, Strategy
from loguru import logger

from src.data.downloader import OKXDataDownloader
from src.data.processor import DataProcessor
from src.backtest.performance import compute_metrics


@dataclass
class BacktestResult:
    """回测结果数据类"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    # 绩效指标
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    # 原始数据
    equity_curve: Optional[pd.Series] = None
    trades_df: Optional[pd.DataFrame] = None
    # 回测 stats 对象
    _stats: Optional[pd.Series] = None


class BacktestRunner:
    """
    回测执行器

    使用方式：
        runner = BacktestRunner(data_dir="data/raw")
        result = runner.run(
            symbol="BTC/USDT",
            strategy_class=EmaRsiAtrStrategy,
            start="2024-01-01",
            end="2025-06-01",
        )
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% = OKX spot taker fee
        contract_mode: bool = False,  # 永续合约模式（含资金费率）
    ):
        """
        初始化回测执行器

        Args:
            data_dir: 原始数据目录
            initial_cash: 初始资金
            commission: 手续费率
            contract_mode: 是否永续合约模式（包含资金费率计算）
        """
        self.data_dir = Path(data_dir)
        self.initial_cash = initial_cash
        self.commission = commission
        self.contract_mode = contract_mode
        self.downloader = OKXDataDownloader(data_dir)
        logger.info(
            f"回测引擎初始化: 初始资金={initial_cash}, "
            f"手续费={commission:.3%}, 合约模式={contract_mode}"
        )

    def load_data(
        self,
        symbol: str = "BTC/USDT",
        timeframes: tuple[str, ...] = ("1h", "4h"),
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载并合并多时间框架数据

        Args:
            symbol: 交易对
            timeframes: 时间框架列表（第一个为主时间框架）
            since: 起始日期 "YYYY-MM-DD"
            until: 结束日期 "YYYY-MM-DD"

        Returns:
            合并后的 DataFrame（可直接用于 Backtest）
        """
        since_dt = pd.Timestamp(since, tz="utc") if since else None
        until_dt = pd.Timestamp(until, tz="utc") if until else None

        # 加载主时间框架
        primary_tf = timeframes[0]
        df_primary = self.downloader.load(symbol, primary_tf, since_dt, until_dt)
        if df_primary.empty:
            raise ValueError(f"无数据: {symbol} {primary_tf}")

        # 如果只有一个时间框架，直接返回
        if len(timeframes) == 1:
            df_primary = DataProcessor.clean_ohlcv(df_primary)
            return df_primary.set_index("timestamp")

        # 加载并合并额外的时间框架
        result_df = df_primary
        for tf in timeframes[1:]:
            df_higher = self.downloader.load(symbol, tf, since_dt, until_dt)
            if df_higher.empty:
                logger.warning(f"无 {tf} 数据，跳过合并")
                continue

            result_df = DataProcessor.merge_timeframes(
                df_primary, df_higher,
                higher_prefix=tf.replace("h", "h").replace("d", "d"),
                lower_prefix=primary_tf.replace("h", "h").replace("d", "d"),
            )

        # 确保没有 NaN
        result_df = result_df.dropna()
        logger.info(f"数据准备完成: {symbol}，{len(result_df)} 行用于回测")
        return result_df

    def run(
        self,
        symbol: str = "BTC/USDT",
        strategy_class: Type[Strategy] = None,
        timeframes: tuple[str, ...] = ("1h", "4h"),
        since: str = "2024-01-01",
        until: str = "2025-06-01",
        **strategy_kwargs,
    ) -> BacktestResult:
        """
        运行单次回测

        Args:
            symbol: 交易对
            strategy_class: 策略类
            timeframes: 时间框架
            since: 起始日期
            until: 结束日期
            **strategy_kwargs: 策略参数覆盖

        Returns:
            BacktestResult
        """
        if strategy_class is None:
            from src.strategies.ema_rsi_atr import EmaRsiAtrStrategy
            strategy_class = EmaRsiAtrStrategy

        logger.info(f"开始回测: {strategy_class.__name__} on {symbol} {timeframes}")

        # 加载数据
        data = self.load_data(symbol, timeframes, since, until)
        if data.empty:
            raise ValueError("回测数据为空")

        # 初始化 Backtest
        bt = Backtest(
            data=data,
            strategy=strategy_class,
            cash=self.initial_cash,
            commission=self.commission,
            exclusive_orders=True,       # 互斥订单（同一时间只有一个持仓）
            hedge=False,                  # 不允许同时做多做空
        )

        # 运行回测
        stats = bt.run(**strategy_kwargs)

        # 计算扩展指标
        equity_curve = stats["_equity_curve"]["Equity"]
        trades_list = stats["_trades"]
        metrics = compute_metrics(stats, equity_curve, trades_list)

        # === 资金费率调整（永续合约模式） ===
        funding = {"total_funding": 0.0, "impact_pct": 0.0}
        if self.contract_mode and trades_list is not None and len(trades_list) > 0:
            try:
                from src.data.funding_rate import FundingRateModel
                frm = FundingRateModel(data_dir=str(self.data_dir))
                # 使用默认费率估算（不需要API调用）
                avg_rate = frm.DEFAULT_FUNDING_RATE  # 0.01% per 8h
                funding = frm.adjust_backtest_results(
                    trades_list,
                    frm._generate_default_series(
                        data.index[0].to_pydatetime(),
                        data.index[-1].to_pydatetime(),
                    ),
                    self.initial_cash,
                )
                # 调整收益率
                adjusted_return = metrics["total_return_pct"] + funding["impact_pct"]
                logger.info(
                    f"💰 资金费率调整: {funding['impact_pct']:+.2f}% | "
                    f"调整后收益: {adjusted_return:.2f}%"
                )
            except Exception as e:
                logger.warning(f"资金费率计算失败: {e}")

        # 组装结果
        strategy_name = getattr(strategy_class, "__name__", strategy_class.__class__.__name__)
        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe="+".join(timeframes),
            start_date=str(data.index[0].date()),
            end_date=str(data.index[-1].date()),
            total_return_pct=metrics["total_return_pct"],
            cagr_pct=metrics["cagr_pct"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            calmar_ratio=metrics["calmar_ratio"],
            win_rate_pct=metrics["win_rate_pct"],
            profit_factor=metrics["profit_factor"],
            total_trades=metrics["total_trades"],
            equity_curve=equity_curve,
            trades_df=trades_list,
            _stats=stats,
        )
        # 附加资金费率信息
        result.funding_impact = funding["impact_pct"]
        result.adjusted_return = metrics["total_return_pct"] + funding["impact_pct"]

        logger.info(f"回测完成: 收益率={result.total_return_pct:.2%}, "
                     f"夏普={result.sharpe_ratio:.2f}, 最大回撤={result.max_drawdown_pct:.2%}")

        return result

    def run_comparison(
        self,
        symbol: str = "BTC/USDT",
        strategies: list[tuple[str, Type[Strategy]]] = None,
        timeframes: tuple[str, ...] = ("1h", "4h"),
        since: str = "2024-01-01",
        until: str = "2025-06-01",
    ) -> pd.DataFrame:
        """
        运行多策略对比回测

        Args:
            symbol: 交易对
            strategies: [(名称, 策略类), ...]
            timeframes: 时间框架
            since: 起始日期
            until: 结束日期

        Returns:
            对比结果 DataFrame
        """
        if strategies is None:
            from src.strategies import STRATEGY_REGISTRY
            strategies = list(STRATEGY_REGISTRY.items())

        results = []
        for name, strat_class in strategies:
            try:
                result = self.run(symbol, strat_class, timeframes, since, until)
                results.append({
                    "策略": name,
                    "总收益率": f"{result.total_return_pct:.2%}",
                    "年化收益": f"{result.cagr_pct:.2%}",
                    "夏普比率": f"{result.sharpe_ratio:.2f}",
                    "最大回撤": f"{result.max_drawdown_pct:.2%}",
                    "Calmar": f"{result.calmar_ratio:.2f}",
                    "胜率": f"{result.win_rate_pct:.2%}",
                    "盈亏比": f"{result.profit_factor:.2f}",
                    "交易次数": result.total_trades,
                })
            except Exception as e:
                logger.error(f"{name} 回测失败: {e}")
                results.append({"策略": name, "总收益率": f"ERROR: {e}"})

        return pd.DataFrame(results)

    def print_summary(self, result: BacktestResult):
        """打印回测结果摘要"""
        print(f"""
╔══════════════════════════════════════════════════════╗
║              回测结果摘要                              ║
╠══════════════════════════════════════════════════════╣
║  策略: {result.strategy_name:<40} ║
║  交易对: {result.symbol:<38} ║
║  时间: {result.start_date} ~ {result.end_date:<26} ║
╠══════════════════════════════════════════════════════╣
║  总收益率:   {result.total_return_pct:>8.2%}                       ║
║  年化收益:   {result.cagr_pct:>8.2%}                       ║
║  夏普比率:   {result.sharpe_ratio:>8.2f}                       ║
║  最大回撤:   {result.max_drawdown_pct:>8.2%}                       ║
║  胜率:       {result.win_rate_pct:>8.2%}                       ║
║  盈亏比:     {result.profit_factor:>8.2f}                       ║
║  交易次数:   {result.total_trades:>8}                         ║
╚══════════════════════════════════════════════════════╝
        """)
