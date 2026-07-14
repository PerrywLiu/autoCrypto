"""
绩效指标计算模块

计算标准金融指标：
- 总收益率、年化收益率 (CAGR)
- 夏普比率、索提诺比率
- 最大回撤、Calmar 比率
- 胜率、盈亏比（利润因子）、期望值
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_metrics(
    stats: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.02,  # 无风险利率 2%
) -> dict:
    """
    计算综合绩效指标

    Args:
        stats: backtesting.py 的 stats 输出
        equity_curve: 权益曲线 Series
        trades: 交易明细 DataFrame
        risk_free_rate: 年化无风险利率

    Returns:
        指标字典
    """
    metrics = {}

    # ---- 基本指标 ----
    metrics["total_return_pct"] = _safe_float(
        (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    )
    metrics["start_equity"] = _safe_float(equity_curve.iloc[0])
    metrics["end_equity"] = _safe_float(equity_curve.iloc[-1])

    # ---- 年化收益率 (CAGR) ----
    trading_days = len(equity_curve)
    # 假设 1H K线，每年约 365 * 24 = 8760 根
    periods_per_year = 365 * 24  # 1H
    if trading_days > 0:
        years = trading_days / periods_per_year
        if years > 0:
            metrics["cagr_pct"] = _safe_float(
                (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
            )
        else:
            metrics["cagr_pct"] = 0.0
    else:
        metrics["cagr_pct"] = 0.0

    # ---- 日收益率（用于夏普/索提诺） ----
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0:
        # 将 hourly returns 年化
        hourly_rf = risk_free_rate / (365 * 24)
        excess = returns - hourly_rf

        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret > 0:
            metrics["sharpe_ratio"] = _safe_float(
                (mean_ret - hourly_rf) / std_ret * np.sqrt(365 * 24)
            )
        else:
            metrics["sharpe_ratio"] = 0.0

        # 索提诺比率（仅下行波动）
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics["sortino_ratio"] = _safe_float(
                (mean_ret - hourly_rf) / downside.std() * np.sqrt(365 * 24)
            )
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["sortino_ratio"] = 0.0

    # ---- 最大回撤 ----
    metrics["max_drawdown_pct"] = _calc_max_drawdown(equity_curve)

    # ---- Calmar 比率 ----
    if metrics["max_drawdown_pct"] > 0:
        metrics["calmar_ratio"] = _safe_float(
            metrics["cagr_pct"] / abs(metrics["max_drawdown_pct"])
        )
    else:
        metrics["calmar_ratio"] = 0.0

    # ---- 交易统计 ----
    if len(trades) > 0:
        pnls = trades["PnL"].values if "PnL" in trades.columns else trades.get("PnL", [])
        metrics["total_trades"] = len(trades)
        metrics["winning_trades"] = int((pnls > 0).sum())
        metrics["losing_trades"] = int((pnls <= 0).sum())
        metrics["win_rate_pct"] = _safe_float(
            metrics["winning_trades"] / metrics["total_trades"]
        )

        # 利润因子 (Profit Factor)
        gross_profit = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls <= 0].sum())
        if gross_loss > 0:
            metrics["profit_factor"] = _safe_float(gross_profit / gross_loss)
        else:
            metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

        # 平均盈亏比
        avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
        avg_loss = abs(pnls[pnls <= 0].mean()) if (pnls <= 0).any() else 0
        if avg_loss > 0:
            metrics["avg_win_loss_ratio"] = _safe_float(avg_win / avg_loss)
        else:
            metrics["avg_win_loss_ratio"] = float("inf") if avg_win > 0 else 0.0

        # 最大单笔盈利 / 最大单笔亏损
        metrics["max_win"] = _safe_float(pnls.max()) if len(pnls) > 0 else 0
        metrics["max_loss"] = _safe_float(pnls.min()) if len(pnls) > 0 else 0

        # 平均持仓时间（K线数）
        if "EntryBar" in trades.columns and "ExitBar" in trades.columns:
            avg_hold = (trades["ExitBar"] - trades["EntryBar"]).mean()
            metrics["avg_hold_bars"] = _safe_float(avg_hold)
        else:
            metrics["avg_hold_bars"] = 0.0
    else:
        metrics["total_trades"] = 0
        metrics["winning_trades"] = 0
        metrics["losing_trades"] = 0
        metrics["win_rate_pct"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_win_loss_ratio"] = 0.0
        metrics["max_win"] = 0.0
        metrics["max_loss"] = 0.0
        metrics["avg_hold_bars"] = 0.0

    return metrics


def compute_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    计算月度收益率

    Args:
        equity_curve: 权益曲线（index 为 DatetimeIndex）

    Returns:
        月度收益率 Series
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)

    monthly = equity_curve.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()
    return monthly_returns


def compute_rolling_sharpe(
    equity_curve: pd.Series, window: int = 720  # 30天 * 24小时
) -> pd.Series:
    """
    计算滚动夏普比率

    Args:
        equity_curve: 权益曲线
        window: 滚动窗口（K线数）

    Returns:
        滚动夏普比率 Series
    """
    returns = equity_curve.pct_change().dropna()
    if len(returns) < window:
        return pd.Series(dtype=float)

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    hourly_rf = 0.02 / (365 * 24)
    rolling_sharpe = (rolling_mean - hourly_rf) / rolling_std * np.sqrt(365 * 24)
    return rolling_sharpe


def _calc_max_drawdown(equity: pd.Series) -> float:
    """计算最大回撤（百分比）"""
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    return _safe_float(drawdown.min())


def _safe_float(value) -> float:
    """安全转换为 float，处理 NaN 和 Inf"""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, 6)
    except (ValueError, TypeError):
        return 0.0
