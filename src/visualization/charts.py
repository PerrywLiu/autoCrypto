"""
可视化模块

使用 Plotly 生成交互式图表：
- 权益曲线 + 回撤区间（双Y轴）
- 月度收益率热力图
- 交易信号标注（买入/卖出箭头）
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


def plot_equity_curve(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    title: str = "回测权益曲线",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    绘制权益曲线 + 回撤区间叠加图

    Args:
        equity_curve: 权益曲线 Series（index 为 DatetimeIndex）
        trades: 交易明细 DataFrame（可选，用于标注买卖点）
        title: 图表标题
        save_path: 保存为 HTML 文件的路径

    Returns:
        Plotly Figure 对象
    """
    # 计算回撤
    cummax = equity_curve.cummax()
    drawdown_pct = ((equity_curve - cummax) / cummax) * 100

    # 创建双 Y 轴子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "回撤 (%)")
    )

    # ---- 上图：权益曲线 ----
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="权益",
            line=dict(color="#1f77b4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.05)",
        ),
        row=1, col=1,
    )

    # 标注初始资金线
    fig.add_hline(
        y=equity_curve.iloc[0],
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        row=1, col=1,
    )

    # 标注交易买卖点
    if trades is not None and len(trades) > 0:
        # 买入点
        if "EntryPrice" in trades.columns:
            buy_trades = trades[trades["Size"] > 0] if "Size" in trades.columns else trades
            if len(buy_trades) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades["EntryTime"] if "EntryTime" in buy_trades.columns
                          else equity_curve.index[:len(buy_trades)],
                        y=buy_trades["EntryPrice"],
                        mode="markers",
                        name="买入",
                        marker=dict(
                            symbol="triangle-up", size=10,
                            color="green", line=dict(width=1, color="darkgreen")
                        ),
                    ),
                    row=1, col=1,
                )

        # 卖出点
        if "ExitPrice" in trades.columns:
            sell_trades = trades[trades["Size"] < 0] if "Size" in trades.columns else trades
            if len(sell_trades) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades["ExitTime"] if "ExitTime" in sell_trades.columns
                          else equity_curve.index[:len(sell_trades)],
                        y=sell_trades["ExitPrice"],
                        mode="markers",
                        name="卖出",
                        marker=dict(
                            symbol="triangle-down", size=10,
                            color="red", line=dict(width=1, color="darkred")
                        ),
                    ),
                    row=1, col=1,
                )

    # ---- 下图：回撤 ----
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in drawdown_pct.values]
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            mode="lines",
            name="回撤",
            line=dict(color="#e74c3c", width=1),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.1)",
        ),
        row=2, col=1,
    )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

    # 布局
    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        hovermode="x unified",
        height=700,
    )
    fig.update_yaxes(title_text="权益 (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        logger.info(f"图表已保存: {path}")

    return fig


def plot_monthly_heatmap(
    equity_curve: pd.Series,
    title: str = "月度收益率热力图",
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    绘制月度收益率热力图

    Args:
        equity_curve: 权益曲线
        title: 标题
        save_path: HTML 保存路径

    Returns:
        Plotly Figure
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.warning("权益曲线 index 不是 DatetimeIndex，无法绘制月度热力图")
        return go.Figure()

    # 按月重采样计算收益率
    monthly = equity_curve.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    # 构建年度×月份矩阵
    monthly_ret.index = monthly_ret.index.to_period("M")
    pivot = monthly_ret.groupby([
        monthly_ret.index.year,
        monthly_ret.index.month
    ]).first().unstack(level=0)

    # 转换为百分比
    pivot_pct = pivot * 100

    fig = go.Figure(data=go.Heatmap(
        z=pivot_pct.values,
        x=[str(y) for y in pivot_pct.columns],
        y=[f"{m}月" for m in pivot_pct.index],
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row]
              for row in pivot_pct.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="收益率 (%)"),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="年份",
        yaxis_title="月份",
        height=400,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"热力图已保存: {save_path}")

    return fig


def plot_backtest_report(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    metrics: Optional[dict] = None,
    title: str = "回测报告",
    save_dir: str = "outputs/reports",
) -> dict[str, str]:
    """
    生成完整的回测可视化报告

    Args:
        equity_curve: 权益曲线
        trades: 交易明细
        metrics: 绩效指标字典
        title: 报告标题
        save_dir: 输出目录

    Returns:
        {"equity_chart": path, "heatmap": path, "report_html": path}
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    saved = {}

    # 1) 权益曲线
    eq_path = f"{save_dir}/{timestamp}_equity.html"
    plot_equity_curve(equity_curve, trades, title=title, save_path=eq_path)
    saved["equity_chart"] = eq_path

    # 2) 月度热力图
    heat_path = f"{save_dir}/{timestamp}_monthly_heatmap.html"
    plot_monthly_heatmap(equity_curve, title=f"{title} - 月度收益", save_path=heat_path)
    saved["heatmap"] = heat_path

    logger.info(f"回测报告已生成: {save_dir}/")

    return saved
