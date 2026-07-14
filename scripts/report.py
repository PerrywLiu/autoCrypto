#!/usr/bin/env python3
"""
查看纸上交易的收益分析

用法:
    uv run python scripts/report.py
"""

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_trades(path: str = "outputs/live_trades.json") -> list:
    """加载交易记录"""
    f = Path(path)
    if not f.exists():
        print(f"❌ 交易记录文件不存在: {path}")
        print("   先启动 --dry-run 模式跑一段时间后就有了")
        return []
    with open(f) as fh:
        return json.load(fh)


def show_report(trades: list):
    """展示收益分析"""
    if not trades:
        print("暂无交易记录")
        return

    df = pd.DataFrame(trades)

    # ── 基本指标 ──
    total_pnl = df["pnl"].sum()
    wins = (df["pnl"] > 0).sum()
    total = len(df)
    win_rate = wins / total * 100 if total > 0 else 0
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if total - wins > 0 else 0
    best = df["pnl"].max()
    worst = df["pnl"].min()

    # 按模式统计
    by_mode = df.groupby("mode").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).sum() / len(x) * 100)
    ).round(1)

    print("╔══════════════════════════════════════╗")
    print("║       纸上交易收益分析               ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  总交易: {total} 笔                    ")
    print(f"║  总盈亏: {total_pnl:+.2f} USDT           ")
    print(f"║  胜率:   {win_rate:.1f}%                 ")
    print(f"║  均盈:   {avg_win:+.2f}  均亏: {avg_loss:+.2f}")
    print(f"║  最佳:   {best:+.2f}  最差: {worst:+.2f}")
    print(f"║  盈亏比: {abs(avg_win/avg_loss) if avg_loss != 0 else 99:.2f}")
    print("╚══════════════════════════════════════╝")

    print(f"\n📋 按策略模式:")
    print(by_mode.to_string())

    # ── 权益曲线 ──
    df["cum_pnl"] = df["pnl"].cumsum()
    df["drawdown"] = df["cum_pnl"].cummax() - df["cum_pnl"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("累计盈亏 (USDT)", "回撤 (USDT)")
    )

    # 权益曲线
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in df["pnl"]]
    fig.add_trace(go.Scatter(y=df["cum_pnl"], mode="lines+markers",
        name="累计盈亏", line=dict(color="#1f77b4", width=2),
        marker=dict(color=colors, size=8)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # 回撤
    fig.add_trace(go.Scatter(y=df["drawdown"], mode="lines",
        name="回撤", fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="#e74c3c", width=1)), row=2, col=1)

    fig.update_layout(height=600, hovermode="x unified",
                      title="纸上交易绩效", showlegend=False)
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="USDT", row=2, col=1)

    path = "outputs/performance.html"
    Path("outputs").mkdir(exist_ok=True)
    fig.write_html(path)
    print(f"\n📊 图表已保存: {path}")


def main():
    trades = load_trades()
    show_report(trades)


if __name__ == "__main__":
    main()
