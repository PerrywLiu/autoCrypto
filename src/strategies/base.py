"""
趋势交易策略基类

继承 backtesting.Strategy，提供公共能力：
- ATR 动态止损止盈计算
- 基于风险的仓位大小计算
- 置信度评分
- 交易冷却期
- 连续亏损熔断
"""

from backtesting import Strategy
import numpy as np


class BaseStrategy(Strategy):
    """
    趋势交易策略基类

    子类只需实现 init() 和 next()，其余能力由基类提供。
    """

    # === 可被子类覆盖的参数 ===
    # 单笔风险比例（占总资金）
    risk_per_trade: float = 0.02
    # 最大持仓 K 线数（None = 不限）
    max_hold_bars: int | None = None
    # 连续亏损熔断次数
    max_consecutive_losses: int = 5
    # 交易冷却期（K线数）
    cooldown_bars: int = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consecutive_losses: int = 0
        self._bars_since_last_trade: int = 999
        self._entry_bar: int = 0

    def init(self):
        """子类必须实现：预计算所需指标"""
        raise NotImplementedError("子类必须实现 init() 方法")

    def next(self):
        """子类必须实现：每个 K 线的交易逻辑"""
        raise NotImplementedError("子类必须实现 next() 方法")

    # ------------------------------------------------------------------
    # 公共工具方法
    # ------------------------------------------------------------------

    def atr_stop_loss(self, atr_col, multiplier: float = 1.0, side: str = "long") -> float:
        """
        计算 ATR 止损价格

        Args:
            atr_col: ATR 指标的列名（在 self.data.df 中）
            multiplier: ATR 倍数
            side: "long" 或 "short"

        Returns:
            止损价格
        """
        atr_value = self.data.df[atr_col].iloc[-1]
        if side == "long":
            return self.data.Close[-1] - atr_value * multiplier
        else:
            return self.data.Close[-1] + atr_value * multiplier

    def atr_take_profit(self, atr_col, multiplier: float = 1.5, side: str = "long") -> float:
        """
        计算 ATR 止盈价格

        Args:
            atr_col: ATR 指标的列名
            multiplier: ATR 倍数
            side: "long" 或 "short"

        Returns:
            止盈价格
        """
        atr_value = self.data.df[atr_col].iloc[-1]
        if side == "long":
            return self.data.Close[-1] + atr_value * multiplier
        else:
            return self.data.Close[-1] - atr_value * multiplier

    def position_size(self, stop_distance_pct: float) -> float:
        """
        基于风险计算仓位大小

        公式: 仓位 = (总权益 × 风险比例) / 止损距离比例

        Args:
            stop_distance_pct: 止损距离占当前价格的比例（如 0.02 = 2%）

        Returns:
            建议的交易数量（以基础资产计），最大不超过总权益的 95%
        """
        risk_amount = self.equity * self.risk_per_trade
        if stop_distance_pct <= 0:
            return 0.0
        position_value = risk_amount / stop_distance_pct
        # 限制不超过总权益的 95%
        max_position = self.equity * 0.95
        size = min(position_value, max_position) / self.data.Close[-1]
        return max(0.0, size)

    def calc_confidence(self, components: dict[str, float]) -> float:
        """
        计算综合置信度

        Args:
            components: 各分量字典，如 {"rsi_strength": 0.6, "vol": 0.4}

        Returns:
            0~1 之间的置信度分数
        """
        if not components:
            return 0.0
        return round(sum(components.values()) / len(components), 4)

    def bounded_confidence(self, value: float) -> float:
        """将置信度限制在 [0, 0.95] 范围内"""
        return round(max(0.0, min(value, 0.95)), 4)

    # ------------------------------------------------------------------
    # 风险控制
    # ------------------------------------------------------------------

    def can_trade(self) -> bool:
        """检查是否满足交易条件"""
        # 冷却期检查
        if self._bars_since_last_trade < self.cooldown_bars:
            return False
        # 连续亏损熔断
        if self._consecutive_losses >= self.max_consecutive_losses:
            return False
        return True

    def _on_trade_close(self, trade):
        """交易关闭时更新状态（backtesting.py 没有此回调，需在 next 中手动调用）"""
        if trade.pl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        self._bars_since_last_trade = 0

    def _update_counters(self):
        """每个 K 线更新计数器"""
        self._bars_since_last_trade += 1
