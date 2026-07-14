"""
Donchian 通道突破策略（海龟交易法变体）

参考 okx-ai-quant/src/okx_ai_quant/strategy.py DonchianBreakoutStrategy

核心逻辑：
1. 20 期 Donchian 通道（最高价上轨 / 最低价下轨）
2. 4H EMA12/26 趋势方向过滤器
3. 价格突破上轨 + 4H EMA 多头排列 → 做多
4. 价格突破下轨 + 4H EMA 空头排列 → 做空
5. ATR14 动态止损止盈
"""

from backtesting import Strategy
import pandas as pd
import numpy as np
from loguru import logger


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _donchian_channel(high: pd.Series, low: pd.Series, period: int):
    """返回 (upper, lower) - 最高上轨和最低下轨"""
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    return upper, lower


class DonchianBreakoutStrategy(Strategy):
    """
    Donchian 通道突破策略

    入场：价格突破 20 期 Donchian 通道 + 4H EMA 趋势确认
    止损：通道边界外 1x ATR
    止盈：入场价 + 1.5x ATR
    """

    # === Donchian 参数 ===
    donchian_period: int = 20

    # === 趋势过滤器 ===
    ema_fast: int = 12
    ema_slow: int = 26

    # === ATR 参数 ===
    atr_period: int = 14
    atr_stop_mult: float = 1.0
    atr_tp_mult: float = 1.5

    # === 过滤器 ===
    min_expected_move: float = 0.005

    # === 风控 ===
    risk_per_trade: float = 0.02

    def init(self):
        """预计算 Donchian 通道、趋势指标和 ATR"""

        # ---- 1H 级别 Donchian 通道 ----
        high_1h = self.data.High
        low_1h = self.data.Low
        close_1h = self.data.Close

        self.upper, self.lower = self.I(
            _donchian_channel, high_1h, low_1h, self.donchian_period,
            name=("Donchian_Upper", "Donchian_Lower")
        )

        # ---- ATR ----
        df_1h = pd.DataFrame({
            "high": high_1h, "low": low_1h, "close": close_1h
        })
        self.atr = self.I(_atr, df_1h, self.atr_period, name="ATR14")

        # ---- 4H EMA 趋势过滤器 ----
        if "4h_Close" in self.data.df.columns:
            close_4h = self.data.df["4h_Close"]
            self.ema_fast_4h = self.I(_ema, close_4h, self.ema_fast,
                                       overlay=False, name="DC_EMA12_4H")
            self.ema_slow_4h = self.I(_ema, close_4h, self.ema_slow,
                                       overlay=False, name="DC_EMA26_4H")
        else:
            logger.warning("未检测到 4H 数据列，趋势过滤将使用 1H EMA")
            self.ema_fast_4h = self.I(_ema, close_1h, self.ema_fast,
                                       overlay=True, name="DC_EMA12_1H")
            self.ema_slow_4h = self.I(_ema, close_1h, self.ema_slow,
                                       overlay=True, name="DC_EMA26_1H")

    def next(self):
        """每个 K 线的交易决策"""

        close_price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        atr_val = self.atr[-1]
        ema_f = self.ema_fast_4h[-1]
        ema_s = self.ema_slow_4h[-1]

        # 数据有效性
        if (pd.isna(upper) or pd.isna(lower) or pd.isna(atr_val) or
            pd.isna(ema_f) or pd.isna(ema_s) or close_price <= 0):
            return

        # === 4H 趋势方向 ===
        trend_bullish = ema_f > ema_s   # EMA 多头排列
        trend_bearish = ema_f < ema_s   # EMA 空头排列

        # === 入场信号 ===
        if not self.position:
            # 做多：突破上轨 + 趋势向上
            if close_price > upper and trend_bullish:
                expected_move = (close_price - upper + atr_val) / close_price
                if expected_move >= self.min_expected_move:
                    sl_price = upper - atr_val * self.atr_stop_mult
                    tp_price = close_price + atr_val * self.atr_tp_mult
                    size = self._calc_size(close_price, sl_price)
                    if size > 0:
                        self.buy(size=size, sl=sl_price, tp=tp_price)
                        logger.debug(
                            f"📈 Donchian开多 @ {close_price:.2f} | "
                            f"上轨={upper:.2f} 止损={sl_price:.2f}"
                        )

            # 做空：跌破下轨 + 趋势向下
            elif close_price < lower and trend_bearish:
                expected_move = (lower - close_price + atr_val) / close_price
                if expected_move >= self.min_expected_move:
                    sl_price = lower + atr_val * self.atr_stop_mult
                    tp_price = close_price - atr_val * self.atr_tp_mult
                    size = self._calc_size(close_price, sl_price)
                    if size > 0:
                        self.sell(size=size, sl=sl_price, tp=tp_price)
                        logger.debug(
                            f"📉 Donchian开空 @ {close_price:.2f} | "
                            f"下轨={lower:.2f} 止损={sl_price:.2f}"
                        )

        # === 平仓：回归通道中线 + 趋势反转 ===
        mid = (upper + lower) / 2
        if self.position.is_long:
            if close_price < mid and not trend_bullish:
                self.position.close()
                logger.debug(f"🔴 Donchian平多 @ {close_price:.2f}")
        elif self.position.is_short:
            if close_price > mid and not trend_bearish:
                self.position.close()
                logger.debug(f"🟢 Donchian平空 @ {close_price:.2f}")

    def _calc_size(self, entry: float, stop: float) -> float:
        """基于风险计算仓位"""
        dist = abs(entry - stop) / entry
        if dist <= 0:
            return 0.0
        risk_amt = self.equity * self.risk_per_trade
        return min(risk_amt / dist, self.equity * 0.95) / entry
