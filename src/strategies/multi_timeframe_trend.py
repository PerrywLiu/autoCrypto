"""
多时间框架趋势策略

参考 okx-ai-quant/src/okx_ai_quant/strategy.py MultiTimeframeTrendStrategy

核心逻辑：
1. EMA12/26/50 三重排列（快>中>慢 = 多头；快<中<慢 = 空头）
2. 1H + 4H 双时间框架必须同时对齐
3. trend_gap 过滤：EMA 快慢差值/价格 < 0.2% → 趋势太弱，观望
4. RSI 合理区间过滤：45~72（做多）、28~55（做空）
5. 止损 1.2x ATR，止盈 2.0x ATR（趋势越强、空间越大）
"""

from backtesting import Strategy
import pandas as pd
import numpy as np
from loguru import logger


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    result = result.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    result = result.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return result.clip(lower=0.0, upper=100.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


class MultiTimeframeTrendStrategy(Strategy):
    """
    多时间框架 EMA 三重排列策略

    只有 1H 和 4H 的 EMA12/26/50 同时形成多头/空头排列时，
    且趋势强度、RSI 区间均满足条件时才入场。
    """

    # === 三重均线 ===
    ema_fast: int = 12
    ema_mid: int = 26
    ema_slow: int = 50

    # === 趋势强度 ===
    trend_gap_min: float = 0.002    # EMA快慢差/价格 ≥ 0.2%

    # === RSI ===
    rsi_period: int = 14
    rsi_long_lower: float = 45.0
    rsi_long_upper: float = 72.0
    rsi_short_lower: float = 28.0
    rsi_short_upper: float = 55.0

    # === ATR ===
    atr_period: int = 14
    atr_stop_mult: float = 1.2
    atr_tp_mult: float = 2.0

    # === 过滤器 ===
    min_expected_move: float = 0.005

    # === 风控 ===
    risk_per_trade: float = 0.02

    def init(self):
        """预计算 1H 和 4H 的三重 EMA、RSI、ATR"""
        close_1h = self.data.Close
        df_1h = pd.DataFrame({
            "high": self.data.High,
            "low": self.data.Low,
            "close": close_1h,
        })

        # ---- 1H 指标 ----
        self.ema_fast_1h = self.I(_ema, close_1h, self.ema_fast, name="MT_EMA12_1H")
        self.ema_mid_1h = self.I(_ema, close_1h, self.ema_mid, name="MT_EMA26_1H")
        self.ema_slow_1h = self.I(_ema, close_1h, self.ema_slow, name="MT_EMA50_1H")
        self.rsi_1h = self.I(_rsi, close_1h, self.rsi_period, name="MT_RSI14")
        self.atr_1h = self.I(_atr, df_1h, self.atr_period, name="MT_ATR14")

        # ---- 4H 指标（如果有） ----
        has_4h = "4h_Close" in self.data.df.columns
        if has_4h:
            close_4h = self.data.df["4h_Close"]
            self.ema_fast_4h = self.I(_ema, close_4h, self.ema_fast, overlay=False, name="MT_EMA12_4H")
            self.ema_mid_4h = self.I(_ema, close_4h, self.ema_mid, overlay=False, name="MT_EMA26_4H")
            self.ema_slow_4h = self.I(_ema, close_4h, self.ema_slow, overlay=False, name="MT_EMA50_4H")
        else:
            logger.warning("未检测到 4H 数据列，将仅使用 1H")
            self.ema_fast_4h = self.ema_fast_1h
            self.ema_mid_4h = self.ema_mid_1h
            self.ema_slow_4h = self.ema_slow_1h

    def next(self):
        """每个 K 线的交易决策"""
        close_price = self.data.Close[-1]
        rsi_val = self.rsi_1h[-1]
        atr_val = self.atr_1h[-1]

        # ---- 1H 指标 ----
        f1 = self.ema_fast_1h[-1]
        m1 = self.ema_mid_1h[-1]
        s1 = self.ema_slow_1h[-1]

        # ---- 4H 指标 ----
        f4 = self.ema_fast_4h[-1]
        m4 = self.ema_mid_4h[-1]
        s4 = self.ema_slow_4h[-1]

        # 数据有效性
        if any(pd.isna(x) for x in [f1, m1, s1, f4, m4, s4, rsi_val, atr_val]):
            return
        if close_price <= 0:
            return

        # === EMA 排列检查 ===
        # 1H 级别
        one_up = f1 > m1 > s1 and close_price > m1
        one_down = f1 < m1 < s1 and close_price < m1
        # 4H 级别
        four_up = f4 > m4 > s4
        four_down = f4 < m4 < s4

        # === 趋势强度检查 ===
        trend_gap = abs(f1 - s1) / close_price
        if trend_gap < self.trend_gap_min:
            return  # 均线太粘合，趋势不明显

        # === 期望波动检查 ===
        expected_move = max((atr_val * 1.8) / close_price, abs(f1 - s1) / close_price)
        if expected_move < self.min_expected_move:
            return

        # === 入场信号 ===
        if not self.position:
            # 做多：1H 多头排列 + 4H 多头排列 + RSI 合理区间
            if (four_up and one_up and
                self.rsi_long_lower <= rsi_val <= self.rsi_long_upper):
                sl_price = close_price - atr_val * self.atr_stop_mult
                tp_price = close_price + atr_val * self.atr_tp_mult
                size = self._calc_size(close_price, sl_price)
                if size > 0:
                    self.buy(size=size, sl=sl_price, tp=tp_price)
                    logger.debug(
                        f"📈 MTF开多 @ {close_price:.2f} | "
                        f"gap={trend_gap:.3%} RSI={rsi_val:.1f}"
                    )

            # 做空：1H 空头排列 + 4H 空头排列 + RSI 合理区间
            elif (four_down and one_down and
                  self.rsi_short_lower <= rsi_val <= self.rsi_short_upper):
                sl_price = close_price + atr_val * self.atr_stop_mult
                tp_price = close_price - atr_val * self.atr_tp_mult
                size = self._calc_size(close_price, sl_price)
                if size > 0:
                    self.sell(size=size, sl=sl_price, tp=tp_price)
                    logger.debug(
                        f"📉 MTF开空 @ {close_price:.2f} | "
                        f"gap={trend_gap:.3%} RSI={rsi_val:.1f}"
                    )

        # === 平仓：EMA 排列瓦解 ===
        if self.position.is_long and not one_up:
            self.position.close()
            logger.debug(f"🔴 MTF平多 @ {close_price:.2f}（排列瓦解）")
        elif self.position.is_short and not one_down:
            self.position.close()
            logger.debug(f"🟢 MTF平空 @ {close_price:.2f}（排列瓦解）")

    def _calc_size(self, entry: float, stop: float) -> float:
        dist = abs(entry - stop) / entry
        if dist <= 0:
            return 0.0
        risk_amt = self.equity * self.risk_per_trade
        return min(risk_amt / dist, self.equity * 0.95) / entry
