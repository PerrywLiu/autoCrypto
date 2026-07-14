"""
EMA-RSI-ATR 趋势跟踪策略（优化版 v2）

参考 okx-ai-quant/src/okx_ai_quant/strategy.py EmaRsiAtrStrategy

核心改进：
1. 4H EMA 趋势过滤器强制执行（双时间框架必须对齐）
2. 交易冷却期 + 最小持仓K线数，防止过度交易
3. ATR 动态止损止盈（更宽裕的止损 + 更高的盈亏比）
4. 趋势强度过滤（EMA 差值必须超过噪声阈值）
5. RSI 极端值过滤（避免在超买/超卖区域追趋势）

核心逻辑：
- 做多：4H EMA 多头 + 1H EMA 多头 + RSI 合理 + 趋势强度足够
- 做空：4H EMA 空头 + 1H EMA 空头 + RSI 合理 + 趋势强度足够
- 平仓：4H 趋势反转 或 止损/止盈触发
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


# ------------------------------------------------------------------
# NumPy 指标（兼容 backtesting.py 的 self.I()）
# ------------------------------------------------------------------

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA (alpha = 2/(span+1)，与 pandas ewm(adjust=False) 等价)"""
    if span <= 0 or len(arr) < span:
        result = np.empty_like(arr, dtype=float)
        result[:] = np.nan
        return result
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(arr, dtype=float)
    result[:] = np.nan
    result[span - 1] = np.mean(arr[:span])
    for i in range(span, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Wilder's smoothing)"""
    result = np.empty_like(arr, dtype=float)
    result[:] = np.nan
    if len(arr) < period + 1:
        return result

    delta = np.diff(arr)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    alpha = 1.0 / period

    for i in range(period, len(delta)):
        if avg_loss == 0:
            result[i + 1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        avg_gain = alpha * gain[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * loss[i] + (1 - alpha) * avg_loss

    return np.clip(result, 0.0, 100.0)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR (rolling mean)"""
    result = np.empty_like(close, dtype=float)
    result[:] = np.nan
    if len(close) < period + 1:
        return result

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    for i in range(period - 1, len(tr)):
        result[i] = np.mean(tr[i - period + 1:i + 1])
    return result


# ------------------------------------------------------------------
# 优化版策略类
# ------------------------------------------------------------------

class EmaRsiAtrStrategy(Strategy):
    """
    EMA-RSI-ATR 双时间框架趋势跟踪策略（v2 优化版）

    参数说明见 config/strategies/ema_rsi_atr.yaml
    """

    # === 均线参数 ===
    ema_fast: int = 12
    ema_slow: int = 26

    # === RSI 参数 ===
    rsi_period: int = 14
    rsi_long_min: float = 45.0    # 做多 RSI 合理下限（太低→超卖反弹风险）
    rsi_long_max: float = 72.0    # 做多 RSI 合理上限（太高→追高接盘风险）
    rsi_short_min: float = 28.0   # 做空 RSI 合理下限
    rsi_short_max: float = 55.0   # 做空 RSI 合理上限

    # === ATR 参数 ===
    atr_period: int = 14
    atr_stop_mult: float = 1.5        # 止损 = 入场价 ± 1.5x ATR（比原来更宽裕）
    atr_tp_mult: float = 3.0          # 止盈 = 入场价 ± 3.0x ATR（盈亏比 1:2）

    # === 趋势强度参数 ===
    trend_gap_min: float = 0.002      # EMA快慢线差/价格 ≥ 0.2%

    # === 置信度与过滤 ===
    rsi_weight: float = 0.3
    volatility_weight: float = 0.2
    base_confidence: float = 0.5
    confidence_threshold: float = 0.55  # 提高到 0.55
    min_expected_move: float = 0.008    # 提高到 0.8%（过滤低波动噪声）

    # === 风控参数 ===
    risk_per_trade: float = 0.02       # 单笔风险 2%
    cooldown_bars: int = 6             # 交易冷却期（至少6根K线不重复开仓）
    min_hold_bars: int = 4             # 最小持仓（至少持有4根K线再考虑平仓）

    def init(self):
        """预计算所有技术指标"""

        close_1h = self.data.Close
        high_1h = self.data.High
        low_1h = self.data.Low

        # ---- 1H 指标 ----
        self.ema_fast_1h = self.I(_ema, close_1h, self.ema_fast,
                                   overlay=True, name="EMA12_1H")
        self.ema_slow_1h = self.I(_ema, close_1h, self.ema_slow,
                                   overlay=True, name="EMA26_1H")
        self.rsi_1h = self.I(_rsi, close_1h, self.rsi_period,
                              name="RSI14_1H")
        self.atr_1h = self.I(_atr, high_1h, low_1h, close_1h, self.atr_period,
                              name="ATR14_1H")

        # ---- 4H 指标（从合并数据中获取）----
        has_4h = "4h_Close" in self.data.df.columns
        if has_4h:
            close_4h = self.data.df["4h_Close"].values
            self.ema_fast_4h = self.I(_ema, close_4h, self.ema_fast,
                                       overlay=False, name="EMA12_4H")
            self.ema_slow_4h = self.I(_ema, close_4h, self.ema_slow,
                                       overlay=False, name="EMA26_4H")
            self._4h_enabled = True
            logger.info("4H 趋势过滤器已启用")
        else:
            self.ema_fast_4h = self.ema_fast_1h  # fallback: 用 1H 代替
            self.ema_slow_4h = self.ema_slow_1h
            self._4h_enabled = False
            logger.warning("未检测到 4H 数据列，4H 过滤器降级为 1H")

        # ---- 内部状态 ----
        self._bars_since_entry: int = 999
        self._bars_since_exit: int = 999
        self._entry_bar: int = 0

        logger.info(
            f"EMA-RSI-ATR v2 初始化完成 | "
            f"cooldown={self.cooldown_bars} 最小持仓={self.min_hold_bars} | "
            f"止损={self.atr_stop_mult}xATR 止盈={self.atr_tp_mult}xATR"
        )

    def next(self):
        """每个 1H K 线决策"""

        # ---- 获取当前数据 ----
        close_price = self.data.Close[-1]
        fast_1h = self.ema_fast_1h[-1]
        slow_1h = self.ema_slow_1h[-1]
        rsi_val = self.rsi_1h[-1]
        atr_val = self.atr_1h[-1]

        # 4H 趋势
        fast_4h = self.ema_fast_4h[-1]
        slow_4h = self.ema_slow_4h[-1]

        # 数据有效性检查
        if (np.isnan(fast_1h) or np.isnan(slow_1h) or
            np.isnan(rsi_val) or np.isnan(atr_val) or
            np.isnan(fast_4h) or np.isnan(slow_4h) or close_price <= 0):
            return

        # ---- 更新计数器 ----
        self._bars_since_exit += 1
        if self.position:
            self._bars_since_entry += 1

        # ============================================
        # 趋势判断
        # ============================================
        one_h_up = fast_1h > slow_1h
        one_h_down = fast_1h < slow_1h
        four_h_up = fast_4h > slow_4h
        four_h_down = fast_4h < slow_4h

        # 趋势强度：EMA 快慢线差必须大于趋势阈值
        trend_gap = abs(fast_1h - slow_1h) / close_price
        trend_strong = trend_gap >= self.trend_gap_min

        # 期望波动
        atr_rate = atr_val / close_price
        expected_move = max(atr_rate * 1.8, abs(fast_1h - slow_1h) / close_price)

        # ============================================
        # 持仓管理
        # ============================================
        if self.position:
            # 最小持仓保护：刚入场不久，不急于平仓（让止损/止盈自己触发）
            if self._bars_since_entry < self.min_hold_bars:
                return

            # 平仓条件：4H 趋势反转（弱化 1H 抖动的影响）
            if self.position.is_long:
                # 做多平仓：4H 转空头（而非 1H 一时抖动）
                if four_h_down or close_price < fast_1h - atr_val * 0.5:
                    self.position.close()
                    self._bars_since_entry = 999
                    self._bars_since_exit = 0
                    logger.info(
                        f"🔴 平多 @ {close_price:.0f} | "
                        f"4H={'↓' if four_h_down else '↑'}  "
                        f"持有={self._bars_since_entry}k"
                    )
                    return

            elif self.position.is_short:
                # 做空平仓：4H 转多头
                if four_h_up or close_price > fast_1h + atr_val * 0.5:
                    self.position.close()
                    self._bars_since_entry = 999
                    self._bars_since_exit = 0
                    logger.info(
                        f"🟢 平空 @ {close_price:.0f} | "
                        f"4H={'↑' if four_h_up else '↓'}  "
                        f"持有={self._bars_since_entry}k"
                    )
                    return

            # 还在持仓中，不检查开仓
            return

        # ============================================
        # 开仓信号（无持仓时）
        # ============================================

        # ---- 冷却期检查 ----
        if self._bars_since_exit < self.cooldown_bars:
            return

        # ---- 波动过滤 ----
        if expected_move < self.min_expected_move:
            return

        # ---- 趋势强度过滤 ----
        if not trend_strong:
            return  # 均线太粘合，没有明确趋势

        # ---- 置信度计算 ----
        rsi_component = min(abs(rsi_val - 50.0) / 50.0, 1.0)
        move_component = min(expected_move * 10.0, 1.0)
        confidence = round(
            self.base_confidence +
            (self.rsi_weight * rsi_component) +
            (self.volatility_weight * move_component),
            4
        )
        if confidence < self.confidence_threshold:
            return

        # ============================================
        # 做多信号
        # ============================================
        # 条件：4H 多头 + 1H 多头 + RSI 在合理区间 + 趋势够强
        long_conditions = (
            four_h_up and one_h_up and                          # 双时间框架同向
            self.rsi_long_min <= rsi_val <= self.rsi_long_max   # RSI 合理区间
        )

        if long_conditions:
            sl_price = close_price - atr_val * self.atr_stop_mult
            tp_price = close_price + atr_val * self.atr_tp_mult
            fraction = self._calc_fraction(close_price, sl_price)
            if fraction > 0:
                self.buy(size=fraction, sl=sl_price, tp=tp_price)
                self._bars_since_entry = 0
                self._bars_since_exit = 999
                logger.info(
                    f"📈 开多 @ {close_price:.0f} | "
                    f"TP={tp_price:.0f}(+{self.atr_tp_mult}xATR) "
                    f"SL={sl_price:.0f}(-{self.atr_stop_mult}xATR) | "
                    f"RSI={rsi_val:.1f} gap={trend_gap:.2%} 置信度={confidence:.2%}"
                )
                return

        # ============================================
        # 做空信号
        # ============================================
        short_conditions = (
            four_h_down and one_h_down and
            self.rsi_short_min <= rsi_val <= self.rsi_short_max
        )

        if short_conditions:
            sl_price = close_price + atr_val * self.atr_stop_mult
            tp_price = close_price - atr_val * self.atr_tp_mult
            fraction = self._calc_fraction(close_price, sl_price)
            if fraction > 0:
                self.sell(size=fraction, sl=sl_price, tp=tp_price)
                self._bars_since_entry = 0
                self._bars_since_exit = 999
                logger.info(
                    f"📉 开空 @ {close_price:.0f} | "
                    f"TP={tp_price:.0f}(+{self.atr_tp_mult}xATR) "
                    f"SL={sl_price:.0f}(-{self.atr_stop_mult}xATR) | "
                    f"RSI={rsi_val:.1f} gap={trend_gap:.2%} 置信度={confidence:.2%}"
                )

    def _calc_fraction(self, entry_price: float, stop_price: float) -> float:
        """基于风险计算仓位比例（equity 的分数）"""
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0
        fraction = min(self.risk_per_trade / stop_dist_pct, 0.95)
        return max(0.0, fraction)
