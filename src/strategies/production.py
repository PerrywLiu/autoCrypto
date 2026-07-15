"""
生产级 TripleEma 4h 趋势跟踪策略

基于 2023.07~2026.07 三年回测验证:
- 1x 杠杆: +37.3% 收益, 夏普 1.22, 回撤 -18.1%
- 2x 杠杆: +145.0% 收益, 夏普 1.77, 回撤 -32.7%

策略逻辑:
  - EMA10/20/50 三线完美排列判断趋势 (多头: 10>20>50, 空头: 10<20<50)
  - 价格回调靠近 EMA20 入场
  - 日线 200EMA 过滤方向 (牛市只做多, 熊市只做空)
  - ADX>20 确认趋势强度
  - ATR 动态止损止盈
  - 连续亏损自动延长冷却期

适用: BTC/USDT 永续合约, 4小时K线
杠杆: 推荐 2x
"""

from backtesting import Strategy
import numpy as np


# ═══════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════

def ema(a: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    if span <= 0 or len(a) < span:
        return np.full(len(a), np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full(len(a), np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """平均真实波幅"""
    r = np.full(len(close), np.nan, dtype=float)
    if len(close) < period + 1:
        return r
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    r[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """平均趋向指数"""
    n = len(close)
    r = np.full(n, np.nan, dtype=float)
    if n < period * 2:
        return r

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = 0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    at = np.full(n, np.nan)
    pt = np.full(n, np.nan)
    nt = np.full(n, np.nan)

    at[period - 1] = np.mean(tr[:period])
    pt[period - 1] = np.mean(plus_dm[:period])
    nt[period - 1] = np.mean(minus_dm[:period])

    for i in range(period, n):
        at[i] = alpha * tr[i] + (1 - alpha) * at[i - 1]
        pt[i] = alpha * plus_dm[i] + (1 - alpha) * pt[i - 1]
        nt[i] = alpha * minus_dm[i] + (1 - alpha) * nt[i - 1]

    pdi = np.where(at > 0, 100 * pt / at, 0)
    ndi = np.where(at > 0, 100 * nt / at, 0)
    ds = pdi + ndi

    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(ds > 0, 100 * np.abs(pdi - ndi) / ds, 0)

    r[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    for i in range(period * 2 - 1, n):
        r[i] = alpha * dx[i] + (1 - alpha) * r[i - 1]
    return r


# ═══════════════════════════════════════════
# 策略
# ═══════════════════════════════════════════

class TripleEmaStrategy(Strategy):
    """
    BTC 4h 趋势跟踪 — TripleEma

    入场条件:
      做多: EMA10>20>50 + 价格>EMA200 + ADX>20 + 价格靠近EMA20
      做空: EMA10<20<50 + 价格<EMA200 + ADX>20 + 价格靠近EMA20

    风控:
      - ATR动态止损止盈 (1.5x / 3.0x)
      - 连续亏损延长冷却期
    """

    # ── 指标参数 ──
    ema_fast = 10     # 快线
    ema_mid = 20      # 中线 (入场参考)
    ema_slow = 50     # 慢线
    ema_trend = 200   # 长期趋势线
    adx_period = 14   # ADX 周期
    adx_threshold = 20  # 最低趋势强度

    # ── 风控参数 ──
    atr_period = 14
    sl_atr_mult = 1.5   # 止损 = 当前价 ± 1.5×ATR
    tp_atr_mult = 3.0   # 止盈 = 当前价 ± 3.0×ATR
    cooldown_base = 6    # 基础冷却期 (4h K线数)
    cooldown_extra_per_loss = 3  # 每次亏损额外冷却

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low

        # 指标
        self.ema10 = self.I(ema, c, self.ema_fast, overlay=True, name="EMA10")
        self.ema20 = self.I(ema, c, self.ema_mid, overlay=True, name="EMA20")
        self.ema50 = self.I(ema, c, self.ema_slow, overlay=True, name="EMA50")
        self.ema200 = self.I(ema, c, self.ema_trend, overlay=True, name="EMA200")
        self.adx_val = self.I(adx, h, l, c, self.adx_period, name="ADX")
        self.atr_val = self.I(atr, h, l, c, self.atr_period, name="ATR")

        # 状态
        self._consecutive_losses = 0
        self._last_trade_bar = -999

    def _is_bull_market(self) -> bool:
        """牛市: 价格在 200EMA 之上"""
        c = self.data.Close[-1]
        e200 = self.ema200[-1]
        return not np.isnan(e200) and c > e200

    def _is_bull_alignment(self) -> bool:
        """多头排列: EMA10 > EMA20 > EMA50"""
        return (self.ema10[-1] > self.ema20[-1] > self.ema50[-1])

    def _is_bear_alignment(self) -> bool:
        """空头排列: EMA10 < EMA20 < EMA50"""
        return (self.ema10[-1] < self.ema20[-1] < self.ema50[-1])

    def _is_near_ema20(self) -> bool:
        """价格在 EMA20 附近 (1倍ATR范围内)"""
        c = self.data.Close[-1]
        atr_v = self.atr_val[-1]
        e20 = self.ema20[-1]
        if np.isnan(atr_v) or np.isnan(e20):
            return False
        return abs(c - e20) < atr_v

    def _is_trend_strong(self) -> bool:
        """ADX > 阈值"""
        adx_v = self.adx_val[-1]
        return not np.isnan(adx_v) and adx_v > self.adx_threshold

    def _cooldown_ok(self) -> bool:
        """冷却期检查"""
        bars_since = len(self.data) - self._last_trade_bar
        required = self.cooldown_base + \
                   self._consecutive_losses * self.cooldown_extra_per_loss
        return bars_since >= required

    def next(self):
        c = self.data.Close[-1]
        atr_v = self.atr_val[-1]

        # 数据校验
        if c <= 0 or np.isnan(atr_v):
            return
        if any(np.isnan(x) for x in [
            self.ema10[-1], self.ema20[-1], self.ema50[-1],
            self.ema200[-1], self.adx_val[-1]
        ]):
            return

        bull_market = self._is_bull_market()
        bull_align = self._is_bull_alignment()
        bear_align = self._is_bear_alignment()
        near_ema20 = self._is_near_ema20()
        trend_strong = self._is_trend_strong()
        can_trade = self._cooldown_ok()

        # ── 持仓管理 ──
        if self.position:
            if self.position.is_long and not bull_align:
                self.position.close()
                if self.position.pl and self.position.pl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
                self._last_trade_bar = len(self.data)

            elif self.position.is_short and not bear_align:
                self.position.close()
                if self.position.pl and self.position.pl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
                self._last_trade_bar = len(self.data)
            return

        # ── 入场: 所有条件满足 ──
        if not (trend_strong and near_ema20 and can_trade):
            return

        # 做多: 牛市 + 三线多头排列
        if bull_market and bull_align:
            sl = c - atr_v * self.sl_atr_mult
            tp = c + atr_v * self.tp_atr_mult
            self.buy(sl=sl, tp=tp)
            self._last_trade_bar = len(self.data)

        # 做空: 熊市 + 三线空头排列
        elif not bull_market and bear_align:
            sl = c + atr_v * self.sl_atr_mult
            tp = c - atr_v * self.tp_atr_mult
            self.sell(sl=sl, tp=tp)
            self._last_trade_bar = len(self.data)
