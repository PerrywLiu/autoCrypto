"""
15分钟K线专用策略集

基于学术研究和社区回测结果精选的策略：
1. Bollinger Bands 均值回归 — 15m BTC 学术最佳
2. SuperTrend 动量 — BTC 15m 社区验证
3. RSI + EMA 趋势过滤 — 最高胜率和夏普
4. EMA 三线排列 — 最高总收益
5. 波动率突破 — 高波动加密货币特化
6. 量价突破 — 成交量确认的趋势策略
"""

from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# 纯 NumPy 指标实现 (避免 pandas-ta 依赖问题)
# ═══════════════════════════════════════════════════════════

def _ema_np(a, span):
    """指数移动平均"""
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan, dtype=float)
    alpha = 2.0 / (span + 1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _sma_np(a, span):
    """简单移动平均"""
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan, dtype=float)
    r = np.full_like(a, np.nan, dtype=float)
    for i in range(span - 1, len(a)):
        r[i] = np.mean(a[i - span + 1:i + 1])
    return r


def _rsi_np(a, period=14):
    """相对强弱指数 (Wilder's smoothing)"""
    r = np.full_like(a, np.nan, dtype=float)
    if len(a) < period + 1:
        return r
    d = np.diff(a)
    gains = np.maximum(d, 0)
    losses = np.maximum(-d, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    alpha = 1.0 / period
    for i in range(period, len(d)):
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            r[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        else:
            r[i + 1] = 100.0 if avg_gain > 0 else 50.0
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    return np.clip(r, 0, 100)


def _atr_np(h, l, c, period=14):
    """平均真实波幅"""
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < period + 1:
        return r
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_c),
        np.abs(l - prev_c)
    ])
    # SMA 初始化
    r[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def _bollinger_np(a, period=20, nbdev=2.0):
    """
    布林带
    Returns: (upper, middle, lower)
    """
    middle = _sma_np(a, period)
    std = np.full_like(a, np.nan, dtype=float)
    for i in range(period - 1, len(a)):
        std[i] = np.std(a[i - period + 1:i + 1], ddof=1)
    upper = middle + nbdev * std
    lower = middle - nbdev * std
    return upper, middle, lower


def _supertrend_np(h, l, c, period=10, multiplier=3.0):
    """
    SuperTrend 指标
    Returns: (trend, upper_band, lower_band)
    trend: 1=上升趋势, -1=下降趋势
    """
    n = len(c)
    atr = _atr_np(h, l, c, period)
    hl_avg = (h + l) / 2.0

    upper = hl_avg + multiplier * atr
    lower = hl_avg - multiplier * atr
    trend = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        # 上升趋势
        if c[i] > upper[i - 1]:
            trend[i] = 1
        # 下降趋势
        elif c[i] < lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1

        # 调整 bands
        if trend[i] == 1 and lower[i] < lower[i - 1]:
            lower[i] = lower[i - 1]
        if trend[i] == -1 and upper[i] > upper[i - 1]:
            upper[i] = upper[i - 1]

    return trend.astype(float), upper, lower


def _macd_np(a, fast=12, slow=26, signal=9):
    """
    MACD 指标
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = _ema_np(a, fast)
    ema_slow = _ema_np(a, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_np(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx_np(h, l, c, period=14):
    """平均趋向指数"""
    n = len(c)
    r = np.full(n, np.nan, dtype=float)
    if n < period * 2:
        return r
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    up_move = h - np.roll(h, 1)
    down_move = np.roll(l, 1) - l
    up_move[0] = down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    atr_s = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    atr_s[period - 1] = np.mean(tr[:period])
    plus_di[period - 1] = np.mean(plus_dm[:period])
    minus_di[period - 1] = np.mean(minus_dm[:period])

    for i in range(period, n):
        atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
        plus_di[i] = alpha * plus_dm[i] + (1 - alpha) * plus_di[i - 1]
        minus_di[i] = alpha * minus_dm[i] + (1 - alpha) * minus_di[i - 1]

    pdi = np.where(atr_s > 0, 100 * plus_di / atr_s, 0)
    ndi = np.where(atr_s > 0, 100 * minus_di / atr_s, 0)
    ds = pdi + ndi
    dx = np.where(ds > 0, 100 * np.abs(pdi - ndi) / ds, 0)

    r[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    for i in range(period * 2 - 1, n):
        r[i] = alpha * dx[i] + (1 - alpha) * r[i - 1]
    return r


def _rolling_max(a, window):
    """滚动最大值"""
    r = np.full_like(a, np.nan, dtype=float)
    for i in range(window - 1, len(a)):
        r[i] = np.max(a[i - window + 1:i + 1])
    return r


def _rolling_min(a, window):
    """滚动最小值"""
    r = np.full_like(a, np.nan, dtype=float)
    for i in range(window - 1, len(a)):
        r[i] = np.min(a[i - window + 1:i + 1])
    return r


def _volume_sma(v, period=20):
    """成交量均线"""
    return _sma_np(v, period)


# ═══════════════════════════════════════════════════════════
# 策略 1: 布林带均值回归 (学术最佳 15m BTC)
# ═══════════════════════════════════════════════════════════

class BollingerReversion(Strategy):
    """
    布林带均值回归策略

    学术研究发现这是 15 分钟 BTC 最优策略：
    - 价格触及下轨 + RSI < 35 → 做多
    - 价格触及上轨 + RSI > 65 → 做空
    - 止盈: 中轨 (SMA)
    - 止损: 2x ATR 或 布林带外轨

    参数优化范围:
    - bb_period: 15-25
    - bb_std: 1.5-2.5
    - rsi_period: 10-20
    """
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    atr_period = 14
    atr_sl_mult = 2.0
    risk_per_trade = 0.02

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        # 布林带
        self.bb_upper, self.bb_mid, self.bb_lower = \
            self.I(_bollinger_np, c, self.bb_period, self.bb_std,
                   name="BB", overlay=True)

        # RSI
        self.rsi = self.I(_rsi_np, c, self.rsi_period, name="RSI")

        # ATR 用于止损
        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        bb_l = self.bb_lower[-1]
        bb_u = self.bb_upper[-1]
        bb_m = self.bb_mid[-1]

        if any(np.isnan(x) for x in [atr, rsi, bb_l, bb_u, bb_m]) or c <= 0:
            return

        # 已有持仓 → RSI 回到中性平仓
        if self.position:
            if self.position.is_long and rsi > 55:
                self.position.close()
            elif self.position.is_short and rsi < 45:
                self.position.close()
            return

        # 做多: 价格触及下轨 + RSI 超卖
        if c <= bb_l * 1.005 and rsi < self.rsi_oversold:
            sl = c - atr * self.atr_sl_mult
            tp = bb_m  # 止盈到中轨
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空: 价格触及上轨 + RSI 超买
        elif c >= bb_u * 0.995 and rsi > self.rsi_overbought:
            sl = c + atr * self.atr_sl_mult
            tp = bb_m
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略 2: SuperTrend 动量 (BTC 15m 社区验证)
# ═══════════════════════════════════════════════════════════

class SuperTrendMomentum(Strategy):
    """
    SuperTrend 动量策略

    基于社区在 BTC 15m 上验证过的策略：
    - SuperTrend 由下降转上升 → 做多
    - SuperTrend 由上升转下降 → 做空
    - ADX > 20 过滤（趋势市才交易）
    - RSI 确认方向

    参数:
    - st_period: 7-14
    - st_mult: 2.0-4.0
    """
    st_period = 10
    st_multiplier = 3.0
    adx_period = 14
    adx_threshold = 20
    rsi_period = 14
    atr_sl_mult = 1.5
    risk_per_trade = 0.02

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        # SuperTrend
        self.st_trend, self.st_upper, self.st_lower = \
            self.I(_supertrend_np, h, l, c, self.st_period, self.st_multiplier,
                   name="SuperTrend")

        # ADX 趋势过滤
        self.adx = self.I(_adx_np, h, l, c, self.adx_period, name="ADX")

        # RSI 确认
        self.rsi = self.I(_rsi_np, c, self.rsi_period, name="RSI")

        # ATR 动态止损
        self.atr = self.I(_atr_np, h, l, c, 14, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]
        rsi = self.rsi[-1]

        if any(np.isnan(x) for x in [atr, adx, rsi]) or c <= 0:
            return

        st_now = self.st_trend[-1]
        st_prev = self.st_trend[-2]

        # SuperTrend 翻转信号
        st_flip_up = st_prev == -1 and st_now == 1   # 转上升
        st_flip_down = st_prev == 1 and st_now == -1   # 转下降

        # 平仓条件：ST 翻转
        if self.position:
            if self.position.is_long and st_flip_down:
                self.position.close()
            elif self.position.is_short and st_flip_up:
                self.position.close()
            return

        # 趋势过滤：ADX 必须足够强
        if adx < self.adx_threshold:
            return

        # 做多：ST 转上升 + RSI > 45
        if st_flip_up and rsi > 45:
            sl = c - atr * self.atr_sl_mult
            tp = c + atr * 3.0
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空：ST 转下降 + RSI < 55
        elif st_flip_down and rsi < 55:
            sl = c + atr * self.atr_sl_mult
            tp = c - atr * 3.0
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略 3: RSI + EMA 双趋势过滤 (最高胜率)
# ═══════════════════════════════════════════════════════════

class RsiEmaTrend(Strategy):
    """
    RSI + EMA 趋势过滤策略

    社区回测中胜率最高的策略模式 (74%+)：
    - 做多: EMA20 > EMA50 (上升趋势) + RSI 回调到 40-50 + 价格靠近 EMA20
    - 做空: EMA20 < EMA50 (下降趋势) + RSI 反弹到 50-60 + 价格靠近 EMA20
    - 止盈: 2x ATR
    - 止损: 1.5x ATR 或 EMA50

    参数:
    - ema_fast: 12-25
    - ema_slow: 40-60
    """
    ema_fast = 20
    ema_slow = 50
    rsi_period = 14
    rsi_entry_low = 45    # 做多 RSI 上限
    rsi_entry_high = 55   # 做空 RSI 下限
    atr_period = 14
    atr_sl_mult = 1.5
    atr_tp_mult = 2.5
    risk_per_trade = 0.02

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.ema_f = self.I(_ema_np, c, self.ema_fast, overlay=True, name="EMAfast")
        self.ema_s = self.I(_ema_np, c, self.ema_slow, overlay=True, name="EMAslow")
        self.rsi = self.I(_rsi_np, c, self.rsi_period, name="RSI")
        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        ema_f = self.ema_f[-1]
        ema_s = self.ema_s[-1]

        if any(np.isnan(x) for x in [atr, rsi, ema_f, ema_s]) or c <= 0:
            return

        bull_trend = ema_f > ema_s
        bear_trend = ema_f < ema_s
        near_ema = abs(c - ema_f) <= atr * 1.2  # 价格靠近快线

        # 平仓: 趋势反转
        if self.position:
            if self.position.is_long and not bull_trend:
                self.position.close()
            elif self.position.is_short and not bear_trend:
                self.position.close()
            return

        # 做多: 上升趋势 + RSI 回调 + 靠近 EMA
        if bull_trend and 30 < rsi < self.rsi_entry_low and near_ema:
            sl = max(c - atr * self.atr_sl_mult, ema_s)  # 止损不低于 EMA50
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (abs(c - sl) / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空: 下降趋势 + RSI 反弹 + 靠近 EMA
        elif bear_trend and self.rsi_entry_high < rsi < 70 and near_ema:
            sl = min(c + atr * self.atr_sl_mult, ema_s)
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (abs(c - sl) / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略 4: EMA 三线排列 + MACD 确认
# ═══════════════════════════════════════════════════════════

class EmaRibbonMacd(Strategy):
    """
    EMA 三线排列 + MACD 确认策略

    趋势最明确时才入场，追求高盈亏比：
    - 三线多头排列 (EMA10>20>50) + MACD > 0 + 回调到 EMA10 → 做多
    - 三线空头排列 (EMA10<20<50) + MACD < 0 + 反弹到 EMA10 → 做空
    """
    ema1 = 10
    ema2 = 20
    ema3 = 50
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    atr_sl_mult = 1.5
    atr_tp_mult = 3.0
    risk_per_trade = 0.02

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.ema10 = self.I(_ema_np, c, self.ema1, overlay=True, name="EMA10")
        self.ema20 = self.I(_ema_np, c, self.ema2, overlay=True, name="EMA20")
        self.ema50 = self.I(_ema_np, c, self.ema3, overlay=True, name="EMA50")

        self.macd, self.macd_sig, self.macd_hist = \
            self.I(_macd_np, c, self.macd_fast, self.macd_slow, self.macd_signal,
                   name="MACD")

        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        atr = self.atr[-1]
        e10, e20, e50 = self.ema10[-1], self.ema20[-1], self.ema50[-1]
        macd_val = self.macd[-1]
        macd_prev = self.macd[-2]

        if any(np.isnan(x) for x in [atr, e10, e20, e50, macd_val]) or c <= 0:
            return

        bull_align = e10 > e20 > e50  # 多头排列
        bear_align = e10 < e20 < e50  # 空头排列
        near_ema10 = abs(c - e10) <= atr * 0.8
        macd_rising = macd_val > macd_prev
        macd_falling = macd_val < macd_prev

        # 平仓：EMA 排列破坏
        if self.position:
            if self.position.is_long and not bull_align:
                self.position.close()
            elif self.position.is_short and not bear_align:
                self.position.close()
            return

        # 做多：三线多头 + MACD 回升 + 回调到 EMA10
        if bull_align and macd_rising and near_ema10 and macd_val > 0:
            sl = e20  # 止损到 EMA20
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (abs(c - sl) / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空：三线空头 + MACD 回落 + 反弹到 EMA10
        elif bear_align and macd_falling and near_ema10 and macd_val < 0:
            sl = e20
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (abs(c - sl) / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略 5: 波动率突破策略 (加密货币特化)
# ═══════════════════════════════════════════════════════════

class VolatilityBreakout(Strategy):
    """
    波动率突破策略

    利用加密货币高波动特性：
    - 当 ATR 扩张超过阈值 → 波动率爆发
    - 价格突破前 N 根 K 线高低点 + 波动确认
    - 快速止盈，适合 15m 快进快出
    """
    lookback = 20         # 突破周期
    vol_expand = 1.5      # ATR 扩张倍数（当前 ATR > N 期均值的倍数）
    atr_period = 14
    atr_ma_period = 50    # ATR 长期均值
    atr_sl_mult = 1.5
    atr_tp_mult = 2.0
    risk_per_trade = 0.02
    cooldown = 5          # 冷却期 K 线数

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.highest = self.I(_rolling_max, h, self.lookback, name="HH")
        self.lowest = self.I(_rolling_min, l, self.lookback, name="LL")
        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")
        self.atr_ma = self.I(_ema_np, self.atr, self.atr_ma_period, name="ATR_MA")

        self._bars_since_trade = 999

    def next(self):
        c = self.data.Close[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        highest = self.highest[-1]
        lowest = self.lowest[-1]

        self._bars_since_trade += 1

        if any(np.isnan(x) for x in [atr, atr_ma, highest, lowest]) or c <= 0:
            return

        # 波动率扩张检查
        vol_expanding = atr_ma > 0 and atr > atr_ma * self.vol_expand

        # 持仓管理
        if self.position:
            if self.position.is_long and c < self.data.Low[-2]:
                self.position.close()
                self._bars_since_trade = 0
            elif self.position.is_short and c > self.data.High[-2]:
                self.position.close()
                self._bars_since_trade = 0
            return

        if self._bars_since_trade < self.cooldown:
            return

        # 做多：突破最高点 + 波动扩张
        if h >= highest * 0.999 and vol_expanding:
            sl = c - atr * self.atr_sl_mult
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)
                self._bars_since_trade = 999

        # 做空：跌破最低点 + 波动扩张
        elif l <= lowest * 1.001 and vol_expanding:
            sl = c + atr * self.atr_sl_mult
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)
                self._bars_since_trade = 999


# ═══════════════════════════════════════════════════════════
# 策略 6: 量价突破策略
# ═══════════════════════════════════════════════════════════

class VolumeBreakout(Strategy):
    """
    量价突破策略

    放量突破前高/前低，成交量确认趋势：
    - 价格突破 N 期高点 + 成交量 > 均量的 1.5 倍 → 做多
    - 价格跌破 N 期低点 + 成交量 > 均量的 1.5 倍 → 做空
    - 追踪止损
    """
    lookback = 20
    vol_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_sl_mult = 2.0
    atr_tp_mult = 3.0
    risk_per_trade = 0.02

    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume

        self.highest = self.I(_rolling_max, h, self.lookback, name="HH")
        self.lowest = self.I(_rolling_min, l, self.lookback, name="LL")
        self.vol_ma = self.I(_volume_sma, v, self.vol_period, name="VolMA")
        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        v = self.data.Volume[-1]
        atr = self.atr[-1]
        vol_ma = self.vol_ma[-1]
        highest = self.highest[-1]
        lowest = self.lowest[-1]

        if any(np.isnan(x) for x in [atr, vol_ma, highest, lowest]) or c <= 0:
            return

        vol_surge = vol_ma > 0 and v > vol_ma * self.vol_mult

        # 平仓: 反向突破
        if self.position:
            if self.position.is_long and l < self.data.Low[-2]:
                self.position.close()
            elif self.position.is_short and h > self.data.High[-2]:
                self.position.close()
            return

        # 做多: 放量突破前高
        if h > highest and vol_surge:
            sl = c - atr * self.atr_sl_mult
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空: 放量跌破前低
        elif l < lowest and vol_surge:
            sl = c + atr * self.atr_sl_mult
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略 7: 多时间框架确认 (15m + 1h/4h)
# ═══════════════════════════════════════════════════════════

class MultiTfConfirmation(Strategy):
    """
    多时间框架确认策略

    在 15 分钟线上使用 1H/4H 级别的趋势方向进行过滤：
    - 1H EMA 排列决定大方向 (只做顺势)
    - 15m RSI 回调给出入场时机
    - 多级止盈

    数据必须有 1h_Close, 4h_Close 等高时间框架列 (通过 DataProcessor.merge_timeframes 合并)
    """
    ema_fast = 20
    ema_slow = 50
    rsi_period = 14
    rsi_low = 40
    rsi_high = 60
    atr_period = 14
    atr_sl_mult = 1.5
    atr_tp_mult = 2.5
    risk_per_trade = 0.02

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        # 15m 指标
        self.ema15_f = self.I(_ema_np, c, self.ema_fast, overlay=True, name="15m_EMA20")
        self.ema15_s = self.I(_ema_np, c, self.ema_slow, overlay=True, name="15m_EMA50")
        self.rsi = self.I(_rsi_np, c, self.rsi_period, name="RSI")
        self.atr = self.I(_atr_np, h, l, c, self.atr_period, name="ATR")

        # 检查是否有高时间框架列
        self._has_1h = "1h_Close" in self.data.df.columns
        self._has_4h = "4h_Close" in self.data.df.columns

        if self._has_1h:
            self.ema1h = self.I(_ema_np, self.data.df["1h_Close"], 50,
                                overlay=False, name="1h_EMA50")
        if self._has_4h:
            self.ema4h = self.I(_ema_np, self.data.df["4h_Close"], 50,
                                overlay=False, name="4h_EMA50")

    def next(self):
        c = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        ema15_f = self.ema15_f[-1]
        ema15_s = self.ema15_s[-1]

        if any(np.isnan(x) for x in [atr, rsi, ema15_f, ema15_s]) or c <= 0:
            return

        # 高时间框架趋势确认
        ht_bull = True   # 默认允许做多
        ht_bear = True   # 默认允许做空

        if self._has_1h:
            try:
                c_1h = self.data.df["1h_Close"].iloc[-1]
                ema1h_v = self.ema1h[-1]
                if not np.isnan(ema1h_v):
                    ht_bull = c_1h > ema1h_v  # 1H 价格在 EMA50 之上才做多
                    ht_bear = c_1h < ema1h_v  # 1H 价格在 EMA50 之下才做空
            except (IndexError, KeyError):
                pass

        if self._has_4h:
            try:
                c_4h = self.data.df["4h_Close"].iloc[-1]
                ema4h_v = self.ema4h[-1]
                if not np.isnan(ema4h_v):
                    ht_bull = ht_bull and (c_4h > ema4h_v)
                    ht_bear = ht_bear and (c_4h < ema4h_v)
            except (IndexError, KeyError):
                pass

        # 15m 趋势方向
        m15_bull = ema15_f > ema15_s
        m15_bear = ema15_f < ema15_s

        # 平仓
        if self.position:
            if self.position.is_long and not m15_bull:
                self.position.close()
            elif self.position.is_short and not m15_bear:
                self.position.close()
            return

        # 做多：HT上升 + 15m上升 + RSI回调
        if ht_bull and m15_bull and rsi < self.rsi_low:
            sl = c - atr * self.atr_sl_mult
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)

        # 做空：HT下降 + 15m下降 + RSI反弹
        elif ht_bear and m15_bear and rsi > self.rsi_high:
            sl = c + atr * self.atr_sl_mult
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════
# 策略注册
# ═══════════════════════════════════════════════════════════

FIFTEEN_MIN_STRATEGIES = {
    "bollinger_reversion": BollingerReversion,
    "supertrend_momentum": SuperTrendMomentum,
    "rsi_ema_trend": RsiEmaTrend,
    "ema_ribbon_macd": EmaRibbonMacd,
    "volatility_breakout": VolatilityBreakout,
    "volume_breakout": VolumeBreakout,
    "multi_tf_confirm": MultiTfConfirmation,
}
