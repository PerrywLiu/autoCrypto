"""
15分钟K线专用策略集 v2 — 修复版

策略：
1. Bollinger Bands 均值回归 (纯做多)
2. SuperTrend 趋势跟踪
3. RSI + EMA 回调入场
4. MACD 零轴金叉
5. 波动率扩张突破
6. 多时间框架顺势
"""

from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np


# ═══════════════════════════════════════════
# 纯 NumPy 指标
# ═══════════════════════════════════════════

def _ema(a, span):
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _sma(a, span):
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan)
    r = np.full_like(a, np.nan, dtype=float)
    for i in range(span - 1, len(a)):
        r[i] = np.mean(a[i - span + 1:i + 1])
    return r


def _rsi(a, period=14):
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
            r[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            r[i + 1] = 100.0 if avg_gain > 0 else 50.0
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    return np.clip(r, 0, 100)


def _atr(h, l, c, period=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < period + 1:
        return r
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    r[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def _adx(h, l, c, period=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n < period * 2: return r
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    up_move = h - np.roll(h, 1); down_move = np.roll(l, 1) - l
    up_move[0] = down_move[0] = 0
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    alpha = 1.0 / period
    atr_s = np.full(n, np.nan); pdi = np.full(n, np.nan); ndi = np.full(n, np.nan)
    atr_s[period - 1] = np.mean(tr[:period])
    pdi[period - 1] = np.mean(pdm[:period])
    ndi[period - 1] = np.mean(ndm[:period])
    for i in range(period, n):
        atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
        pdi[i] = alpha * pdm[i] + (1 - alpha) * pdi[i - 1]
        ndi[i] = alpha * ndm[i] + (1 - alpha) * ndi[i - 1]
    _pdi = np.where(atr_s > 0, 100 * pdi / atr_s, 0)
    _ndi = np.where(atr_s > 0, 100 * ndi / atr_s, 0)
    ds = _pdi + _ndi
    dx = np.where(ds > 0, 100 * np.abs(_pdi - _ndi) / ds, 0)
    r[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    for i in range(period * 2 - 1, n):
        r[i] = alpha * dx[i] + (1 - alpha) * r[i - 1]
    return r


def _bollinger(a, period=20, nbdev=2.0):
    mid = _sma(a, period)
    std = np.full_like(a, np.nan)
    for i in range(period - 1, len(a)):
        std[i] = np.std(a[i - period + 1:i + 1], ddof=1)
    return mid + nbdev * std, mid, mid - nbdev * std


def _supertrend(h, l, c, period=10, mult=3.0):
    n = len(c)
    atr_v = _atr(h, l, c, period)
    hl_avg = (h + l) / 2.0
    upper = hl_avg + mult * atr_v
    lower = hl_avg - mult * atr_v
    trend = np.zeros(n)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]): continue
        if c[i] > upper[i - 1]: trend[i] = 1
        elif c[i] < lower[i - 1]: trend[i] = -1
        else: trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1
        if trend[i] == 1 and lower[i] < lower[i - 1]: lower[i] = lower[i - 1]
        if trend[i] == -1 and upper[i] > upper[i - 1]: upper[i] = upper[i - 1]
    return trend, upper, lower


def _macd(a, fast=12, slow=26, signal=9):
    ef = _ema(a, fast); es = _ema(a, slow)
    macd_line = ef - es
    sig_line = _ema(macd_line, signal)
    return macd_line, sig_line, macd_line - sig_line


def _rolling_high(a, window):
    r = np.full_like(a, np.nan)
    for i in range(window - 1, len(a)):
        r[i] = np.max(a[i - window + 1:i + 1])
    return r


def _rolling_low(a, window):
    r = np.full_like(a, np.nan)
    for i in range(window - 1, len(a)):
        r[i] = np.min(a[i - window + 1:i + 1])
    return r


# ═══════════════════════════════════════════
# 策略 1: 布林带均值回归 (纯做多，学术最优)
# ═══════════════════════════════════════════

class BollingerReversionV2(Strategy):
    """
    布林带均值回归 - 只做多 (15m BTC 最优)

    价格触及下轨 + RSI超卖 → 买入
    价格回到中轨或RSI超买 → 卖出
    """
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.bb_u, self.bb_m, self.bb_l = self.I(_bollinger, c, self.bb_period, self.bb_std,
                                                   overlay=True, name="BB")
        self.rsi = self.I(_rsi, c, self.rsi_period, name="RSI")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        rsi_v = self.rsi[-1]
        bb_l = self.bb_l[-1]
        bb_m = self.bb_m[-1]
        atr_v = self.atr[-1]

        if any(np.isnan(x) for x in [rsi_v, bb_l, bb_m, atr_v]) or c <= 0:
            return

        if self.position:
            # 止盈：回到中轨上方 或 RSI > 65
            if c >= bb_m or rsi_v > self.rsi_overbought * 0.9:
                self.position.close()
            # 止损：跌破下轨下方 1 ATR
            elif c < bb_l - atr_v * 0.5:
                self.position.close()
            return

        # 买入：价格触及/跌破下轨 + RSI 超卖
        if c <= bb_l * 1.003 and rsi_v < self.rsi_oversold:
            # 用总资金100%买入
            self.buy()


# ═══════════════════════════════════════════
# 策略 2: SuperTrend 趋势跟踪
# ═══════════════════════════════════════════

class SuperTrendV2(Strategy):
    """
    SuperTrend 趋势跟踪

    ST 由空转多 + ADX 确认 → 买入
    ST 由多转空 → 平仓/做空
    """
    st_period = 10
    st_mult = 3.0
    adx_period = 14
    adx_min = 22

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.st_trend, self.st_u, self.st_l = self.I(
            _supertrend, h, l, c, self.st_period, self.st_mult, name="ST")
        self.adx = self.I(_adx, h, l, c, self.adx_period, name="ADX")

    def next(self):
        c = self.data.Close[-1]
        adx_v = self.adx[-1]
        st_now = self.st_trend[-1]
        st_prev = self.st_trend[-2]

        if any(np.isnan(x) for x in [adx_v, st_now, st_prev]) or c <= 0:
            return

        # 信号：ST 翻转
        bull_signal = (st_prev <= 0 and st_now > 0)
        bear_signal = (st_prev > 0 and st_now <= 0)

        if self.position:
            if self.position.is_long and bear_signal:
                self.position.close()
            elif self.position.is_short and bull_signal:
                self.position.close()
            return

        # 入场：趋势确认 (ADX > threshold)
        if bull_signal and adx_v > self.adx_min:
            self.buy()
        elif bear_signal and adx_v > self.adx_min:
            self.sell()


# ═══════════════════════════════════════════
# 策略 3: RSI + EMA 回调入场
# ═══════════════════════════════════════════

class RsiEmaPullback(Strategy):
    """
    RSI + EMA 回调入场

    上升趋势中 RSI 回调到低位 + 价格靠近 EMA → 买入
    下降趋势中 RSI 反弹到高位 + 价格靠近 EMA → 做空
    """
    ema_fast = 20
    ema_slow = 50
    rsi_period = 14
    rsi_buy_zone = 40   # RSI < 此值考虑做多
    rsi_sell_zone = 60  # RSI > 此值考虑做空

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema_f = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA20")
        self.ema_s = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA50")
        self.rsi = self.I(_rsi, c, self.rsi_period, name="RSI")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        ema_f = self.ema_f[-1]; ema_s = self.ema_s[-1]
        rsi_v = self.rsi[-1]; atr_v = self.atr[-1]

        if any(np.isnan(x) for x in [ema_f, ema_s, rsi_v, atr_v]) or c <= 0:
            return

        bull_trend = ema_f > ema_s
        bear_trend = ema_f < ema_s
        near_ema = abs(c - ema_f) < atr_v * 1.5

        if self.position:
            # 趋势反转平仓
            if self.position.is_long and ema_f < ema_s:
                self.position.close()
            elif self.position.is_short and ema_f > ema_s:
                self.position.close()
            return

        # 做多：上升趋势 + RSI 回调 + 靠近快线
        if bull_trend and rsi_v < self.rsi_buy_zone and near_ema:
            self.buy()

        # 做空：下降趋势 + RSI 反弹 + 靠近快线
        elif bear_trend and rsi_v > self.rsi_sell_zone and near_ema:
            self.sell()


# ═══════════════════════════════════════════
# 策略 4: MACD 零轴策略
# ═══════════════════════════════════════════

class MacdZeroCross(Strategy):
    """
    MACD 零轴金叉死叉

    MACD 在零轴上方金叉 → 做多
    MACD 在零轴下方死叉 → 做空
    信号线反穿 → 平仓
    """
    macd_fast = 12
    macd_slow = 26
    macd_sig = 9

    def init(self):
        c = self.data.Close
        self.macd_l, self.macd_s, self.macd_h = self.I(
            _macd, c, self.macd_fast, self.macd_slow, self.macd_sig, name="MACD")

    def next(self):
        c = self.data.Close[-1]
        if c <= 0: return

        ml = self.macd_l[-1]; ms = self.macd_s[-1]
        ml_p = self.macd_l[-2]; ms_p = self.macd_s[-2]

        if any(np.isnan(x) for x in [ml, ms, ml_p, ms_p]):
            return

        # 金叉/死叉
        golden = ml_p <= ms_p and ml > ms
        dead = ml_p >= ms_p and ml < ms

        if self.position:
            if self.position.is_long and dead:
                self.position.close()
            elif self.position.is_short and golden:
                self.position.close()
            return

        # 金叉在零轴上方 → 做多
        if golden and ml > 0:
            self.buy()
        # 死叉在零轴下方 → 做空
        elif dead and ml < 0:
            self.sell()


# ═══════════════════════════════════════════
# 策略 5: 波动率扩张突破
# ═══════════════════════════════════════════

class VolatilityExpansion(Strategy):
    """
    波动率扩张突破

    ATR 扩张到均值的 1.5 倍 + 价格突破近期高低点 → 入场
    持有 N 根K线或反向信号 → 出场
    """
    lookback = 20
    atr_period = 14
    atr_ma_period = 100
    vol_ratio = 1.4
    max_hold = 48  # 最大持有 12 小时 (48*15min)

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.hh = self.I(_rolling_high, h, self.lookback, name="HH")
        self.ll = self.I(_rolling_low, l, self.lookback, name="LL")
        self.atr = self.I(_atr, h, l, c, self.atr_period, name="ATR")
        self.atr_ma = self.I(_ema, self.atr, self.atr_ma_period, name="ATR_MA")
        self._entry_bar = 0

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        atr_v = self.atr[-1]; atr_ma_v = self.atr_ma[-1]
        hh = self.hh[-1]; ll = self.ll[-1]

        if any(np.isnan(x) for x in [atr_v, atr_ma_v, hh, ll]) or c <= 0:
            return

        # 波动率是否扩张
        vol_expand = atr_ma_v > 0 and atr_v > atr_ma_v * self.vol_ratio
        bars_held = len(self.data) - self._entry_bar if self._entry_bar else 0

        if self.position:
            # 持有超时
            if bars_held >= self.max_hold:
                self.position.close()
            # 反向突破
            elif self.position.is_long and c < ll * 1.005:
                self.position.close()
            elif self.position.is_short and c > hh * 0.995:
                self.position.close()
            return

        # 入场：波动扩张 + 突破
        if vol_expand and h > hh:
            self.buy()
            self._entry_bar = len(self.data)
        elif vol_expand and l < ll:
            self.sell()
            self._entry_bar = len(self.data)


# ═══════════════════════════════════════════
# 策略 6: 多时间框架顺势
# ═══════════════════════════════════════════

class MultiTfTrend(Strategy):
    """
    多时框顺势 (15m + 1h + 4h)

    只在 1h 和 4h 趋势一致时顺势交易。
    数据需含 '1h_Close' 和 '4h_Close' 列。
    """
    ema_period = 50
    rsi_period = 14
    rsi_low = 45
    rsi_high = 55

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema15_50 = self.I(_ema, c, self.ema_period, overlay=True, name="EMA50_15m")
        self.rsi = self.I(_rsi, c, self.rsi_period, name="RSI")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

        # 高时框 EMA
        df = self.data.df
        if "1h_Close" in df.columns:
            self.ema1h_50 = self.I(_ema, df["1h_Close"], self.ema_period, name="EMA50_1h")
            self._has_1h = True
        else:
            self._has_1h = False

        if "4h_Close" in df.columns:
            self.ema4h_50 = self.I(_ema, df["4h_Close"], self.ema_period, name="EMA50_4h")
            self._has_4h = True
        else:
            self._has_4h = False

    def next(self):
        c = self.data.Close[-1]
        ema15 = self.ema15_50[-1]
        rsi_v = self.rsi[-1]
        atr_v = self.atr[-1]

        if any(np.isnan(x) for x in [ema15, rsi_v, atr_v]) or c <= 0:
            return

        # 高时框方向
        ht_bull = True; ht_bear = True
        if self._has_1h:
            try:
                c1h = self.data.df["1h_Close"].iloc[-1]
                e1h = self.ema1h_50[-1]
                if not np.isnan(e1h):
                    ht_bull = c1h > e1h
                    ht_bear = c1h < e1h
            except: pass
        if self._has_4h:
            try:
                c4h = self.data.df["4h_Close"].iloc[-1]
                e4h = self.ema4h_50[-1]
                if not np.isnan(e4h):
                    ht_bull = ht_bull and c4h > e4h
                    ht_bear = ht_bear and c4h < e4h
            except: pass

        m15_bull = c > ema15
        m15_bear = c < ema15

        if self.position:
            if self.position.is_long and m15_bear:
                self.position.close()
            elif self.position.is_short and m15_bull:
                self.position.close()
            return

        # 做多：1h+4h上升 + 15m上升 + RSI弱势(回调买入)
        if ht_bull and m15_bull and rsi_v < self.rsi_low:
            self.buy()
        # 做空：1h+4h下降 + 15m下降 + RSI强势(反弹卖出)
        elif ht_bear and m15_bear and rsi_v > self.rsi_high:
            self.sell()


# ═══════════════════════════════════════════
# 策略 7: EMA 金叉 + ADX 过滤
# ═══════════════════════════════════════════

class EmaCrossAdx(Strategy):
    """
    EMA 金叉 + ADX 趋势过滤

    EMA10 上穿 EMA30 且 ADX > 20 → 做多
    EMA10 下穿 EMA30 且 ADX > 20 → 做空
    """
    ema_short = 10
    ema_long = 30
    adx_period = 14
    adx_min = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema_s = self.I(_ema, c, self.ema_short, overlay=True, name="EMA10")
        self.ema_l = self.I(_ema, c, self.ema_long, overlay=True, name="EMA30")
        self.adx = self.I(_adx, h, l, c, self.adx_period, name="ADX")

    def next(self):
        c = self.data.Close[-1]
        if c <= 0: return

        ema_s = self.ema_s[-1]; ema_l = self.ema_l[-1]
        ema_sp = self.ema_s[-2]; ema_lp = self.ema_l[-2]
        adx_v = self.adx[-1]

        if any(np.isnan(x) for x in [ema_s, ema_l, ema_sp, ema_lp, adx_v]):
            return

        cross_up = ema_sp <= ema_lp and ema_s > ema_l
        cross_down = ema_sp >= ema_lp and ema_s < ema_l

        if self.position:
            if self.position.is_long and cross_down:
                self.position.close()
            elif self.position.is_short and cross_up:
                self.position.close()
            return

        if cross_up and adx_v > self.adx_min:
            self.buy()
        elif cross_down and adx_v > self.adx_min:
            self.sell()


# ═══════════════════════════════════════════
# 策略 8: 结合型：布林+RSI 双向
# ═══════════════════════════════════════════

class BbRsiDual(Strategy):
    """
    布林带 + RSI 双向策略

    做多: 价格 < 下轨 + RSI < 35
    做空: 价格 > 上轨 + RSI > 65
    平仓: 回到中轨
    """
    bb_p = 20; bb_std = 2.0
    rsi_p = 14

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.bb_u, self.bb_m, self.bb_l = self.I(_bollinger, c, self.bb_p, self.bb_std, overlay=True, name="BB")
        self.rsi = self.I(_rsi, c, self.rsi_p, name="RSI")

    def next(self):
        c = self.data.Close[-1]
        rsi_v = self.rsi[-1]
        bb_u = self.bb_u[-1]; bb_m = self.bb_m[-1]; bb_l = self.bb_l[-1]

        if any(np.isnan(x) for x in [rsi_v, bb_u, bb_m, bb_l]) or c <= 0:
            return

        if self.position:
            crossed_mid_up = self.data.Close[-2] <= bb_m and c > bb_m
            crossed_mid_down = self.data.Close[-2] >= bb_m and c < bb_m

            if self.position.is_long and (c >= bb_m or crossed_mid_up):
                self.position.close()
            elif self.position.is_short and (c <= bb_m or crossed_mid_down):
                self.position.close()
            return

        # 做多
        if c <= bb_l and rsi_v < 35:
            self.buy()
        # 做空
        elif c >= bb_u and rsi_v > 65:
            self.sell()


# ═══════════════════════════════════════════
# 注册
# ═══════════════════════════════════════════

ALL_STRATEGIES = {
    "1_bollinger_reversion": BollingerReversionV2,
    "2_supertrend": SuperTrendV2,
    "3_rsi_ema_pullback": RsiEmaPullback,
    "4_macd_zero_cross": MacdZeroCross,
    "5_vol_expansion": VolatilityExpansion,
    "6_mtf_trend": MultiTfTrend,
    "7_ema_cross_adx": EmaCrossAdx,
    "8_bb_rsi_dual": BbRsiDual,
}
