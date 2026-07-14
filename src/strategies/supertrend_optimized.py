"""
SuperTrend 优化版 — 带市场状态过滤

基于回测发现：SuperTrend 在牛市赚 +22.3%，但在熊市/震荡市亏损。
优化方向：
1. 用 200EMA 判断长期趋势方向（牛市只做多，熊市只做空）
2. 优化 ST 参数适配 15m (周期 7, 乘数 2.5)
3. 动态止损: 1.5x ATR 止损, 3x ATR 止盈
4. 连续亏损后冷却
5. 1h 趋势确认
"""

from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np


# ═══════════════════════
# 指标
# ═══════════════════════

def _ema(a, span):
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _atr(h, l, c, p=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < p + 1: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    r[p - 1] = np.mean(tr[:p]); a_ = 1.0 / p
    for i in range(p, len(tr)): r[i] = a_ * tr[i] + (1 - a_) * r[i - 1]
    return r


def _supertrend(h, l, c, period=7, mult=2.5):
    """SuperTrend 指标"""
    atr_v = _atr(h, l, c, period)
    hl_avg = (h + l) / 2.0
    upper = hl_avg + mult * atr_v
    lower = hl_avg - mult * atr_v
    n = len(c)
    trend = np.zeros(n)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]): continue
        if c[i] > upper[i - 1]: trend[i] = 1
        elif c[i] < lower[i - 1]: trend[i] = -1
        else: trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1
        if trend[i] == 1 and lower[i] < lower[i - 1]: lower[i] = lower[i - 1]
        if trend[i] == -1 and upper[i] > upper[i - 1]: upper[i] = upper[i - 1]
    return trend, upper, lower


# ═══════════════════════
# 策略 1: 牛市SuperTrend (只做多)
# ═══════════════════════

class BullSuperTrend(Strategy):
    """
    牛市 SuperTrend — 只做多

    条件:
    - 价格 > 200EMA (长期牛市确认)
    - 1h 价格 > 1h EMA50 (中期趋势向上)
    - SuperTrend 转多 → 入场
    - 1.5x ATR 止损, 3x ATR 止盈
    """
    st_period = 7
    st_mult = 2.5
    ema_long = 200
    atr_period = 14

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        # SuperTrend
        self.st_trend, self.st_u, self.st_l = self.I(
            _supertrend, h, l, c, self.st_period, self.st_mult, name="ST")

        # 长期趋势 (200 EMA)
        self.ema200 = self.I(_ema, c, self.ema_long, overlay=True, name="EMA200")

        # ATR 止损止盈
        self.atr = self.I(_atr, h, l, c, self.atr_period, name="ATR")

        # 1h 趋势
        self._has_1h_ema = False
        if "1h_Close" in self.data.df.columns:
            self.ema1h = self.I(_ema, self.data.df["1h_Close"], 50, name="1hEMA50")
            self._has_1h_ema = True

        self._consecutive_losses = 0
        self._last_trade_bar = -999

    def next(self):
        c = self.data.Close[-1]
        atr_v = self.atr[-1]
        ema200_v = self.ema200[-1]
        st_now = self.st_trend[-1]
        st_prev = self.st_trend[-2]

        if any(np.isnan(x) for x in [atr_v, ema200_v, st_now, st_prev]) or c <= 0:
            return

        # 市场状态: 必须是牛市 (价格 > 200EMA)
        bull_market = c > ema200_v

        # 1h 确认
        if self._has_1h_ema:
            try:
                c1h = self.data.df["1h_Close"].iloc[-1]
                e1h = self.ema1h[-1]
                if not np.isnan(e1h):
                    bull_market = bull_market and (c1h > e1h)
            except: pass

        # SuperTrend 翻转信号
        flip_up = st_prev <= 0 and st_now > 0
        flip_down = st_prev > 0 and st_now <= 0

        # 冷却: 连续亏损
        bars_since = len(self.data) - self._last_trade_bar
        cooldown = max(12, self._consecutive_losses * 6)  # 每次亏损冷却延长

        # 平仓
        if self.position:
            if self.position.is_long and flip_down:
                self.position.close()
                # 记录盈亏
                if self.position.pl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
                self._last_trade_bar = len(self.data)
            return

        # 做多：牛市 + ST翻转向上 + 未冷却
        if bull_market and flip_up and bars_since >= cooldown:
            sl = c - atr_v * 1.5
            tp = c + atr_v * 3.0
            self.buy(sl=sl, tp=tp)
            self._last_trade_bar = len(self.data)


# ═══════════════════════
# 策略 2: 双向SuperTrend + 市场状态自适应
# ═══════════════════════

class AdaptiveSuperTrend(Strategy):
    """
    自适应 SuperTrend — 根据 200EMA 决定方向

    - 牛市 (价格 > 200EMA): 只做多
    - 熊市 (价格 < 200EMA): 只做空
    - ADX < 20 (震荡): 不交易
    """
    st_period = 7
    st_mult = 2.5
    ema_long = 200
    atr_period = 14
    adx_min = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.st_trend, self.st_u, self.st_l = self.I(
            _supertrend, h, l, c, self.st_period, self.st_mult, name="ST")
        self.ema200 = self.I(_ema, c, self.ema_long, overlay=True, name="EMA200")
        self.atr = self.I(_atr, h, l, c, self.atr_period, name="ATR")
        self.adx = self.I(_atr, h, l, c, 14, name="ADX")  # 用 ATR 近似
        self._consecutive_losses = 0
        self._last_trade_bar = -999

        # 正确的 ADX
        from src.strategies.fifteen_min_v3 import _adx
        self.adx_real = self.I(_adx, h, l, c, 14, name="ADX")

    def next(self):
        c = self.data.Close[-1]
        atr_v = self.atr[-1]
        ema200_v = self.ema200[-1]
        adx_v = self.adx_real[-1]
        st_now = self.st_trend[-1]
        st_prev = self.st_trend[-2]

        if any(np.isnan(x) for x in [atr_v, ema200_v, adx_v, st_now, st_prev]) or c <= 0:
            return
        if adx_v < self.adx_min: return  # 震荡市不交易

        bull_market = c > ema200_v  # 长期牛市
        bear_market = c < ema200_v  # 长期熊市

        flip_up = st_prev <= 0 and st_now > 0
        flip_down = st_prev > 0 and st_now <= 0

        bars_since = len(self.data) - self._last_trade_bar
        cooldown = max(12, self._consecutive_losses * 6)

        if self.position:
            if self.position.is_long:
                if flip_down:
                    self.position.close()
                    self._consecutive_losses = (self._consecutive_losses + 1
                        if self.position.pl < 0 else 0)
                    self._last_trade_bar = len(self.data)
            elif self.position.is_short:
                if flip_up:
                    self.position.close()
                    self._consecutive_losses = (self._consecutive_losses + 1
                        if self.position.pl < 0 else 0)
                    self._last_trade_bar = len(self.data)
            return

        if bars_since < cooldown: return

        # 牛市只做多
        if bull_market and flip_up:
            self.buy(sl=c-atr_v*1.5, tp=c+atr_v*3.0)
            self._last_trade_bar = len(self.data)

        # 熊市只做空
        elif bear_market and flip_down:
            self.sell(sl=c+atr_v*1.5, tp=c-atr_v*3.0)
            self._last_trade_bar = len(self.data)


# ═══════════════════════
# 策略 3: 简单双EMA金叉 + 200EMA过滤
# ═══════════════════════

class SimpleEmaCross(Strategy):
    """
    最简单的趋势策略: EMA9/21金叉 + 200EMA牛市过滤

    对标 #1 策略，但极简化：
    - 只在牛市做多 (价格 > 200EMA)
    - EMA9上穿EMA21 → 入场
    - 固定 1:2 盈亏比 (1% SL, 2% TP)
    """
    ema_fast = 9
    ema_slow = 21
    ema_trend = 200

    def init(self):
        c = self.data.Close
        self.ef = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA9")
        self.es = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA21")
        self.e200 = self.I(_ema, c, self.ema_trend, overlay=True, name="EMA200")

    def next(self):
        c = self.data.Close[-1]
        ef = self.ef[-1]; es = self.es[-1]
        ef_p = self.ef[-2]; es_p = self.es[-2]
        e200 = self.e200[-1]

        if any(np.isnan(x) for x in [ef, es, ef_p, es_p, e200]) or c <= 0: return

        # 牛市过滤
        bull = c > e200
        cross_up = ef_p <= es_p and ef > es
        cross_down = ef_p >= es_p and ef < es

        if self.position:
            if self.position.is_long and cross_down:
                self.position.close()
            elif self.position.is_short and cross_up:
                self.position.close()
            return

        # 牛市做多
        if bull and cross_up:
            self.buy(sl=c*0.985, tp=c*1.02)  # 1.5% SL, 2% TP
        # 熊市做空
        elif not bull and cross_down:
            self.sell(sl=c*1.015, tp=c*0.98)


# ═══════════════════════
# 策略 4: 布林带挤压突破
# ═══════════════════════

class BollingerSqueeze(Strategy):
    """
    布林带收缩后的突破交易

    当布林带宽度收缩到近期最低 → 即将变盘
    突破上轨 + 放量 → 做多
    跌破下轨 + 放量 → 做空
    """
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 50

    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume
        # 布林带
        mid = _ema(c, self.bb_period)  # 用 SMA 近似
        std_arr = np.full_like(c, np.nan)
        for i in range(self.bb_period - 1, len(c)):
            std_arr[i] = np.std(c[i-self.bb_period+1:i+1], ddof=1)

        upper = mid + self.bb_std * std_arr
        lower = mid - self.bb_std * std_arr
        bandwidth = (upper - lower) / mid  # 带宽百分比

        # 带宽滚动最低
        squeeze_threshold = np.full_like(bandwidth, np.nan)
        for i in range(self.squeeze_lookback - 1, len(bandwidth)):
            squeeze_threshold[i] = np.min(
                bandwidth[i-self.squeeze_lookback+1:i+1]
            ) * 1.1  # 比历史最低宽10%就算挤压结束

        self.bb_u = self.I(lambda: upper, name="BB_U", overlay=True)
        self.bb_l = self.I(lambda: lower, name="BB_L", overlay=True)
        self.bb_m = self.I(lambda: mid, name="BB_M", overlay=True)
        self.bw = self.I(lambda: bandwidth, name="BW")
        self.sq_th = self.I(lambda: squeeze_threshold, name="Squeeze")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")
        self.vol_ma = self.I(_ema, v, 20, name="VolMA")

        # 1h 趋势
        self._1h = False
        if "1h_Close" in self.data.df.columns:
            self.e1h = self.I(_ema, self.data.df["1h_Close"], 50, name="1hEMA50")
            self._1h = True

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        v = self.data.Volume[-1]
        atr_v = self.atr[-1]
        bw_v = self.bw[-1]; sq_th = self.sq_th[-1]
        bb_u = self.bb_u[-1]; bb_l = self.bb_l[-1]
        vol_ma = self.vol_ma[-1]

        if any(np.isnan(x) for x in [atr_v, bw_v, sq_th, bb_u, bb_l, vol_ma]) or c <= 0:
            return

        # 挤压中 (带宽低于近期最低) → 等待
        is_squeezing = bw_v < sq_th
        vol_surge = v > vol_ma * 1.3

        ht_bull = True; ht_bear = True
        if self._1h:
            try:
                c1h = self.data.df["1h_Close"].iloc[-1]
                e1h = self.e1h[-1]
                if not np.isnan(e1h):
                    ht_bull = c1h > e1h; ht_bear = c1h < e1h
            except: pass

        if self.position:
            if self.position.is_long and c < self.bb_m[-1]:
                self.position.close()
            elif self.position.is_short and c > self.bb_m[-1]:
                self.position.close()
            return

        # 突破交易 (挤压后放量突破)
        if not is_squeezing and vol_surge:
            if h > bb_u and ht_bull:
                self.buy(sl=l*0.995, tp=c+atr_v*2.0)
            elif l < bb_l and ht_bear:
                self.sell(sl=h*1.005, tp=c-atr_v*2.0)


# ═══════════════════════
# 注册
# ═══════════════════════

STRATEGIES_OPT = {
    "bull_supertrend": BullSuperTrend,
    "adaptive_supertrend": AdaptiveSuperTrend,
    "simple_ema_cross": SimpleEmaCross,
    "bollinger_squeeze": BollingerSqueeze,
}
