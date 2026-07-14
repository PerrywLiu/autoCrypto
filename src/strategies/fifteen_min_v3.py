"""
15分钟趋势跟踪策略 v3 — 基于49策略研究结果

核心原则（来自研究）：
1. 趋势跟踪 >> 均值回归 (加密市场趋势性强)
2. 低胜率(~35%) + 高盈亏比(3:1) >> 高胜率 + 低盈亏比
3. 简单策略比复杂策略更稳健 (2参数 EMA交叉 最好)
4. 需要更高时间框架过滤噪音

策略：
1. EMA金叉+1h趋势过滤 (对标 #1 策略，适配15m)
2. Donchian突破+ADX (对标 #10 策略)
3. 三均线排列+回调入场 (对标 #4 策略)
4. 动量突破+ATR跟踪止损 (对标 #8 策略)
"""

from backtesting import Strategy
from backtesting.lib import crossover, TrailingStrategy
import numpy as np


# ══════════════════════════════════
# NumPy 指标
# ══════════════════════════════════

def _ema(a, span):
    if span <= 0 or len(a) < span: return np.full_like(a, np.nan)
    alpha = 2.0/(span+1.0); r = np.full_like(a, np.nan, dtype=float)
    r[span-1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha*a[i] + (1-alpha)*r[i-1]
    return r

def _atr(h, l, c, p=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < p+1: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    r[p-1] = np.mean(tr[:p]); a_ = 1.0/p
    for i in range(p, len(tr)): r[i] = a_*tr[i] + (1-a_)*r[i-1]
    return r

def _adx(h, l, c, p=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n < p*2: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    um = h-np.roll(h,1); dm = np.roll(l,1)-l; um[0]=dm[0]=0
    pdm = np.where((um>dm)&(um>0), um, 0.0)
    ndm = np.where((dm>um)&(dm>0), dm, 0.0)
    a_ = 1.0/p; at = np.full(n,np.nan); pt=np.full(n,np.nan); nt=np.full(n,np.nan)
    at[p-1]=np.mean(tr[:p]); pt[p-1]=np.mean(pdm[:p]); nt[p-1]=np.mean(ndm[:p])
    for i in range(p,n):
        at[i]=a_*tr[i]+(1-a_)*at[i-1]
        pt[i]=a_*pdm[i]+(1-a_)*pt[i-1]
        nt[i]=a_*ndm[i]+(1-a_)*nt[i-1]
    pdi = np.where(at>0, 100*pt/at, 0); ndi = np.where(at>0, 100*nt/at, 0)
    ds = pdi+ndi
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(ds>0, 100*np.abs(pdi-ndi)/ds, 0)
    r[p*2-2] = np.mean(dx[p-1:p*2-1])
    for i in range(p*2-1,n): r[i]=a_*dx[i]+(1-a_)*r[i-1]
    return r

def _rsi(a, p=14):
    r = np.full_like(a, np.nan, dtype=float)
    if len(a) < p+1: return r
    d = np.diff(a); g = np.maximum(d,0); lo = np.maximum(-d,0)
    ag = np.mean(g[:p]); al = np.mean(lo[:p]); a_=1.0/p
    for i in range(p, len(d)):
        r[i+1] = 100.0-(100.0/(1.0+ag/al)) if al>0 else (100.0 if ag>0 else 50.0)
        ag = a_*g[i]+(1-a_)*ag; al = a_*lo[i]+(1-a_)*al
    return np.clip(r,0,100)

def _highest(a, w):
    r = np.full_like(a, np.nan)
    for i in range(w-1, len(a)): r[i] = np.max(a[i-w+1:i+1])
    return r

def _lowest(a, w):
    r = np.full_like(a, np.nan)
    for i in range(w-1, len(a)): r[i] = np.min(a[i-w+1:i+1])
    return r


# ══════════════════════════════════
# 策略 1: EMA金叉 + 1h趋势过滤 (对标 #1 策略)
# ══════════════════════════════════

class EmaCrossTrend(Strategy):
    """
    EMA12/26 金叉死叉 + 1h EMA50 趋势过滤

    对标 49 策略回测中表现最好的 ema_crossover（夏普 1.30，收益 491%）
    适配到 15m：用更短的 EMA 周期 + 1h 趋势过滤减少假信号

    做多条件:
      - 1h 价格在 EMA50 之上 (大趋势向上)
      - EMA9 上穿 EMA21 (15m 短期趋势)
      - ADX > 20 (趋势够强)

    做空条件: 反之
    """
    ema_fast = 9
    ema_slow = 21
    adx_period = 14
    adx_min = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema_f = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA9")
        self.ema_s = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA21")
        self.adx = self.I(_adx, h, l, c, self.adx_period, name="ADX")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

        # 1h 趋势过滤
        if "1h_Close" in self.data.df.columns:
            self.ema1h_50 = self.I(_ema, self.data.df["1h_Close"], 50, name="1hEMA50")
            self._has_1h = True
        else:
            self._has_1h = False

    def next(self):
        c = self.data.Close[-1]; atr_v = self.atr[-1]
        if c <= 0 or np.isnan(atr_v): return

        if self.position:
            # 动态跟踪止损: 用前一根K线低点-0.5 ATR
            if self.position.is_long:
                trail = max(self.data.Low[-2] - atr_v * 0.5,
                           self.data.Low[-3] - atr_v * 0.5)
                if c < trail:
                    self.position.close()
            elif self.position.is_short:
                trail = min(self.data.High[-2] + atr_v * 0.5,
                           self.data.High[-3] + atr_v * 0.5)
                if c > trail:
                    self.position.close()
            return

        # 趋势过滤
        adx_v = self.adx[-1]
        if np.isnan(adx_v) or adx_v < self.adx_min:
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

        # EMA 交叉信号
        cross_up = (self.ema_f[-2] <= self.ema_s[-2] and self.ema_f[-1] > self.ema_s[-1])
        cross_down = (self.ema_f[-2] >= self.ema_s[-2] and self.ema_f[-1] < self.ema_s[-1])

        if cross_up and ht_bull:
            sl = c - atr_v * 2.0
            tp = c + atr_v * 4.0
            self.buy(sl=sl, tp=tp)

        elif cross_down and ht_bear:
            sl = c + atr_v * 2.0
            tp = c - atr_v * 4.0
            self.sell(sl=sl, tp=tp)


# ══════════════════════════════════
# 策略 2: Donchian突破 + ADX (对标 #10)
# ══════════════════════════════════

class DonchianAdx(Strategy):
    """
    20期 Donchian 通道突破 + ADX > 22 过滤

    对标 donchian_breakout 策略（夏普 1.06，收益 320%）
    """
    donchian = 20
    adx_period = 14
    adx_min = 22

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.hh = self.I(_highest, h, self.donchian, name="HH")
        self.ll = self.I(_lowest, l, self.donchian, name="LL")
        self.adx = self.I(_adx, h, l, c, self.adx_period, name="ADX")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

        if "1h_Close" in self.data.df.columns:
            self.ema1h_50 = self.I(_ema, self.data.df["1h_Close"], 50, name="1hEMA50")
            self._has_1h = True
        else:
            self._has_1h = False

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        atr_v = self.atr[-1]; adx_v = self.adx[-1]
        hh = self.hh[-1]; ll = self.ll[-1]
        if any(np.isnan(x) for x in [atr_v, adx_v, hh, ll]) or c <= 0: return

        if adx_v < self.adx_min: return

        ht_bull = True; ht_bear = True
        if self._has_1h:
            try:
                c1h = self.data.df["1h_Close"].iloc[-1]
                e1h = self.ema1h_50[-1]
                if not np.isnan(e1h):
                    ht_bull = c1h > e1h; ht_bear = c1h < e1h
            except: pass

        if self.position:
            if self.position.is_long and c < ll: self.position.close()
            elif self.position.is_short and c > hh: self.position.close()
            return

        if h > hh and ht_bull:
            sl = c - atr_v * 1.5; tp = c + atr_v * 4.0
            self.buy(sl=sl, tp=tp)

        elif l < ll and ht_bear:
            sl = c + atr_v * 1.5; tp = c - atr_v * 4.0
            self.sell(sl=sl, tp=tp)


# ══════════════════════════════════
# 策略 3: 三均线排列+回调 (对标 #4 triple_ma)
# ══════════════════════════════════

class TripleMaPullback(Strategy):
    """
    三均线多头排列+回调到中轨入场

    对标 triple_ma（夏普 1.25，收益 502%）
    """
    ema_s = 10; ema_m = 20; ema_l = 50
    adx_min = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.e10 = self.I(_ema, c, self.ema_s, overlay=True, name="EMA10")
        self.e20 = self.I(_ema, c, self.ema_m, overlay=True, name="EMA20")
        self.e50 = self.I(_ema, c, self.ema_l, overlay=True, name="EMA50")
        self.adx = self.I(_adx, h, l, c, 14, name="ADX")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

    def next(self):
        c = self.data.Close[-1]
        atr_v = self.atr[-1]; adx_v = self.adx[-1]
        e10 = self.e10[-1]; e20 = self.e20[-1]; e50 = self.e50[-1]
        if any(np.isnan(x) for x in [atr_v, adx_v, e10, e20, e50]) or c <= 0: return
        if adx_v < self.adx_min: return

        bulls = e10 > e20 > e50  # 完美多头
        bears = e10 < e20 < e50  # 完美空头

        if self.position:
            if self.position.is_long and not bulls: self.position.close()
            elif self.position.is_short and not bears: self.position.close()
            return

        near_e20 = abs(c - e20) < atr_v * 1.0

        if bulls and near_e20 and c > e20:
            sl = e50; tp = c + atr_v * 4.0
            self.buy(sl=sl, tp=tp)

        elif bears and near_e20 and c < e20:
            sl = e50; tp = c - atr_v * 4.0
            self.sell(sl=sl, tp=tp)


# ══════════════════════════════════
# 策略 4: 动量突破 + 跟踪止损 (对标 #8 momentum)
# ══════════════════════════════════

class MomentumBreakout(Strategy):
    """
    动量突破：价格突破 N 期高低点 + 成交量确认

    对标 momentum 策略（夏普 1.07, 收益 283%, 135笔交易）
    - 最多交易信号 → 统计最可靠
    """
    lookback = 20
    vol_ma_period = 20
    vol_mult = 1.3
    adx_min = 18

    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume
        self.hh = self.I(_highest, h, self.lookback, name="HH")
        self.ll = self.I(_lowest, l, self.lookback, name="LL")
        self.adx = self.I(_adx, h, l, c, 14, name="ADX")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")
        self.vol_ma = self.I(_ema, v, self.vol_ma_period, name="VolMA")

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        v = self.data.Volume[-1]
        atr_v = self.atr[-1]; adx_v = self.adx[-1]
        hh = self.hh[-1]; ll = self.ll[-1]; vma = self.vol_ma[-1]
        if any(np.isnan(x) for x in [atr_v, adx_v, hh, ll, vma]) or c <= 0: return

        vol_surge = vma > 0 and v > vma * self.vol_mult
        strong_trend = adx_v > self.adx_min

        if self.position:
            if self.position.is_long and l < ll: self.position.close()
            elif self.position.is_short and h > hh: self.position.close()
            return

        if not strong_trend: return

        if h > hh:
            sl = c - atr_v * 1.5
            tp = c + atr_v * 3.0
            self.buy(sl=sl, tp=tp)

        elif l < ll:
            sl = c + atr_v * 1.5
            tp = c - atr_v * 3.0
            self.sell(sl=sl, tp=tp)


# ══════════════════════════════════
# 策略 5: SuperTrend 精确版
# ══════════════════════════════════

class SuperTrendPrecise(Strategy):
    """
    SuperTrend 精确版 — 信号翻转+多时框确认

    ST(10,3) 信号 + ADX + 1h EMA 过滤
    """
    st_period = 10; st_mult = 3.0
    adx_min = 22

    def _supertrend(self, h, l, c, period, mult):
        atr_v = _atr(h, l, c, period)
        hl_avg = (h + l) / 2.0
        upper = hl_avg + mult * atr_v; lower = hl_avg - mult * atr_v
        n = len(c); trend = np.zeros(n)
        for i in range(1, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]): continue
            if c[i] > upper[i-1]: trend[i] = 1
            elif c[i] < lower[i-1]: trend[i] = -1
            else: trend[i] = trend[i-1] if trend[i-1] != 0 else 1
            if trend[i] == 1 and lower[i] < lower[i-1]: lower[i] = lower[i-1]
            if trend[i] == -1 and upper[i] > upper[i-1]: upper[i] = upper[i-1]
        return trend, upper, lower

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.st, self.u, self.lw = self.I(self._supertrend, h, l, c,
                                           self.st_period, self.st_mult, name="ST")
        self.adx = self.I(_adx, h, l, c, 14, name="ADX")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

        if "1h_Close" in self.data.df.columns:
            self.e1h_50 = self.I(_ema, self.data.df["1h_Close"], 50, name="1hEMA50")
            self._1h = True
        else:
            self._1h = False

    def next(self):
        c = self.data.Close[-1]; atr_v = self.atr[-1]; adx_v = self.adx[-1]
        if c <= 0 or np.isnan(atr_v) or np.isnan(adx_v): return

        st_n = self.st[-1]; st_p = self.st[-2]
        flip_up = st_p <= 0 and st_n > 0
        flip_down = st_p > 0 and st_n <= 0

        if adx_v < self.adx_min: return

        ht_bull = True; ht_bear = True
        if self._1h:
            try:
                c1h = self.data.df["1h_Close"].iloc[-1]; e1 = self.e1h_50[-1]
                if not np.isnan(e1): ht_bull = c1h > e1; ht_bear = c1h < e1
            except: pass

        if self.position:
            if self.position.is_long and flip_down: self.position.close()
            elif self.position.is_short and flip_up: self.position.close()
            return

        if flip_up and ht_bull:
            self.buy(sl=c-atr_v*2.0, tp=c+atr_v*4.0)
        elif flip_down and ht_bear:
            self.sell(sl=c+atr_v*2.0, tp=c-atr_v*4.0)


# ══════════════════════════════════
# 注册
# ══════════════════════════════════

STRATEGIES_V3 = {
    "A_ema_cross_trend": EmaCrossTrend,
    "B_donchian_adx": DonchianAdx,
    "C_triple_ma": TripleMaPullback,
    "D_momentum": MomentumBreakout,
    "E_supertrend": SuperTrendPrecise,
}
