"""
多时间框架趋势策略 — 4H 定方向 + 1H 精确入场

4H: EMA20/50/100 排列 → 只做多(多头排列) 或 只做空(空头排列)
1H: 价格回调到 EMA20 附近 + ADX>20 → 精确入场
出场: TP=2.5xATR(1H) | SL=1.5xATR(1H) | 或4H趋势反转
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


def _ema(a, span):
    if span <= 0 or len(a) < span: return np.full_like(a, np.nan, dtype=float)
    alpha = 2.0/(span+1.0); r = np.full_like(a, np.nan, dtype=float)
    r[span-1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha*a[i] + (1-alpha)*r[i-1]
    return r

def _atr(h,l,c,p=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c)<p+1: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    for i in range(p-1, len(tr)): r[i] = np.mean(tr[i-p+1:i+1])
    return r

def _adx(h,l,c,p=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n<p*2: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    um = h-np.roll(h,1); dm = np.roll(l,1)-l; um[0]=dm[0]=0
    pdm=np.where((um>dm)&(um>0),um,0.0); ndm=np.where((dm>um)&(dm>0),dm,0.0)
    a=1.0/p; at=np.full(n,np.nan); pt=np.full(n,np.nan); nt=np.full(n,np.nan)
    at[p-1]=np.mean(tr[:p]); pt[p-1]=np.mean(pdm[:p]); nt[p-1]=np.mean(ndm[:p])
    for i in range(p,n):
        at[i]=a*tr[i]+(1-a)*at[i-1]; pt[i]=a*pdm[i]+(1-a)*pt[i-1]; nt[i]=a*ndm[i]+(1-a)*nt[i-1]
    pdi=np.where(at>0,100*pt/at,0); ndi=np.where(at>0,100*nt/at,0)
    ds=pdi+ndi; dx=np.where(ds>0,100*np.abs(pdi-ndi)/ds,0)
    r[p*2-2]=np.mean(dx[p-1:p*2-1])
    for i in range(p*2-1,n): r[i]=a*dx[i]+(1-a)*r[i-1]
    return r


class MTFTrendStrategy(Strategy):
    """4H方向 + 1H入场"""

    # 1H 参数
    ema_fast = 20; ema_mid = 50; ema_slow = 100
    adx_entry = 25   # 1H ADX 更高门槛 (过滤噪声)
    atr_period = 14
    sl_mult = 1.5
    tp_mult = 3.0    # 更大止盈空间 (1H波动大)
    entry_dist = 0.5 # 更接近 EMA 才入场

    risk_per_trade = 0.02
    cooldown_bars = 12  # 更长冷却 (避免频繁交易)

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        # 1H 指标
        self.ef = self.I(_ema, c, self.ema_fast, overlay=True, name="1h_EMA20")
        self.em = self.I(_ema, c, self.ema_mid, overlay=True, name="1h_EMA50")
        self.es = self.I(_ema, c, self.ema_slow, overlay=True, name="1h_EMA100")
        self.atr = self.I(_atr, h, l, c, self.atr_period, name="1h_ATR")
        self.adx = self.I(_adx, h, l, c, self.atr_period, name="1h_ADX")

        # 4H 方向（从合并数据中读取）
        self._has_4h = "4h_Close" in self.data.df.columns
        if self._has_4h:
            c4 = self.data.df["4h_Close"].values
            self.ef4 = self.I(_ema, c4, self.ema_fast, overlay=False, name="4h_EMA20")
            self.em4 = self.I(_ema, c4, self.ema_mid, overlay=False, name="4h_EMA50")
            logger.info("4H方向过滤: 已启用")
        else:
            self.ef4 = self.ef; self.em4 = self.em
            logger.info("4H方向过滤: 无4H数据, 用1H代替")

        self._bars_since_exit = 999
        self._entry_bar = 0

    def next(self):
        c = self.data.Close[-1]; atr = self.atr[-1]
        ef = self.ef[-1]; em = self.em[-1]; es = self.es[-1]; adx = self.adx[-1]

        # 4H 方向
        ef4 = self.ef4[-1]; em4 = self.em4[-1]

        if any(np.isnan(x) for x in [atr, ef, em, es, adx, ef4, em4]) or c <= 0:
            return

        self._bars_since_exit += 1

        # ── 4H 大方向 ──
        bull_4h = ef4 > em4   # 4H 多头 → 只做多
        bear_4h = ef4 < em4   # 4H 空头 → 只做空

        # ── 1H 入场条件 ──
        near_ema20 = abs(c - ef) <= atr * self.entry_dist
        trend_1h_ok = adx >= self.adx_entry

        # ── 持仓 ──
        if self.position:
            # 4H 方向反转 → 平仓
            if self.position.is_long and bear_4h:
                self.position.close(); self._bars_since_exit = 0
            elif self.position.is_short and bull_4h:
                self.position.close(); self._bars_since_exit = 0
            return

        if self._bars_since_exit < self.cooldown_bars:
            return

        # ── 1H 趋势必须与 4H 同向 ──
        bull_1h = ef > em   # 1H 也多头
        bear_1h = ef < em   # 1H 也空头

        # ── 只朝 4H 方向交易 (1H 也需同向) ──
        if bull_4h and bull_1h and trend_1h_ok and near_ema20:
            sl = c - atr * self.sl_mult
            tp = c + atr * self.tp_mult
            frac = min(self.risk_per_trade / (self.sl_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)
                self._bars_since_exit = 999

        elif bear_4h and bear_1h and trend_1h_ok and near_ema20:
            sl = c + atr * self.sl_mult
            tp = c - atr * self.tp_mult
            frac = min(self.risk_per_trade / (self.sl_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)
                self._bars_since_exit = 999
