"""
自适应混合策略 — 根据市场状态切换

震荡市 (ADX<20): 均值回归 — RSI超卖买/超买卖, 高胜率
趋势市 (ADX>25): 趋势跟踪 — EMA排列+回调入场, 高盈亏比
过渡期 (ADX 20-25): 不交易, 等待明确信号
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


def _ema(a, span):
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan)
    alpha = 2.0/(span+1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span-1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha*a[i]+(1-alpha)*r[i-1]
    return r

def _rsi(a, p=14):
    r = np.full_like(a, np.nan, dtype=float)
    if len(a) < p+1: return r
    d = np.diff(a); g = np.maximum(d,0); lo = np.maximum(-d,0)
    ag = np.mean(g[:p]); al = np.mean(lo[:p]); a_ = 1.0/p
    for i in range(p, len(d)):
        r[i+1] = 100.0-(100.0/(1.0+ag/al)) if al>0 else (100.0 if ag>0 else 50.0)
        ag = a_*g[i]+(1-a_)*ag; al = a_*lo[i]+(1-a_)*al
    return np.clip(r,0,100)

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


class AdaptiveHybridStrategy(Strategy):
    """
    自适应混合 — ADX 决定用哪个子策略

    ADX < 20: 均值回归模式
      买入: RSI < 30 + 价格近EMA200
      卖出: RSI > 70 + 价格远离EMA20
      TP=2xATR, SL=2xATR

    ADX > 25: 趋势跟踪模式
      买入: EMA20>50>100 + 价格近EMA20
      卖出: EMA20<50<100 + 价格近EMA20
      TP=2.5xATR, SL=1.5xATR

    风险: 2%/笔, 冷却12根K线
    """

    # 回归参数
    rsi_period = 14; rsi_os = 30; rsi_ob = 70
    ema_long = 200; ema_short = 20
    mr_tp = 2.0; mr_sl = 2.0

    # 趋势参数
    ema_f = 20; ema_m = 50; ema_s = 100
    tr_tp = 2.5; tr_sl = 1.5; tr_entry = 1.0

    # 切换阈值
    adx_ranging = 20   # ADX <= 此值 → 震荡模式
    adx_trending = 25  # ADX >= 此值 → 趋势模式

    atr_period = 14; adx_period = 14
    risk_per_trade = 0.02; cooldown_bars = 12

    def init(self):
        c,h,l = self.data.Close, self.data.High, self.data.Low

        self.rsi_v = self.I(_rsi, c, self.rsi_period, name="RSI")
        self.ema200_v = self.I(_ema, c, self.ema_long, overlay=True, name="EMA200")
        self.ema20_v = self.I(_ema, c, self.ema_short, overlay=True, name="EMA20")
        self.tr_f = self.I(_ema, c, self.ema_f, overlay=True, name="TrEMA20")
        self.tr_m = self.I(_ema, c, self.ema_m, overlay=True, name="TrEMA50")
        self.tr_s = self.I(_ema, c, self.ema_s, overlay=True, name="TrEMA100")
        self.atr_v = self.I(_atr, h,l,c, self.atr_period, name="ATR")
        self.adx_v = self.I(_adx, h,l,c, self.adx_period, name="ADX")

        self._bars_since_exit = 999
        logger.info(f"自适应混合: 震荡(ADX<{self.adx_ranging})→回归 趋势(ADX>{self.adx_trending})→跟踪")

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        atr = self.atr_v[-1]; adx = self.adx_v[-1]; rsi = self.rsi_v[-1]
        e_l = self.ema200_v[-1]; e_s = self.ema20_v[-1]
        ef = self.tr_f[-1]; em = self.tr_m[-1]; esl = self.tr_s[-1]

        if any(np.isnan(x) for x in [atr,adx,rsi,e_l,e_s,ef,em,esl]) or c<=0: return
        self._bars_since_exit += 1

        # ── 持仓: RSI 回到中性区就平 ──
        if self.position:
            if self.position.is_long and rsi > 55: self.position.close(); self._bars_since_exit=0
            elif self.position.is_short and rsi < 45: self.position.close(); self._bars_since_exit=0
            return

        if self._bars_since_exit < self.cooldown_bars: return

        mode = "trend" if adx >= self.adx_trending else ("range" if adx <= self.adx_ranging else None)
        if mode is None: return  # 过渡期不交易

        if mode == "range":
            # ═══ 均值回归模式 ═══
            near_support = abs(c - e_l) <= atr * 2.5
            far_above = c > e_s + atr * 3

            if rsi < self.rsi_os and near_support:
                sl = c-atr*self.mr_sl; tp = c+atr*self.mr_tp
                frac = min(self.risk_per_trade/(self.mr_sl*atr/c), 0.95)
                if frac>0: self.buy(size=frac,sl=sl,tp=tp); self._bars_since_exit=999
                return

            if rsi > self.rsi_ob and far_above:
                sl = c+atr*self.mr_sl; tp = c-atr*self.mr_tp
                frac = min(self.risk_per_trade/(self.mr_sl*atr/c), 0.95)
                if frac>0: self.sell(size=frac,sl=sl,tp=tp); self._bars_since_exit=999

        else:
            # ═══ 趋势跟踪模式 ═══
            bull = ef>em>esl; bear = ef<em<esl
            near = abs(c-ef) <= atr*self.tr_entry

            if bull and near:
                sl = c-atr*self.tr_sl; tp = c+atr*self.tr_tp
                frac = min(self.risk_per_trade/(self.tr_sl*atr/c), 0.95)
                if frac>0: self.buy(size=frac,sl=sl,tp=tp); self._bars_since_exit=999

            elif bear and near:
                sl = c+atr*self.tr_sl; tp = c-atr*self.tr_tp
                frac = min(self.risk_per_trade/(self.tr_sl*atr/c), 0.95)
                if frac>0: self.sell(size=frac,sl=sl,tp=tp); self._bars_since_exit=999
