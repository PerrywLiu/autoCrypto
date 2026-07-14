"""
均值回归 + 趋势确认加强 — 双策略高胜率方案

方案 A: 超严格趋势 (swing_trend.py 已就绪，仅调参数)
  - ADX>30, 日线强制, 第二触及EMA, SL=1.5x, TP=2.5x

方案 B: 均值回归 (本文件)
  - RSI<30 + 价格近EMA200 → 超卖买入
  - RSI>70 + 价格远高于EMA20 → 超买卖出
  - TP=2x ATR, SL=2x ATR
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


def _ema(a, span):
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _rsi(a, p=14):
    r = np.full_like(a, np.nan, dtype=float)
    if len(a) < p+1: return r
    d = np.diff(a); g = np.maximum(d,0); lo = np.maximum(-d,0)
    ag = np.mean(g[:p]); al = np.mean(lo[:p]); alpha = 1.0/p
    for i in range(p, len(d)):
        r[i+1] = 100.0-(100.0/(1.0+ag/al)) if al>0 else (100.0 if ag>0 else 50.0)
        ag = alpha*g[i]+(1-alpha)*ag; al = alpha*lo[i]+(1-alpha)*al
    return np.clip(r, 0, 100)


def _atr(h, l, c, p=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < p+1: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    for i in range(p-1, len(tr)): r[i] = np.mean(tr[i-p+1:i+1])
    return r


def _adx(h, l, c, p=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n < p*2: return r
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    um = h-np.roll(h,1); dm = np.roll(l,1)-l; um[0]=dm[0]=0
    pdm = np.where((um>dm)&(um>0),um,0.0); ndm = np.where((dm>um)&(dm>0),dm,0.0)
    a=1.0/p; at=np.full(n,np.nan); pt=np.full(n,np.nan); nt=np.full(n,np.nan)
    at[p-1]=np.mean(tr[:p]); pt[p-1]=np.mean(pdm[:p]); nt[p-1]=np.mean(ndm[:p])
    for i in range(p,n):
        at[i]=a*tr[i]+(1-a)*at[i-1]; pt[i]=a*pdm[i]+(1-a)*pt[i-1]; nt[i]=a*ndm[i]+(1-a)*nt[i-1]
    pdi=np.where(at>0,100*pt/at,0); ndi=np.where(at>0,100*nt/at,0)
    ds=pdi+ndi; dx=np.where(ds>0,100*np.abs(pdi-ndi)/ds,0)
    r[p*2-2]=np.mean(dx[p-1:p*2-1])
    for i in range(p*2-1,n): r[i]=a*dx[i]+(1-a)*r[i-1]
    return r


# ═══════════════════════════════════════
# 方案 B: 均值回归策略
# ═══════════════════════════════════════

class MeanRevertStrategy(Strategy):
    """
    均值回归 — 超卖买入, 超买卖出

    买入: RSI < 30 + 价格在EMA200附近 (超卖反弹)
    卖出: RSI > 70 + 价格远高于EMA20 (超买回落)
    止盈: 2x ATR | 止损: 2x ATR
    冷却: 12根K线 (2天)
    """

    rsi_period = 14
    rsi_oversold = 30       # 超卖阈值
    rsi_overbought = 70     # 超买阈值

    ema_trend = 200          # 长期均线 (支撑/阻力参考)
    ema_fast = 20            # 短期均线

    atr_period = 14
    atr_stop_mult = 2.0      # 止损 ATR 倍数
    atr_tp_mult = 2.0        # 止盈 ATR 倍数

    risk_per_trade = 0.02
    cooldown_bars = 12       # 2天冷却

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.rsi_v = self.I(_rsi, c, self.rsi_period, name="RSI14")
        self.ema200 = self.I(_ema, c, self.ema_trend, overlay=True, name="EMA200")
        self.ema20 = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA20")
        self.atr_v = self.I(_atr, h, l, c, self.atr_period, name="ATR14")

        self._bars_since_exit = 999
        logger.info(f"均值回归就绪: RSI<{self.rsi_oversold}买 RSI>{self.rsi_overbought}卖")

    def next(self):
        c = self.data.Close[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        atr = self.atr_v[-1]
        rsi = self.rsi_v[-1]
        ema200 = self.ema200[-1]
        ema20 = self.ema20[-1]

        if any(np.isnan(x) for x in [atr, rsi, ema200, ema20]) or c <= 0:
            return

        self._bars_since_exit += 1

        # ── 持仓管理 (纯固定止盈止损, 不追踪) ──
        if self.position:
            if self.position.is_long and rsi > 55:  # 恢复到中性区就平仓
                self.position.close()
                self._bars_since_exit = 0
            elif self.position.is_short and rsi < 45:
                self.position.close()
                self._bars_since_exit = 0
            return

        if self._bars_since_exit < self.cooldown_bars:
            return

        # ── 买入: RSI超卖 + 价格接近EMA200支撑 ──
        near_support = abs(c - ema200) <= atr * 2.5
        if rsi < self.rsi_oversold and near_support:
            sl = c - atr * self.atr_stop_mult
            tp = c + atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_stop_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)
                self._bars_since_exit = 999
                logger.info(f"🟢 超卖买入 @{c:.0f} RSI={rsi:.0f} EMA200={ema200:.0f}")

        # ── 卖出: RSI超买 + 价格远离EMA20 ──
        far_above_ema = c > ema20 + atr * 3
        if rsi > self.rsi_overbought and far_above_ema:
            sl = c + atr * self.atr_stop_mult
            tp = c - atr * self.atr_tp_mult
            frac = min(self.risk_per_trade / (self.atr_stop_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)
                self._bars_since_exit = 999
                logger.info(f"🔴 超买卖出 @{c:.0f} RSI={rsi:.0f} EMA20={ema20:.0f}")


# ═══════════════════════════════════════
# 方案 A: 超严格趋势 (基于 swing_trend 参数强化)
# ═══════════════════════════════════════

class StrictTrendStrategy(Strategy):
    """
    超严格趋势跟踪 — 追求高胜率

    入场条件:
      1. EMA20>50>100 排列 (4H级别)
      2. ADX > 30 (极强趋势)
      3. 价格第二次触及EMA20 (第一次触及后已确认反弹)
      4. 日线EMA20>EMA50 同向确认

    出场: TP=2.5x ATR | SL=1.5x ATR
    """

    ema_fast = 20; ema_mid = 50; ema_slow = 100
    adx_period = 14; adx_threshold = 30
    atr_period = 14; atr_entry_mult = 1.0
    atr_stop_mult = 1.5; atr_tp_mult = 2.5
    risk_per_trade = 0.02; cooldown_bars = 12

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema_f = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA20")
        self.ema_m = self.I(_ema, c, self.ema_mid,  overlay=True, name="EMA50")
        self.ema_s = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA100")
        self.atr_v = self.I(_atr, h, l, c, self.atr_period, name="ATR14")
        self.adx_v = self.I(_adx, h, l, c, self.adx_period, name="ADX14")

        # 日线
        self._has_daily = "1d_Close" in self.data.df.columns
        if self._has_daily:
            dc = self.data.df["1d_Close"].values
            self.d_ef = self.I(_ema, dc, 20, overlay=False, name="D20")
            self.d_em = self.I(_ema, dc, 50, overlay=False, name="D50")

        self._bars_since_exit = 999
        self._touched_ema_before = False  # 追踪"第二次触及"
        logger.info(f"严格趋势就绪: ADX>{self.adx_threshold} TP={self.atr_tp_mult}x SL={self.atr_stop_mult}x")

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        ef = self.ema_f[-1]; em = self.ema_m[-1]; es = self.ema_s[-1]
        atr = self.atr_v[-1]; adx = self.adx_v[-1]

        if any(np.isnan(x) for x in [ef,em,es,atr,adx]) or c <= 0: return
        self._bars_since_exit += 1

        bull = ef > em > es; bear = ef < em < es
        strong = adx > self.adx_threshold
        near_ema = abs(c - ef) <= atr * self.atr_entry_mult

        # 日线过滤
        if self._has_daily:
            d_ef = self.d_ef[-1]; d_em = self.d_em[-1]
            if not (np.isnan(d_ef) or np.isnan(d_em)):
                if d_ef > d_em: bear = False
                if d_ef < d_em: bull = False

        # ── 第二次触及跟踪 ──
        if near_ema:
            self._touched_ema_before = True
        elif abs(c - ef) > atr * 3:
            self._touched_ema_before = False  # 价格远离EMA，重置

        # ── 持仓 ──
        if self.position:
            if self.position.is_long and ef < em: self.position.close(); self._bars_since_exit=0
            elif self.position.is_short and ef > em: self.position.close(); self._bars_since_exit=0
            return

        if self._bars_since_exit < self.cooldown_bars: return
        if not strong: return

        # ── 入场: 第二次触及EMA后才入场 ──
        if self._touched_ema_before and near_ema:
            if bull:
                sl = c - atr * self.atr_stop_mult
                tp = c + atr * self.atr_tp_mult
                frac = min(self.risk_per_trade/(self.atr_stop_mult*atr/c), 0.95)
                if frac>0: self.buy(size=frac, sl=sl, tp=tp); self._bars_since_exit=999; self._touched_ema_before=False
            elif bear:
                sl = c + atr * self.atr_stop_mult
                tp = c - atr * self.atr_tp_mult
                frac = min(self.risk_per_trade/(self.atr_stop_mult*atr/c), 0.95)
                if frac>0: self.sell(size=frac, sl=sl, tp=tp); self._bars_since_exit=999; self._touched_ema_before=False
