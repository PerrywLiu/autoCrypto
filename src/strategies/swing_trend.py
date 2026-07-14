"""
大周期趋势跟踪 — 4H 主周期 + 日线方向过滤

设计原则:
  - 4H K线: 日均6根, 噪音比1H低75%
  - 日线 EMA20/50: 大方向过滤器 (日线多头→只做多, 日线空头→只做空)
  - ADX>25: 只在强趋势市场交易
  - 回调入场: 价格回踩EMA20才入场, 不追高
  - Trailing Stop: 3~4x ATR追踪止损 (大周期给更多空间)
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


# ═══════════════════════════════════════
# NumPy 指标
# ═══════════════════════════════════════

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
    for i in range(p - 1, len(tr)):
        r[i] = np.mean(tr[i - p + 1:i + 1])
    return r


def _adx(h, l, c, p=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n < p * 2: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    um = h - np.roll(h, 1); dm = np.roll(l, 1) - l; um[0] = dm[0] = 0
    pdm = np.where((um > dm) & (um > 0), um, 0.0)
    ndm = np.where((dm > um) & (dm > 0), dm, 0.0)
    a = 1.0 / p
    at = np.full(n, np.nan); pt = np.full(n, np.nan); nt = np.full(n, np.nan)
    at[p-1]=np.mean(tr[:p]); pt[p-1]=np.mean(pdm[:p]); nt[p-1]=np.mean(ndm[:p])
    for i in range(p, n):
        at[i]=a*tr[i]+(1-a)*at[i-1]; pt[i]=a*pdm[i]+(1-a)*pt[i-1]; nt[i]=a*ndm[i]+(1-a)*nt[i-1]
    pdi=np.where(at>0,100*pt/at,0); ndi=np.where(at>0,100*nt/at,0)
    ds=pdi+ndi; dx=np.where(ds>0,100*np.abs(pdi-ndi)/ds,0)
    r[p*2-2]=np.mean(dx[p-1:p*2-1])
    for i in range(p*2-1,n): r[i]=a*dx[i]+(1-a)*r[i-1]
    return r


# ═══════════════════════════════════════
# 大周期趋势策略
# ═══════════════════════════════════════

class SwingTrendStrategy(Strategy):
    """
    大周期趋势跟踪 — 4H K线, 极简逻辑

    入场条件 (全部满足):
      1. EMA排列: 20 > 50 > 100 (多头) 或 20 < 50 < 100 (空头)
      2. ADX > 25: 强趋势市场
      3. 价格回调到 EMA20 附近 (≤ 1.5x ATR)

    出场:
      - Trailing Stop 3x ATR (从趋势极值回撤)
      - 或 EMA20/50 死叉/金叉

    风控:
      2% 单笔风险, 每日最多1笔
    """

    # ── 均线 ──
    ema_fast = 20
    ema_mid = 50
    ema_slow = 100

    # ── 趋势 ──
    adx_period = 14
    adx_threshold = 25       # 强趋势门槛

    # ── 止损/止盈 ──
    atr_period = 14
    atr_stop_mult = 3.0       # Trailing Stop 倍数
    atr_entry_mult = 1.5      # 回调容忍度
    use_fixed_tp = True         # 固定止盈 (测试集验证优于纯TS)
    tp_mult = 3.0              # 止盈倍数 3x ATR

    # ── 波动率自适应仓位 ──
    vol_adaptive = True        # 低波加仓/高波减仓
    vol_lookback = 50          # 波动率参考周期
    vol_min_risk = 0.01        # 最小风险 (高波动时)
    vol_max_risk = 0.03        # 最大风险 (低波动时)

    # ── 风控 ──
    risk_per_trade = 0.02      # 基准单笔风险
    cooldown_bars = 6          # 4H * 6 = 1天冷却

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low

        self.ema_f = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA20")
        self.ema_m = self.I(_ema, c, self.ema_mid,  overlay=True, name="EMA50")
        self.ema_s = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA100")
        self.atr_v = self.I(_atr, h, l, c, self.atr_period, name="ATR14")
        self.adx_v = self.I(_adx, h, l, c, self.adx_period, name="ADX14")

        # 日线方向过滤
        has_daily = "1d_Close" in self.data.df.columns
        if has_daily:
            dc = self.data.df["1d_Close"].values
            self.d_ema_f = self.I(_ema, dc, self.ema_fast, overlay=False, name="D_EMA20")
            self.d_ema_m = self.I(_ema, dc, self.ema_mid,  overlay=False, name="D_EMA50")
            self._daily_filter = True
            logger.info(f"日线方向过滤: 已启用")
        else:
            self._daily_filter = False
            logger.info(f"日线方向过滤: 无日线数据, 跳过")

        self._highest = 0.0
        self._lowest = float("inf")
        self._trail_stop = 0.0
        self._bars_since_exit = 999
        self._atr_rates = []

    def _adaptive_risk(self, atr: float, price: float) -> float:
        """波动率自适应风险: 低波→加仓, 高波→减仓"""
        atr_rate = atr / price
        self._atr_rates.append(atr_rate)
        if len(self._atr_rates) > self.vol_lookback:
            self._atr_rates.pop(0)
        if len(self._atr_rates) < 10:
            return self.risk_per_trade
        avg = sum(self._atr_rates) / len(self._atr_rates)
        if avg <= 0: return self.risk_per_trade
        ratio = avg / atr_rate  # >1=低波, <1=高波
        ratio = max(0.5, min(ratio, 2.0))  # 限制在 0.5~2.0 之间
        risk = self.risk_per_trade * ratio
        return max(self.vol_min_risk, min(risk, self.vol_max_risk))

    def next(self):
        c = self.data.Close[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        atr = self.atr_v[-1]

        ef = self.ema_f[-1]; em = self.ema_m[-1]; es = self.ema_s[-1]
        adx = self.adx_v[-1]

        if any(np.isnan(x) for x in [ef, em, es, atr, adx]) or c <= 0:
            return

        self._bars_since_exit += 1

        bull = ef > em > es
        bear = ef < em < es
        strong = adx > self.adx_threshold

        # 日线方向过滤: 日线多头→只做多, 日线空头→只做空
        if self._daily_filter:
            d_ef = self.d_ema_f[-1]; d_em = self.d_ema_m[-1]
            if not (np.isnan(d_ef) or np.isnan(d_em)):
                daily_bull = d_ef > d_em
                daily_bear = d_ef < d_em
                if daily_bull: bear = False    # 日线多头 → 禁止做空
                if daily_bear: bull = False    # 日线空头 → 禁止做多

        # ── 持仓管理 ──
        if self.position:
            if self.position.is_long:
                if h > self._highest:
                    self._highest = h
                    self._trail_stop = h - atr * self.atr_stop_mult
                if l <= self._trail_stop or ef < em:  # TS触发 或 死叉
                    self.position.close()
                    self._bars_since_exit = 0

            elif self.position.is_short:
                if l < self._lowest:
                    self._lowest = l
                    self._trail_stop = l + atr * self.atr_stop_mult
                if h >= self._trail_stop or ef > em:  # TS触发 或 金叉
                    self.position.close()
                    self._bars_since_exit = 0
            return

        # ── 冷却 + 强趋势 ──
        if self._bars_since_exit < self.cooldown_bars:
            return
        if not strong:
            return

        # ── 入场: 回调到 EMA20 附近 ──
        near_ema = abs(c - ef) <= atr * self.atr_entry_mult

        # 做多
        if bull and near_ema:
            sl = c - atr * self.atr_stop_mult
            tp = c + atr * self.tp_mult if self.use_fixed_tp else None
            risk = self._adaptive_risk(atr, c) if self.vol_adaptive else self.risk_per_trade
            frac = min(risk / (self.atr_stop_mult * atr / c), 0.95)
            if frac > 0:
                self.buy(size=frac, sl=sl, tp=tp)
                self._highest = h; self._trail_stop = sl
                self._bars_since_exit = 999
                logger.info(f"📈 LONG @{c:.0f} ADX={adx:.0f} 风险={risk:.1%}")

        # 做空
        elif bear and near_ema:
            sl = c + atr * self.atr_stop_mult
            tp = c - atr * self.tp_mult if self.use_fixed_tp else None
            risk = self._adaptive_risk(atr, c) if self.vol_adaptive else self.risk_per_trade
            frac = min(risk / (self.atr_stop_mult * atr / c), 0.95)
            if frac > 0:
                self.sell(size=frac, sl=sl, tp=tp)
                self._lowest = l; self._trail_stop = sl
                self._bars_since_exit = 999
                logger.info(f"📉 SHORT @{c:.0f} ADX={adx:.0f} 风险={risk:.1%}")
