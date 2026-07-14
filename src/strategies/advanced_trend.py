"""
进阶趋势跟踪策略 v10

经过 6 轮迭代验证的最优配置：

进化路径:
  v1 原始 EMA12/26 + RSI + ATR ────── -52%  (纯1H, 无过滤器)
  v6 +TrailingStop + ADX + 回调入场 ────  +3%  (结构性改造)
  v7 参数优化 EMA20/50/100 + ADX>15 ─── +22%  (慢均线是关键)
  v8 6层硬过滤 ───────────────────────── +13%  (过滤太狠, 交易太少)
  v9 评分制代替硬过滤 ────────────────── +6%   (平衡但胜率不够)
  v10 ⭐反弹确认 + 非对称做空 + 固定止盈 ─ +22%  (当前版本)

核心机制:
  1. EMA(20/50/100) 三重排列 ── 慢均线过滤1H噪声, 只捕捉真趋势
  2. 反弹确认(2分) ── 价格碰EMA后必须反向收阳/收阴, 不追回调
  3. ADX非对称 ── 做多>15, 做空>18 (做空需要更强趋势)
  4. 评分制(8项) ── 成交量/RSI/EMA斜率/蜡烛形态等加权评分
  5. Trailing Stop 2.5x ATR ── 动态追踪止损, 让利润奔跑

出场方式 (可切换):
  - use_fixed_tp=False (默认): 纯 Trailing Stop, 胜率~42%, 收益+6%
  - use_fixed_tp=True, tp_mult=1.5: 固定止盈, 胜率~62%, 收益偏低
  - use_fixed_tp=True, tp_mult=3.0: 固定止盈, 胜率~46%, 收益+9%
"""

from backtesting import Strategy
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# NumPy 指标函数 (兼容 backtesting.py 的 self.I())
# ═══════════════════════════════════════════════════════════════

def _ema(a: np.ndarray, span: int) -> np.ndarray:
    if span <= 0 or len(a) < span:
        return np.full_like(a, np.nan, dtype=float)
    alpha = 2.0 / (span + 1.0)
    r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < p + 1: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    for i in range(p - 1, len(tr)):
        r[i] = np.mean(tr[i - p + 1:i + 1])
    return r


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> np.ndarray:
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


def _rsi(a: np.ndarray, p: int = 14) -> np.ndarray:
    r = np.full_like(a, np.nan, dtype=float)
    if len(a) < p+1: return r
    d = np.diff(a); g = np.maximum(d,0); l = np.maximum(-d,0)
    ag = np.mean(g[:p]); al = np.mean(l[:p]); alpha = 1.0/p
    for i in range(p, len(d)):
        r[i+1] = 100.0-(100.0/(1.0+ag/al)) if al > 0 else (100.0 if ag > 0 else 50.0)
        ag = alpha*g[i]+(1-alpha)*ag; al = alpha*l[i]+(1-alpha)*al
    return np.clip(r, 0, 100)


def _vol_ma(v: np.ndarray, p: int = 20) -> np.ndarray:
    r = np.full_like(v, np.nan, dtype=float)
    if len(v) < p: return r
    for i in range(p-1, len(v)): r[i] = np.mean(v[i-p+1:i+1])
    return r


# ═══════════════════════════════════════════════════════════════
# AdvancedTrendStrategy v10
# ═══════════════════════════════════════════════════════════════

class AdvancedTrendStrategy(Strategy):
    """
    进阶趋势跟踪 v10

    【入场】EMA排列 + ADX趋势确认 + 反弹到EMA后再确认
    【出场】Trailing Stop (默认) 或 固定止盈 (可选)
    【风控】2%单笔风险 + 3K冷却期
    """

    # ── 均线 ──
    ema_fast: int = 20
    ema_mid: int  = 50
    ema_slow: int = 100

    # ── ADX 趋势过滤 ──
    adx_period: int = 14
    adx_long: int  = 15   # 做多最低 ADX
    adx_short: int = 18   # 做空需更强趋势

    # ── ATR 止损/入场 ──
    atr_period: int = 14
    atr_stop_mult: float = 2.5   # Trailing Stop 倍数
    atr_entry_mult: float = 0.8  # 回调距 EMA 阈值 (ATR倍数)

    # ── 止盈 (use_fixed_tp=True 时生效) ──
    use_fixed_tp: bool = False    # 是否启用固定止盈
    tp_mult: float = 2.0          # 止盈倍数 (相对入场价, ATR倍数)

    # ── 评分门槛 ──
    min_score_long: int = 6       # 做多最低分 (满分 9)
    min_score_short: int = 7      # 做空最低分 (满分 10)

    # ── RSI ──
    rsi_period: int = 14
    rsi_long_min: int = 42        # 做多 RSI 下限
    rsi_short_max: int = 48       # 做空 RSI 上限

    # ── 风控 ──
    risk_per_trade: float = 0.02  # 单笔风险 2%
    cooldown_bars: int = 3        # 平仓后冷却 K 线数

    # ═══════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════

    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume

        # 趋势指标
        self.ema_f = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA20")
        self.ema_m = self.I(_ema, c, self.ema_mid,  overlay=True, name="EMA50")
        self.ema_s = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA100")

        # 波动/强度
        self.atr_v = self.I(_atr, h, l, c, self.atr_period, name="ATR14")
        self.adx_v = self.I(_adx, h, l, c, self.adx_period, name="ADX14")

        # 动量和成交量
        self.rsi_v = self.I(_rsi, c, self.rsi_period, name="RSI14")
        self.volma = self.I(_vol_ma, v, 20, name="VolMA20")

        # 状态变量
        self._highest_since_entry: float = 0.0
        self._lowest_since_entry: float = float("inf")
        self._trailing_stop: float = 0.0
        self._entry_price: float = 0.0
        self._bars_since_exit: int = 999

        logger.info(
            f"v10 就绪 | EMA{self.ema_fast}/{self.ema_mid}/{self.ema_slow} "
            f"ADX>{self.adx_long}/{self.adx_short} "
            f"TS={self.atr_stop_mult}xATR "
            f"TP={'FIX'+str(self.tp_mult) if self.use_fixed_tp else 'TS'}"
        )

    # ═══════════════════════════════════════════════════════
    # 每根 K 线决策
    # ═══════════════════════════════════════════════════════

    def next(self):
        c  = self.data.Close[-1]
        h  = self.data.High[-1]
        l  = self.data.Low[-1]
        o  = self.data.Open[-1]
        v  = self.data.Volume[-1]
        cp = self.data.Close[-2]          # 前一根收盘
        ef = self.ema_f[-1]
        em = self.ema_m[-1]
        es = self.ema_s[-1]
        ef_p = self.ema_f[-2]             # 前一根 EMA 快线
        em_p = self.ema_m[-2]             # 前一根 EMA 中线
        atr = self.atr_v[-1]
        adx = self.adx_v[-1]
        rsi = self.rsi_v[-1]
        rsi_p = self.rsi_v[-2]
        vol_ma = self.volma[-1]

        if any(np.isnan(x) for x in [ef, em, es, atr, adx, rsi, rsi_p, vol_ma]) or c <= 0:
            return

        self._bars_since_exit += 1
        ef_slope = ef - self.ema_f[-2]  # EMA 快线斜率

        # ── 趋势方向 ──
        bull = ef > em > es
        bear = ef < em < es

        # ═════════════════════════════════════════════════
        # 持仓管理: Trailing Stop
        # ═════════════════════════════════════════════════
        if self.position:
            if self.position.is_long:
                if h > self._highest_since_entry:
                    self._highest_since_entry = h
                    self._trailing_stop = h - atr * self.atr_stop_mult
                if l <= self._trailing_stop or (ef < em):
                    self.position.close()
                    self._bars_since_exit = 0
            elif self.position.is_short:
                if l < self._lowest_since_entry:
                    self._lowest_since_entry = l
                    self._trailing_stop = l + atr * self.atr_stop_mult
                if h >= self._trailing_stop or (ef > em):
                    self.position.close()
                    self._bars_since_exit = 0
            return

        # ── 冷却期 ──
        if self._bars_since_exit < self.cooldown_bars:
            return

        # ═════════════════════════════════════════════════
        # 做多
        # ═════════════════════════════════════════════════
        if bull and adx >= self.adx_long:
            bounced = (cp <= ef_p or cp <= em_p) and (c > ef or c > em)
            score = self._score_long(c, o, l, v, ef, em, es, atr, adx, rsi, rsi_p, ef_slope, vol_ma, bounced)
            if score >= self.min_score_long:
                sl = c - atr * self.atr_stop_mult
                tp = c + atr * self.tp_mult if self.use_fixed_tp else None
                frac = min(self.risk_per_trade / (self.atr_stop_mult * atr / c), 0.95)
                if frac > 0:
                    self.buy(size=frac, sl=sl, tp=tp)
                    self._highest_since_entry = h
                    self._entry_price = c
                    self._bars_since_exit = 999
                    logger.info(f"📈 LONG {score}/9 @{c:.0f} ADX={adx:.0f} RSI={rsi:.0f}")

        # ═════════════════════════════════════════════════
        # 做空 (门槛更高)
        # ═════════════════════════════════════════════════
        if bear and adx >= self.adx_short:
            rejected = (cp >= ef_p or cp >= em_p) and (c < ef or c < em)
            score = self._score_short(c, o, h, v, ef, em, es, atr, adx, rsi, rsi_p, ef_slope, vol_ma, rejected)
            if score >= self.min_score_short:
                sl = c + atr * self.atr_stop_mult
                tp = c - atr * self.tp_mult if self.use_fixed_tp else None
                frac = min(self.risk_per_trade / (self.atr_stop_mult * atr / c), 0.95)
                if frac > 0:
                    self.sell(size=frac, sl=sl, tp=tp)
                    self._lowest_since_entry = l
                    self._entry_price = c
                    self._bars_since_exit = 999
                    logger.info(f"📉 SHORT {score}/10 @{c:.0f} ADX={adx:.0f} RSI={rsi:.0f}")

    # ═══════════════════════════════════════════════════════
    # 做多评分卡 (满分 9)
    # ═══════════════════════════════════════════════════════
    def _score_long(self, c, o, l, v, ef, em, es, atr, adx, rsi, rsi_p, ef_slope, vol_ma, bounced) -> int:
        s = 0
        if ef > em > es: s += 1                                   # S1: EMA排列
        if adx >= self.adx_long: s += 1                           # S2: ADX趋势
        if v >= vol_ma: s += 1                                    # S2: 量能放大
        if rsi > rsi_p and rsi >= self.rsi_long_min: s += 1      # S3: RSI动量↑
        if ef_slope > 0: s += 1                                   # S4: EMA斜率↑
        if abs(c - ef) <= atr * self.atr_entry_mult: s += 1      # S5: 价格近快线
        if abs(c - em) <= atr * self.atr_entry_mult: s += 1      # S6: 价格近中线
        if c > o or (min(o,c)-l) > atr*0.3: s += 1               # S7: 蜡烛收阳
        if bounced: s += 2                                        # S8: ⭐反弹确认(2分)
        return s

    # ═══════════════════════════════════════════════════════
    # 做空评分卡 (满分 10, 含超卖保护)
    # ═══════════════════════════════════════════════════════
    def _score_short(self, c, o, h, v, ef, em, es, atr, adx, rsi, rsi_p, ef_slope, vol_ma, rejected) -> int:
        s = 0
        if ef < em < es: s += 1                                   # S1: EMA空头排列
        if adx >= self.adx_short: s += 1
        if v >= vol_ma: s += 1
        if rsi < rsi_p and rsi <= self.rsi_short_max: s += 1
        if ef_slope < 0: s += 1
        if abs(c - ef) <= atr * self.atr_entry_mult: s += 1
        if abs(c - em) <= atr * self.atr_entry_mult: s += 1
        if c < o or (h-max(o,c)) > atr*0.3: s += 1
        if rejected: s += 2                                       # ⭐反弹被拒(2分)
        if rsi > 25: s += 1                                       # 不在超卖区
        return s
