"""
BTC 日线/4h 趋势跟踪策略

基于49策略回测研究 (dev.to/maymay5692)，适配日线和4小时线：
1. EMA金叉 (对标 #1, 日线夏普1.30/收益491%)
2. 三均线排列 (对标 #4, 日线夏普1.25/收益502%)
3. MACD趋势 (对标 #5, 日线夏普1.17/收益428%)
4. Donchian突破 (对标 #10, 日线夏普1.06/收益320%)

核心改进：
- 200EMA市场状态过滤 (牛市只做多，熊市只做空)
- ADX > 20 趋势强度确认
- ATR动态止损止盈 (1.5x/3x)
- 连续亏损自动冷却
"""

from backtesting import Strategy
import numpy as np


# ═══════════════════════════════════════════
# NumPy 指标
# ═══════════════════════════════════════════

def _ema(a, span):
    if span <= 0 or len(a) < span: return np.full_like(a, np.nan)
    alpha = 2.0 / (span + 1.0); r = np.full_like(a, np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _atr(h, l, c, period=14):
    r = np.full_like(c, np.nan, dtype=float)
    if len(c) < period + 1: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    r[period - 1] = np.mean(tr[:period]); alpha = 1.0 / period
    for i in range(period, len(tr)): r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def _adx(h, l, c, period=14):
    n = len(c); r = np.full(n, np.nan, dtype=float)
    if n < period * 2: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    up_move = h - np.roll(h, 1); down_move = np.roll(l, 1) - l
    up_move[0] = down_move[0] = 0
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    alpha = 1.0 / period
    at = np.full(n, np.nan); pt = np.full(n, np.nan); nt = np.full(n, np.nan)
    at[period - 1] = np.mean(tr[:period])
    pt[period - 1] = np.mean(pdm[:period]); nt[period - 1] = np.mean(ndm[:period])
    for i in range(period, n):
        at[i] = alpha * tr[i] + (1 - alpha) * at[i - 1]
        pt[i] = alpha * pdm[i] + (1 - alpha) * pt[i - 1]
        nt[i] = alpha * ndm[i] + (1 - alpha) * nt[i - 1]
    pdi = np.where(at > 0, 100 * pt / at, 0); ndi = np.where(at > 0, 100 * nt / at, 0)
    ds = pdi + ndi
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(ds > 0, 100 * np.abs(pdi - ndi) / ds, 0)
    r[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    for i in range(period * 2 - 1, n): r[i] = alpha * dx[i] + (1 - alpha) * r[i - 1]
    return r


def _macd(a, fast=12, slow=26, signal=9):
    ef = _ema(a, fast); es = _ema(a, slow)
    macd_line = ef - es; sig_line = _ema(macd_line, signal)
    return macd_line, sig_line, macd_line - sig_line


def _highest(a, w):
    r = np.full_like(a, np.nan)
    for i in range(w - 1, len(a)): r[i] = np.max(a[i - w + 1:i + 1])
    return r


def _lowest(a, w):
    r = np.full_like(a, np.nan)
    for i in range(w - 1, len(a)): r[i] = np.min(a[i - w + 1:i + 1])
    return r


# ═══════════════════════════════════════════
# 基类: 公共风险管理和市场状态
# ═══════════════════════════════════════════

class BaseBtcStrategy(Strategy):
    """BTC策略基类：200EMA市场状态 + 冷却期 + 动态止损 + 杠杆"""
    ema_trend = 200       # 长期趋势EMA
    adx_period = 14
    adx_min = 20          # ADX最低阈值
    atr_period = 14
    atr_sl_mult = 1.5     # 止损 ATR倍数
    atr_tp_mult = 3.0     # 止盈 ATR倍数
    cooldown_base = 3     # 基础冷却期(根K线)
    cooldown_per_loss = 2 # 每次亏损额外冷却
    leverage = 3           # 杠杆倍数 (1=现货, 3=3x)

    def _init_base(self):
        """子类在 init() 中调用此方法"""
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ema200 = self.I(_ema, c, self.ema_trend, overlay=True, name="EMA200")
        self.adx = self.I(_adx, h, l, c, self.adx_period, name="ADX")
        self.atr = self.I(_atr, h, l, c, self.atr_period, name="ATR")
        self._consecutive_losses = 0
        self._last_trade_bar = -999
        self._entry_price = 0

    def _market_bull(self) -> bool:
        """牛市判断：价格 > 200EMA"""
        c = self.data.Close[-1]; e200 = self.ema200[-1]
        return not np.isnan(e200) and c > e200

    def _trend_strong(self) -> bool:
        """趋势足够强：ADX > 阈值"""
        adx_v = self.adx[-1]
        return not np.isnan(adx_v) and adx_v > self.adx_min

    def _can_trade(self) -> bool:
        """检查是否满足交易条件（冷却期+趋势确认）"""
        if not self._trend_strong(): return False
        bars_since = len(self.data) - self._last_trade_bar
        cooldown = self.cooldown_base + self._consecutive_losses * self.cooldown_per_loss
        return bars_since >= cooldown

    def _on_close(self):
        """平仓时记录盈亏，更新冷却状态"""
        if self.position and self.position.pl is not None:
            if self.position.pl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = max(0, self._consecutive_losses - 1)
        self._last_trade_bar = len(self.data)

    def _buy_with_stops(self):
        """带ATR止损止盈的做多（含杠杆）"""
        c = self.data.Close[-1]; atr_v = self.atr[-1]
        # 杠杆越高，止损越紧（同比例缩小ATR倍数以控制风险）
        sl_adj = self.atr_sl_mult / self.leverage
        tp_adj = self.atr_tp_mult / self.leverage
        sl = c - atr_v * max(sl_adj, 0.3)
        tp = c + atr_v * max(tp_adj, 0.5)
        self.buy(size=self.leverage, sl=sl, tp=tp)

    def _sell_with_stops(self):
        """带ATR止损止盈的做空（含杠杆）"""
        c = self.data.Close[-1]; atr_v = self.atr[-1]
        sl_adj = self.atr_sl_mult / self.leverage
        tp_adj = self.atr_tp_mult / self.leverage
        sl = c + atr_v * max(sl_adj, 0.3)
        tp = c - atr_v * max(tp_adj, 0.5)
        self.sell(size=self.leverage, sl=sl, tp=tp)


# ═══════════════════════════════════════════
# 策略1: EMA金叉 (对标 #1 最佳策略)
# ═══════════════════════════════════════════

class EmaCrossTrend(BaseBtcStrategy):
    """
    EMA12/26金叉死叉 + 200EMA市场状态过滤

    对标研究中 #1 ema_crossover (日线夏普1.30, 收益491%)
    - 牛市(价格>200EMA): 金叉做多，死叉平仓
    - 熊市(价格<200EMA): 死叉做空，金叉平仓
    - ADX > 20 确认趋势强度
    """
    ema_fast = 12
    ema_slow = 26

    def init(self):
        c = self.data.Close
        self.ef = self.I(_ema, c, self.ema_fast, overlay=True, name="EMA12")
        self.es = self.I(_ema, c, self.ema_slow, overlay=True, name="EMA26")
        self._init_base()

    def next(self):
        c = self.data.Close[-1]
        ef = self.ef[-1]; es = self.es[-1]
        ef_p = self.ef[-2]; es_p = self.es[-2]

        if any(np.isnan(x) for x in [ef, es, ef_p, es_p]) or c <= 0: return

        cross_up = ef_p <= es_p and ef > es      # 金叉
        cross_down = ef_p >= es_p and ef < es    # 死叉
        bull = self._market_bull()

        # 平仓: 反向信号
        if self.position:
            if self.position.is_long and cross_down:
                self.position.close(); self._on_close()
            elif self.position.is_short and cross_up:
                self.position.close(); self._on_close()
            return

        if not self._can_trade(): return

        # 牛市做多，熊市做空
        if bull and cross_up:
            self._buy_with_stops()
        elif not bull and cross_down:
            self._sell_with_stops()


# ═══════════════════════════════════════════
# 策略2: 三均线排列 (对标 #4)
# ═══════════════════════════════════════════

class TripleEmaTrend(BaseBtcStrategy):
    """
    EMA10/20/50三线排列 + 回调入场

    对标 #4 triple_ma (日线夏普1.25, 收益502%)
    - 完美多头排列(EMA10>20>50)且价格靠近EMA20 → 做多
    - 完美空头排列(EMA10<20<50)且价格靠近EMA20 → 做空
    """
    ema1 = 10; ema2 = 20; ema3 = 50

    def init(self):
        c = self.data.Close
        self.e10 = self.I(_ema, c, self.ema1, overlay=True, name="EMA10")
        self.e20 = self.I(_ema, c, self.ema2, overlay=True, name="EMA20")
        self.e50 = self.I(_ema, c, self.ema3, overlay=True, name="EMA50")
        self._init_base()

    def next(self):
        c = self.data.Close[-1]
        e10 = self.e10[-1]; e20 = self.e20[-1]; e50 = self.e50[-1]
        atr_v = self.atr[-1]

        if any(np.isnan(x) for x in [e10, e20, e50, atr_v]) or c <= 0: return

        bull_align = e10 > e20 > e50   # 完美多头排列
        bear_align = e10 < e20 < e50   # 完美空头排列
        near_ema20 = abs(c - e20) < atr_v * 1.0  # 价格在EMA20附近
        bull = self._market_bull()

        # 平仓: 排列破坏
        if self.position:
            if self.position.is_long and not bull_align:
                self.position.close(); self._on_close()
            elif self.position.is_short and not bear_align:
                self.position.close(); self._on_close()
            return

        if not self._can_trade(): return

        if bull and bull_align and near_ema20:
            self._buy_with_stops()
        elif not bull and bear_align and near_ema20:
            self._sell_with_stops()


# ═══════════════════════════════════════════
# 策略3: MACD趋势 (对标 #5)
# ═══════════════════════════════════════════

class MacdTrend(BaseBtcStrategy):
    """
    MACD(12,26,9)零轴金叉死叉 + 200EMA过滤

    对标 #5 macd (日线夏普1.17, 收益428%)
    - MACD从零轴下方上穿信号线 + 牛市 → 做多
    - MACD从零轴上方下穿信号线 + 熊市 → 做空
    """
    macd_fast = 12; macd_slow = 26; macd_sig = 9

    def init(self):
        c = self.data.Close
        self.ml, self.ms, self.mh = self.I(_macd, c, self.macd_fast,
                                            self.macd_slow, self.macd_sig, name="MACD")
        self._init_base()

    def next(self):
        c = self.data.Close[-1]
        ml = self.ml[-1]; ms = self.ms[-1]
        ml_p = self.ml[-2]; ms_p = self.ms[-2]

        if any(np.isnan(x) for x in [ml, ms, ml_p, ms_p]) or c <= 0: return

        golden = ml_p <= ms_p and ml > ms   # 金叉
        dead = ml_p >= ms_p and ml < ms     # 死叉
        bull = self._market_bull()

        if self.position:
            if self.position.is_long and dead:
                self.position.close(); self._on_close()
            elif self.position.is_short and golden:
                self.position.close(); self._on_close()
            return

        if not self._can_trade(): return

        # 零轴上金叉做多，零轴下死叉做空
        if bull and golden and ml > 0:
            self._buy_with_stops()
        elif not bull and dead and ml < 0:
            self._sell_with_stops()


# ═══════════════════════════════════════════
# 策略4: Donchian突破 (对标 #10)
# ═══════════════════════════════════════════

class DonchianTrend(BaseBtcStrategy):
    """
    20日 Donchian通道突破 + ADX + 200EMA

    对标 #10 donchian_breakout (日线夏普1.06, 收益320%)
    - 价格突破20日最高 + 牛市 + ADX>20 → 做多
    - 价格跌破20日最低 + 熊市 + ADX>20 → 做空
    """
    donchian = 20

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.hh = self.I(_highest, h, self.donchian, name="HH20")
        self.ll = self.I(_lowest, l, self.donchian, name="LL20")
        self._init_base()

    def next(self):
        c = self.data.Close[-1]; h = self.data.High[-1]; l = self.data.Low[-1]
        hh = self.hh[-1]; ll = self.ll[-1]

        if any(np.isnan(x) for x in [hh, ll]) or c <= 0: return

        bull = self._market_bull()

        if self.position:
            if self.position.is_long and c < ll:
                self.position.close(); self._on_close()
            elif self.position.is_short and c > hh:
                self.position.close(); self._on_close()
            return

        if not self._can_trade(): return

        if bull and h > hh:
            self._buy_with_stops()
        elif not bull and l < ll:
            self._sell_with_stops()


# ═══════════════════════════════════════════
# 策略注册
# ═══════════════════════════════════════════

DAILY_STRATEGIES = {
    "ema_cross": EmaCrossTrend,
    "triple_ema": TripleEmaTrend,
    "macd_trend": MacdTrend,
    "donchian": DonchianTrend,
}
