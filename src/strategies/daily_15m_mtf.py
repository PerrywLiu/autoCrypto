"""
15分钟 + 日线趋势过滤 + 3x杠杆 v2

日线EMA在数据准备阶段预计算（避免前向填充重复值失真）。
策略直接读取合并后的 1d_Close, 1d_ema200, 1d_ema50 列。
"""

from backtesting import Strategy
import numpy as np


def _ema(a, span):
    if span <= 0 or len(a) < span: return np.full(len(a), np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full(len(a), np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def _atr(h, l, c, period=14):
    r = np.full(len(c), np.nan, dtype=float)
    if len(c) < period + 1: return r
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    r[period - 1] = np.mean(tr[:period]); alpha = 1.0 / period
    for i in range(period, len(tr)): r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def _supertrend(h, l, c, period=10, mult=3.0):
    atr_v = _atr(h, l, c, period)
    hl_avg = (h + l) / 2.0
    upper = hl_avg + mult * atr_v; lower = hl_avg - mult * atr_v
    n = len(c); trend = np.zeros(n)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]): continue
        if c[i] > upper[i - 1]: trend[i] = 1
        elif c[i] < lower[i - 1]: trend[i] = -1
        else: trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1
        if trend[i] == 1 and lower[i] < lower[i - 1]: lower[i] = lower[i - 1]
        if trend[i] == -1 and upper[i] > upper[i - 1]: upper[i] = upper[i - 1]
    return trend


# ═══════════════════════
# 策略1: 日线EMA方向 + 15m EMA金叉
# ═══════════════════════

class DailyTrend15mEntryV2(Strategy):
    """日线EMA200定方向 + 15m EMA9/21金叉入场 + 3x杠杆"""
    ema_fast = 9; ema_slow = 21
    atr_sl_mult = 2.0; atr_tp_mult = 5.0
    leverage = 3

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.ef = self.I(_ema, c, self.ema_fast, overlay=True, name="15m_EMA9")
        self.es = self.I(_ema, c, self.ema_slow, overlay=True, name="15m_EMA21")
        self.atr = self.I(_atr, h, l, c, 14, name="15m_ATR")

        self._has_daily = "1d_ema200" in self.data.df.columns
        self._consecutive_losses = 0
        self._last_trade_bar = -999

    def _daily_dir(self) -> int:
        """1=强牛, -1=强熊, 0=中性"""
        if not self._has_daily: return 0
        i = len(self.data) - 1
        df = self.data.df
        try:
            c1d = df["1d_Close"].iloc[i]
            e200 = df["1d_ema200"].iloc[i]
            e50 = df["1d_ema50"].iloc[i]
        except (IndexError, KeyError): return 0
        if np.isnan(e200) or np.isnan(e50): return 0
        if c1d > e50 > e200: return 1
        if c1d < e50 < e200: return -1
        return 0

    def next(self):
        c = self.data.Close[-1]; atr_v = self.atr[-1]
        if c <= 0 or np.isnan(atr_v): return

        ef = self.ef[-1]; es = self.es[-1]
        ef_p = self.ef[-2]; es_p = self.es[-2]
        if any(np.isnan(x) for x in [ef, es, ef_p, es_p]): return

        cross_up = ef_p <= es_p and ef > es
        cross_down = ef_p >= es_p and ef < es
        daily = self._daily_dir()

        bars_since = len(self.data) - self._last_trade_bar
        cooldown = 12 + self._consecutive_losses * 6

        if self.position:
            if self.position.is_long and cross_down:
                self.position.close()
                self._consecutive_losses = (self._consecutive_losses + 1
                    if self.position.pl and self.position.pl < 0 else 0)
                self._last_trade_bar = len(self.data)
            elif self.position.is_short and cross_up:
                self.position.close()
                self._consecutive_losses = (self._consecutive_losses + 1
                    if self.position.pl and self.position.pl < 0 else 0)
                self._last_trade_bar = len(self.data)
            return

        if bars_since < cooldown: return

        if daily == 1 and cross_up:
            self.buy(size=self.leverage,
                     sl=c-atr_v*self.atr_sl_mult, tp=c+atr_v*self.atr_tp_mult)
            self._last_trade_bar = len(self.data)
        elif daily == -1 and cross_down:
            self.sell(size=self.leverage,
                      sl=c+atr_v*self.atr_sl_mult, tp=c-atr_v*self.atr_tp_mult)
            self._last_trade_bar = len(self.data)


# ═══════════════════════
# 策略2: 日线方向 + 15m SuperTrend
# ═══════════════════════

class DailySuperTrend15mV2(Strategy):
    """日线EMA定方向 + 15m SuperTrend(7,2.5) + 3x杠杆"""
    st_period = 7; st_mult = 2.5
    atr_sl_mult = 1.5; atr_tp_mult = 4.0
    leverage = 3

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.st = self.I(_supertrend, h, l, c, self.st_period, self.st_mult, name="ST")
        self.atr = self.I(_atr, h, l, c, 14, name="ATR")

        self._has_daily = "1d_ema200" in self.data.df.columns
        self._consecutive_losses = 0
        self._last_trade_bar = -999

    def _daily_dir(self) -> int:
        if not self._has_daily: return 0
        i = len(self.data) - 1
        df = self.data.df
        try:
            c1d = df["1d_Close"].iloc[i]
            e200 = df["1d_ema200"].iloc[i]
            e50 = df["1d_ema50"].iloc[i]
        except: return 0
        if np.isnan(e200) or np.isnan(e50): return 0
        if c1d > e50 > e200: return 1
        if c1d < e50 < e200: return -1
        return 0

    def next(self):
        c = self.data.Close[-1]; atr_v = self.atr[-1]
        if c <= 0 or np.isnan(atr_v): return

        st_now = self.st[-1]; st_prev = self.st[-2]
        if np.isnan(st_now) or np.isnan(st_prev): return

        flip_up = st_prev <= 0 and st_now > 0
        flip_down = st_prev > 0 and st_now <= 0
        daily = self._daily_dir()

        bars_since = len(self.data) - self._last_trade_bar
        cooldown = 24 + self._consecutive_losses * 8

        if self.position:
            if self.position.is_long and flip_down:
                self.position.close()
                self._consecutive_losses = (self._consecutive_losses + 1
                    if self.position.pl and self.position.pl < 0 else 0)
                self._last_trade_bar = len(self.data)
            elif self.position.is_short and flip_up:
                self.position.close()
                self._consecutive_losses = (self._consecutive_losses + 1
                    if self.position.pl and self.position.pl < 0 else 0)
                self._last_trade_bar = len(self.data)
            return

        if bars_since < cooldown: return

        if flip_up and daily == 1:
            self.buy(size=self.leverage,
                     sl=c-atr_v*self.atr_sl_mult, tp=c+atr_v*self.atr_tp_mult)
            self._last_trade_bar = len(self.data)
        elif flip_down and daily == -1:
            self.sell(size=self.leverage,
                      sl=c+atr_v*self.atr_sl_mult, tp=c-atr_v*self.atr_tp_mult)
            self._last_trade_bar = len(self.data)


MTF_STRATEGIES = {
    "daily_ema_15m": DailyTrend15mEntryV2,
    "daily_st_15m": DailySuperTrend15mV2,
}
