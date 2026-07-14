"""
实盘交易机器人 — 双策略并行

策略 A: 自适应混合趋势跟踪 (4H K线)
策略 B: 资金费率套利 (现货多+永续空, 每8h收资金费率, delta中性)
"""

import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════
# 策略指标 (NumPy, 与回测一致)
# ═══════════════════════════════════════════════

def _ema(a, span):
    if span <= 0 or len(a) < span: return np.full_like(a, np.nan, dtype=float)
    alpha = 2.0/(span+1.0); r = np.full_like(a, np.nan, dtype=float)
    r[span-1] = np.mean(a[:span])
    for i in range(span, len(a)): r[i] = alpha*a[i] + (1-alpha)*r[i-1]
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


# ═══════════════════════════════════════════════
# 实盘机器人
# ═══════════════════════════════════════════════

class LiveTradingBot:
    """
    实盘交易机器人

    用法:
        bot = LiveTradingBot(api_key, secret, passphrase, testnet=True)
        bot.start()
    """

    def __init__(
        self,
        api_key: str, secret: str, passphrase: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "4h",
        testnet: bool = True,
        initial_capital: float = 500,
        risk_per_trade: float = 0.02,
        leverage: int = 3,
        max_daily_loss_pct: float = 0.05,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.testnet = testnet
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.max_daily_loss_pct = max_daily_loss_pct

        # OKX 客户端
        self.exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # 模式
        self.dry_run = False  # 纸上交易模式（只看不下单）

        # 趋势策略状态
        self._running = False
        self._position: Optional[dict] = None
        self._trade_log: list = []

        # 费率套利状态
        self._funding_pnl: float = 0.0          # 累计费率收入
        self._funding_hedge_open: bool = False   # 对冲仓位是否已开
        self._funding_hedge_price: float = 0.0   # 开仓价
        self._funding_notional: float = 0.0      # 名义价值
        self._funding_leverage: int = 5           # 套利杠杆
        self._funding_alloc: float = 0.4          # 40%资金分配给套利
        self._last_funding_check: float = 0.0     # 上次检查费率的时间
        self._total_pnl: float = 0.0
        self._last_trade_day: int = 0
        self._candles: list = []  # 历史 K 线缓存
        self._ohlcv: dict = {"open":[],"high":[],"low":[],"close":[],"volume":[]}

        # 指标缓存
        self._indicators: dict = {}

        mode_str = "🧪纸上交易" if self.dry_run else ("模拟盘" if testnet else "⚠️实盘")
        logger.info(f"实盘机器人就绪: {symbol} {timeframe} {mode_str} 杠杆={leverage}x")

    # ═══════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════

    def load_history(self, limit: int = 500):
        """加载历史 K 线用于指标预热"""
        ccxt_sym = self.symbol + ":USDT"
        candles = self.exchange.fetch_ohlcv(ccxt_sym, self.timeframe, limit=limit)
        self._candles = candles
        self._ohlcv = {
            "open":   np.array([c[1] for c in candles], dtype=float),
            "high":   np.array([c[2] for c in candles], dtype=float),
            "low":    np.array([c[3] for c in candles], dtype=float),
            "close":  np.array([c[4] for c in candles], dtype=float),
            "volume": np.array([c[5] for c in candles], dtype=float),
        }
        logger.info(f"已加载 {len(candles)} 根历史K线")

    def _update_indicators(self):
        """更新所有技术指标"""
        c, h, l = self._ohlcv["close"], self._ohlcv["high"], self._ohlcv["low"]
        self._indicators = {
            "ema_f": _ema(c, 20), "ema_m": _ema(c, 50), "ema_s": _ema(c, 100),
            "ema200": _ema(c, 200), "ema20": _ema(c, 20),
            "atr": _atr(h, l, c, 14), "adx": _adx(h, l, c, 14),
            "rsi": _rsi(c, 14),
        }

    # ═══════════════════════════════════════
    # 信号生成
    # ═══════════════════════════════════════

    def evaluate(self) -> Optional[dict]:
        """评估当前是否应该开仓。返回信号 dict 或 None"""
        self._update_indicators()
        ind = self._indicators

        c = self._ohlcv["close"][-1]
        atr = ind["atr"][-1]
        adx = ind["adx"][-1]
        rsi = ind["rsi"][-1]
        ef, em, es = ind["ema_f"][-1], ind["ema_m"][-1], ind["ema_s"][-1]
        e200, e20 = ind["ema200"][-1], ind["ema20"][-1]

        if any(np.isnan(x) for x in [atr, adx, rsi, ef, em, es, e200, e20]) or c <= 0:
            return None

        # 模式判断
        if adx <= 20:  # 震荡 → 均值回归
            near_support = abs(c - e200) <= atr * 2.5
            far_above = c > e20 + atr * 3
            if rsi < 30 and near_support:
                return {"side": "long", "mode": "mean_revert", "entry": c, "sl": c-atr*2.0, "tp": c+atr*2.0}
            if rsi > 70 and far_above:
                return {"side": "short", "mode": "mean_revert", "entry": c, "sl": c+atr*2.0, "tp": c-atr*2.0}

        elif adx >= 25:  # 趋势 → 趋势跟踪
            near_ema = abs(c - ef) <= atr * 1.0
            if ef > em > es and near_ema:
                return {"side": "long", "mode": "trend", "entry": c, "sl": c-atr*1.5, "tp": c+atr*2.5}
            if ef < em < es and near_ema:
                return {"side": "short", "mode": "trend", "entry": c, "sl": c+atr*1.5, "tp": c-atr*2.5}

        return None

    # ═══════════════════════════════════════
    # 策略 B: 资金费率套利
    # ═══════════════════════════════════════

    def _funding_arb_update(self):
        """
        每8h调用一次：结算资金费率
        现货多 + 永续空 = delta中性, 只赚费率
        """
        now = time.time()
        if now - self._last_funding_check < 8 * 3600:  # 不到8小时
            return

        self._last_funding_check = now

        if not self._funding_hedge_open:
            return

        # 获取当前费率
        try:
            ccxt_sym = self.symbol + ":USDT"
            rate_info = self.exchange.fetch_funding_rate(ccxt_sym)
            # OKX 返回格式: {"fundingRate": 0.0001, "nextFundingRate": ..., ...}
            raw_rate = rate_info.get("fundingRate") or rate_info.get("funding_rate") or 0
            current_rate = float(raw_rate) if raw_rate else 0.0001
            raw_next = rate_info.get("nextFundingRate") or rate_info.get("next_funding_rate") or current_rate
            next_rate = float(raw_next) if raw_next else current_rate

            # 本次结算收益 = 名义价值 × 费率
            settlement = self._funding_notional * current_rate
            self._funding_pnl += settlement

            logger.info(
                f"💰 费率结算: +{settlement:.2f}U "
                f"(费率={current_rate*100:.4f}% 累计={self._funding_pnl:+.1f}U "
                f"下次预测={next_rate*100:.4f}%)"
            )
        except Exception as e:
            logger.warning(f"费率查询失败: {e}")

    def _funding_arb_status(self) -> str:
        """费率套利状态报告"""
        if not self._funding_hedge_open:
            return "⏳ 未开仓"
        current = self._funding_notional
        capital_used = self.initial_capital * self._funding_alloc
        pnl_pct = self._funding_pnl / capital_used * 100 if capital_used > 0 else 0
        return (f"📊 对冲仓位: {current:.0f}U名义 | 入场价:{self._funding_hedge_price:.0f} | "
                f"费率收益:{self._funding_pnl:+.1f}U ({pnl_pct:+.1f}%)")

    def _funding_arb_open(self, price: float):
        """开仓费率套利对冲"""
        if self._funding_hedge_open:
            return

        capital = self.initial_capital * self._funding_alloc
        self._funding_notional = capital * self._funding_leverage
        self._funding_hedge_price = price
        self._funding_hedge_open = True

        logger.info(
            f"🔒 费率套利开仓: 现货多+永续空 | "
            f"名义{self._funding_notional:.0f}U | {self._funding_leverage}x | "
            f"资金{capital:.0f}U"
        )

        # 实盘模式: 实际下单
        if not self.dry_run:
            try:
                ccxt_sym = self.symbol + ":USDT"
                size = self._funding_notional / price
                # 开永续空头
                self.exchange.create_order(ccxt_sym, "market", "sell",
                    self.exchange.amount_to_precision(ccxt_sym, size),
                    params={"tdMode": "isolated", "leverage": self._funding_leverage})
                # 开现货多头 (需切换到 spot)
                self.exchange.options["defaultType"] = "spot"
                self.exchange.create_order(self.symbol, "market", "buy",
                    self.exchange.amount_to_precision(self.symbol, size))
                self.exchange.options["defaultType"] = "swap"
                logger.info("✅ 对冲仓位已实际开仓")
            except Exception as e:
                logger.error(f"开仓失败: {e}")

    # ═══════════════════════════════════════
    # 交易执行
    # ═══════════════════════════════════════

    def get_balance(self) -> float:
        """获取 USDT 余额"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_position(self, symbol: str = None) -> Optional[dict]:
        """查询当前持仓"""
        if symbol is None:
            symbol = self.symbol + ":USDT"
        try:
            positions = self.exchange.fetch_positions([symbol])
            for p in positions:
                if float(p.get("contracts", 0)) != 0:
                    return p
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
        return None

    def set_leverage(self):
        """设置杠杆"""
        try:
            ccxt_sym = self.symbol + ":USDT"
            self.exchange.set_leverage(self.leverage, ccxt_sym)
            logger.info(f"杠杆已设为 {self.leverage}x")
        except Exception as e:
            logger.warning(f"设置杠杆失败(可能已设置): {e}")

    def place_order(self, signal: dict) -> bool:
        """根据信号下单"""
        ccxt_sym = self.symbol + ":USDT"

        # 检查现有持仓
        pos = self.get_position()
        if pos:
            logger.info("已有持仓，跳过开仓")
            return False

        # 日亏损熔断
        today = datetime.now(timezone.utc).day
        if today != self._last_trade_day:
            self._daily_pnl = 0.0
            self._last_trade_day = today

        if abs(self._daily_pnl) > self.initial_capital * self.max_daily_loss_pct:
            logger.warning(f"日亏损熔断: PnL={self._daily_pnl:.1f}")
            return False

        # 计算仓位
        balance = self.get_balance()
        if balance <= 0:
            logger.error("余额为0")
            return False

        risk_amount = balance * self.risk_per_trade * self.leverage
        entry = signal["entry"]
        sl = signal["sl"]
        stop_pct = abs(entry - sl) / entry
        position_value = risk_amount / stop_pct if stop_pct > 0 else 0
        size = position_value / entry

        # 精度处理
        try:
            market = self.exchange.market(ccxt_sym)
            amount = self.exchange.amount_to_precision(ccxt_sym, size)
        except Exception:
            amount = round(size, 4)

        side = "buy" if signal["side"] == "long" else "sell"

        try:
            # 下限价单
            order = self.exchange.create_order(
                symbol=ccxt_sym,
                type="limit",
                side=side,
                amount=amount,
                price=entry,
                params={"tdMode": "isolated", "leverage": self.leverage},
            )
            logger.info(f"📈 下单成功: {side} {amount}张 @ {entry} 模式={signal['mode']}")

            # 设置止损止盈
            try:
                tp_side = "sell" if side == "buy" else "buy"
                self.exchange.create_order(ccxt_sym, "limit", tp_side,
                    amount=amount, price=signal["tp"],
                    params={"reduceOnly": True, "tdMode": "isolated"})
                self.exchange.create_order(ccxt_sym, "stop_market",
                    "sell" if side == "buy" else "buy",
                    amount=amount, params={"stopPrice": signal["sl"],
                    "reduceOnly": True, "tdMode": "isolated"})
            except Exception as e:
                logger.warning(f"止盈止损设置失败: {e}")

            return True

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return False

    # ═══════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════

    def start(self):
        """启动实盘机器人"""
        logger.info("🚀 启动实盘机器人...")

        # 预热
        self.load_history(500)
        self.set_leverage()

        # 检查是否已有持仓
        pos = self.get_position()
        if pos:
            logger.info(f"检测到现有持仓: {pos.get('side')} {pos.get('contracts')}张")

        self._running = True
        logger.info("✅ 双策略机器人已启动")

        # 开仓费率套利对冲
        current_price = self._ohlcv["close"][-1]
        self._funding_arb_open(current_price)

        # 启动时立即评估趋势策略状态
        self._update_indicators()
        ind = self._indicators
        c = self._ohlcv["close"][-1]
        adx = ind["adx"][-1]; rsi = ind["rsi"][-1]
        ef = ind["ema_f"][-1]; em = ind["ema_m"][-1]
        mode = "震荡(回归)" if adx <= 20 else ("趋势(跟踪)" if adx >= 25 else "过渡(等待)")
        logger.info(f"📊 当前状态: 价格={c:.0f} ADX={adx:.0f} RSI={rsi:.0f} 模式={mode}")
        logger.info(f"   EMA: 20={ef:.0f} 50={em:.0f} 100={ind['ema_s'][-1]:.0f}")
        logger.info(f"   {self._funding_arb_status()}")

        signal = self.evaluate()
        if signal:
            logger.info(f"🎯 当前信号: {signal['side']} 模式={signal['mode']} "
                        f"入场={signal['entry']:.1f} SL={signal['sl']:.1f} TP={signal['tp']:.1f}")
            if self.dry_run:
                self._position = {"side":signal["side"], "entry":signal["entry"],
                                  "sl":signal["sl"], "tp":signal["tp"], "mode":signal["mode"]}
                logger.info(f"📝 模拟开仓: {signal['side']} @{signal['entry']:.1f}")
        else:
            logger.info("⏳ 当前无信号，等待条件满足")
        logger.info("")

        last_candle_ts = self._candles[-1][0] if self._candles else 0

        while self._running:
            try:
                # ── 费率套利: 每8h自动结算 ──
                self._funding_arb_update()

                # ── 每分钟巡检：检查当前未收盘K线是否异常 ──
                ccxt_sym = self.symbol + ":USDT"
                ticker = self.exchange.fetch_ticker(ccxt_sym)
                current_price = ticker["last"]
                daily_change = ticker.get("percentage", 0) or 0

                # 如果 24h 涨跌幅 > 10% → 大新闻
                if abs(daily_change) > 10:
                    direction = "暴涨" if daily_change > 0 else "暴跌"
                    logger.warning(f"⚠️ 24h{direction}={daily_change:+.1f}% 疑似重大新闻!")

                    # 检查 EMA 是否确认方向
                    self._update_indicators()
                    price_above_ema50 = current_price > self._indicators["ema_m"][-1]
                    trend_confirmed = (daily_change > 0 and price_above_ema50) or \
                                     (daily_change < 0 and not price_above_ema50)

                    if trend_confirmed:
                        logger.info(f"⚡ EMA确认趋势方向, 继续监控")
                    else:
                        # 逆向持仓 → 平仓
                        if self.dry_run and self._position:
                            pos_side = self._position["side"]
                            if (pos_side == "short" and daily_change > 0) or \
                               (pos_side == "long" and daily_change < 0):
                                logger.warning(f"🛑 新闻逆向平仓: {pos_side} @{current_price:.0f}")
                                self._position = None
                            else:
                                logger.info(f"📊 持仓顺势: {pos_side} 继续持有")
                    time.sleep(120)  # 等2分钟再检查
                    continue

                # 获取最新K线
                candles = self.exchange.fetch_ohlcv(ccxt_sym, self.timeframe, limit=1)
                if not candles:
                    time.sleep(60)
                    continue

                latest = candles[-1]
                if latest[0] == last_candle_ts:
                    # 同一根K线，等待
                    time.sleep(60)
                    continue

                # 新K线！更新数据
                last_candle_ts = latest[0]
                self._candles.append(latest)
                for i, key in enumerate(["open","high","low","close","volume"]):
                    self._ohlcv[key] = np.append(self._ohlcv[key], latest[i+1])

                ts = datetime.fromtimestamp(latest[0]/1000, tz=timezone.utc)

                # ⚡ 异常波动检测：如果刚收盘的K线振幅超过 3x ATR → 新闻冲击
                atr_val = self._indicators.get("atr", np.array([0]))[-1]
                candle_range_pct = (latest[2] - latest[3]) / latest[4] * 100 if latest[4] > 0 else 0
                avg_atr_pct = (atr_val / latest[4] * 100) if latest[4] > 0 else 0
                is_shock = candle_range_pct > avg_atr_pct * 3 if avg_atr_pct > 0 else False

                if is_shock:
                    direction = "↑暴涨" if latest[4] > latest[1] else "↓暴跌"
                    logger.warning(f"⚡ 新闻冲击! {direction} 振幅={candle_range_pct:.1f}% (ATR={avg_atr_pct:.1f}%)")

                    # 冲击后顺势追：EMA 是否确认了新方向？
                    shock_bull = latest[4] > self._indicators["ema_m"][-1]  # 价格站上 EMA50
                    shock_bear = latest[4] < self._indicators["ema_m"][-1]  # 价格跌破 EMA50

                    if self.dry_run and self._position:
                        # 持仓与冲击方向相反 → 平仓避险
                        pos_side = self._position["side"]
                        if (pos_side == "short" and shock_bull) or (pos_side == "long" and shock_bear):
                            exit_p = latest[4]
                            pnl_pct = (exit_p/self._position['entry']-1)*100 if pos_side=='long' else (self._position['entry']/exit_p-1)*100
                            logger.warning(f"🛑 冲击逆向平仓: {pos_side} @{exit_p:.0f} PnL={pnl_pct:+.1f}%")
                            self._position = None

                    # 顺势追：EMA 确认 + ADX 足够强 → 不跳过，继续评估
                    if (shock_bull or shock_bear) and self._indicators["adx"][-1] > 22:
                        logger.info(f"⚡ 冲击后顺势: EMA确认方向, 继续评估信号")
                        # 不 continue，正常走信号评估
                    else:
                        logger.warning(f"⚠️ 冲击方向不明(EMA未确认), 跳过本轮")
                        continue

                logger.info(f"🕐 新K线 {ts} O={latest[1]:.1f} H={latest[2]:.1f} L={latest[3]:.1f} C={latest[4]:.1f}" +
                            (f" ⚡振幅{candle_range_pct:.1f}%" if is_shock else ""))

                # 检查持仓状态
                if self.dry_run:
                    # ── 纸上交易模式 ──
                    sim_pos = self._position
                    if sim_pos:
                        # 检查是否触发 SL/TP
                        hit_sl = (sim_pos["side"] == "long" and latest[3] <= sim_pos["sl"]) or \
                                 (sim_pos["side"] == "short" and latest[2] >= sim_pos["sl"])
                        hit_tp = (sim_pos["side"] == "long" and latest[2] >= sim_pos["tp"]) or \
                                 (sim_pos["side"] == "short" and latest[3] <= sim_pos["tp"])
                        if hit_sl or hit_tp:
                            exit_price = sim_pos["sl"] if hit_sl else sim_pos["tp"]
                            pnl_pct = (exit_price / sim_pos["entry"] - 1) * 100
                            if sim_pos["side"] == "short": pnl_pct = -pnl_pct
                            pnl_usdt = self.initial_capital * self.risk_per_trade * self.leverage * pnl_pct / 100 * (abs(sim_pos['entry']-sim_pos['sl'])/sim_pos['entry'])**-1
                            # 简化：用百分比计算
                            stop_pct = abs(sim_pos['entry']-sim_pos['sl'])/sim_pos['entry']
                            risk_usdt = self.initial_capital * self.risk_per_trade * self.leverage
                            actual_pnl = risk_usdt * pnl_pct / (stop_pct*100) if stop_pct > 0 else 0
                            self._total_pnl += actual_pnl
                            reason = "SL" if hit_sl else "TP"
                            logger.info(f'📊 模拟平仓: {sim_pos["side"]} @{exit_price:.0f} {reason} PnL={actual_pnl:+.1f}U (累计{self._total_pnl:+.1f}U)')
                            trade_record = {"side":sim_pos["side"],"entry":float(sim_pos["entry"]),"exit":float(exit_price),"pnl":round(actual_pnl,2),"reason":reason,"mode":sim_pos["mode"],"time":str(ts)}
                            self._trade_log.append(trade_record)
                            # 实时保存到文件
                            import json
                            Path("outputs").mkdir(exist_ok=True)
                            with open("outputs/live_trades.json", "w") as f:
                                json.dump(self._trade_log, f, indent=2, ensure_ascii=False, default=str)
                            self._position = None
                        else:
                            unrealised = ((latest[4] / sim_pos["entry"] - 1) * 100) if sim_pos["side"]=="long" else ((sim_pos["entry"] / latest[4] - 1) * 100)
                            logger.info(f"📊 模拟持仓: {sim_pos['side']} @{sim_pos['entry']:.0f} 现价={latest[4]:.0f} 浮盈={unrealised:+.1f}%")
                    else:
                        # 无持仓，评估信号
                        signal = self.evaluate()
                        if signal:
                            logger.info(f"🎯 信号: {signal['side']} 模式={signal['mode']} "
                                        f"入场={signal['entry']:.1f} SL={signal['sl']:.1f} TP={signal['tp']:.1f}")
                            self._position = {"side":signal["side"], "entry":signal["entry"],
                                              "sl":signal["sl"], "tp":signal["tp"], "mode":signal["mode"]}
                            logger.info(f"📝 模拟开仓: {signal['side']} @{signal['entry']:.1f} "
                                        f"SL={signal['sl']:.1f} TP={signal['tp']:.1f}")
                else:
                    # ── 真实交易模式 ──
                    pos = self.get_position()
                    if pos:
                        pnl = float(pos.get("unrealizedPnl", 0))
                        logger.info(f"📊 持仓中: {pos.get('side')} PnL={pnl:.1f}")
                    else:
                        signal = self.evaluate()
                        if signal:
                            logger.info(f"🎯 信号: {signal['side']} 模式={signal['mode']} "
                                        f"入场={signal['entry']:.1f} SL={signal['sl']:.1f} TP={signal['tp']:.1f}")
                            self.place_order(signal)

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                time.sleep(60)

    def stop(self):
        """停止机器人"""
        self._running = False
        logger.info("🛑 机器人已停止")
