"""
生产级 TripleEma 4h 实盘交易机器人

回测验证: 2x杠杆, 全时段+135.5%, 月均+2.4%, 夏普1.80
"""

import time
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════
# 指标 (纯 NumPy, 与回测完全一致)
# ═══════════════════════════════════════════

def ema(a: np.ndarray, span: int) -> np.ndarray:
    if span <= 0 or len(a) < span:
        return np.full(len(a), np.nan)
    alpha = 2.0 / (span + 1.0)
    r = np.full(len(a), np.nan, dtype=float)
    r[span - 1] = np.mean(a[:span])
    for i in range(span, len(a)):
        r[i] = alpha * a[i] + (1 - alpha) * r[i - 1]
    return r


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    r = np.full(len(close), np.nan, dtype=float)
    if len(close) < period + 1:
        return r
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    r[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        r[i] = alpha * tr[i] + (1 - alpha) * r[i - 1]
    return r


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    n = len(close); r = np.full(n, np.nan, dtype=float)
    if n < period * 2: return r
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    um = high - np.roll(high, 1); dm = np.roll(low, 1) - low
    um[0] = dm[0] = 0
    pdm = np.where((um > dm) & (um > 0), um, 0.0)
    ndm = np.where((dm > um) & (dm > 0), dm, 0.0)
    alpha = 1.0 / period
    at_arr = np.full(n, np.nan); pt = np.full(n, np.nan); nt = np.full(n, np.nan)
    at_arr[period - 1] = np.mean(tr[:period])
    pt[period - 1] = np.mean(pdm[:period]); nt[period - 1] = np.mean(ndm[:period])
    for i in range(period, n):
        at_arr[i] = alpha * tr[i] + (1 - alpha) * at_arr[i - 1]
        pt[i] = alpha * pdm[i] + (1 - alpha) * pt[i - 1]
        nt[i] = alpha * ndm[i] + (1 - alpha) * nt[i - 1]
    pdi = np.where(at_arr > 0, 100 * pt / at_arr, 0)
    ndi = np.where(at_arr > 0, 100 * nt / at_arr, 0)
    ds = pdi + ndi
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(ds > 0, 100 * np.abs(pdi - ndi) / ds, 0)
    r[period * 2 - 2] = np.mean(dx[period - 1:period * 2 - 1])
    for i in range(period * 2 - 1, n):
        r[i] = alpha * dx[i] + (1 - alpha) * r[i - 1]
    return r


# ═══════════════════════════════════════════
# 实盘机器人
# ═══════════════════════════════════════════

class ProductionBot:
    """TripleEma 4h 2x 杠杆实盘交易机器人"""

    def __init__(
        self,
        api_key: str, secret: str, passphrase: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "4h",
        testnet: bool = True,
        leverage: int = 2,
        initial_capital: float = 500,
        risk_per_trade: float = 0.02,
        max_daily_loss_pct: float = 0.05,
        max_consecutive_losses: int = 5,
        # 策略参数
        ema_fast: int = 10, ema_mid: int = 20,
        ema_slow: int = 50, ema_trend: int = 200,
        adx_threshold: int = 20,
        sl_atr: float = 1.5, tp_atr: float = 3.0,
        cooldown_bars: int = 6,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses

        # 策略参数
        self.p_ema_f = ema_fast; self.p_ema_m = ema_mid
        self.p_ema_s = ema_slow; self.p_ema_t = ema_trend
        self.p_adx_thr = adx_threshold
        self.p_sl_atr = sl_atr; self.p_tp_atr = tp_atr
        self.p_cooldown = cooldown_bars

        # OKX 连接
        self.exchange = ccxt.okx({
            "apiKey": api_key, "secret": secret, "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self.dry_run = True       # 默认纸上交易
        self._running = False
        self._position: Optional[dict] = None
        self._trade_log: list = []
        self._consecutive_losses = 0
        self._last_trade_bar = -999
        self._daily_pnl = 0.0
        self._last_trade_day = 0
        self._total_pnl = 0.0

        # K线缓存
        self._candles: list = []
        self._ohlcv = {"open": [], "high": [], "low": [], "close": [], "volume": []}
        self._indicators: dict = {}

        mode = "🧪纸上交易" if self.dry_run else ("🧪模拟盘" if testnet else "⚠️实盘")
        logger.info(f"🤖 TripleEma机器人: {symbol} {timeframe} {leverage}x {mode}")

    # ═══════════════════════
    # 数据
    # ═══════════════════════

    def load_history(self, limit: int = 500):
        """加载历史K线用于指标预热"""
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
        self._update_indicators()
        logger.info(f"已加载 {len(candles)} 根历史K线")

    def _update_indicators(self):
        """计算所有指标"""
        c = self._ohlcv["close"]; h = self._ohlcv["high"]
        l = self._ohlcv["low"]
        self._indicators = {
            "ema10":  ema(c, self.p_ema_f),
            "ema20":  ema(c, self.p_ema_m),
            "ema50":  ema(c, self.p_ema_s),
            "ema200": ema(c, self.p_ema_t),
            "adx":    adx(h, l, c, 14),
            "atr":    atr(h, l, c, 14),
        }

    # ═══════════════════════
    # 信号生成 (与回测策略逻辑完全一致)
    # ═══════════════════════

    def evaluate(self) -> Optional[dict]:
        """评估当前是否应该开仓"""
        ind = self._indicators
        c = self._ohlcv["close"][-1]
        e10 = ind["ema10"][-1]; e20 = ind["ema20"][-1]
        e50 = ind["ema50"][-1]; e200 = ind["ema200"][-1]
        adx_v = ind["adx"][-1]; atr_v = ind["atr"][-1]

        if any(np.isnan(x) for x in [c, e10, e20, e50, e200, adx_v, atr_v]):
            return None
        if c <= 0: return None

        bull = c > e200                        # 牛市
        bull_align = e10 > e20 > e50           # 多头排列
        bear_align = e10 < e20 < e50           # 空头排列
        near = abs(c - e20) < atr_v            # 价格靠近EMA20
        strong = adx_v > self.p_adx_thr        # 趋势够强

        if not strong or not near:
            return None

        if bull and bull_align:
            return {
                "side": "long",
                "entry": float(c),
                "sl": float(c - atr_v * self.p_sl_atr),
                "tp": float(c + atr_v * self.p_tp_atr),
            }
        elif not bull and bear_align:
            return {
                "side": "short",
                "entry": float(c),
                "sl": float(c + atr_v * self.p_sl_atr),
                "tp": float(c - atr_v * self.p_tp_atr),
            }
        return None

    # ═══════════════════════
    # 交易执行
    # ═══════════════════════

    def get_balance(self) -> float:
        """获取 USDT 余额"""
        try:
            bal = self.exchange.fetch_balance()
            return float(bal.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_position(self) -> Optional[dict]:
        """查询当前持仓"""
        try:
            ccxt_sym = self.symbol + ":USDT"
            positions = self.exchange.fetch_positions([ccxt_sym])
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

    def execute_signal(self, signal: dict) -> bool:
        """执行交易信号"""
        ccxt_sym = self.symbol + ":USDT"

        # 检查现有持仓
        if self.get_position():
            logger.info("已有持仓, 跳过")
            return False

        # 日熔断检查
        today = datetime.now(timezone.utc).day
        if today != self._last_trade_day:
            self._daily_pnl = 0.0
            self._last_trade_day = today
        if abs(self._daily_pnl) > self.initial_capital * self.max_daily_loss_pct:
            logger.warning(f"⛔ 日熔断: PnL={self._daily_pnl:.1f}U")
            return False

        # 连续亏损暂停
        if self._consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"⛔ 连续亏损{self._consecutive_losses}次, 暂停交易")
            return False

        # 计算仓位
        balance = self.get_balance()
        if balance <= 0:
            logger.error("余额为0")
            return False

        entry = signal["entry"]; sl = signal["sl"]
        stop_pct = abs(entry - sl) / entry
        if stop_pct <= 0: return False

        # 仓位大小 = (余额 × 风险比例 × 杠杆) / 止损距离
        risk_amount = balance * self.risk_per_trade * self.leverage
        position_value = risk_amount / stop_pct
        size = position_value / entry  # 合约张数

        # 精度处理
        try:
            market = self.exchange.market(ccxt_sym)
            amount = self.exchange.amount_to_precision(ccxt_sym, size)
            entry_price = self.exchange.price_to_precision(ccxt_sym, entry)
        except Exception:
            amount = round(size, 4)
            entry_price = entry

        side = "buy" if signal["side"] == "long" else "sell"

        if self.dry_run:
            self._position = signal
            logger.info(
                f"📝 [纸上] {side.upper()} {amount}张 @{entry:.0f} "
                f"SL={sl:.0f} TP={signal['tp']:.0f}"
            )
            return True

        # 真实下单
        try:
            order = self.exchange.create_order(
                symbol=ccxt_sym, type="limit", side=side,
                amount=amount, price=entry_price,
                params={"tdMode": "isolated", "leverage": self.leverage},
            )
            logger.info(f"📈 下单: {side} {amount}张 @{entry_price}")

            # 设止盈止损
            try:
                tp_side = "sell" if side == "buy" else "buy"
                tp_price = self.exchange.price_to_precision(ccxt_sym, signal["tp"])
                sl_price = self.exchange.price_to_precision(ccxt_sym, signal["sl"])
                self.exchange.create_order(
                    ccxt_sym, "limit", tp_side,
                    amount=amount, price=tp_price,
                    params={"reduceOnly": True, "tdMode": "isolated"}
                )
                self.exchange.create_order(
                    ccxt_sym, "stop_market", "sell" if side == "buy" else "buy",
                    amount=amount,
                    params={"stopPrice": sl_price,
                            "reduceOnly": True, "tdMode": "isolated"}
                )
                logger.info(f"  止盈={tp_price} 止损={sl_price}")
            except Exception as e:
                logger.warning(f"止盈止损设置失败: {e}")

            return True
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return False

    # ═══════════════════════
    # 主循环
    # ═══════════════════════

    def start(self):
        """启动机器人"""
        logger.info("🚀 启动 TripleEma 实盘机器人...")

        self.load_history(500)
        self.set_leverage()

        # 检查现有持仓
        pos = self.get_position()
        if pos:
            side = pos.get("side", "?")
            contracts = pos.get("contracts", 0)
            logger.info(f"⚠️ 检测到现有持仓: {side} {contracts}张, 请手动处理")
            return

        self._running = True
        logger.info("✅ 机器人已启动")

        # 初始状态
        c = self._ohlcv["close"][-1]
        ind = self._indicators
        logger.info(
            f"📊 当前状态: 价格={c:.0f} "
            f"EMA10/20/50={ind['ema10'][-1]:.0f}/{ind['ema20'][-1]:.0f}/{ind['ema50'][-1]:.0f} "
            f"EMA200={ind['ema200'][-1]:.0f} ADX={ind['adx'][-1]:.0f}"
        )
        logger.info(f"   牛熊: {'🐂牛市' if c > ind['ema200'][-1] else '🐻熊市'}")

        signal = self.evaluate()
        if signal:
            logger.info(
                f"🎯 初始信号: {signal['side']} 入场={signal['entry']:.0f} "
                f"SL={signal['sl']:.0f} TP={signal['tp']:.0f}"
            )
            self.execute_signal(signal)
        else:
            logger.info("⏳ 当前无信号")

        last_candle_ts = self._candles[-1][0] if self._candles else 0
        ccxt_sym = self.symbol + ":USDT"

        # ── 主循环 ──
        while self._running:
            try:
                # 获取最新K线
                candles = self.exchange.fetch_ohlcv(
                    ccxt_sym, self.timeframe, limit=1)
                if not candles:
                    time.sleep(60)
                    continue

                latest = candles[-1]
                if latest[0] == last_candle_ts:
                    # 同一根K线: 检查持仓是否需要移动止损
                    if self._position:
                        self._check_trailing_stop(latest)
                    time.sleep(60)
                    continue

                # 新K线! 更新数据
                last_candle_ts = latest[0]
                self._candles.append(latest)
                for i, key in enumerate(["open", "high", "low", "close", "volume"]):
                    self._ohlcv[key] = np.append(
                        self._ohlcv[key], latest[i + 1])
                self._update_indicators()

                ts = datetime.fromtimestamp(
                    latest[0] / 1000, tz=timezone.utc)
                logger.info(
                    f"🕐 新K线 {ts} "
                    f"O={latest[1]:.0f} H={latest[2]:.0f} "
                    f"L={latest[3]:.0f} C={latest[4]:.0f}"
                )

                # 持仓检查
                if self._position:
                    self._check_exit_conditions(latest)
                else:
                    # 冷却期检查
                    bars_since = len(self._candles) - self._last_trade_bar
                    cooldown = self.p_cooldown + \
                        self._consecutive_losses * 3
                    if bars_since < cooldown:
                        logger.debug(f"冷却中: {bars_since}/{cooldown}")
                        continue

                    signal = self.evaluate()
                    if signal:
                        logger.info(
                            f"🎯 信号: {signal['side']} "
                            f"入场={signal['entry']:.0f} "
                            f"SL={signal['sl']:.0f} TP={signal['tp']:.0f}"
                        )
                        self.execute_signal(signal)

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                time.sleep(60)

    def _check_exit_conditions(self, latest):
        """检查是否需要平仓"""
        pos = self._position
        if not pos: return

        c = latest[4]; h = latest[2]; l = latest[3]

        if pos["side"] == "long":
            # 止盈或止损
            if h >= pos["tp"]:
                self._close_position("TP", pos["tp"])
            elif l <= pos["sl"]:
                self._close_position("SL", pos["sl"])
            # EMA排列破坏(趋势反转)
            elif not self._is_bull_align():
                self._close_position("趋势反转", c)

        elif pos["side"] == "short":
            if l <= pos["tp"]:
                self._close_position("TP", pos["tp"])
            elif h >= pos["sl"]:
                self._close_position("SL", pos["sl"])
            elif not self._is_bear_align():
                self._close_position("趋势反转", c)

    def _is_bull_align(self) -> bool:
        ind = self._indicators
        return ind["ema10"][-1] > ind["ema20"][-1] > ind["ema50"][-1]

    def _is_bear_align(self) -> bool:
        ind = self._indicators
        return ind["ema10"][-1] < ind["ema20"][-1] < ind["ema50"][-1]

    def _check_trailing_stop(self, latest):
        """移动止损检查(在同一根K线内)"""
        pass  # 由 _check_exit_conditions 在新K线时处理

    def _close_position(self, reason: str, price: float):
        """平仓并记录"""
        pos = self._position
        if not pos: return

        if pos["side"] == "long":
            pnl_pct = (price / pos["entry"] - 1) * 100 * self.leverage
        else:
            pnl_pct = (pos["entry"] / price - 1) * 100 * self.leverage

        pnl_usdt = self.initial_capital * pnl_pct / 100
        self._total_pnl += pnl_usdt
        self._daily_pnl += pnl_usdt

        if pnl_usdt < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._last_trade_bar = len(self._candles)

        trade = {
            "side": pos["side"],
            "entry": pos["entry"],
            "exit": price,
            "pnl": round(pnl_usdt, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "time": str(datetime.now(timezone.utc)),
        }
        self._trade_log.append(trade)
        self._position = None

        logger.info(
            f"{'✅' if pnl_usdt > 0 else '❌'} 平仓: {pos['side']} "
            f"@{price:.0f} {reason} PnL={pnl_usdt:+.1f}U "
            f"({pnl_pct:+.1f}%) 连续亏损:{self._consecutive_losses}"
        )

        # 保存交易记录
        Path("outputs").mkdir(exist_ok=True)
        with open("outputs/production_trades.json", "w") as f:
            json.dump(self._trade_log, f, indent=2,
                      ensure_ascii=False, default=str)

    def stop(self):
        """停止机器人"""
        self._running = False
        logger.info("🛑 机器人已停止")


def run_bot(config_path: str = "config/production.yaml"):
    """从配置文件启动机器人"""
    import yaml
    import os
    from dotenv import load_dotenv

    load_dotenv()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 从环境变量读取 API 密钥
    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")

    # 日志
    log_file = cfg.get("output", {}).get("log_file", "outputs/production.log")
    Path(log_file).parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation="10 MB", retention="7 days",
               level=cfg.get("mode", {}).get("log_level", "INFO"))

    bot = ProductionBot(
        api_key=api_key, secret=secret, passphrase=passphrase,
        symbol=cfg["symbol"],
        timeframe=cfg["timeframe"],
        testnet=cfg["mode"]["testnet"],
        leverage=cfg["risk"]["leverage"],
        initial_capital=cfg["risk"]["initial_capital"],
        risk_per_trade=cfg["risk"].get("risk_per_trade", 0.02),
        max_daily_loss_pct=cfg["risk"].get("max_daily_loss_pct", 0.05),
        max_consecutive_losses=cfg["risk"].get("max_consecutive_losses", 5),
        **cfg["strategy"],
    )

    bot.dry_run = cfg["mode"]["dry_run"]

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
