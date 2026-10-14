"""
生产级 TripleEma 4h 实盘交易机器人

回测验证: 2x杠杆, 全时段+135.5%, 月均+2.4%, 夏普1.80
"""

import time
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import ccxt
import numpy as np
from loguru import logger

from src.live.alerts import LocalAlertManager
from src.live.okx_account import OKXAccountReader, reconcile_account_state
from src.live.order_executor import (
    OrderExecutor,
    OrderPlan,
    build_spot_core_rebalance_plan,
)
from src.live.portfolio import (
    LIVE_TRADING_MODES,
    PortfolioConfig,
    cap_swing_notional,
    classify_latest_regime,
    load_portfolio_config,
    plan_core_rebalance,
    portfolio_exposure,
)
from src.live.risk import calculate_position_size, loss_pause_decision


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

def closed_candles_after(candles: list, last_ts: int,
                         interval_ms: int, now_ms: int) -> list:
    """只返回上次处理后、已经走完完整周期的K线。"""
    return sorted(
        (c for c in candles if last_ts < c[0] and c[0] + interval_ms <= now_ms),
        key=lambda c: c[0],
    )


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
        loss_pause_bars: int = 18,
        max_notional_pct: float = 0.95,
        # 策略参数
        ema_fast: int = 10, ema_mid: int = 20,
        ema_slow: int = 50, ema_trend: int = 200,
        adx_threshold: int = 20,
        ema_slope_lookback: int = 0,
        min_ema_spread: float = 0.0,
        min_atr_percentile: float = 0.0,
        atr_percentile_window: int = 180,
        sl_atr: float = 1.5, tp_atr: float = 3.0,
        cooldown_bars: int = 6,
        cooldown_extra_per_loss: int = 3,
        portfolio_config: PortfolioConfig | None = None,
        state_file: str = "outputs/production_state.json",
        max_api_errors: int = 3,
        alert_manager: LocalAlertManager | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.loss_pause_bars = loss_pause_bars
        self.max_notional_pct = max_notional_pct

        # 策略参数
        self.p_ema_f = ema_fast; self.p_ema_m = ema_mid
        self.p_ema_s = ema_slow; self.p_ema_t = ema_trend
        self.p_adx_thr = adx_threshold
        self.p_ema_slope_lookback = ema_slope_lookback
        self.p_min_ema_spread = min_ema_spread
        self.p_min_atr_percentile = min_atr_percentile
        self.p_atr_percentile_window = atr_percentile_window
        self.p_sl_atr = sl_atr; self.p_tp_atr = tp_atr
        self.p_cooldown = cooldown_bars
        self.p_cooldown_extra_per_loss = cooldown_extra_per_loss
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.state_file = Path(state_file)
        self.max_api_errors = max_api_errors
        self.alert_manager = alert_manager

        # OKX 连接
        self.exchange = ccxt.okx({
            "apiKey": api_key, "secret": secret, "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self.account_reader = OKXAccountReader(self.exchange, self.symbol)
        self.order_executor = OrderExecutor(self.exchange)

        self.dry_run = True       # 默认纸上交易
        self._running = False
        self._position: Optional[dict] = None
        self._trade_log: list = []
        self._consecutive_losses = 0
        self._last_trade_bar = -999
        self._last_trade_candle_ts: int | None = None
        self._daily_pnl = 0.0
        self._last_trade_day = 0
        self._total_pnl = 0.0
        self._paused_reason: str | None = None
        self._api_error_count = 0
        self._market_error_count = 0
        self._last_market_success_at: str | None = None
        self._last_signal: dict = {"decision": "waiting", "reason": "startup"}
        self._core_trade_log: list = []
        self._paper_core_btc = 0.0
        self._paper_cash = initial_capital

        # K线缓存
        self._candles: list = []
        self._ohlcv = {"open": [], "high": [], "low": [], "close": [], "volume": []}
        self._indicators: dict = {}

        mode = "🧪纸上交易" if self.dry_run else ("🧪模拟盘" if testnet else "⚠️实盘")
        logger.info(f"🤖 TripleEma机器人: {symbol} {timeframe} {leverage}x {mode}")
        if self.portfolio_config.enabled:
            logger.info(
                "组合风控已启用: "
                f"mode={self.portfolio_config.mode} "
                f"net<={self.portfolio_config.max_net_exposure:.0%} "
                f"gross<={self.portfolio_config.max_gross_exposure:.0%}"
            )

    def _state_payload(self) -> dict:
        return {
            "version": 1,
            "schema": "production_bot_state",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dry_run": self.dry_run,
            "paper_cash": self._paper_cash,
            "paper_core_btc": self._paper_core_btc,
            "position": self._position,
            "total_pnl": self._total_pnl,
            "daily_pnl": self._daily_pnl,
            "last_trade_day": self._last_trade_day,
            "consecutive_losses": self._consecutive_losses,
            "last_trade_bar": self._last_trade_bar,
            "last_trade_candle_ts": self._last_trade_candle_ts,
            "trade_log": self._trade_log,
            "core_trade_log": self._core_trade_log,
            "updated_at": str(datetime.now(timezone.utc)),
        }

    def _save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
            with open(temporary, "w") as f:
                json.dump(self._state_payload(), f, indent=2, ensure_ascii=False, default=str)
            temporary.replace(self.state_file)
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            reason = self._validate_state_payload(state)
            if reason:
                self._paused_reason = f"state_invalid:{reason}"
                logger.warning(f"状态文件校验失败，跳过恢复: {reason}")
                return
            if state.get("symbol") != self.symbol or state.get("timeframe") != self.timeframe:
                logger.warning("状态文件交易对或周期不匹配，跳过恢复")
                return
            self._paper_cash = float(state.get("paper_cash", self._paper_cash))
            self._paper_core_btc = float(state.get("paper_core_btc", self._paper_core_btc))
            self._position = state.get("position")
            self._total_pnl = float(state.get("total_pnl", self._total_pnl))
            self._daily_pnl = float(state.get("daily_pnl", self._daily_pnl))
            self._last_trade_day = int(state.get("last_trade_day", self._last_trade_day))
            self._consecutive_losses = int(
                state.get("consecutive_losses", self._consecutive_losses)
            )
            self._last_trade_bar = int(state.get("last_trade_bar", self._last_trade_bar))
            self._trade_log = list(state.get("trade_log", self._trade_log))
            self._core_trade_log = list(state.get("core_trade_log", []))
            self._last_trade_candle_ts = state.get("last_trade_candle_ts")
            if self._last_trade_candle_ts is None and self._trade_log:
                trade = self._trade_log[-1]
                self._last_trade_candle_ts = trade.get("candle_time")
                if self._last_trade_candle_ts is None and trade.get("time"):
                    # Legacy records timestamp the execution just after the bar closed.
                    stamp = datetime.fromisoformat(trade["time"])
                    if stamp.tzinfo is None:
                        raise ValueError("legacy trade time has no timezone")
                    interval = self.exchange.parse_timeframe(self.timeframe) * 1000
                    self._last_trade_candle_ts = int(stamp.timestamp() * 1000) // interval * interval - interval
            if self._last_trade_bar >= 0 and self._last_trade_candle_ts is None:
                self._paused_reason = "state_invalid:cooldown_timestamp_missing"
                logger.warning("无法恢复交易冷却时间，暂停开仓")
            logger.info(
                f"已恢复交易状态: cash={self._paper_cash:.1f}U "
                f"core_btc={self._paper_core_btc:.8f} position={'yes' if self._position else 'no'}"
            )
        except Exception as e:
            self._paused_reason = "state_invalid:restore_failed"
            logger.warning(f"恢复状态失败: {e}")

    def _bars_since_trade(self) -> int:
        stamp = getattr(self, "_last_trade_candle_ts", None)
        if stamp is None:
            # Only used for never-traded states and older in-memory callers.
            return len(self._candles) - self._last_trade_bar
        if not self._candles:
            return 0
        interval = self.exchange.parse_timeframe(self.timeframe) * 1000
        return max(0, (int(self._candles[-1][0]) - int(stamp)) // interval)

    def _cooldown_status(self) -> dict:
        elapsed = self._bars_since_trade()
        required = self.p_cooldown + self._consecutive_losses * self.p_cooldown_extra_per_loss
        if self.max_consecutive_losses > 0 and self._consecutive_losses >= self.max_consecutive_losses:
            required = self.loss_pause_bars if elapsed < self.loss_pause_bars else self.p_cooldown
        return {"elapsed_bars": elapsed, "required_bars": required,
                "remaining_bars": max(0, required - elapsed), "timeframe": self.timeframe}

    def _write_health(self):
        path = self.state_file.with_name("production_health.json")
        payload = {
            "updated_at": str(datetime.now(timezone.utc)),
            "running": self._running, "mode": "paper" if self.dry_run else "exchange",
            "paused_reason": self._paused_reason,
            "api_error_count": self._api_error_count,
            "market_error_count": self._market_error_count,
            "last_market_success_at": self._last_market_success_at,
            "last_closed_candle_ts": self._candles[-1][0] if self._candles else None,
            "cooldown": self._cooldown_status(), "signal": self._last_signal,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            temporary.replace(path)
        except OSError as error:
            logger.warning(f"写入机器人心跳失败: {error}")

    def _validate_state_payload(self, state: dict[str, Any]) -> str:
        if int(state.get("version", 0) or 0) != 1:
            return "version"
        if state.get("schema") not in (None, "production_bot_state"):
            return "schema"
        if state.get("symbol") != self.symbol or state.get("timeframe") != self.timeframe:
            return "identity"
        stamp = state.get("last_trade_candle_ts")
        if stamp is not None and (isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0):
            return "cooldown_timestamp"
        if stamp is not None and stamp > self.exchange.milliseconds():
            return "cooldown_timestamp"

        position = state.get("position")
        if position is None:
            return ""
        if not isinstance(position, dict):
            return "position_type"

        required = ("side", "entry", "notional")
        for key in required:
            if key not in position:
                return f"position_missing_{key}"
        if str(position.get("side")).lower() not in ("long", "short"):
            return "position_side"
        for key in ("entry", "notional"):
            try:
                if float(position.get(key)) <= 0:
                    return f"position_{key}"
            except (TypeError, ValueError):
                return f"position_{key}"

        if self.dry_run:
            return ""

        live_required = ("size", "entry_order_id", "protective_orders")
        for key in live_required:
            if key not in position:
                return f"live_position_missing_{key}"
        try:
            if float(position.get("size")) <= 0:
                return "live_position_size"
        except (TypeError, ValueError):
            return "live_position_size"

        protective = position.get("protective_orders")
        if not isinstance(protective, dict):
            return "protective_orders_type"
        if not protective.get("take_profit"):
            return "protective_take_profit"
        if not protective.get("stop_loss"):
            return "protective_stop_loss"
        if not position.get("entry_order_id"):
            return "entry_order_id"
        return ""

    def _order_id(self, result) -> str | None:
        order = getattr(result, "order", None) or {}
        value = order.get("id") or order.get("orderId") or (order.get("info") or {}).get("ordId")
        return str(value) if value else None

    def _order_average_price(self, result, fallback: float) -> float:
        order = getattr(result, "order", None) or {}
        info = order.get("info", {}) or {}
        for key in ("average", "avgPrice", "price"):
            value = order.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                return parsed
        for key in ("avgPx", "fillPx", "px"):
            try:
                parsed = float(info.get(key))
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                return parsed
        return fallback

    def _order_filled_amount(self, result, fallback: float) -> float:
        filled = float(getattr(result, "filled", 0.0) or 0.0)
        if filled > 0:
            return filled
        order = getattr(result, "order", None) or {}
        info = order.get("info", {}) or {}
        for key in ("filled", "filledSz", "accFillSz"):
            try:
                parsed = float(order.get(key) or info.get(key))
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                return parsed
        return fallback

    def _order_fee(self, result) -> dict[str, Any]:
        return {
            "cost": float(getattr(result, "fee_cost", 0.0) or 0.0),
            "currency": str(getattr(result, "fee_currency", "") or ""),
        }

    def _order_snapshot(self, result) -> dict[str, Any]:
        order = getattr(result, "order", None) or {}
        return {
            "id": self._order_id(result),
            "status": getattr(result, "status", ""),
            "filled": float(getattr(result, "filled", 0.0) or 0.0),
            "remaining": float(getattr(result, "remaining", 0.0) or 0.0),
            "fee": self._order_fee(result),
            "raw": order,
        }

    def _protective_order_ids(self) -> set[str]:
        if not self._position:
            return set()
        orders = self._position.get("protective_orders") or {}
        return {str(order_id) for order_id in orders.values() if order_id}

    def _open_order_ids(self, orders: list[dict[str, Any]]) -> set[str]:
        ids = set()
        for order in orders:
            info = order.get("info", {}) or {}
            value = order.get("id") or order.get("orderId") or info.get("ordId")
            if value:
                ids.add(str(value))
        return ids

    def _protective_orders_present(self, orders: list[dict[str, Any]]) -> bool:
        expected = self._protective_order_ids()
        return bool(expected) and expected.issubset(self._open_order_ids(orders))

    def _record_api_success(self):
        self._api_error_count = 0

    def _record_api_failure(self, reason: str, error: Exception | str) -> bool:
        if reason == "market_data_failed":
            self._market_error_count += 1
            if self._market_error_count >= self.max_api_errors and self._paused_reason in (None, "market_data_unavailable"):
                self._paused_reason = "market_data_unavailable"
            logger.warning(f"行情错误({self._market_error_count}/{self.max_api_errors}): {error}")
            return False
        self._api_error_count += 1
        if self._api_error_count >= self.max_api_errors:
            self._paused_reason = "api_errors_exceeded"
            logger.error(
                f"连续 API 错误达到上限({self._api_error_count}/{self.max_api_errors})，"
                f"暂停交易: {reason}: {error}"
            )
            if self.alert_manager:
                self.alert_manager.send(
                    "critical",
                    "api_errors_exceeded",
                    "连续 API 错误达到上限，交易已暂停",
                    {
                        "reason": reason,
                        "error": str(error),
                        "count": self._api_error_count,
                        "max_api_errors": self.max_api_errors,
                    },
                )
            return False
        self._paused_reason = reason
        logger.error(
            f"API 错误({self._api_error_count}/{self.max_api_errors}): {reason}: {error}"
        )
        return False

    def _cancel_protective_orders(self, symbol: str) -> tuple[bool, str]:
        for order_id in self._protective_order_ids():
            result = self.order_executor.cancel_order(order_id, symbol)
            if not result.ok:
                return False, f"cancel_protective_failed:{order_id}:{result.reason}"
        return True, ""

    def _execute_live_close(self, reason: str, price: float) -> dict[str, Any] | None:
        pos = self._position
        if not pos:
            return None

        ccxt_sym = self.symbol + ":USDT"
        try:
            snapshot = self.account_reader.snapshot(price)
        except Exception as e:
            self._record_api_failure("account_snapshot_failed", e)
            return None

        close_order_id = None
        close_order = None
        close_fee = {"cost": 0.0, "currency": ""}
        exit_price = price
        filled_size = float(pos.get("size", 0.0) or 0.0)
        if snapshot.swap_position:
            close_side = "sell" if pos["side"] == "long" else "buy"
            size = filled_size or snapshot.swap_position.contracts
            try:
                amount = self.exchange.amount_to_precision(ccxt_sym, size)
            except Exception:
                amount = round(size, 4)
            close_result = self.order_executor.submit(OrderPlan(
                symbol=ccxt_sym,
                order_type="market",
                side=close_side,
                amount=amount,
                params={"reduceOnly": True, "tdMode": "isolated"},
                purpose="close",
            ))
            if not close_result.ok:
                self._paused_reason = f"close_order_failed:{close_result.reason}"
                logger.error(f"平仓下单失败，暂停交易: {close_result.reason}")
                return None
            fill_result = self.order_executor.wait_for_fill(close_result.order, ccxt_sym)
            if not fill_result.ok:
                self._paused_reason = f"close_not_filled:{fill_result.reason}"
                logger.error(f"平仓未成交，暂停交易: {fill_result.reason}")
                return None
            close_order_id = self._order_id(fill_result)
            close_order = self._order_snapshot(fill_result)
            close_fee = self._order_fee(fill_result)
            exit_price = self._order_average_price(fill_result, price)
            filled_size = self._order_filled_amount(fill_result, filled_size)

        cancelled, cancel_reason = self._cancel_protective_orders(ccxt_sym)
        if not cancelled:
            self._paused_reason = cancel_reason
            logger.error(f"保护单撤销失败，暂停交易: {cancel_reason}")
            return None

        try:
            after = self.account_reader.snapshot(exit_price)
        except Exception as e:
            self._record_api_failure("account_snapshot_failed", e)
            return None
        if after.swap_position:
            self._paused_reason = "close_position_not_flat"
            logger.error("平仓后交易所仍有合约仓位，暂停交易")
            return None

        return {
            "exit_price": exit_price,
            "filled_size": filled_size,
            "close_order_id": close_order_id,
            "close_order": close_order,
            "close_fee": close_fee,
            "exchange_already_flat": snapshot.swap_position is None,
        }

    # ═══════════════════════
    # 数据
    # ═══════════════════════

    def _fetch_market_candles(self, limit: int = 10) -> list:
        """A valid fresh public poll recovers only the market-data pause."""
        try:
            candles = self.exchange.fetch_ohlcv(self.symbol + ":USDT", self.timeframe, limit=limit)
            interval = self.exchange.parse_timeframe(self.timeframe) * 1000
            now = self.exchange.milliseconds()
            if not candles:
                raise ValueError("行情返回空K线")
            if any(len(row) < 6 or not all(np.isfinite(value) for value in row[:6]) for row in candles):
                raise ValueError("行情K线字段无效")
            if any(current[0] <= previous[0] for previous, current in zip(candles, candles[1:])):
                raise ValueError("行情K线重复或顺序错误")
            if candles[-1][0] > now or now - candles[-1][0] > 2 * interval:
                raise ValueError("行情K线过期或时间在未来")
        except (ccxt.BaseError, OSError, ValueError, TypeError) as error:
            self._record_api_failure("market_data_failed", error)
            return []
        self._last_market_success_at = str(datetime.now(timezone.utc))
        self._market_error_count = 0
        if self._paused_reason == "market_data_unavailable":
            if self.dry_run:
                self._paused_reason = None
                logger.info("行情恢复，解除行情暂停")
            else:
                # Exchange modes must reconcile positions/protective orders first.
                self._reconcile_live_state(float(candles[-1][4]))
        return candles

    def load_history(self, limit: int = 500):
        """加载历史K线用于指标预热"""
        ccxt_sym = self.symbol + ":USDT"
        candles = self.exchange.fetch_ohlcv(ccxt_sym, self.timeframe, limit=limit)
        # OKX 返回的最后一根通常仍在形成，不能用于收盘信号。
        interval_ms = self.exchange.parse_timeframe(self.timeframe) * 1000
        now_ms = self.exchange.milliseconds()
        candles = [c for c in candles if c[0] + interval_ms <= now_ms]
        if not candles:
            raise ValueError("没有已收盘K线可用于初始化")
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

    def _process_closed_candle(self, candle):
        """每根已收盘K线只处理一次，指标和信号均使用最终 OHLCV。"""
        self._candles.append(candle)
        for i, key in enumerate(["open", "high", "low", "close", "volume"]):
            self._ohlcv[key] = np.append(self._ohlcv[key], candle[i + 1])
        self._update_indicators()

        ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
        logger.info(
            f"🕐 已收盘K线 {ts} "
            f"O={candle[1]:.0f} H={candle[2]:.0f} "
            f"L={candle[3]:.0f} C={candle[4]:.0f}"
        )
        self._update_portfolio_plan(candle)
        self._write_account_snapshot(float(candle[4]))

        if self._position:
            self._last_signal = {"decision": "holding", "reason": "position_open"}
            self._check_exit_conditions(candle)
            return

        bars_since = self._bars_since_trade()
        pause = loss_pause_decision(
            consecutive_losses=self._consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            current_bar=len(self._candles),
            last_trade_bar=len(self._candles) - bars_since,
            loss_pause_bars=self.loss_pause_bars,
        )
        if not pause.can_trade:
            self._last_signal = {"decision": "cooldown", "reason": "loss_pause"}
            logger.info(f"连续亏损暂停中: 剩余{pause.remaining_bars}根K线")
            return
        if pause.reset_losses:
            logger.info("连续亏损暂停结束，恢复交易")
            self._consecutive_losses = 0
            self._save_state()

        cooldown = self.p_cooldown + \
            self._consecutive_losses * self.p_cooldown_extra_per_loss
        if bars_since < cooldown:
            self._last_signal = {"decision": "cooldown", "reason": "trade_cooldown"}
            logger.info(f"冷却中: {bars_since}/{cooldown}")
            return

        signal = self.evaluate()
        if signal:
            logger.info(
                f"🎯 信号: {signal['side']} "
                f"入场={signal['entry']:.0f} "
                f"SL={signal['sl']:.0f} TP={signal['tp']:.0f}"
            )
            executed = self.execute_signal(signal)
            self._last_signal = {**self._last_signal, "decision": "opened" if executed else "blocked",
                                 "side": signal["side"], "paused_reason": self._paused_reason}
        else:
            ind = self._indicators
            logger.info(
                "⏳ 收盘无信号: "
                f"ADX={ind['adx'][-1]:.1f} "
                f"距EMA20={abs(candle[4] - ind['ema20'][-1]):.0f} "
                f"ATR={ind['atr'][-1]:.0f} "
                f"EMA10/20/50={ind['ema10'][-1]:.0f}/"
                f"{ind['ema20'][-1]:.0f}/{ind['ema50'][-1]:.0f}"
            )

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
        self._last_signal = {"decision": "no_signal", "reason": "invalid_indicators"}
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

        spread = abs(e10 - e50) / c
        if spread < self.p_min_ema_spread:
            self._last_signal["reason"] = "ema_spread_too_small"
            return None

        if self.p_min_atr_percentile > 0:
            window = self.p_atr_percentile_window
            atr_series = self._indicators["atr"]
            close_series = self._ohlcv["close"]
            if len(atr_series) <= window:
                return None
            atr_pct = atr_series[-window - 1:-1] / close_series[-window - 1:-1]
            atr_pct = atr_pct[~np.isnan(atr_pct)]
            if len(atr_pct) == 0:
                return None
            current_atr_pct = atr_v / c
            if current_atr_pct < np.percentile(atr_pct, self.p_min_atr_percentile):
                return None

        if not strong or not near:
            self._last_signal["reason"] = "adx_too_low" if not strong else "too_far_from_ema20"
            return None

        if self.p_ema_slope_lookback > 0:
            lookback = self.p_ema_slope_lookback
            if len(self._ohlcv["close"]) <= lookback:
                return None
            previous_e20 = ind["ema20"][-1 - lookback]
            if np.isnan(previous_e20):
                return None
            bull_slope = e20 > previous_e20
            bear_slope = e20 < previous_e20
        else:
            bull_slope = True
            bear_slope = True

        self._last_signal = {
            "decision": "no_signal", "reason": "direction_filters_not_aligned",
            "above_ema200": bool(bull), "bull_alignment": bool(bull_align),
            "bear_alignment": bool(bear_align), "bull_slope": bool(bull_slope),
            "bear_slope": bool(bear_slope),
        }
        if bull and bull_align and bull_slope:
            self._last_signal.update(decision="signal", reason="long_conditions_met")
            return {
                "side": "long",
                "entry": float(c),
                "sl": float(c - atr_v * self.p_sl_atr),
                "tp": float(c + atr_v * self.p_tp_atr),
            }
        elif not bull and bear_align and bear_slope:
            self._last_signal.update(decision="signal", reason="short_conditions_met")
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
        if self.dry_run:
            return None
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
        if self.dry_run:
            return
        try:
            ccxt_sym = self.symbol + ":USDT"
            self.exchange.set_leverage(self.leverage, ccxt_sym)
            logger.info(f"杠杆已设为 {self.leverage}x")
        except Exception as e:
            logger.warning(f"设置杠杆失败(可能已设置): {e}")

    def _portfolio_cash(self) -> float:
        if self.dry_run:
            return self._paper_cash
        return self.get_balance()

    def _portfolio_equity(self, price: float) -> float:
        if self.dry_run:
            equity = self._paper_cash + self._paper_core_btc * price + self._total_pnl
            if self._position:
                notional = float(self._position.get("notional", 0.0))
                entry = float(self._position.get("entry", price))
                if self._position.get("side") == "long":
                    equity += notional * (price / entry - 1)
                else:
                    equity += notional * (entry / price - 1)
            return equity
        try:
            return self.account_reader.snapshot(price).equity_usdt
        except Exception as e:
            self._record_api_failure("account_snapshot_failed", e)
            return self.get_balance()

    def _account_snapshot(self, price: float) -> dict:
        if not self.dry_run:
            snapshot = self.account_reader.snapshot(price)
            return {
                "mode": "live_readonly",
                "price": price,
                **snapshot.as_dict(),
            }
        equity = self._portfolio_equity(price)
        exposure = portfolio_exposure(
            equity=equity,
            price=price,
            core_btc=self._paper_core_btc if self.dry_run else 0.0,
            swing_side=self._position.get("side") if self._position else None,
            swing_notional=float(self._position.get("notional", 0.0)) if self._position else 0.0,
        )
        return {
            "mode": "paper" if self.dry_run else "live_readonly",
            "price": price,
            "equity": equity,
            "cash": self._portfolio_cash(),
            "core_btc": self._paper_core_btc if self.dry_run else 0.0,
            "position": self._position,
            "net_exposure": exposure.net,
            "gross_exposure": exposure.gross,
            "core_weight": exposure.core_weight,
            "swing_weight": exposure.swing_weight,
            "updated_at": str(datetime.now(timezone.utc)),
        }

    def _write_account_snapshot(self, price: float):
        output = self.state_file.parent / "production_account_snapshot.json"
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(self._account_snapshot(price), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"保存账户快照失败: {e}")

    def _reconcile_live_state(self, price: float) -> bool:
        if self.dry_run:
            return True
        try:
            snapshot = self.account_reader.snapshot(price)
        except Exception as e:
            self._record_api_failure("account_snapshot_failed", e)
            return False

        result = reconcile_account_state(
            snapshot,
            self._position,
            allow_open_orders=bool(self._protective_order_ids()),
        )
        if result.ok:
            if self._position and not self._protective_orders_present(snapshot.open_orders):
                self._paused_reason = "protective_orders_missing"
                logger.warning("保护单缺失，暂停交易")
                return False
            self._record_api_success()
            self._paused_reason = None
            logger.info("账户对账通过")
            return True

        self._paused_reason = result.reason
        logger.warning(f"账户对账失败，暂停交易: {result.reason}")
        return False

    def _paper_performance_snapshot(self, price):
        equity = self._portfolio_equity(price)
        return {"equity": equity, "initial_capital": self.initial_capital,
                "total_pnl": equity - self.initial_capital,
                "return_pct": (equity / self.initial_capital - 1) * 100 if self.initial_capital > 0 else None,
                "price": price, "time": str(datetime.now(timezone.utc))}

    def _entry_analysis(self):
        return {**self._last_signal, "indicators": {
            key: float(values[-1]) for key, values in self._indicators.items()
            if len(values) and np.isfinite(values[-1])}}

    def _update_portfolio_plan(self, latest: list | None = None):
        """Log live portfolio targets; paper mode also simulates core spot rebalancing."""
        if not self.portfolio_config.enabled or len(self._ohlcv["close"]) == 0:
            return

        close = float(self._ohlcv["close"][-1])
        open_price = float(self._ohlcv["open"][-1])
        ema200 = float(self._indicators.get("ema200", [np.nan])[-1])
        regime = classify_latest_regime(self._ohlcv["close"])
        target_weight = self.portfolio_config.core_weights.get(
            regime,
            self.portfolio_config.core_weights["neutral"],
        )
        equity = self._portfolio_equity(close)
        cash = self._portfolio_cash()
        live_snapshot = None
        if not self.dry_run:
            try:
                live_snapshot = self.account_reader.snapshot(close)
            except Exception as e:
                self._record_api_failure("account_snapshot_failed", e)
                return
        core_btc = self._paper_core_btc if self.dry_run else live_snapshot.core_btc
        swing_side = self._position.get("side") if self._position else None
        swing_notional = float(self._position.get("notional", 0.0)) if self._position else 0.0
        if live_snapshot and live_snapshot.swap_position:
            swing_side = live_snapshot.swap_position.side
            swing_notional = live_snapshot.swap_position.notional
        exposure = portfolio_exposure(
            equity=equity,
            price=close,
            core_btc=core_btc,
            swing_side=swing_side,
            swing_notional=swing_notional,
        )
        bar_return = close / open_price - 1 if open_price > 0 else 0.0
        plan = plan_core_rebalance(
            current_weight=exposure.core_weight,
            target_weight=target_weight,
            regime_label=regime,
            bar_return=bar_return,
            close=close,
            ema200=ema200,
            config=self.portfolio_config,
        )

        logger.info(
            "组合计划: "
            f"regime={regime} equity={equity:.1f}U cash={cash:.1f}U "
            f"core={exposure.core_weight:.1%}->{target_weight:.1%} "
            f"net={exposure.net:.1%} gross={exposure.gross:.1%} "
            f"plan={plan.reason or 'hold'} delta={plan.weight_delta:+.1%}"
        )

        if not self.dry_run:
            spot_plan = build_spot_core_rebalance_plan(
                symbol=self.symbol,
                price=close,
                equity=equity,
                current_core_btc=core_btc,
                target_weight=target_weight,
                threshold=self.portfolio_config.rebalance_threshold,
                max_notional=self.portfolio_config.max_order_notional_usdt,
                min_notional=self.portfolio_config.min_order_notional_usdt,
            )
            if spot_plan:
                logger.info(
                    "现货核心仓计划: "
                    f"{spot_plan.side.upper()} {spot_plan.symbol} "
                    f"amount={spot_plan.amount} max_notional="
                    f"{self.portfolio_config.max_order_notional_usdt:.1f}U "
                    f"allow_spot={self.portfolio_config.allow_spot_trading}"
                )
            return

        if not self.dry_run or plan.weight_delta == 0:
            return

        analysis = {
            "reason": plan.reason, "regime": regime, "urgent": plan.urgent,
            "weight_before": exposure.core_weight, "target_weight": target_weight,
            "planned_weight": exposure.core_weight + plan.weight_delta,
            "bar_return": bar_return, "close": close,
            "ema200": ema200 if np.isfinite(ema200) else None,
            "below_ema200": bool(np.isfinite(ema200) and close < ema200),
            "sharp_drop": bool(bar_return <= -abs(self.portfolio_config.urgent_drop_pct)),
        }
        target_after_step = exposure.core_weight + plan.weight_delta
        target_notional = equity * target_after_step
        current_notional = self._paper_core_btc * close
        diff = target_notional - current_notional
        if diff > 0:
            buy_notional = min(diff, self._paper_cash / (1 + self.portfolio_config.spot_fee))
            if buy_notional <= 0:
                return
            fee = buy_notional * self.portfolio_config.spot_fee
            self._paper_cash -= buy_notional + fee
            self._paper_core_btc += buy_notional / close
            self._core_trade_log.append({
                "time": str(datetime.now(timezone.utc)), "mode": "paper", "side": "buy",
                "candle_time": self._candles[-1][0], "price": close,
                "size": buy_notional / close, "notional": buy_notional, "fee": fee,
                "analysis": analysis, "account_after": self._paper_performance_snapshot(close),
            })
            logger.info(
                f"📝 [纸上核心仓] BUY BTC 名义={buy_notional:.1f}U fee={fee:.2f}U"
            )
            self._save_state()
        else:
            sell_notional = min(abs(diff), current_notional)
            if sell_notional <= 0:
                return
            fee = sell_notional * self.portfolio_config.spot_fee
            self._paper_cash += sell_notional - fee
            self._paper_core_btc -= sell_notional / close
            self._core_trade_log.append({
                "time": str(datetime.now(timezone.utc)), "mode": "paper", "side": "sell",
                "candle_time": self._candles[-1][0], "price": close,
                "size": sell_notional / close, "notional": sell_notional, "fee": fee,
                "analysis": analysis, "account_after": self._paper_performance_snapshot(close),
            })
            logger.info(
                f"📝 [纸上核心仓] SELL BTC 名义={sell_notional:.1f}U fee={fee:.2f}U"
                f"{' urgent' if plan.urgent else ''}"
            )
            self._save_state()

    def execute_signal(self, signal: dict) -> bool:
        """执行交易信号"""
        ccxt_sym = self.symbol + ":USDT"

        if self._paused_reason:
            logger.warning(f"交易暂停中: {self._paused_reason}")
            return False

        if self._position:
            logger.info("已有纸上持仓, 跳过")
            return False

        if (
            self.portfolio_config.enabled
            and self.portfolio_config.mode not in LIVE_TRADING_MODES
            and not self.dry_run
        ):
            logger.warning(
                "组合风控处于非 live 模式，禁止真实下单；"
                "请先完成核心仓实盘执行与成交恢复后再切换到 testnet_live/live_small/live_scaled"
            )
            return False
        if (
            self.portfolio_config.enabled
            and self.portfolio_config.mode in LIVE_TRADING_MODES
            and not self.portfolio_config.allow_swap_trading
            and not self.dry_run
        ):
            self._paused_reason = "swap_trading_disabled"
            logger.warning("合约交易硬开关未开启，禁止真实下单")
            return False

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

        pause = loss_pause_decision(
            consecutive_losses=self._consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            current_bar=len(self._candles),
            last_trade_bar=len(self._candles) - self._bars_since_trade(),
            loss_pause_bars=self.loss_pause_bars,
        )
        if not pause.can_trade:
            logger.warning(
                f"⛔ 连续亏损{self._consecutive_losses}次, "
                f"暂停交易剩余{pause.remaining_bars}根K线"
            )
            return False
        if pause.reset_losses:
            logger.info("连续亏损暂停结束，恢复交易")
            self._consecutive_losses = 0

        # 计算仓位
        balance = self.initial_capital + self._total_pnl if self.dry_run \
            else self.get_balance()
        if balance <= 0:
            logger.error("余额为0")
            return False

        entry = signal["entry"]; sl = signal["sl"]
        sizing = calculate_position_size(
            balance=balance,
            entry=entry,
            stop_loss=sl,
            risk_per_trade=self.risk_per_trade,
            leverage=self.leverage,
            max_notional_pct=self.max_notional_pct,
        )
        if not sizing.tradable:
            logger.warning(f"仓位计算失败: {sizing.reason}")
            return False

        notional = sizing.notional
        if self.portfolio_config.enabled:
            regime = classify_latest_regime(self._ohlcv["close"])
            equity = self._portfolio_equity(float(signal["entry"]))
            if self.dry_run:
                core_notional = self._paper_core_btc * float(signal["entry"])
                cash = self._portfolio_cash()
            else:
                live_snapshot = self.account_reader.snapshot(float(signal["entry"]))
                core_notional = live_snapshot.core_notional
                cash = live_snapshot.usdt.free
            swing_limit = self.portfolio_config.swing_limits.get(
                regime,
                self.portfolio_config.swing_limits["neutral"],
            ).get(signal["side"], 0.0)
            capped_notional = cap_swing_notional(
                side=signal["side"],
                requested=notional,
                equity=equity,
                cash=cash,
                core_notional=core_notional,
                regime_limit=swing_limit,
                leverage=self.leverage,
                config=self.portfolio_config,
            )
            if capped_notional <= 0:
                logger.info(
                    f"组合风控拦截{signal['side']}信号: "
                    f"regime={regime} limit={swing_limit:.0%}"
                )
                return False
            if capped_notional < notional:
                logger.info(
                    f"组合风控缩小仓位: {notional:.1f}U -> {capped_notional:.1f}U"
                )
                notional = capped_notional

        size = notional / entry

        # 精度处理
        try:
            amount = self.exchange.amount_to_precision(ccxt_sym, size)
            entry_price = self.exchange.price_to_precision(ccxt_sym, entry)
        except Exception:
            amount = round(size, 4)
            entry_price = entry

        side = "buy" if signal["side"] == "long" else "sell"

        if self.dry_run:
            self._position = {
                **signal,
                "entry_analysis": self._entry_analysis(),
                "size": size,
                "notional": notional,
                "raw_notional": sizing.raw_notional,
                "max_notional": sizing.max_notional,
                "capped": sizing.capped or notional < sizing.notional,
            }
            logger.info(
                f"📝 [纸上] {side.upper()} {amount}张 @{entry:.0f} "
                f"名义={notional:.1f}U "
                f"SL={sl:.0f} TP={signal['tp']:.0f}"
                f"{' capped' if sizing.capped or notional < sizing.notional else ''}"
            )
            self._save_state()
            return True

        tp_side = "sell" if side == "buy" else "buy"
        tp_price = self.exchange.price_to_precision(ccxt_sym, signal["tp"])
        sl_price = self.exchange.price_to_precision(ccxt_sym, signal["sl"])
        result = self.order_executor.submit_bracket(
            entry=OrderPlan(
                symbol=ccxt_sym,
                order_type="limit",
                side=side,
                amount=amount,
                price=entry_price,
                params={"tdMode": "isolated", "leverage": self.leverage},
                purpose="entry",
            ),
            take_profit=OrderPlan(
                symbol=ccxt_sym,
                order_type="limit",
                side=tp_side,
                amount=amount,
                price=tp_price,
                params={"reduceOnly": True, "tdMode": "isolated"},
                purpose="take_profit",
            ),
            stop_loss=OrderPlan(
                symbol=ccxt_sym,
                order_type="stop_market",
                side=tp_side,
                amount=amount,
                params={
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                    "tdMode": "isolated",
                },
                purpose="stop_loss",
            )
        )
        if not result.ok:
            logger.error(f"下单失败: {result.reason}")
            self._paused_reason = f"order_failed:{result.reason}"
            return False

        filled_size = self._order_filled_amount(result.entry, float(size))
        fill_price = self._order_average_price(result.entry, float(entry))
        live_notional = filled_size * fill_price
        self._position = {
            **signal,
            "entry": fill_price,
            "size": filled_size,
            "notional": live_notional,
            "raw_notional": sizing.raw_notional,
            "max_notional": sizing.max_notional,
            "capped": sizing.capped or notional < sizing.notional,
            "entry_order_id": self._order_id(result.entry),
            "entry_order": self._order_snapshot(result.entry),
            "entry_fee": self._order_fee(result.entry),
            "protective_orders": {
                "take_profit": self._order_id(result.take_profit),
                "stop_loss": self._order_id(result.stop_loss),
            },
            "protective_order_details": {
                "take_profit": self._order_snapshot(result.take_profit),
                "stop_loss": self._order_snapshot(result.stop_loss),
            },
            "opened_at": str(datetime.now(timezone.utc)),
            "live": True,
        }
        self._save_state()

        logger.info(
            f"📈 下单: {side} {amount}张 @{entry_price} "
            f"名义={notional:.1f}U"
            f"{' capped' if sizing.capped or notional < sizing.notional else ''}"
        )
        logger.info(f"  止盈={tp_price} 止损={sl_price}")
        return True

    # ═══════════════════════
    # 主循环
    # ═══════════════════════

    def start(self):
        """启动机器人"""
        logger.info("🚀 启动 TripleEma 实盘机器人...")

        self.load_history(500)
        self._load_state()
        self.set_leverage()
        if self._paused_reason:
            self._write_health()
            return

        if not self.dry_run:
            price = float(self._ohlcv["close"][-1])
            if not self._reconcile_live_state(price):
                return

        self._running = True
        self._last_market_success_at = str(datetime.now(timezone.utc))
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
        if self.dry_run:
            self._reconcile_live_state(float(c))
        self._update_portfolio_plan()
        self._write_account_snapshot(float(c))
        self._write_health()

        # 启动时不重放历史收盘信号；等待下一根K线收盘后再评估。
        last_candle_ts = self._candles[-1][0]
        ccxt_sym = self.symbol + ":USDT"
        interval_ms = self.exchange.parse_timeframe(self.timeframe) * 1000

        # ── 主循环 ──
        while self._running:
            try:
                # 获取最近K线。最后一根可能仍在形成，只处理已收盘部分。
                candles = self._fetch_market_candles()
                if not candles:
                    self._write_health()
                    time.sleep(60)
                    continue
                now_ms = self.exchange.milliseconds()
                closed = closed_candles_after(
                    candles, last_candle_ts, interval_ms, now_ms)
                for candle in closed:
                    if candle[0] - last_candle_ts > interval_ms:
                        logger.warning("K线缺口超过轮询窗口，重新加载历史K线")
                        self.load_history(500)
                        last_candle_ts = self._candles[-1][0]
                        break
                    self._process_closed_candle(candle)
                    last_candle_ts = candle[0]
                self._write_health()
                time.sleep(60)

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self._record_api_failure("main_loop_error", e)
                self._write_health()
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

        live_close = None
        if not self.dry_run:
            live_close = self._execute_live_close(reason, price)
            if live_close is None:
                return
            price = float(live_close["exit_price"])

        notional = pos.get("notional", self.initial_capital * self.leverage)
        if pos["side"] == "long":
            pnl_usdt = notional * (price / pos["entry"] - 1)
        else:
            pnl_usdt = notional * (pos["entry"] / price - 1)

        pnl_pct = (pnl_usdt / max(self.initial_capital, 1)) * 100
        self._total_pnl += pnl_usdt
        self._daily_pnl += pnl_usdt

        if pnl_usdt < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._last_trade_bar = len(self._candles)
        self._last_trade_candle_ts = self._candles[-1][0] if self._candles else None

        trade = {
            "side": pos["side"],
            "entry": pos["entry"],
            "exit": price,
            "notional": round(notional, 2),
            "pnl": round(pnl_usdt, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "time": str(datetime.now(timezone.utc)),
            "candle_time": self._last_trade_candle_ts,
        }
        if live_close:
            entry_fee = pos.get("entry_fee") or {"cost": 0.0, "currency": ""}
            close_fee = live_close["close_fee"]
            trade.update({
                "close_order_id": live_close["close_order_id"],
                "close_order": live_close["close_order"],
                "exchange_already_flat": live_close["exchange_already_flat"],
                "filled_size": live_close["filled_size"],
                "entry_order_id": pos.get("entry_order_id"),
                "entry_order": pos.get("entry_order"),
                "entry_fee": entry_fee,
                "close_fee": close_fee,
                "total_fee": round(
                    float(entry_fee.get("cost", 0.0) or 0.0)
                    + float(close_fee.get("cost", 0.0) or 0.0),
                    8,
                ),
                "protective_orders": pos.get("protective_orders"),
            })
        self._position = None
        if self.dry_run:
            trade["entry_analysis"] = pos.get("entry_analysis")
            trade["account_after"] = self._paper_performance_snapshot(price)
        self._trade_log.append(trade)

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
        self._save_state()

    def stop(self):
        """停止机器人"""
        self._running = False
        self._write_health()
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

    strategy_cfg = _normalize_strategy_config(cfg.get("strategy", {}))

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
        loss_pause_bars=cfg["risk"].get("loss_pause_bars", 18),
        max_notional_pct=cfg["risk"].get("max_notional_pct", 0.95),
        portfolio_config=load_portfolio_config(cfg.get("portfolio")),
        state_file=cfg.get("output", {}).get("state_file", "outputs/production_state.json"),
        max_api_errors=int(cfg.get("live", {}).get("max_api_errors", 3)),
        alert_manager=LocalAlertManager(
            cfg.get("output", {}).get("alert_log", "outputs/alerts.jsonl")
        ),
        **strategy_cfg,
    )

    bot.dry_run = cfg["mode"]["dry_run"]

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()


def _normalize_strategy_config(strategy_cfg: dict) -> dict:
    """Map YAML strategy keys onto ProductionBot constructor parameters."""
    cfg = dict(strategy_cfg)
    cfg.pop("name", None)
    cfg.pop("adx_period", None)
    cfg.pop("atr_period", None)

    key_map = {
        "sl_atr_mult": "sl_atr",
        "tp_atr_mult": "tp_atr",
        "cooldown_base": "cooldown_bars",
    }
    for old_key, new_key in key_map.items():
        if old_key in cfg:
            cfg[new_key] = cfg.pop(old_key)
    return cfg
