"""Order execution abstractions for live trading adapters."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    order_type: str
    side: str
    amount: float | str
    price: float | str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = "entry"


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    order: dict[str, Any] | None = None
    reason: str = ""
    status: str = ""
    filled: float = 0.0
    remaining: float = 0.0
    fee_cost: float = 0.0
    fee_currency: str = ""


@dataclass(frozen=True)
class BracketOrderResult:
    ok: bool
    entry: OrderResult
    take_profit: OrderResult | None = None
    stop_loss: OrderResult | None = None
    reason: str = ""


class OrderExecutor:
    """Small execution adapter. It owns submit calls, not strategy decisions."""

    def __init__(self, exchange, poll_interval: float = 1.0, fill_timeout: float = 30.0):
        self.exchange = exchange
        self.poll_interval = poll_interval
        self.fill_timeout = fill_timeout

    def submit(self, plan: OrderPlan) -> OrderResult:
        try:
            order = self.exchange.create_order(
                symbol=plan.symbol,
                type=plan.order_type,
                side=plan.side,
                amount=plan.amount,
                price=plan.price,
                params=plan.params,
            )
            return order_result_from_order(order)
        except Exception as e:
            return OrderResult(False, reason=str(e))

    def wait_for_fill(
        self,
        order: dict[str, Any],
        symbol: str,
        timeout: float | None = None,
    ) -> OrderResult:
        order_id = order.get("id")
        if not order_id:
            return OrderResult(False, order=order, reason="missing_order_id")

        timeout = self.fill_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        latest = order
        while True:
            try:
                latest = self.exchange.fetch_order(order_id, symbol)
            except Exception as e:
                return OrderResult(False, order=latest, reason=f"fetch_order_failed:{e}")

            result = order_result_from_order(latest)
            if result.status == "closed":
                return OrderResult(True, order=latest, status=result.status,
                                   filled=result.filled, remaining=result.remaining,
                                   fee_cost=result.fee_cost,
                                   fee_currency=result.fee_currency)
            if result.status in ("canceled", "cancelled", "rejected", "expired"):
                return OrderResult(False, order=latest, reason=f"order_{result.status}",
                                   status=result.status, filled=result.filled,
                                   remaining=result.remaining,
                                   fee_cost=result.fee_cost,
                                   fee_currency=result.fee_currency)
            if time.monotonic() >= deadline:
                cancel = self.cancel_order(order_id, symbol)
                reason = "timeout_cancelled" if cancel.ok else f"timeout_cancel_failed:{cancel.reason}"
                return OrderResult(False, order=latest, reason=reason,
                                   status=result.status, filled=result.filled,
                                   remaining=result.remaining,
                                   fee_cost=result.fee_cost,
                                   fee_currency=result.fee_currency)
            time.sleep(self.poll_interval)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        try:
            order = self.exchange.cancel_order(order_id, symbol)
            return order_result_from_order(order)
        except Exception as e:
            return OrderResult(False, reason=str(e))

    def confirm_open_order(self, order: dict[str, Any], symbol: str) -> OrderResult:
        order_id = order.get("id")
        if not order_id:
            return OrderResult(False, order=order, reason="missing_order_id")
        try:
            latest = self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            return OrderResult(False, order=order, reason=f"fetch_order_failed:{e}")

        result = order_result_from_order(latest)
        if result.status in ("open", "closed"):
            return OrderResult(True, order=latest, status=result.status,
                               filled=result.filled, remaining=result.remaining,
                               fee_cost=result.fee_cost,
                               fee_currency=result.fee_currency)
        return OrderResult(False, order=latest, reason=f"protective_order_{result.status}",
                           status=result.status, filled=result.filled,
                           remaining=result.remaining,
                           fee_cost=result.fee_cost,
                           fee_currency=result.fee_currency)

    def submit_bracket(
        self,
        entry: OrderPlan,
        take_profit: OrderPlan,
        stop_loss: OrderPlan,
    ) -> BracketOrderResult:
        entry_result = self.submit(entry)
        if not entry_result.ok:
            return BracketOrderResult(False, entry=entry_result, reason="entry_failed")
        fill_result = self.wait_for_fill(entry_result.order, entry.symbol)
        if not fill_result.ok:
            return BracketOrderResult(False, entry=fill_result, reason="entry_not_filled")

        tp_result = self.submit(take_profit)
        if not tp_result.ok:
            return BracketOrderResult(
                False,
                entry=entry_result,
                take_profit=tp_result,
                reason="take_profit_failed",
            )
        tp_confirm = self.confirm_open_order(tp_result.order, take_profit.symbol)
        if not tp_confirm.ok:
            return BracketOrderResult(
                False,
                entry=entry_result,
                take_profit=tp_confirm,
                reason="take_profit_unconfirmed",
            )

        sl_result = self.submit(stop_loss)
        if not sl_result.ok:
            return BracketOrderResult(
                False,
                entry=entry_result,
                take_profit=tp_confirm,
                stop_loss=sl_result,
                reason="stop_loss_failed",
            )
        sl_confirm = self.confirm_open_order(sl_result.order, stop_loss.symbol)
        if not sl_confirm.ok:
            return BracketOrderResult(
                False,
                entry=entry_result,
                take_profit=tp_confirm,
                stop_loss=sl_confirm,
                reason="stop_loss_unconfirmed",
            )

        return BracketOrderResult(
            True,
            entry=fill_result,
            take_profit=tp_confirm,
            stop_loss=sl_confirm,
        )


def order_result_from_order(order: dict[str, Any] | None) -> OrderResult:
    order = order or {}
    status = str(order.get("status") or "").lower()
    filled = _to_float(order.get("filled"))
    remaining = _to_float(order.get("remaining"))
    fee_cost, fee_currency = _parse_order_fee(order)
    if status in ("rejected", "expired"):
        return OrderResult(False, order=order, reason=f"order_{status}",
                           status=status, filled=filled, remaining=remaining,
                           fee_cost=fee_cost, fee_currency=fee_currency)
    return OrderResult(True, order=order, status=status, filled=filled, remaining=remaining,
                       fee_cost=fee_cost, fee_currency=fee_currency)


def _parse_order_fee(order: dict[str, Any]) -> tuple[float, str]:
    fee = order.get("fee")
    if isinstance(fee, dict):
        cost = _to_float(fee.get("cost"))
        currency = str(fee.get("currency") or "")
        if cost:
            return cost, currency

    fees = order.get("fees")
    if isinstance(fees, list):
        total = 0.0
        currency = ""
        for item in fees:
            if not isinstance(item, dict):
                continue
            total += _to_float(item.get("cost"))
            currency = currency or str(item.get("currency") or "")
        if total:
            return total, currency

    info = order.get("info", {}) or {}
    cost = _to_float(info.get("fee") or info.get("fillFee"))
    currency = str(info.get("feeCcy") or info.get("feeCurrency") or "")
    return cost, currency


def build_spot_core_rebalance_plan(
    symbol: str,
    price: float,
    equity: float,
    current_core_btc: float,
    target_weight: float,
    threshold: float,
    max_notional: float,
    min_notional: float,
) -> OrderPlan | None:
    if price <= 0 or equity <= 0:
        return None

    target_notional = equity * target_weight
    current_notional = current_core_btc * price
    diff = target_notional - current_notional
    if abs(diff) / equity < threshold:
        return None

    notional = min(abs(diff), max_notional)
    if notional < min_notional:
        return None

    if diff > 0:
        return OrderPlan(
            symbol=symbol,
            order_type="market",
            side="buy",
            amount=notional,
            params={"tdMode": "cash", "quoteOrderQty": notional},
            purpose="spot_core_rebalance",
        )

    return OrderPlan(
        symbol=symbol,
        order_type="market",
        side="sell",
        amount=notional / price,
        params={"tdMode": "cash", "notional": notional},
        purpose="spot_core_rebalance",
    )


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
