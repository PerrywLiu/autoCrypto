"""Read-only OKX account snapshot parsing for portfolio reconciliation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.live.portfolio import portfolio_exposure


@dataclass(frozen=True)
class AssetBalance:
    currency: str
    free: float = 0.0
    used: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class SwapPosition:
    symbol: str
    side: str
    contracts: float
    notional: float
    entry_price: float
    unrealized_pnl: float = 0.0
    margin: float = 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    usdt: AssetBalance
    btc: AssetBalance
    swap_position: SwapPosition | None
    open_orders: list[dict[str, Any]]
    equity_usdt: float
    core_btc: float
    core_notional: float
    swing_notional: float
    net_exposure: float
    gross_exposure: float
    core_weight: float
    swing_weight: float
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "usdt": self.usdt.__dict__,
            "btc": self.btc.__dict__,
            "swap_position": self.swap_position.__dict__ if self.swap_position else None,
            "open_orders": self.open_orders,
            "equity_usdt": self.equity_usdt,
            "core_btc": self.core_btc,
            "core_notional": self.core_notional,
            "swing_notional": self.swing_notional,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "core_weight": self.core_weight,
            "swing_weight": self.swing_weight,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ReconcileResult:
    ok: bool
    reason: str = ""


def parse_asset_balance(balance: dict[str, Any], currency: str) -> AssetBalance:
    value = balance.get(currency, {}) or {}
    return AssetBalance(
        currency=currency,
        free=_to_float(value.get("free")),
        used=_to_float(value.get("used")),
        total=_to_float(value.get("total")),
    )


def parse_swap_position(position: dict[str, Any] | None, fallback_symbol: str) -> SwapPosition | None:
    if not position:
        return None

    contracts = abs(_to_float(position.get("contracts") or position.get("contractSize")))
    notional = abs(_to_float(position.get("notional")))
    entry_price = _to_float(
        position.get("entryPrice")
        or position.get("entry_price")
        or position.get("average")
    )

    info = position.get("info", {}) or {}
    if notional <= 0:
        notional = abs(_to_float(info.get("notionalUsd") or info.get("notional")))
    if entry_price <= 0:
        entry_price = _to_float(info.get("avgPx") or info.get("avgPrice"))
    if contracts <= 0:
        contracts = abs(_to_float(info.get("pos")))
    if contracts <= 0 and notional <= 0:
        return None

    side = str(position.get("side") or info.get("posSide") or "").lower()
    if side not in ("long", "short"):
        raw_contracts = _to_float(position.get("contracts") or info.get("pos"))
        side = "short" if raw_contracts < 0 else "long"

    return SwapPosition(
        symbol=str(position.get("symbol") or fallback_symbol),
        side=side,
        contracts=contracts,
        notional=notional,
        entry_price=entry_price,
        unrealized_pnl=_to_float(
            position.get("unrealizedPnl")
            or position.get("unrealized_pnl")
            or info.get("upl")
        ),
        margin=_to_float(position.get("initialMargin") or position.get("margin") or info.get("imr")),
    )


def build_account_snapshot(
    balance: dict[str, Any],
    positions: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    price: float,
    symbol: str = "BTC/USDT",
    swap_symbol: str = "BTC/USDT:USDT",
) -> AccountSnapshot:
    usdt = parse_asset_balance(balance, "USDT")
    btc = parse_asset_balance(balance, "BTC")
    swap_position = None
    for position in positions:
        candidate = parse_swap_position(position, swap_symbol)
        if candidate and candidate.symbol in (swap_symbol, symbol):
            swap_position = candidate
            break

    core_btc = btc.total
    core_notional = core_btc * price
    swing_notional = swap_position.notional if swap_position else 0.0
    unrealized = swap_position.unrealized_pnl if swap_position else 0.0
    equity = usdt.total + core_notional + unrealized
    exposure = portfolio_exposure(
        equity=equity,
        price=price,
        core_btc=core_btc,
        swing_side=swap_position.side if swap_position else None,
        swing_notional=swing_notional,
    )
    return AccountSnapshot(
        usdt=usdt,
        btc=btc,
        swap_position=swap_position,
        open_orders=list(open_orders or []),
        equity_usdt=equity,
        core_btc=core_btc,
        core_notional=core_notional,
        swing_notional=swing_notional,
        net_exposure=exposure.net,
        gross_exposure=exposure.gross,
        core_weight=exposure.core_weight,
        swing_weight=exposure.swing_weight,
        updated_at=str(datetime.now(timezone.utc)),
    )


class OKXAccountReader:
    """Read-only account adapter. It must not submit, amend, or cancel orders."""

    def __init__(self, exchange, symbol: str = "BTC/USDT"):
        self.exchange = exchange
        self.symbol = symbol
        self.swap_symbol = symbol + ":USDT"

    def snapshot(self, price: float) -> AccountSnapshot:
        balance = self.exchange.fetch_balance()
        positions = self.exchange.fetch_positions([self.swap_symbol])
        spot_orders = self.exchange.fetch_open_orders(self.symbol)
        swap_orders = self.exchange.fetch_open_orders(self.swap_symbol)
        return build_account_snapshot(
            balance=balance,
            positions=positions,
            open_orders=[*spot_orders, *swap_orders],
            price=price,
            symbol=self.symbol,
            swap_symbol=self.swap_symbol,
        )


def reconcile_account_state(
    snapshot: AccountSnapshot,
    local_position: dict[str, Any] | None,
    allow_open_orders: bool = False,
) -> ReconcileResult:
    if snapshot.open_orders and not allow_open_orders:
        return ReconcileResult(False, "open_orders_present")

    has_exchange_position = snapshot.swap_position is not None
    has_local_position = local_position is not None
    if has_exchange_position and not has_local_position:
        return ReconcileResult(False, "exchange_position_without_local_state")
    if has_local_position and not has_exchange_position:
        return ReconcileResult(False, "local_position_without_exchange_position")
    if has_exchange_position and has_local_position:
        local_side = str(local_position.get("side", "")).lower()
        if local_side and local_side != snapshot.swap_position.side:
            return ReconcileResult(False, "position_side_mismatch")

        local_notional = abs(_to_float(local_position.get("notional")))
        if local_notional > 0:
            diff = abs(local_notional - snapshot.swap_position.notional)
            tolerance = max(local_notional, snapshot.swap_position.notional) * 0.10
            if diff > tolerance:
                return ReconcileResult(False, "position_notional_mismatch")

    return ReconcileResult(True)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
