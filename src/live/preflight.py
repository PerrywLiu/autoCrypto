"""Preflight checks before enabling OKX live trading modes."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.live.okx_account import AccountSnapshot, build_account_snapshot
from src.live.portfolio import LIVE_TRADING_MODES, load_portfolio_config


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    severity: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class PreflightReport:
    checks: list[PreflightCheck] = field(default_factory=list)
    snapshot: AccountSnapshot | None = None

    @property
    def ok(self) -> bool:
        return all(check.ok or check.severity != "error" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
            "snapshot": self.snapshot.as_dict() if self.snapshot else None,
        }


def validate_preflight_config(cfg: dict[str, Any], env: Mapping[str, str]) -> list[PreflightCheck]:
    checks = []
    portfolio = load_portfolio_config(cfg.get("portfolio"))
    dry_run = bool(cfg.get("mode", {}).get("dry_run", True))
    testnet = bool(cfg.get("mode", {}).get("testnet", True))

    checks.append(PreflightCheck(
        "portfolio_enabled",
        portfolio.enabled,
        "error",
        "portfolio.enabled 必须为 true",
    ))
    known_mode = portfolio.mode in {"paper", "readonly", *LIVE_TRADING_MODES}
    checks.append(PreflightCheck(
        "portfolio_mode_known",
        known_mode,
        "error",
        f"portfolio.mode={portfolio.mode}",
    ))

    has_api_keys = all(env.get(key) for key in ("OKX_API_KEY", "OKX_SECRET", "OKX_PASSPHRASE"))
    needs_api = (not dry_run) or portfolio.mode in {"readonly", *LIVE_TRADING_MODES}
    checks.append(PreflightCheck(
        "okx_api_keys",
        has_api_keys or not needs_api,
        "error",
        "非 dry-run/readonly/live 模式必须配置 OKX_API_KEY、OKX_SECRET、OKX_PASSPHRASE",
    ))

    live_mode = portfolio.mode in LIVE_TRADING_MODES and not dry_run
    checks.append(PreflightCheck(
        "live_mode_switches",
        (not live_mode) or portfolio.allow_spot_trading or portfolio.allow_swap_trading,
        "error",
        "live 模式必须显式打开至少一个真实交易硬开关",
    ))
    checks.append(PreflightCheck(
        "testnet_mode_match",
        not live_mode or portfolio.mode != "testnet_live" or testnet,
        "error",
        "testnet_live 必须使用 mode.testnet=true",
    ))
    checks.append(PreflightCheck(
        "real_mode_match",
        not live_mode or portfolio.mode == "testnet_live" or not testnet,
        "error",
        "live_small/live_scaled 必须使用 mode.testnet=false",
    ))
    checks.append(PreflightCheck(
        "single_order_cap",
        float(portfolio.max_order_notional_usdt) > 0,
        "error",
        "max_order_notional_usdt 必须大于 0",
    ))
    checks.append(PreflightCheck(
        "min_order_cap",
        float(portfolio.min_order_notional_usdt) > 0,
        "error",
        "min_order_notional_usdt 必须大于 0",
    ))
    return checks


def needs_okx_account(cfg: dict[str, Any]) -> bool:
    portfolio = load_portfolio_config(cfg.get("portfolio"))
    dry_run = bool(cfg.get("mode", {}).get("dry_run", True))
    return (not dry_run) or portfolio.mode in {"readonly", *LIVE_TRADING_MODES}


def validate_account_snapshot(
    snapshot: AccountSnapshot,
    allow_existing_state: bool = False,
    state: dict[str, Any] | None = None,
) -> list[PreflightCheck]:
    state_checks = validate_live_state_position(state or {}, allow_empty=False)
    can_restore_state = bool(state_checks) and all(check.ok for check in state_checks)
    checks = [
        PreflightCheck(
            "account_equity_positive",
            snapshot.equity_usdt > 0,
            "error",
            f"账户权益={snapshot.equity_usdt:.2f}U",
        ),
        PreflightCheck(
            "no_unexpected_open_orders",
            not snapshot.open_orders,
            "error",
            f"未完成挂单数量={len(snapshot.open_orders)}",
        ),
        PreflightCheck(
            "no_unexpected_swap_position",
            snapshot.swap_position is None or (allow_existing_state and can_restore_state),
            "error",
            "交易所有合约仓位，必须先确认本地状态可恢复",
        ),
    ]
    return checks


def load_state_file(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def validate_live_state_position(
    state: dict[str, Any],
    allow_empty: bool = True,
) -> list[PreflightCheck]:
    position = state.get("position")
    if position is None:
        return [PreflightCheck(
            "state_position",
            allow_empty,
            "error",
            "本地状态没有可恢复 live 仓位",
        )]
    ok = isinstance(position, dict)
    if not ok:
        return [PreflightCheck("state_position", False, "error", "position 必须是对象")]

    required = ("side", "entry", "notional", "size", "entry_order_id", "protective_orders")
    missing = [key for key in required if key not in position]
    protective = position.get("protective_orders") if isinstance(position, dict) else {}
    if isinstance(protective, dict):
        if not protective.get("take_profit"):
            missing.append("protective_orders.take_profit")
        if not protective.get("stop_loss"):
            missing.append("protective_orders.stop_loss")
    else:
        missing.append("protective_orders")

    return [PreflightCheck(
        "state_position_schema",
        not missing,
        "error",
        "本地 live 仓位状态字段完整" if not missing else f"缺少字段: {', '.join(missing)}",
    )]


def build_snapshot_from_exchange(exchange, symbol: str, price: float) -> AccountSnapshot:
    swap_symbol = symbol + ":USDT"
    return build_account_snapshot(
        balance=exchange.fetch_balance(),
        positions=exchange.fetch_positions([swap_symbol]),
        open_orders=[
            *exchange.fetch_open_orders(symbol),
            *exchange.fetch_open_orders(swap_symbol),
        ],
        price=price,
        symbol=symbol,
        swap_symbol=swap_symbol,
    )
