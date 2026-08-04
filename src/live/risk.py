"""Shared production risk controls for live trading and offline replay."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSize:
    tradable: bool
    size: float = 0.0
    notional: float = 0.0
    raw_notional: float = 0.0
    max_notional: float = 0.0
    stop_pct: float = 0.0
    risk_amount: float = 0.0
    capped: bool = False
    reason: str = ""


@dataclass(frozen=True)
class LossPauseDecision:
    can_trade: bool
    reset_losses: bool = False
    remaining_bars: int = 0
    reason: str = ""


def calculate_position_size(
    balance: float,
    entry: float,
    stop_loss: float,
    risk_per_trade: float,
    leverage: float,
    max_notional_pct: float = 0.95,
) -> PositionSize:
    """Calculate notional exposure from account risk and leverage cap."""
    if balance <= 0:
        return PositionSize(False, reason="balance_non_positive")
    if entry <= 0:
        return PositionSize(False, reason="entry_non_positive")
    if stop_loss <= 0:
        return PositionSize(False, reason="stop_non_positive")
    if risk_per_trade <= 0:
        return PositionSize(False, reason="risk_non_positive")
    if leverage <= 0:
        return PositionSize(False, reason="leverage_non_positive")
    if max_notional_pct <= 0 or max_notional_pct > 1:
        return PositionSize(False, reason="max_notional_pct_invalid")

    stop_pct = abs(entry - stop_loss) / entry
    if stop_pct <= 0:
        return PositionSize(False, reason="stop_distance_zero")

    risk_amount = balance * risk_per_trade
    raw_notional = risk_amount / stop_pct
    max_notional = balance * leverage * max_notional_pct
    notional = min(raw_notional, max_notional)
    if notional <= 0:
        return PositionSize(False, reason="notional_non_positive")

    return PositionSize(
        tradable=True,
        size=notional / entry,
        notional=notional,
        raw_notional=raw_notional,
        max_notional=max_notional,
        stop_pct=stop_pct,
        risk_amount=risk_amount,
        capped=raw_notional > max_notional,
    )


def loss_pause_decision(
    consecutive_losses: int,
    max_consecutive_losses: int,
    current_bar: int,
    last_trade_bar: int,
    loss_pause_bars: int,
) -> LossPauseDecision:
    """Decide whether the consecutive-loss pause should block or reset."""
    if max_consecutive_losses <= 0:
        return LossPauseDecision(can_trade=True)
    if consecutive_losses < max_consecutive_losses:
        return LossPauseDecision(can_trade=True)

    if loss_pause_bars <= 0:
        return LossPauseDecision(can_trade=True, reset_losses=True)

    bars_since_trade = current_bar - last_trade_bar
    if bars_since_trade >= loss_pause_bars:
        return LossPauseDecision(can_trade=True, reset_losses=True)

    remaining = max(loss_pause_bars - bars_since_trade, 0)
    return LossPauseDecision(
        can_trade=False,
        remaining_bars=remaining,
        reason="loss_pause",
    )
