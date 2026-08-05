#!/usr/bin/env python
"""Backtest BTC core-satellite portfolio blends with the online 4h strategy."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.backtest_online import (
    COMMISSION,
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE,
    calculate_indicators,
    check_exit,
    compute_buy_hold_equity,
    compute_equity_metrics,
    compute_monthly_stats,
    evaluate_signal,
    load_config,
    load_local_ohlcv,
    replay,
)
from src.live.portfolio import (
    DEFAULT_CORE_WEIGHTS as REGIME_CORE_WEIGHTS,
    PortfolioConfig,
    align_regime_core_weights,
    align_regime_labels,
    cap_swing_notional,
    compute_daily_regime,
    plan_core_rebalance,
)
from src.live.risk import calculate_position_size, loss_pause_decision


PERIODS_PER_YEAR = 365 * 6


@dataclass
class SwingPosition:
    side: str
    entry: float
    sl: float
    tp: float
    notional: float
    raw_notional: float
    max_notional: float
    size: float
    stop_pct: float
    capped: bool
    open_fee: float
    entry_time: pd.Timestamp
    entry_bar: int

    def as_exit_dict(self) -> dict:
        return {
            "side": self.side,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "notional": self.notional,
            "entry_time": self.entry_time,
            "entry_bar": self.entry_bar,
        }


@dataclass
class PortfolioState:
    cash: float
    core_btc: float = 0.0
    swing: SwingPosition | None = None
    total_fee: float = 0.0
    total_funding: float = 0.0
    rebalance_count: int = 0
    core_buy_count: int = 0
    core_sell_count: int = 0
    urgent_rebalance_count: int = 0
    swing_trades: int = 0
    capped_orders: int = 0
    pause_events: int = 0
    max_net_exposure: float = 0.0
    max_gross_exposure: float = 0.0


def parse_weights(value: str) -> list[float]:
    weights = [float(v.strip()) for v in value.split(",") if v.strip()]
    for weight in weights:
        if weight < 0 or weight > 1:
            raise ValueError(f"core weight must be in [0, 1]: {weight}")
    return weights


def blend_equity(
    strategy_equity: pd.Series,
    buy_hold_equity: pd.Series,
    core_weight: float,
) -> pd.Series:
    strategy_aligned, buy_hold_aligned = strategy_equity.align(
        buy_hold_equity,
        join="inner",
    )
    return core_weight * buy_hold_aligned + (1 - core_weight) * strategy_aligned


def yearly_returns(equity: pd.Series) -> dict[str, float]:
    if equity.empty:
        return {}
    rows = {}
    for year, group in equity.groupby(equity.index.year):
        if len(group) < 2:
            continue
        rows[str(year)] = float(group.iloc[-1] / group.iloc[0] - 1)
    return rows


def summarize_portfolio(
    equity: pd.Series,
    core_weight: float | None,
    label: str,
    avg_core_weight: float | None = None,
) -> dict:
    metrics = compute_equity_metrics(equity, PERIODS_PER_YEAR)
    monthly = compute_monthly_stats(equity)
    years = yearly_returns(equity)
    return {
        "label": label,
        "core_weight": core_weight,
        "avg_core_weight": avg_core_weight if avg_core_weight is not None else core_weight,
        "strategy_weight": 1 - core_weight if core_weight is not None else None,
        "return": metrics["return"],
        "max_drawdown": metrics["max_drawdown"],
        "cagr": metrics["cagr"],
        "calmar": metrics["calmar"],
        "sharpe": metrics["sharpe"],
        "monthly_win_rate": monthly["monthly_win_rate"],
        "worst_month": monthly["worst_month"],
        "worst_year": min(years.values()) if years else 0.0,
        **{f"return_{year}": value for year, value in years.items()},
    }


def run_fixed_core(df: pd.DataFrame, cfg: dict, weights: list[float]) -> pd.DataFrame:
    _, _, strategy_equity = replay(df, cfg, return_equity=True)
    buy_hold_equity = compute_buy_hold_equity(df, strategy_equity.iloc[0])

    rows = []
    for weight in weights:
        portfolio = blend_equity(strategy_equity, buy_hold_equity, weight)
        rows.append(summarize_portfolio(portfolio, weight, f"fixed_{weight:.2f}"))
    return pd.DataFrame(rows).sort_values(
        ["calmar", "return"],
        ascending=[False, False],
    )


def dynamic_blend_equity(
    strategy_equity: pd.Series,
    buy_hold_equity: pd.Series,
    core_weights: pd.Series,
) -> pd.Series:
    strategy_aligned, buy_hold_aligned = strategy_equity.align(
        buy_hold_equity,
        join="inner",
    )
    core_weights = core_weights.reindex(strategy_aligned.index, method="ffill")
    core_weights = core_weights.fillna(REGIME_CORE_WEIGHTS["neutral"])

    strategy_returns = strategy_aligned.pct_change().fillna(0)
    buy_hold_returns = buy_hold_aligned.pct_change().fillna(0)

    equity = [float(strategy_aligned.iloc[0])]
    for i in range(1, len(strategy_aligned)):
        weight = float(core_weights.iloc[i - 1])
        portfolio_return = (
            weight * buy_hold_returns.iloc[i]
            + (1 - weight) * strategy_returns.iloc[i]
        )
        equity.append(equity[-1] * (1 + portfolio_return))

    return pd.Series(equity, index=strategy_aligned.index, dtype=float)


def swing_unrealized_pnl(position: SwingPosition | None, price: float) -> float:
    if position is None:
        return 0.0
    if position.side == "long":
        return position.notional * (price / position.entry - 1)
    return position.notional * (position.entry / price - 1)


def portfolio_equity(state: PortfolioState, price: float) -> float:
    return state.cash + state.core_btc * price + swing_unrealized_pnl(state.swing, price)


def rebalance_core(
    state: PortfolioState,
    equity: float,
    price: float,
    target_weight: float,
    spot_fee: float,
    threshold: float,
    max_step_pct: float = 1.0,
    urgent: bool = False,
) -> None:
    if equity <= 0 or price <= 0:
        return
    target_notional = equity * target_weight
    current_notional = state.core_btc * price
    diff = target_notional - current_notional
    if abs(diff) / equity < threshold:
        return

    if diff > 0:
        buy_notional = min(diff, equity * max_step_pct, state.cash / (1 + spot_fee))
        if buy_notional <= 0:
            return
        fee = buy_notional * spot_fee
        state.cash -= buy_notional + fee
        state.core_btc += buy_notional / price
        state.core_buy_count += 1
    else:
        sell_notional = min(abs(diff), equity * max_step_pct, current_notional)
        if sell_notional <= 0:
            return
        fee = sell_notional * spot_fee
        state.cash += sell_notional - fee
        state.core_btc -= sell_notional / price
        state.core_sell_count += 1
    state.total_fee += fee
    state.rebalance_count += 1
    if urgent:
        state.urgent_rebalance_count += 1


def update_exposure_stats(state: PortfolioState, equity: float, price: float) -> None:
    if equity <= 0:
        return
    core_notional = state.core_btc * price
    swing_signed = 0.0
    if state.swing:
        swing_signed = state.swing.notional if state.swing.side == "long" else -state.swing.notional
    net = (core_notional + swing_signed) / equity
    gross = (abs(core_notional) + abs(swing_signed)) / equity
    state.max_net_exposure = max(state.max_net_exposure, net)
    state.max_gross_exposure = max(state.max_gross_exposure, gross)


def run_real_regime_portfolio(
    df: pd.DataFrame,
    cfg: dict,
    rebalance_threshold: float,
    bear_rebalance_threshold: float,
    urgent_drop_pct: float,
    core_buy_step: float,
    core_sell_step: float,
    urgent_sell_step: float,
    spot_fee: float,
    max_net_exposure: float,
    max_gross_exposure: float,
    max_short_exposure: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    portfolio_cfg = PortfolioConfig(
        enabled=True,
        mode="backtest",
        rebalance_threshold=rebalance_threshold,
        bear_rebalance_threshold=bear_rebalance_threshold,
        urgent_drop_pct=urgent_drop_pct,
        core_buy_step=core_buy_step,
        core_sell_step=core_sell_step,
        urgent_sell_step=urgent_sell_step,
        spot_fee=spot_fee,
        max_net_exposure=max_net_exposure,
        max_gross_exposure=max_gross_exposure,
        max_short_exposure=max_short_exposure,
    )
    strategy = cfg["strategy"]
    risk = cfg["risk"]
    indicators = calculate_indicators(df, strategy)
    regime = compute_daily_regime(df)
    core_weights = align_regime_core_weights(df, regime, portfolio_cfg.core_weights)
    regime_labels = align_regime_labels(df, regime)

    leverage = float(risk.get("leverage", 2))
    risk_per_trade = float(risk.get("risk_per_trade", 0.02))
    max_notional_pct = float(risk.get("max_notional_pct", 0.95))
    max_consecutive_losses = int(risk.get("max_consecutive_losses", 5))
    loss_pause_bars = int(risk.get("loss_pause_bars", 18))
    cooldown_base = int(strategy.get("cooldown_base", 6))
    cooldown_extra = int(strategy.get("cooldown_extra_per_loss", 3))

    state = PortfolioState(cash=float(risk["initial_capital"]))
    consecutive_losses = 0
    last_trade_bar = -999
    in_pause = False
    equity_curve = []
    trades = []
    last_rebalance_day = None

    for i, row in enumerate(df.itertuples(index=False)):
        timestamp = row.timestamp
        close = float(row.Close)
        equity = portfolio_equity(state, close)
        if equity <= 0:
            equity_curve.append(max(equity, 0.0))
            continue

        current_core_weight = state.core_btc * close / equity
        target_core_weight = float(core_weights.loc[timestamp])
        regime_label = str(regime_labels.loc[timestamp])
        open_price = float(row.Open)
        bar_return = close / open_price - 1 if open_price > 0 else 0.0
        plan = plan_core_rebalance(
            current_weight=current_core_weight,
            target_weight=target_core_weight,
            regime_label=regime_label,
            bar_return=bar_return,
            close=close,
            ema200=float(indicators["ema200"][i]),
            config=portfolio_cfg,
        )
        current_day = timestamp.floor("D")
        should_rebalance = current_day != last_rebalance_day or plan.urgent
        if should_rebalance and plan.weight_delta != 0:
            planned_target = current_core_weight + plan.weight_delta
            rebalance_core(
                state,
                equity=equity,
                price=close,
                target_weight=planned_target,
                spot_fee=spot_fee,
                threshold=0.0,
                max_step_pct=1.0,
                urgent=plan.urgent,
            )
            if current_day != last_rebalance_day:
                last_rebalance_day = current_day
            equity = portfolio_equity(state, close)
        elif current_day != last_rebalance_day:
            last_rebalance_day = current_day

        update_exposure_stats(state, equity, close)
        equity_curve.append(equity)

        if state.swing:
            exit_signal = check_exit(row, indicators, i, state.swing.as_exit_dict())
            if not exit_signal:
                continue

            reason, exit_price = exit_signal
            position = state.swing
            if position.side == "long":
                gross = position.notional * (exit_price / position.entry - 1)
            else:
                gross = position.notional * (position.entry / exit_price - 1)

            hours = (timestamp - position.entry_time).total_seconds() / 3600
            settlements = max(1, int(np.ceil(hours / FUNDING_INTERVAL_HOURS))) \
                if hours > 0 else 0
            funding = -position.notional * FUNDING_RATE * settlements \
                if position.side == "long" else \
                position.notional * FUNDING_RATE * settlements
            close_fee = position.notional * COMMISSION
            pnl = gross - close_fee + funding
            state.cash += pnl
            state.total_fee += close_fee
            state.total_funding += funding
            state.swing = None
            state.swing_trades += 1
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
            last_trade_bar = i

            trades.append({
                **position.as_exit_dict(),
                "exit_time": timestamp,
                "exit": exit_price,
                "reason": reason,
                "gross": gross,
                "open_fee": position.open_fee,
                "close_fee": close_fee,
                "fee": position.open_fee + close_fee,
                "funding": funding,
                "pnl": pnl,
                "equity": portfolio_equity(state, close),
            })
            continue

        pause = loss_pause_decision(
            consecutive_losses=consecutive_losses,
            max_consecutive_losses=max_consecutive_losses,
            current_bar=i,
            last_trade_bar=last_trade_bar,
            loss_pause_bars=loss_pause_bars,
        )
        if not pause.can_trade:
            if not in_pause:
                state.pause_events += 1
                in_pause = True
            continue
        if pause.reset_losses:
            consecutive_losses = 0
            in_pause = False

        bars_since = i - last_trade_bar
        cooldown = cooldown_base + consecutive_losses * cooldown_extra
        if bars_since < cooldown:
            continue

        signal = evaluate_signal(row, indicators, i, strategy)
        if not signal:
            continue

        regime_label = str(regime_labels.loc[timestamp])
        swing_limit = portfolio_cfg.swing_limits.get(
            regime_label,
            portfolio_cfg.swing_limits["neutral"],
        ).get(signal["side"], 0.0)
        if swing_limit <= 0:
            continue

        sizing = calculate_position_size(
            balance=equity,
            entry=signal["entry"],
            stop_loss=signal["sl"],
            risk_per_trade=risk_per_trade,
            leverage=leverage,
            max_notional_pct=max_notional_pct,
        )
        if not sizing.tradable:
            continue

        notional = cap_swing_notional(
            side=signal["side"],
            requested=sizing.notional,
            equity=equity,
            cash=state.cash,
            core_notional=state.core_btc * close,
            regime_limit=swing_limit,
            leverage=leverage,
            config=portfolio_cfg,
        )
        if notional <= 0:
            continue

        open_fee = notional * COMMISSION
        if open_fee >= state.cash:
            continue
        state.cash -= open_fee
        state.total_fee += open_fee
        if sizing.capped or notional < sizing.notional:
            state.capped_orders += 1
        state.swing = SwingPosition(
            side=signal["side"],
            entry=signal["entry"],
            sl=signal["sl"],
            tp=signal["tp"],
            notional=notional,
            raw_notional=sizing.raw_notional,
            max_notional=sizing.max_notional,
            size=notional / signal["entry"],
            stop_pct=sizing.stop_pct,
            capped=sizing.capped or notional < sizing.notional,
            open_fee=open_fee,
            entry_time=timestamp,
            entry_bar=i,
        )

    equity = pd.Series(equity_curve, index=df["timestamp"], dtype=float)
    row = summarize_portfolio(
        equity,
        core_weight=None,
        label="real_regime",
        avg_core_weight=float(core_weights.mean()),
    )
    regime_counts = regime.value_counts(normalize=True).to_dict()
    for regime_name in ["strong_bull", "bull", "neutral", "bear"]:
        row[f"regime_{regime_name}_pct"] = regime_counts.get(regime_name, 0.0)
    row.update({
        "rebalance_count": state.rebalance_count,
        "core_buy_count": state.core_buy_count,
        "core_sell_count": state.core_sell_count,
        "urgent_rebalance_count": state.urgent_rebalance_count,
        "swing_trades": state.swing_trades,
        "capped_orders": state.capped_orders,
        "pause_events": state.pause_events,
        "total_fee": state.total_fee,
        "total_funding": state.total_funding,
        "max_net_exposure": state.max_net_exposure,
        "max_gross_exposure": state.max_gross_exposure,
        "ending_cash": state.cash,
        "ending_core_btc": state.core_btc,
    })
    return pd.DataFrame([row]), equity, pd.DataFrame(trades)


def run_regime_core(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _, _, strategy_equity = replay(df, cfg, return_equity=True)
    buy_hold_equity = compute_buy_hold_equity(df, strategy_equity.iloc[0])
    regime = compute_daily_regime(df)
    core_weights = align_regime_core_weights(df, regime)
    portfolio = dynamic_blend_equity(strategy_equity, buy_hold_equity, core_weights)

    row = summarize_portfolio(
        portfolio,
        core_weight=None,
        label="regime_core",
        avg_core_weight=float(core_weights.mean()),
    )
    regime_counts = regime.value_counts(normalize=True).to_dict()
    for regime_name in ["strong_bull", "bull", "neutral", "bear"]:
        row[f"regime_{regime_name}_pct"] = regime_counts.get(regime_name, 0.0)
    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser(description="Backtest BTC core-satellite portfolios")
    parser.add_argument("--config", default="config/production.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--mode", choices=["fixed-core", "regime-core", "real-regime", "all"],
                        default="all")
    parser.add_argument("--weights", default="0,0.3,0.5,0.7,1.0")
    parser.add_argument("--output", default="outputs/portfolio_fixed_core.csv")
    parser.add_argument("--trades-output", default="outputs/portfolio_real_regime_trades.csv")
    parser.add_argument("--rebalance-threshold", type=float, default=0.05)
    parser.add_argument("--bear-rebalance-threshold", type=float, default=0.01)
    parser.add_argument("--urgent-drop-pct", type=float, default=0.04)
    parser.add_argument("--core-buy-step", type=float, default=0.50)
    parser.add_argument("--core-sell-step", type=float, default=0.50)
    parser.add_argument("--urgent-sell-step", type=float, default=1.00)
    parser.add_argument("--spot-fee", type=float, default=0.001)
    parser.add_argument("--max-net-exposure", type=float, default=0.90)
    parser.add_argument("--max-gross-exposure", type=float, default=1.20)
    parser.add_argument("--max-short-exposure", type=float, default=0.30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_local_ohlcv(args.data_dir, args.start, args.end)

    results = []
    if args.mode in ("fixed-core", "all"):
        results.append(run_fixed_core(df, cfg, parse_weights(args.weights)))
    if args.mode in ("regime-core", "all"):
        results.append(run_regime_core(df, cfg))
    if args.mode in ("real-regime", "all"):
        real_result, _, real_trades = run_real_regime_portfolio(
            df=df,
            cfg=cfg,
            rebalance_threshold=args.rebalance_threshold,
            bear_rebalance_threshold=args.bear_rebalance_threshold,
            urgent_drop_pct=args.urgent_drop_pct,
            core_buy_step=args.core_buy_step,
            core_sell_step=args.core_sell_step,
            urgent_sell_step=args.urgent_sell_step,
            spot_fee=args.spot_fee,
            max_net_exposure=args.max_net_exposure,
            max_gross_exposure=args.max_gross_exposure,
            max_short_exposure=args.max_short_exposure,
        )
        results.append(real_result)
        if not real_trades.empty:
            trades_out = Path(args.trades_output)
            trades_out.parent.mkdir(parents=True, exist_ok=True)
            real_trades.to_csv(trades_out, index=False)
    result = pd.concat(results, ignore_index=True)
    result = result.sort_values(["calmar", "return"], ascending=[False, False])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("BTC core-satellite portfolio backtest")
    print(f"  range: {args.start} -> {args.end}")
    print(f"  mode: {args.mode}")
    print(f"  saved: {out}")
    columns = [
        "label",
        "core_weight",
        "avg_core_weight",
        "strategy_weight",
        "return",
        "max_drawdown",
        "calmar",
        "sharpe",
        "monthly_win_rate",
        "worst_year",
        "rebalance_count",
        "core_buy_count",
        "core_sell_count",
        "urgent_rebalance_count",
        "swing_trades",
        "total_fee",
        "max_net_exposure",
        "max_gross_exposure",
    ]
    print(result.reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main()
