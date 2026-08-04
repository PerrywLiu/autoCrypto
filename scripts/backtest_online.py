#!/usr/bin/env python
"""Replay the production bot logic on local BTC/USDT 4h history."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.live.production_bot import adx, atr, ema
from src.live.risk import calculate_position_size, loss_pause_decision


COMMISSION = 0.0004
FUNDING_RATE = 0.0001
FUNDING_INTERVAL_HOURS = 8


def load_local_ohlcv(data_dir: str, since: str, until: str) -> pd.DataFrame:
    base = Path(data_dir) / "swap" / "BTC_USDT" / "4h"
    files = sorted(base.rglob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no local 4h data under {base}")

    df = pd.concat(
        [pd.read_csv(f, compression="gzip", parse_dates=["timestamp"]) for f in files],
        ignore_index=True,
    )
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    start = pd.Timestamp(since, tz="utc")
    end = pd.Timestamp(until, tz="utc")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if df.empty:
        raise ValueError(f"no data in range {since} -> {until}")
    return df.reset_index(drop=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def calculate_indicators(df: pd.DataFrame, strategy: dict) -> dict[str, np.ndarray]:
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    atr_values = atr(high, low, close, strategy.get("atr_period", 14))
    atr_pct = np.divide(
        atr_values,
        close,
        out=np.full(len(close), np.nan, dtype=float),
        where=close != 0,
    )
    min_atr_percentile = float(strategy.get("min_atr_percentile", 0) or 0)
    atr_percentile_threshold = np.full(len(close), np.nan, dtype=float)
    if min_atr_percentile > 0:
        window = int(strategy.get("atr_percentile_window", 180) or 180)
        atr_pct_series = pd.Series(atr_pct)
        atr_percentile_threshold = atr_pct_series.rolling(window).quantile(
            min_atr_percentile / 100
        ).shift(1).to_numpy(dtype=float)
    return {
        "ema10": ema(close, strategy.get("ema_fast", 10)),
        "ema20": ema(close, strategy.get("ema_mid", 20)),
        "ema50": ema(close, strategy.get("ema_slow", 50)),
        "ema200": ema(close, strategy.get("ema_trend", 200)),
        "adx": adx(high, low, close, strategy.get("adx_period", 14)),
        "atr": atr_values,
        "atr_pct": atr_pct,
        "atr_percentile_threshold": atr_percentile_threshold,
    }


def evaluate_signal(row, ind: dict[str, np.ndarray], i: int, strategy: dict) -> dict | None:
    close = float(row.Close)
    values = [
        close,
        ind["ema10"][i],
        ind["ema20"][i],
        ind["ema50"][i],
        ind["ema200"][i],
        ind["adx"][i],
        ind["atr"][i],
    ]
    if close <= 0 or any(np.isnan(x) for x in values):
        return None

    bull = close > ind["ema200"][i]
    bull_align = ind["ema10"][i] > ind["ema20"][i] > ind["ema50"][i]
    bear_align = ind["ema10"][i] < ind["ema20"][i] < ind["ema50"][i]
    near_ema20 = abs(close - ind["ema20"][i]) < ind["atr"][i]
    strong = ind["adx"][i] > strategy.get("adx_threshold", 20)
    ema_slope_lookback = int(strategy.get("ema_slope_lookback", 0) or 0)
    if ema_slope_lookback > 0:
        if i < ema_slope_lookback or np.isnan(ind["ema20"][i - ema_slope_lookback]):
            return None
        bull_slope = ind["ema20"][i] > ind["ema20"][i - ema_slope_lookback]
        bear_slope = ind["ema20"][i] < ind["ema20"][i - ema_slope_lookback]
    else:
        bull_slope = True
        bear_slope = True

    min_ema_spread = float(strategy.get("min_ema_spread", 0.0) or 0.0)
    ema_spread = abs(ind["ema10"][i] - ind["ema50"][i]) / close
    spread_ok = ema_spread >= min_ema_spread

    min_atr_percentile = float(strategy.get("min_atr_percentile", 0) or 0)
    atr_ok = True
    if min_atr_percentile > 0:
        threshold = ind["atr_percentile_threshold"][i]
        if np.isnan(threshold):
            return None
        atr_ok = ind["atr_pct"][i] >= threshold

    if not (strong and near_ema20 and spread_ok and atr_ok):
        return None

    sl_mult = strategy.get("sl_atr_mult", 1.5)
    tp_mult = strategy.get("tp_atr_mult", 3.0)
    if bull and bull_align and bull_slope:
        return {
            "side": "long",
            "entry": close,
            "sl": close - ind["atr"][i] * sl_mult,
            "tp": close + ind["atr"][i] * tp_mult,
        }
    if (not bull) and bear_align and bear_slope:
        return {
            "side": "short",
            "entry": close,
            "sl": close + ind["atr"][i] * sl_mult,
            "tp": close - ind["atr"][i] * tp_mult,
        }
    return None


def check_exit(row, ind: dict[str, np.ndarray], i: int, position: dict) -> tuple[str, float] | None:
    close = float(row.Close)
    high = float(row.High)
    low = float(row.Low)

    if position["side"] == "long":
        if high >= position["tp"]:
            return "TP", position["tp"]
        if low <= position["sl"]:
            return "SL", position["sl"]
        if not (ind["ema10"][i] > ind["ema20"][i] > ind["ema50"][i]):
            return "trend", close
    else:
        if low <= position["tp"]:
            return "TP", position["tp"]
        if high >= position["sl"]:
            return "SL", position["sl"]
        if not (ind["ema10"][i] < ind["ema20"][i] < ind["ema50"][i]):
            return "trend", close
    return None


def unrealized_equity(balance: float, position: dict | None, close: float) -> float:
    if not position:
        return balance
    if position["side"] == "long":
        return balance + position["notional"] * (close / position["entry"] - 1)
    return balance + position["notional"] * (position["entry"] / close - 1)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def _cagr(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _sharpe(equity: pd.Series, periods_per_year: int) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 2 or returns.std() <= 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_equity_metrics(equity: pd.Series, periods_per_year: int) -> dict:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) \
        if len(equity) > 1 and equity.iloc[0] > 0 else 0.0
    max_drawdown = _max_drawdown(equity)
    cagr = _cagr(equity, periods_per_year)
    return {
        "return": total_return,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "sharpe": _sharpe(equity, periods_per_year),
    }


def compute_monthly_stats(equity: pd.Series) -> dict:
    if not isinstance(equity.index, pd.DatetimeIndex) or equity.empty:
        return {"monthly_win_rate": 0.0, "worst_month": 0.0}
    monthly = equity.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()
    if monthly_returns.empty:
        return {"monthly_win_rate": 0.0, "worst_month": 0.0}
    return {
        "monthly_win_rate": float((monthly_returns > 0).mean()),
        "worst_month": float(monthly_returns.min()),
    }


def compute_buy_hold_equity(df: pd.DataFrame, initial: float) -> pd.Series:
    equity = initial * df["Close"] / df["Close"].iloc[0]
    return pd.Series(equity.to_numpy(), index=df["timestamp"], dtype=float)


def replay(
    df: pd.DataFrame,
    cfg: dict,
    strategy_overrides: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    strategy = {**cfg["strategy"], **(strategy_overrides or {})}
    risk = cfg["risk"]
    indicators = calculate_indicators(df, strategy)

    balance = float(risk["initial_capital"])
    initial = balance
    leverage = float(risk.get("leverage", 2))
    risk_per_trade = float(risk.get("risk_per_trade", 0.02))
    max_notional_pct = float(risk.get("max_notional_pct", 0.95))
    max_consecutive_losses = int(risk.get("max_consecutive_losses", 5))
    loss_pause_bars = int(risk.get("loss_pause_bars", 18))
    cooldown_base = int(strategy.get("cooldown_base", 6))
    cooldown_extra = int(strategy.get("cooldown_extra_per_loss", 3))

    position = None
    trades = []
    consecutive_losses = 0
    last_trade_bar = -999
    peak = initial
    max_drawdown = 0.0
    capped_orders = 0
    pause_events = 0
    in_pause = False
    equity_curve = []
    in_market_bars = 0
    exposure_values = []

    for i, row in enumerate(df.itertuples(index=False)):
        close = float(row.Close)
        equity = unrealized_equity(balance, position, close)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity - peak) / peak if peak > 0 else 0.0)
        equity_curve.append(equity)
        if position:
            in_market_bars += 1
            exposure_values.append(position["notional"] / equity if equity > 0 else 0.0)

        if position:
            exit_signal = check_exit(row, indicators, i, position)
            if not exit_signal:
                continue

            reason, exit_price = exit_signal
            if position["side"] == "long":
                gross = position["notional"] * (exit_price / position["entry"] - 1)
            else:
                gross = position["notional"] * (position["entry"] / exit_price - 1)

            hours = (row.timestamp - position["entry_time"]).total_seconds() / 3600
            settlements = max(1, int(np.ceil(hours / FUNDING_INTERVAL_HOURS))) \
                if hours > 0 else 0
            funding = -position["notional"] * FUNDING_RATE * settlements \
                if position["side"] == "long" else \
                position["notional"] * FUNDING_RATE * settlements
            fee = position["notional"] * COMMISSION * 2
            pnl = gross - fee + funding
            balance += pnl
            consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
            last_trade_bar = i

            trades.append({
                **position,
                "exit_time": row.timestamp,
                "exit": exit_price,
                "reason": reason,
                "gross": gross,
                "fee": fee,
                "funding": funding,
                "pnl": pnl,
                "balance": balance,
            })
            position = None
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
                pause_events += 1
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

        sizing = calculate_position_size(
            balance=balance,
            entry=signal["entry"],
            stop_loss=signal["sl"],
            risk_per_trade=risk_per_trade,
            leverage=leverage,
            max_notional_pct=max_notional_pct,
        )
        if not sizing.tradable:
            continue
        if sizing.capped:
            capped_orders += 1

        position = {
            **signal,
            "entry_time": row.timestamp,
            "entry_bar": i,
            "notional": sizing.notional,
            "raw_notional": sizing.raw_notional,
            "max_notional": sizing.max_notional,
            "size": sizing.size,
            "stop_pct": sizing.stop_pct,
            "capped": sizing.capped,
        }

    final_equity = equity_curve[-1] if equity_curve else balance
    trades_df = pd.DataFrame(trades)
    wins = int((trades_df["pnl"] > 0).sum()) if not trades_df.empty else 0
    strategy_equity = pd.Series(equity_curve, index=df["timestamp"], dtype=float)
    buy_hold_equity = compute_buy_hold_equity(df, initial)
    strategy_metrics = compute_equity_metrics(strategy_equity, 365 * 6)
    buy_hold_metrics = compute_equity_metrics(buy_hold_equity, 365 * 6)
    strategy_monthly = compute_monthly_stats(strategy_equity)
    buy_hold_monthly = compute_monthly_stats(buy_hold_equity)

    gross_pnl = trades_df["gross"].sum() if not trades_df.empty else 0.0
    total_fee = trades_df["fee"].sum() if not trades_df.empty else 0.0
    total_funding = trades_df["funding"].sum() if not trades_df.empty else 0.0
    avg_notional = trades_df["notional"].mean() if not trades_df.empty else 0.0

    summary = {
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "initial": initial,
        "final_equity": final_equity,
        "total_return": strategy_metrics["return"],
        "max_drawdown": strategy_metrics["max_drawdown"],
        "strategy_cagr": strategy_metrics["cagr"],
        "strategy_calmar": strategy_metrics["calmar"],
        "sharpe": strategy_metrics["sharpe"],
        "trades": len(trades_df),
        "win_rate": wins / len(trades_df) if len(trades_df) else 0.0,
        "capped_orders": capped_orders,
        "pause_events": pause_events,
        "total_fee": total_fee,
        "total_funding": total_funding,
        "gross_pnl": gross_pnl,
        "fee_drag_pct": total_fee / abs(gross_pnl) if gross_pnl != 0 else 0.0,
        "time_in_market": in_market_bars / len(df) if len(df) else 0.0,
        "avg_notional": avg_notional,
        "avg_exposure": float(np.mean(exposure_values)) if exposure_values else 0.0,
        "strategy_monthly_win_rate": strategy_monthly["monthly_win_rate"],
        "strategy_worst_month": strategy_monthly["worst_month"],
        "buy_hold": buy_hold_metrics["return"],
        "buy_hold_max_drawdown": buy_hold_metrics["max_drawdown"],
        "buy_hold_cagr": buy_hold_metrics["cagr"],
        "buy_hold_calmar": buy_hold_metrics["calmar"],
        "buy_hold_sharpe": buy_hold_metrics["sharpe"],
        "buy_hold_monthly_win_rate": buy_hold_monthly["monthly_win_rate"],
        "buy_hold_worst_month": buy_hold_monthly["worst_month"],
        "excess": strategy_metrics["return"] - buy_hold_metrics["return"],
    }
    return summary, trades_df


def main():
    parser = argparse.ArgumentParser(description="Replay online production logic locally")
    parser.add_argument("--config", default="config/production.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--output", default="outputs/online_4h_replay.csv")
    parser.add_argument("--trades-output", default="outputs/online_4h_trades.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_local_ohlcv(args.data_dir, args.start, args.end)
    summary, trades = replay(df, cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out, index=False)
    if not trades.empty:
        trades.to_csv(args.trades_output, index=False)

    print("Online 4h replay")
    print(f"  range: {summary['start']} -> {summary['end']}")
    print(f"  return: {summary['total_return']:.2%}")
    print(f"  max_drawdown: {summary['max_drawdown']:.2%}")
    print(f"  calmar: {summary['strategy_calmar']:.2f}")
    print(f"  sharpe: {summary['sharpe']:.2f}")
    print(f"  trades: {summary['trades']}")
    print(f"  win_rate: {summary['win_rate']:.2%}")
    print(f"  monthly_win_rate: {summary['strategy_monthly_win_rate']:.2%}")
    print(f"  time_in_market: {summary['time_in_market']:.2%}")
    print(f"  capped_orders: {summary['capped_orders']}")
    print(f"  pause_events: {summary['pause_events']}")
    print(f"  fee: {summary['total_fee']:.2f}")
    print(f"  funding: {summary['total_funding']:.2f}")
    print(f"  buy_hold: {summary['buy_hold']:.2%}")
    print(f"  buy_hold_max_drawdown: {summary['buy_hold_max_drawdown']:.2%}")
    print(f"  buy_hold_calmar: {summary['buy_hold_calmar']:.2f}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
