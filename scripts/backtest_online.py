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
    return {
        "ema10": ema(close, strategy.get("ema_fast", 10)),
        "ema20": ema(close, strategy.get("ema_mid", 20)),
        "ema50": ema(close, strategy.get("ema_slow", 50)),
        "ema200": ema(close, strategy.get("ema_trend", 200)),
        "adx": adx(high, low, close, strategy.get("adx_period", 14)),
        "atr": atr(high, low, close, strategy.get("atr_period", 14)),
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
    if not (strong and near_ema20):
        return None

    sl_mult = strategy.get("sl_atr_mult", 1.5)
    tp_mult = strategy.get("tp_atr_mult", 3.0)
    if bull and bull_align:
        return {
            "side": "long",
            "entry": close,
            "sl": close - ind["atr"][i] * sl_mult,
            "tp": close + ind["atr"][i] * tp_mult,
        }
    if (not bull) and bear_align:
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


def replay(df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    strategy = cfg["strategy"]
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

    for i, row in df.iterrows():
        close = float(row.Close)
        equity = unrealized_equity(balance, position, close)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity - peak) / peak if peak > 0 else 0.0)
        equity_curve.append(equity)

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
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6) \
        if len(returns) > 1 and returns.std() > 0 else 0.0
    buy_hold = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1

    summary = {
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "initial": initial,
        "final_equity": final_equity,
        "total_return": final_equity / initial - 1,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": len(trades_df),
        "win_rate": wins / len(trades_df) if len(trades_df) else 0.0,
        "capped_orders": capped_orders,
        "pause_events": pause_events,
        "total_fee": trades_df["fee"].sum() if not trades_df.empty else 0.0,
        "total_funding": trades_df["funding"].sum() if not trades_df.empty else 0.0,
        "buy_hold": buy_hold,
        "excess": final_equity / initial - 1 - buy_hold,
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
    print(f"  sharpe: {summary['sharpe']:.2f}")
    print(f"  trades: {summary['trades']}")
    print(f"  win_rate: {summary['win_rate']:.2%}")
    print(f"  capped_orders: {summary['capped_orders']}")
    print(f"  pause_events: {summary['pause_events']}")
    print(f"  fee: {summary['total_fee']:.2f}")
    print(f"  funding: {summary['total_funding']:.2f}")
    print(f"  buy_hold: {summary['buy_hold']:.2%}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
