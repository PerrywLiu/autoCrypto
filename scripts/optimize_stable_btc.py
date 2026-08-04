#!/usr/bin/env python
"""Grid-search low-drawdown filters for the online BTC 4h strategy."""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.backtest_online import load_config, load_local_ohlcv, replay


def _parse_csv_numbers(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def score_result(summary: dict) -> float:
    buy_hold_return = summary["buy_hold"]
    return_ratio = summary["total_return"] / buy_hold_return \
        if buy_hold_return > 0 else 0.0
    years = max((pd.Timestamp(summary["end"]) - pd.Timestamp(summary["start"])).days / 365.25, 1)
    trades_per_year = summary["trades"] / years
    return (
        summary["strategy_calmar"]
        + 0.5 * max(0.0, return_ratio)
        - 0.2 * max(0.0, trades_per_year - 60) / 60
        - 0.2 * max(0.0, summary["fee_drag_pct"] - 0.15) / 0.15
    )


def run_grid(df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    rows = []
    grid = itertools.product(
        args.adx_thresholds,
        args.ema_slope_lookbacks,
        args.min_ema_spreads,
        args.min_atr_percentiles,
    )

    for adx_threshold, slope_lookback, min_spread, min_atr_pct in grid:
        overrides = {
            "adx_threshold": adx_threshold,
            "ema_slope_lookback": slope_lookback,
            "min_ema_spread": min_spread,
            "min_atr_percentile": min_atr_pct,
            "atr_percentile_window": args.atr_percentile_window,
        }
        summary, _ = replay(df, cfg, strategy_overrides=overrides)
        rows.append({
            **overrides,
            "score": score_result(summary),
            "return": summary["total_return"],
            "max_drawdown": summary["max_drawdown"],
            "calmar": summary["strategy_calmar"],
            "sharpe": summary["sharpe"],
            "trades": summary["trades"],
            "win_rate": summary["win_rate"],
            "monthly_win_rate": summary["strategy_monthly_win_rate"],
            "fee_drag_pct": summary["fee_drag_pct"],
            "time_in_market": summary["time_in_market"],
            "avg_exposure": summary["avg_exposure"],
            "capped_orders": summary["capped_orders"],
            "pause_events": summary["pause_events"],
            "buy_hold_return": summary["buy_hold"],
            "buy_hold_max_drawdown": summary["buy_hold_max_drawdown"],
            "buy_hold_calmar": summary["buy_hold_calmar"],
            "excess": summary["excess"],
        })

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["score", "calmar", "return"],
        ascending=[False, False, False],
    )


def main():
    parser = argparse.ArgumentParser(description="Optimize stable BTC filters")
    parser.add_argument("--config", default="config/production.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--output", default="outputs/stable_btc_optimization.csv")
    parser.add_argument("--adx-thresholds", default="20,22,25,28")
    parser.add_argument("--ema-slope-lookbacks", default="0,3,6,12")
    parser.add_argument("--min-ema-spreads", default="0,0.003,0.006,0.01")
    parser.add_argument("--min-atr-percentiles", default="0,20,30")
    parser.add_argument("--atr-percentile-window", type=int, default=180)
    args = parser.parse_args()

    args.adx_thresholds = _parse_csv_numbers(args.adx_thresholds, float)
    args.ema_slope_lookbacks = _parse_csv_numbers(args.ema_slope_lookbacks, int)
    args.min_ema_spreads = _parse_csv_numbers(args.min_ema_spreads, float)
    args.min_atr_percentiles = _parse_csv_numbers(args.min_atr_percentiles, float)

    cfg = load_config(args.config)
    df = load_local_ohlcv(args.data_dir, args.start, args.end)
    results = run_grid(df, cfg, args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    print("Stable BTC optimization")
    print(f"  candidates: {len(results)}")
    print(f"  saved: {out}")
    print("\nTop 10")
    columns = [
        "score",
        "adx_threshold",
        "ema_slope_lookback",
        "min_ema_spread",
        "min_atr_percentile",
        "return",
        "max_drawdown",
        "calmar",
        "trades",
        "fee_drag_pct",
        "time_in_market",
    ]
    print(results[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
