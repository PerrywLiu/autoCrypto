"""Shared BTC core-satellite portfolio controls for replay and live checks."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_CORE_WEIGHTS = {
    "strong_bull": 0.60,
    "bull": 0.45,
    "neutral": 0.25,
    "bear": 0.05,
}
DEFAULT_SWING_LIMITS = {
    "strong_bull": {"long": 0.30, "short": 0.00},
    "bull": {"long": 0.30, "short": 0.00},
    "neutral": {"long": 0.20, "short": 0.10},
    "bear": {"long": 0.10, "short": 0.30},
}


@dataclass(frozen=True)
class PortfolioConfig:
    enabled: bool = False
    mode: str = "paper"
    core_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CORE_WEIGHTS))
    swing_limits: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            name: dict(limits) for name, limits in DEFAULT_SWING_LIMITS.items()
        }
    )
    rebalance_threshold: float = 0.05
    bear_rebalance_threshold: float = 0.01
    urgent_drop_pct: float = 0.04
    core_buy_step: float = 0.50
    core_sell_step: float = 0.50
    urgent_sell_step: float = 1.00
    spot_fee: float = 0.001
    max_net_exposure: float = 0.90
    max_gross_exposure: float = 1.20
    max_short_exposure: float = 0.30


@dataclass(frozen=True)
class CoreRebalancePlan:
    weight_delta: float
    urgent: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PortfolioExposure:
    net: float
    gross: float
    core_weight: float
    swing_weight: float


def load_portfolio_config(cfg: dict | None) -> PortfolioConfig:
    raw = dict(cfg or {})
    defaults = PortfolioConfig()
    return PortfolioConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        mode=str(raw.get("mode", defaults.mode)),
        core_weights={**defaults.core_weights, **raw.get("core_weights", {})},
        swing_limits=_merge_swing_limits(defaults.swing_limits, raw.get("swing_limits", {})),
        rebalance_threshold=float(raw.get("rebalance_threshold", defaults.rebalance_threshold)),
        bear_rebalance_threshold=float(
            raw.get("bear_rebalance_threshold", defaults.bear_rebalance_threshold)
        ),
        urgent_drop_pct=float(raw.get("urgent_drop_pct", defaults.urgent_drop_pct)),
        core_buy_step=float(raw.get("core_buy_step", defaults.core_buy_step)),
        core_sell_step=float(raw.get("core_sell_step", defaults.core_sell_step)),
        urgent_sell_step=float(raw.get("urgent_sell_step", defaults.urgent_sell_step)),
        spot_fee=float(raw.get("spot_fee", defaults.spot_fee)),
        max_net_exposure=float(raw.get("max_net_exposure", defaults.max_net_exposure)),
        max_gross_exposure=float(raw.get("max_gross_exposure", defaults.max_gross_exposure)),
        max_short_exposure=float(raw.get("max_short_exposure", defaults.max_short_exposure)),
    )


def _merge_swing_limits(
    defaults: dict[str, dict[str, float]],
    override: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged = {name: dict(limits) for name, limits in defaults.items()}
    for regime, limits in override.items():
        merged.setdefault(regime, {})
        merged[regime].update(limits)
    return merged


def ema_series(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_daily_regime(df_4h: pd.DataFrame) -> pd.Series:
    daily_close = (
        df_4h.set_index("timestamp")["Close"]
        .resample("1D")
        .last()
        .dropna()
    )
    ema50 = ema_series(daily_close, 50)
    ema200 = ema_series(daily_close, 200)
    ema200_slope = ema200 > ema200.shift(20)

    regime = pd.Series("neutral", index=daily_close.index)
    bull = (daily_close > ema200) & (ema50 > ema200)
    bear = (daily_close < ema200) & (ema50 < ema200)
    strong_bull = bull & ema200_slope

    regime.loc[bull] = "bull"
    regime.loc[strong_bull] = "strong_bull"
    regime.loc[bear] = "bear"
    return regime.shift(1).fillna("neutral")


def align_regime_labels(df_4h: pd.DataFrame, regime: pd.Series) -> pd.Series:
    timestamps = pd.DatetimeIndex(df_4h["timestamp"])
    aligned = regime.reindex(timestamps, method="ffill").fillna("neutral")
    return pd.Series(aligned.to_numpy(dtype=str), index=timestamps)


def align_regime_core_weights(
    df_4h: pd.DataFrame,
    regime: pd.Series,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    weights = weights or DEFAULT_CORE_WEIGHTS
    timestamps = pd.DatetimeIndex(df_4h["timestamp"])
    aligned = regime.reindex(timestamps, method="ffill").fillna("neutral")
    core_weights = aligned.map(weights).fillna(weights["neutral"])
    return pd.Series(core_weights.to_numpy(dtype=float), index=timestamps)


def classify_latest_regime(close: np.ndarray) -> str:
    if len(close) < 221:
        return "neutral"
    series = pd.Series(close, dtype=float)
    ema50 = ema_series(series, 50)
    ema200 = ema_series(series, 200)
    latest_close = float(series.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    latest_ema200 = float(ema200.iloc[-1])
    if np.isnan(latest_ema50) or np.isnan(latest_ema200):
        return "neutral"
    if latest_close > latest_ema200 and latest_ema50 > latest_ema200:
        if latest_ema200 > float(ema200.iloc[-21]):
            return "strong_bull"
        return "bull"
    if latest_close < latest_ema200 and latest_ema50 < latest_ema200:
        return "bear"
    return "neutral"


def plan_core_rebalance(
    current_weight: float,
    target_weight: float,
    regime_label: str,
    bar_return: float,
    close: float,
    ema200: float,
    config: PortfolioConfig,
) -> CoreRebalancePlan:
    diff = target_weight - current_weight
    if abs(diff) <= 0:
        return CoreRebalancePlan(0.0)

    bearish_break = not np.isnan(ema200) and close < ema200
    sharp_drop = bar_return <= -abs(config.urgent_drop_pct)
    reducing = diff < 0
    urgent = reducing and (regime_label == "bear" or bearish_break or sharp_drop)
    threshold = config.bear_rebalance_threshold if reducing else config.rebalance_threshold
    if urgent:
        threshold = min(threshold, config.bear_rebalance_threshold)
    if abs(diff) < threshold:
        return CoreRebalancePlan(0.0)

    if diff > 0:
        step = min(abs(diff), config.core_buy_step)
        reason = "slow_add"
    elif urgent:
        step = min(abs(diff), config.urgent_sell_step)
        reason = "urgent_reduce"
    else:
        step = min(abs(diff), config.core_sell_step)
        reason = "normal_reduce"
    return CoreRebalancePlan(
        weight_delta=step if diff > 0 else -step,
        urgent=urgent,
        reason=reason,
    )


def cap_swing_notional(
    side: str,
    requested: float,
    equity: float,
    cash: float,
    core_notional: float,
    regime_limit: float,
    leverage: float,
    config: PortfolioConfig,
) -> float:
    if requested <= 0 or equity <= 0 or cash <= 0 or regime_limit <= 0:
        return 0.0

    caps = [
        requested,
        equity * regime_limit,
        cash * leverage,
        max(0.0, equity * config.max_gross_exposure - abs(core_notional)),
    ]
    if side == "long":
        caps.append(max(0.0, equity * config.max_net_exposure - core_notional))
    else:
        caps.append(equity * config.max_short_exposure)
        caps.append(max(0.0, core_notional + equity * config.max_short_exposure))
    return max(0.0, min(caps))


def portfolio_exposure(
    equity: float,
    price: float,
    core_btc: float,
    swing_side: str | None = None,
    swing_notional: float = 0.0,
) -> PortfolioExposure:
    if equity <= 0 or price <= 0:
        return PortfolioExposure(0.0, 0.0, 0.0, 0.0)
    core_notional = core_btc * price
    swing_signed = 0.0
    if swing_side == "long":
        swing_signed = swing_notional
    elif swing_side == "short":
        swing_signed = -swing_notional
    net = (core_notional + swing_signed) / equity
    gross = (abs(core_notional) + abs(swing_signed)) / equity
    return PortfolioExposure(
        net=net,
        gross=gross,
        core_weight=core_notional / equity,
        swing_weight=swing_signed / equity,
    )
