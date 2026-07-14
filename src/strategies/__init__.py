# 策略层：趋势交易策略实现
from src.strategies.base import BaseStrategy
from src.strategies.ema_rsi_atr import EmaRsiAtrStrategy
from src.strategies.donchian_breakout import DonchianBreakoutStrategy
from src.strategies.multi_timeframe_trend import MultiTimeframeTrendStrategy
from src.strategies.advanced_trend import AdvancedTrendStrategy
from src.strategies.swing_trend import SwingTrendStrategy
from src.strategies.mean_revert import MeanRevertStrategy
from src.strategies.adaptive_hybrid import AdaptiveHybridStrategy

# 策略注册表
STRATEGY_REGISTRY = {
    "ema_rsi_atr": EmaRsiAtrStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "multi_timeframe_trend": MultiTimeframeTrendStrategy,
    "advanced_trend": AdvancedTrendStrategy,
    "swing_trend": SwingTrendStrategy,
    "mean_revert": MeanRevertStrategy,
    "adaptive_hybrid": AdaptiveHybridStrategy,
}

def get_strategy(name: str):
    """根据名称获取策略类"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]
