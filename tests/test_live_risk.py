import unittest

import pandas as pd
import yaml

from scripts.backtest_online import (
    compute_buy_hold_equity,
    compute_equity_metrics,
    compute_monthly_stats,
)
from src.live.production_bot import ProductionBot, _normalize_strategy_config
from src.live.risk import calculate_position_size, loss_pause_decision


class PositionSizeTests(unittest.TestCase):
    def test_position_size_uses_account_risk_without_leverage_multiplier(self):
        result = calculate_position_size(
            balance=1000,
            entry=100000,
            stop_loss=98000,
            risk_per_trade=0.02,
            leverage=2,
        )

        self.assertTrue(result.tradable)
        self.assertAlmostEqual(result.stop_pct, 0.02)
        self.assertAlmostEqual(result.risk_amount, 20)
        self.assertAlmostEqual(result.raw_notional, 1000)
        self.assertAlmostEqual(result.notional, 1000)
        self.assertAlmostEqual(result.size, 0.01)
        self.assertFalse(result.capped)

    def test_position_size_caps_notional_at_leverage_limit(self):
        result = calculate_position_size(
            balance=1000,
            entry=100000,
            stop_loss=99500,
            risk_per_trade=0.02,
            leverage=2,
        )

        self.assertTrue(result.tradable)
        self.assertAlmostEqual(result.raw_notional, 4000)
        self.assertAlmostEqual(result.max_notional, 1900)
        self.assertAlmostEqual(result.notional, 1900)
        self.assertTrue(result.capped)

    def test_position_size_rejects_invalid_inputs(self):
        result = calculate_position_size(
            balance=0,
            entry=100000,
            stop_loss=98000,
            risk_per_trade=0.02,
            leverage=2,
        )

        self.assertFalse(result.tradable)
        self.assertEqual(result.reason, "balance_non_positive")


class LossPauseTests(unittest.TestCase):
    def test_below_threshold_can_trade(self):
        decision = loss_pause_decision(
            consecutive_losses=4,
            max_consecutive_losses=5,
            current_bar=100,
            last_trade_bar=95,
            loss_pause_bars=18,
        )

        self.assertTrue(decision.can_trade)
        self.assertFalse(decision.reset_losses)

    def test_threshold_blocks_until_pause_elapsed(self):
        decision = loss_pause_decision(
            consecutive_losses=5,
            max_consecutive_losses=5,
            current_bar=100,
            last_trade_bar=90,
            loss_pause_bars=18,
        )

        self.assertFalse(decision.can_trade)
        self.assertEqual(decision.remaining_bars, 8)
        self.assertEqual(decision.reason, "loss_pause")

    def test_threshold_resets_after_pause_elapsed(self):
        decision = loss_pause_decision(
            consecutive_losses=5,
            max_consecutive_losses=5,
            current_bar=108,
            last_trade_bar=90,
            loss_pause_bars=18,
        )

        self.assertTrue(decision.can_trade)
        self.assertTrue(decision.reset_losses)


class ProductionConfigTests(unittest.TestCase):
    def test_production_strategy_config_matches_bot_constructor(self):
        with open("config/production.yaml") as f:
            cfg = yaml.safe_load(f)

        strategy_cfg = _normalize_strategy_config(cfg["strategy"])
        kwargs = {
            "api_key": "",
            "secret": "",
            "passphrase": "",
            "symbol": cfg["symbol"],
            "timeframe": cfg["timeframe"],
            "testnet": cfg["mode"]["testnet"],
            "leverage": cfg["risk"]["leverage"],
            "initial_capital": cfg["risk"]["initial_capital"],
            "risk_per_trade": cfg["risk"]["risk_per_trade"],
            "max_daily_loss_pct": cfg["risk"]["max_daily_loss_pct"],
            "max_consecutive_losses": cfg["risk"]["max_consecutive_losses"],
            "loss_pause_bars": cfg["risk"]["loss_pause_bars"],
            "max_notional_pct": cfg["risk"]["max_notional_pct"],
            **strategy_cfg,
        }

        bot = ProductionBot(**kwargs)

        self.assertEqual(bot.p_sl_atr, cfg["strategy"]["sl_atr_mult"])
        self.assertEqual(bot.p_tp_atr, cfg["strategy"]["tp_atr_mult"])
        self.assertEqual(bot.p_cooldown, cfg["strategy"]["cooldown_base"])
        self.assertEqual(
            bot.p_cooldown_extra_per_loss,
            cfg["strategy"]["cooldown_extra_per_loss"],
        )
        self.assertEqual(
            bot.p_ema_slope_lookback,
            cfg["strategy"]["ema_slope_lookback"],
        )
        self.assertEqual(bot.p_min_ema_spread, cfg["strategy"]["min_ema_spread"])
        self.assertEqual(
            bot.p_min_atr_percentile,
            cfg["strategy"]["min_atr_percentile"],
        )


class BacktestOnlineMetricTests(unittest.TestCase):
    def test_buy_hold_equity_and_metrics_include_drawdown(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="4h", tz="utc"),
            "Close": [100.0, 120.0, 90.0, 150.0],
        })

        equity = compute_buy_hold_equity(df, initial=1000)
        metrics = compute_equity_metrics(equity, periods_per_year=365 * 6)

        self.assertAlmostEqual(equity.iloc[-1], 1500)
        self.assertAlmostEqual(metrics["return"], 0.5)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.25)
        self.assertGreater(metrics["calmar"], 0)

    def test_monthly_stats_report_win_rate_and_worst_month(self):
        equity = pd.Series(
            [100.0, 110.0, 99.0, 120.0],
            index=pd.to_datetime([
                "2024-01-01",
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
            ], utc=True),
        )

        stats = compute_monthly_stats(equity)

        self.assertAlmostEqual(stats["monthly_win_rate"], 0.5)
        self.assertAlmostEqual(stats["worst_month"], -0.1)


if __name__ == "__main__":
    unittest.main()
