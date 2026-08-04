import unittest

import yaml

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


if __name__ == "__main__":
    unittest.main()
