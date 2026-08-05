import unittest
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from scripts.backtest_online import (
    compute_buy_hold_equity,
    compute_equity_metrics,
    compute_monthly_stats,
)
from scripts.backtest_portfolio import (
    PortfolioState,
    SwingPosition,
    blend_equity,
    dynamic_blend_equity,
    portfolio_equity,
    parse_weights,
    rebalance_core,
)
from src.live.portfolio import (
    PortfolioConfig,
    align_regime_core_weights,
    cap_swing_notional,
    compute_daily_regime,
    load_portfolio_config,
    plan_core_rebalance,
    portfolio_exposure,
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
            "portfolio_config": load_portfolio_config(cfg["portfolio"]),
            "state_file": cfg["output"]["state_file"],
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
        self.assertTrue(bot.portfolio_config.enabled)
        self.assertEqual(bot.portfolio_config.mode, cfg["portfolio"]["mode"])
        self.assertAlmostEqual(
            bot.portfolio_config.core_weights["strong_bull"],
            cfg["portfolio"]["core_weights"]["strong_bull"],
        )

    def test_portfolio_paper_mode_blocks_real_orders(self):
        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="paper"),
        )
        bot.dry_run = False

        allowed = bot.execute_signal({
            "side": "long",
            "entry": 100000.0,
            "sl": 98000.0,
            "tp": 106000.0,
        })

        self.assertFalse(allowed)

    def test_paper_portfolio_state_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                initial_capital=500,
                portfolio_config=PortfolioConfig(enabled=True, mode="paper"),
                state_file=str(state_file),
            )
            bot._paper_cash = 320.0
            bot._paper_core_btc = 0.002
            bot._position = {
                "side": "long",
                "entry": 100000.0,
                "notional": 100.0,
            }
            bot._total_pnl = 12.5
            bot._consecutive_losses = 2
            bot._save_state()

            restored = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                initial_capital=500,
                portfolio_config=PortfolioConfig(enabled=True, mode="paper"),
                state_file=str(state_file),
            )
            restored._load_state()

            self.assertAlmostEqual(restored._paper_cash, 320.0)
            self.assertAlmostEqual(restored._paper_core_btc, 0.002)
            self.assertEqual(restored._position["side"], "long")
            self.assertAlmostEqual(restored._total_pnl, 12.5)
            self.assertEqual(restored._consecutive_losses, 2)

    def test_existing_paper_position_blocks_new_signal(self):
        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            initial_capital=500,
            portfolio_config=PortfolioConfig(enabled=True, mode="paper"),
        )
        bot._position = {
            "side": "long",
            "entry": 100000.0,
            "notional": 100.0,
        }

        allowed = bot.execute_signal({
            "side": "long",
            "entry": 100000.0,
            "sl": 98000.0,
            "tp": 106000.0,
        })

        self.assertFalse(allowed)

    def test_account_snapshot_reports_paper_exposure(self):
        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            initial_capital=500,
            portfolio_config=PortfolioConfig(enabled=True, mode="paper"),
        )
        bot._paper_cash = 300.0
        bot._paper_core_btc = 0.01
        bot._position = {
            "side": "short",
            "entry": 10000.0,
            "notional": 100.0,
        }

        snapshot = bot._account_snapshot(10000.0)

        self.assertAlmostEqual(snapshot["equity"], 400.0)
        self.assertAlmostEqual(snapshot["core_weight"], 0.25)
        self.assertAlmostEqual(snapshot["swing_weight"], -0.25)
        self.assertAlmostEqual(snapshot["net_exposure"], 0.0)
        self.assertAlmostEqual(snapshot["gross_exposure"], 0.5)


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


class PortfolioBacktestTests(unittest.TestCase):
    def test_blend_equity_combines_core_and_strategy_weights(self):
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="utc")
        strategy = pd.Series([100.0, 110.0, 120.0], index=index)
        buy_hold = pd.Series([100.0, 130.0, 160.0], index=index)

        blended = blend_equity(strategy, buy_hold, core_weight=0.5)

        self.assertAlmostEqual(blended.iloc[0], 100.0)
        self.assertAlmostEqual(blended.iloc[1], 120.0)
        self.assertAlmostEqual(blended.iloc[2], 140.0)

    def test_parse_weights_validates_range(self):
        self.assertEqual(parse_weights("0,0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            parse_weights("1.2")

    def test_dynamic_blend_uses_previous_bar_weights(self):
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="utc")
        strategy = pd.Series([100.0, 100.0, 100.0], index=index)
        buy_hold = pd.Series([100.0, 110.0, 121.0], index=index)
        core_weights = pd.Series([0.0, 1.0, 1.0], index=index)

        blended = dynamic_blend_equity(strategy, buy_hold, core_weights)

        self.assertAlmostEqual(blended.iloc[0], 100.0)
        self.assertAlmostEqual(blended.iloc[1], 100.0)
        self.assertAlmostEqual(blended.iloc[2], 110.0)

    def test_daily_regime_is_shifted_before_aligning_to_4h(self):
        timestamps = pd.date_range("2024-01-01", periods=240 * 6, freq="4h", tz="utc")
        close = [100.0] * (220 * 6) + [200.0] * (20 * 6)
        df = pd.DataFrame({"timestamp": timestamps, "Close": close})

        regime = compute_daily_regime(df)
        weights = align_regime_core_weights(df, regime)

        self.assertEqual(len(weights), len(df))
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())

    def test_portfolio_equity_counts_cash_core_and_swing_pnl(self):
        state = PortfolioState(
            cash=500.0,
            core_btc=0.01,
            swing=SwingPosition(
                side="long",
                entry=10000.0,
                sl=9500.0,
                tp=11000.0,
                notional=200.0,
                raw_notional=200.0,
                max_notional=500.0,
                size=0.02,
                stop_pct=0.05,
                capped=False,
                open_fee=0.08,
                entry_time=pd.Timestamp("2024-01-01", tz="utc"),
                entry_bar=0,
            ),
        )

        self.assertAlmostEqual(portfolio_equity(state, 11000.0), 630.0)

    def test_rebalance_core_buys_with_cash_and_fee(self):
        state = PortfolioState(cash=1000.0)

        rebalance_core(
            state,
            equity=1000.0,
            price=10000.0,
            target_weight=0.5,
            spot_fee=0.001,
            threshold=0.01,
        )

        self.assertAlmostEqual(state.core_btc, 0.05)
        self.assertAlmostEqual(state.cash, 499.5)
        self.assertAlmostEqual(state.total_fee, 0.5)
        self.assertEqual(state.rebalance_count, 1)

    def test_core_rebalance_plan_slow_adds_in_bull_regime(self):
        plan = plan_core_rebalance(
            current_weight=0.05,
            target_weight=0.45,
            regime_label="bull",
            bar_return=0.01,
            close=110.0,
            ema200=100.0,
            config=PortfolioConfig(core_buy_step=0.10),
        )

        self.assertAlmostEqual(plan.weight_delta, 0.10)
        self.assertFalse(plan.urgent)
        self.assertEqual(plan.reason, "slow_add")

    def test_core_rebalance_plan_urgently_reduces_on_trend_break(self):
        plan = plan_core_rebalance(
            current_weight=0.60,
            target_weight=0.25,
            regime_label="neutral",
            bar_return=-0.02,
            close=95.0,
            ema200=100.0,
            config=PortfolioConfig(
                core_buy_step=0.10,
                core_sell_step=0.30,
                urgent_sell_step=1.0,
            ),
        )

        self.assertAlmostEqual(plan.weight_delta, -0.35)
        self.assertTrue(plan.urgent)
        self.assertEqual(plan.reason, "urgent_reduce")

    def test_cap_swing_notional_respects_regime_and_exposure_limits(self):
        capped = cap_swing_notional(
            side="long",
            requested=1000.0,
            equity=1000.0,
            cash=1000.0,
            core_notional=600.0,
            regime_limit=0.30,
            leverage=2.0,
            config=PortfolioConfig(),
        )

        self.assertAlmostEqual(capped, 300.0)

    def test_portfolio_exposure_counts_core_and_short_swing(self):
        exposure = portfolio_exposure(
            equity=1000.0,
            price=10000.0,
            core_btc=0.05,
            swing_side="short",
            swing_notional=200.0,
        )

        self.assertAlmostEqual(exposure.core_weight, 0.5)
        self.assertAlmostEqual(exposure.swing_weight, -0.2)
        self.assertAlmostEqual(exposure.net, 0.3)
        self.assertAlmostEqual(exposure.gross, 0.7)


if __name__ == "__main__":
    unittest.main()
