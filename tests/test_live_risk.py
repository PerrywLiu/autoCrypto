import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.backtest_online import (
    compute_buy_hold_equity,
    compute_equity_metrics,
    compute_monthly_stats,
)
from scripts import run_production
from scripts.backtest_portfolio import (
    PortfolioState,
    SwingPosition,
    blend_equity,
    dynamic_blend_equity,
    portfolio_equity,
    parse_weights,
    rebalance_core,
)
from src.live.alerts import LocalAlertManager
from src.live.okx_account import (
    build_account_snapshot,
    parse_asset_balance,
    parse_swap_position,
    reconcile_account_state,
)
from src.live.preflight import (
    needs_okx_account,
    validate_account_snapshot,
    validate_live_state_position,
    validate_preflight_config,
)
from src.live.order_executor import BracketOrderResult, OrderExecutor, OrderPlan, OrderResult
from src.live.order_executor import build_spot_core_rebalance_plan
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
        self.assertFalse(bot.portfolio_config.allow_spot_trading)
        self.assertFalse(bot.portfolio_config.allow_swap_trading)

    def test_testnet_live_example_config_is_safe_by_default(self):
        with open("config/testnet_live.yaml") as f:
            cfg = yaml.safe_load(f)

        portfolio = load_portfolio_config(cfg["portfolio"])

        self.assertFalse(cfg["mode"]["dry_run"])
        self.assertTrue(cfg["mode"]["testnet"])
        self.assertEqual(portfolio.mode, "testnet_live")
        self.assertFalse(portfolio.allow_spot_trading)
        self.assertFalse(portfolio.allow_swap_trading)
        self.assertLessEqual(portfolio.max_order_notional_usdt, 10)
        self.assertLessEqual(cfg["risk"]["max_notional_pct"], 0.50)

    def test_preflight_rejects_live_mode_without_hard_switch(self):
        cfg = {
            "mode": {"dry_run": False, "testnet": True},
            "portfolio": {
                "enabled": True,
                "mode": "testnet_live",
                "allow_spot_trading": False,
                "allow_swap_trading": False,
            },
        }
        checks = validate_preflight_config(
            cfg,
            {
                "OKX_API_KEY": "key",
                "OKX_SECRET": "secret",
                "OKX_PASSPHRASE": "passphrase",
            },
        )

        by_name = {check.name: check for check in checks}
        self.assertFalse(by_name["live_mode_switches"].ok)

    def test_preflight_paper_dry_run_does_not_need_okx_account(self):
        cfg = {
            "mode": {"dry_run": True, "testnet": True},
            "portfolio": {"enabled": True, "mode": "paper"},
        }

        self.assertFalse(needs_okx_account(cfg))

    def test_preflight_rejects_live_small_on_testnet(self):
        cfg = {
            "mode": {"dry_run": False, "testnet": True},
            "portfolio": {
                "enabled": True,
                "mode": "live_small",
                "allow_swap_trading": True,
            },
        }
        checks = validate_preflight_config(
            cfg,
            {
                "OKX_API_KEY": "key",
                "OKX_SECRET": "secret",
                "OKX_PASSPHRASE": "passphrase",
            },
        )

        by_name = {check.name: check for check in checks}
        self.assertFalse(by_name["real_mode_match"].ok)

    def test_preflight_account_snapshot_rejects_open_orders_and_positions(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[{
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 1,
                "notional": 100.0,
                "entryPrice": 10000.0,
            }],
            open_orders=[{"id": "open-1"}],
            price=10000.0,
        )

        checks = validate_account_snapshot(snapshot)
        by_name = {check.name: check for check in checks}

        self.assertFalse(by_name["no_unexpected_open_orders"].ok)
        self.assertFalse(by_name["no_unexpected_swap_position"].ok)

    def test_preflight_existing_position_requires_valid_state(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[{
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 1,
                "notional": 100.0,
                "entryPrice": 10000.0,
            }],
            open_orders=[],
            price=10000.0,
        )
        invalid_state = {
            "position": {
                "side": "long",
                "entry": 10000.0,
                "notional": 100.0,
            },
        }

        state_checks = validate_live_state_position(invalid_state, allow_empty=False)
        account_checks = validate_account_snapshot(
            snapshot,
            allow_existing_state=True,
            state=invalid_state,
        )

        self.assertFalse(state_checks[0].ok)
        by_name = {check.name: check for check in account_checks}
        self.assertFalse(by_name["no_unexpected_swap_position"].ok)

    def test_local_alert_manager_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.jsonl"
            manager = LocalAlertManager(path)

            event = manager.send("critical", "test_code", "测试告警", {"a": 1})

            rows = path.read_text().splitlines()
            payload = json.loads(rows[0])
            self.assertEqual(event.code, "test_code")
            self.assertEqual(payload["severity"], "critical")
        self.assertEqual(payload["context"]["a"], 1)

    def test_run_production_blocks_non_dry_run_when_preflight_fails(self):
        cfg = {
            "mode": {"dry_run": False, "testnet": True},
            "portfolio": {"enabled": True, "mode": "testnet_live"},
            "output": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "production.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            with patch("sys.argv", ["run_production.py", "--config", str(config_path)]), \
                 patch.object(run_production, "run_preflight", return_value=1) as preflight, \
                 patch.object(run_production, "run_bot") as run_bot:
                run_production.main()

            preflight.assert_called_once()
            run_bot.assert_not_called()

    def test_run_production_dry_run_skips_preflight(self):
        cfg = {
            "mode": {"dry_run": True, "testnet": True},
            "portfolio": {"enabled": True, "mode": "paper"},
            "output": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "production.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            with patch("sys.argv", ["run_production.py", "--config", str(config_path)]), \
                 patch.object(run_production, "run_preflight") as preflight, \
                 patch.object(run_production, "run_bot") as run_bot:
                run_production.main()

            preflight.assert_not_called()
            run_bot.assert_called_once_with(str(config_path))

    def test_run_production_skip_preflight_allows_non_dry_run(self):
        cfg = {
            "mode": {"dry_run": False, "testnet": True},
            "portfolio": {"enabled": True, "mode": "testnet_live"},
            "output": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "production.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            with patch(
                "sys.argv",
                ["run_production.py", "--config", str(config_path), "--skip-preflight"],
            ), \
                 patch.object(run_production, "run_preflight") as preflight, \
                 patch.object(run_production, "run_bot") as run_bot:
                run_production.main()

            preflight.assert_not_called()
            run_bot.assert_called_once_with(str(config_path))

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

    def test_portfolio_readonly_mode_blocks_real_orders(self):
        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="readonly"),
        )
        bot.dry_run = False

        allowed = bot.execute_signal({
            "side": "long",
            "entry": 100000.0,
            "sl": 98000.0,
            "tp": 106000.0,
        })

        self.assertFalse(allowed)

    def test_swap_trading_switch_blocks_live_orders(self):
        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
        )
        bot.dry_run = False

        allowed = bot.execute_signal({
            "side": "long",
            "entry": 100000.0,
            "sl": 98000.0,
            "tp": 106000.0,
        })

        self.assertFalse(allowed)
        self.assertEqual(bot._paused_reason, "swap_trading_disabled")

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

    def test_live_state_rejects_missing_protective_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text(json.dumps({
                "version": 1,
                "schema": "production_bot_state",
                "symbol": "BTC/USDT",
                "timeframe": "4h",
                "position": {
                    "side": "long",
                    "entry": 10000.0,
                    "notional": 200.0,
                    "size": 0.02,
                    "entry_order_id": "entry-1",
                },
            }))
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
                state_file=str(state_file),
            )
            bot.dry_run = False

            bot._load_state()

            self.assertIsNone(bot._position)
            self.assertEqual(
                bot._paused_reason,
                "state_invalid:live_position_missing_protective_orders",
            )

    def test_live_state_restores_when_schema_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text(json.dumps({
                "version": 1,
                "schema": "production_bot_state",
                "symbol": "BTC/USDT",
                "timeframe": "4h",
                "paper_cash": 100.0,
                "paper_core_btc": 0.0,
                "position": {
                    "side": "short",
                    "entry": 10000.0,
                    "notional": 200.0,
                    "size": 0.02,
                    "entry_order_id": "entry-1",
                    "protective_orders": {
                        "take_profit": "tp-1",
                        "stop_loss": "sl-1",
                    },
                },
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "last_trade_day": 0,
                "consecutive_losses": 0,
                "last_trade_bar": -1,
                "trade_log": [],
            }))
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
                state_file=str(state_file),
            )
            bot.dry_run = False

            bot._load_state()

            self.assertEqual(bot._position["side"], "short")
            self.assertEqual(bot._protective_order_ids(), {"tp-1", "sl-1"})
            self.assertIsNone(bot._paused_reason)

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

    def test_live_readonly_account_snapshot_uses_okx_reader(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 500.0, "used": 0.0, "total": 500.0},
                        "BTC": {"free": 0.01, "used": 0.0, "total": 0.01},
                    },
                    positions=[{
                        "symbol": "BTC/USDT:USDT",
                        "side": "short",
                        "contracts": 1,
                        "notional": 200.0,
                        "entryPrice": 10000.0,
                        "unrealizedPnl": 5.0,
                    }],
                    open_orders=[],
                    price=price,
                )

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="readonly"),
        )
        bot.dry_run = False
        bot.account_reader = Reader()

        snapshot = bot._account_snapshot(10000.0)

        self.assertEqual(snapshot["mode"], "live_readonly")
        self.assertAlmostEqual(snapshot["equity_usdt"], 605.0)
        self.assertAlmostEqual(snapshot["core_btc"], 0.01)
        self.assertAlmostEqual(snapshot["core_weight"], 100.0 / 605.0)
        self.assertAlmostEqual(snapshot["swing_weight"], -200.0 / 605.0)

    def test_reconcile_failure_pauses_real_orders(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 500.0, "used": 0.0, "total": 500.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[{
                        "symbol": "BTC/USDT:USDT",
                        "side": "long",
                        "contracts": 1,
                        "notional": 100.0,
                        "entryPrice": 10000.0,
                    }],
                    open_orders=[],
                    price=price,
                )

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(
                enabled=True,
                mode="testnet_live",
                allow_swap_trading=True,
            ),
        )
        bot.dry_run = False
        bot.account_reader = Reader()

        ok = bot._reconcile_live_state(10000.0)
        allowed = bot.execute_signal({
            "side": "long",
            "entry": 10000.0,
            "sl": 9800.0,
            "tp": 10600.0,
        })

        self.assertFalse(ok)
        self.assertEqual(bot._paused_reason, "exchange_position_without_local_state")
        self.assertFalse(allowed)

    def test_live_reconcile_accepts_known_protective_orders(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 900.0, "used": 100.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[{
                        "symbol": "BTC/USDT:USDT",
                        "side": "long",
                        "contracts": 1,
                        "notional": 200.0,
                        "entryPrice": 10000.0,
                    }],
                    open_orders=[
                        {"id": "tp-1"},
                        {"id": "sl-1"},
                    ],
                    price=price,
                )

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
        )
        bot.dry_run = False
        bot.account_reader = Reader()
        bot._position = {
            "side": "long",
            "entry": 10000.0,
            "notional": 200.0,
            "protective_orders": {
                "take_profit": "tp-1",
                "stop_loss": "sl-1",
            },
        }

        ok = bot._reconcile_live_state(10000.0)

        self.assertTrue(ok)
        self.assertIsNone(bot._paused_reason)

    def test_live_reconcile_pauses_when_protective_order_missing(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 900.0, "used": 100.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[{
                        "symbol": "BTC/USDT:USDT",
                        "side": "long",
                        "contracts": 1,
                        "notional": 200.0,
                        "entryPrice": 10000.0,
                    }],
                    open_orders=[{"id": "tp-1"}],
                    price=price,
                )

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
        )
        bot.dry_run = False
        bot.account_reader = Reader()
        bot._position = {
            "side": "long",
            "entry": 10000.0,
            "notional": 200.0,
            "protective_orders": {
                "take_profit": "tp-1",
                "stop_loss": "sl-1",
            },
        }

        ok = bot._reconcile_live_state(10000.0)

        self.assertFalse(ok)
        self.assertEqual(bot._paused_reason, "protective_orders_missing")

    def test_api_failures_pause_after_configured_limit(self):
        class Reader:
            def snapshot(self, price):
                raise RuntimeError("network down")

        with tempfile.TemporaryDirectory() as tmp:
            alert_path = Path(tmp) / "alerts.jsonl"
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
                max_api_errors=2,
                alert_manager=LocalAlertManager(alert_path),
            )
            bot.dry_run = False
            bot.account_reader = Reader()

            first = bot._reconcile_live_state(10000.0)
            second = bot._reconcile_live_state(10000.0)

            self.assertFalse(first)
            self.assertFalse(second)
            self.assertEqual(bot._api_error_count, 2)
            self.assertEqual(bot._paused_reason, "api_errors_exceeded")
            payload = json.loads(alert_path.read_text().splitlines()[0])
            self.assertEqual(payload["code"], "api_errors_exceeded")

    def test_live_order_path_uses_order_executor(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[],
                    open_orders=[],
                    price=price,
                )

        class Executor:
            def __init__(self):
                self.calls = []

            def submit_bracket(self, entry, take_profit, stop_loss):
                self.calls = [entry, take_profit, stop_loss]
                return BracketOrderResult(
                    ok=True,
                    entry=OrderResult(
                        ok=True,
                        order={
                            "id": "entry-1",
                            "average": 10001.0,
                            "fee": {"cost": 0.08, "currency": "USDT"},
                        },
                        status="closed",
                        filled=0.02,
                        fee_cost=0.08,
                        fee_currency="USDT",
                    ),
                    take_profit=OrderResult(
                        ok=True,
                        order={"id": "tp-1"},
                        status="open",
                    ),
                    stop_loss=OrderResult(
                        ok=True,
                        order={"id": "sl-1"},
                        status="open",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                portfolio_config=PortfolioConfig(
                    enabled=True,
                    mode="testnet_live",
                    allow_swap_trading=True,
                ),
                state_file=str(state_file),
            )
            bot.dry_run = False
            bot.account_reader = Reader()
            bot.order_executor = Executor()
            bot._ohlcv["close"] = pd.Series([10000.0] * 230).to_numpy()
            bot.exchange.amount_to_precision = lambda symbol, amount: amount
            bot.exchange.price_to_precision = lambda symbol, price: price
            bot.get_balance = lambda: 1000.0
            bot.get_position = lambda: None

            allowed = bot.execute_signal({
                "side": "long",
                "entry": 10000.0,
                "sl": 9800.0,
                "tp": 10600.0,
            })

            self.assertTrue(allowed)
            self.assertEqual(len(bot.order_executor.calls), 3)
            self.assertEqual(bot.order_executor.calls[0].purpose, "entry")
            self.assertEqual(bot.order_executor.calls[1].purpose, "take_profit")
            self.assertEqual(bot.order_executor.calls[2].purpose, "stop_loss")
            self.assertEqual(bot.order_executor.calls[0].params["tdMode"], "isolated")
            self.assertEqual(bot._position["entry_order_id"], "entry-1")
            self.assertEqual(bot._position["protective_orders"]["take_profit"], "tp-1")
            self.assertEqual(bot._position["protective_orders"]["stop_loss"], "sl-1")
            self.assertAlmostEqual(bot._position["entry"], 10001.0)
            self.assertAlmostEqual(bot._position["entry_fee"]["cost"], 0.08)
            self.assertEqual(bot._position["entry_order"]["raw"]["id"], "entry-1")
            self.assertTrue(state_file.exists())

    def test_live_order_failure_pauses_bot(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[],
                    open_orders=[],
                    price=price,
                )

        class Executor:
            def submit_bracket(self, entry, take_profit, stop_loss):
                return type("Result", (), {"ok": False, "reason": "entry_failed"})()

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(
                enabled=True,
                mode="testnet_live",
                allow_swap_trading=True,
            ),
        )
        bot.dry_run = False
        bot.account_reader = Reader()
        bot.order_executor = Executor()
        bot._ohlcv["close"] = pd.Series([10000.0] * 230).to_numpy()
        bot.exchange.amount_to_precision = lambda symbol, amount: amount
        bot.exchange.price_to_precision = lambda symbol, price: price
        bot.get_balance = lambda: 1000.0
        bot.get_position = lambda: None

        allowed = bot.execute_signal({
            "side": "long",
            "entry": 10000.0,
            "sl": 9800.0,
            "tp": 10600.0,
        })

        self.assertFalse(allowed)
        self.assertEqual(bot._paused_reason, "order_failed:entry_failed")

    def test_live_close_submits_reduce_only_and_clears_state(self):
        class Reader:
            def __init__(self):
                self.calls = 0

            def snapshot(self, price):
                self.calls += 1
                positions = []
                if self.calls == 1:
                    positions = [{
                        "symbol": "BTC/USDT:USDT",
                        "side": "long",
                        "contracts": 0.02,
                        "notional": 200.0,
                        "entryPrice": 10000.0,
                    }]
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=positions,
                    open_orders=[],
                    price=price,
                )

        class Executor:
            def __init__(self):
                self.submitted = []
                self.cancelled = []

            def submit(self, plan):
                self.submitted.append(plan)
                return OrderResult(ok=True, order={"id": "close-1", "status": "open"})

            def wait_for_fill(self, order, symbol):
                return OrderResult(
                    ok=True,
                    order={
                        "id": order["id"],
                        "average": 10100.0,
                        "fee": {"cost": 0.06, "currency": "USDT"},
                    },
                    status="closed",
                    filled=0.02,
                    fee_cost=0.06,
                    fee_currency="USDT",
                )

            def cancel_order(self, order_id, symbol):
                self.cancelled.append((order_id, symbol))
                return OrderResult(ok=True, order={"id": order_id, "status": "canceled"})

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            bot = ProductionBot(
                api_key="",
                secret="",
                passphrase="",
                initial_capital=1000,
                portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
                state_file=str(state_file),
            )
            bot.dry_run = False
            bot.account_reader = Reader()
            bot.order_executor = Executor()
            bot.exchange.amount_to_precision = lambda symbol, amount: amount
            bot._position = {
                "side": "long",
                "entry": 10000.0,
                "tp": 10600.0,
                "sl": 9800.0,
                "size": 0.02,
                "notional": 200.0,
                "entry_order_id": "entry-1",
                "entry_fee": {"cost": 0.08, "currency": "USDT"},
                "protective_orders": {
                    "take_profit": "tp-1",
                    "stop_loss": "sl-1",
                },
            }

            bot._close_position("趋势反转", 10080.0)

            self.assertIsNone(bot._position)
            self.assertIsNone(bot._paused_reason)
            self.assertEqual(bot.order_executor.submitted[0].purpose, "close")
            self.assertEqual(bot.order_executor.submitted[0].side, "sell")
            self.assertTrue(bot.order_executor.submitted[0].params["reduceOnly"])
            self.assertEqual(
                set(order_id for order_id, _ in bot.order_executor.cancelled),
                {"tp-1", "sl-1"},
            )
            self.assertEqual(bot._trade_log[-1]["close_order_id"], "close-1")
            self.assertAlmostEqual(bot._trade_log[-1]["exit"], 10100.0)
            self.assertAlmostEqual(bot._trade_log[-1]["close_fee"]["cost"], 0.06)
            self.assertAlmostEqual(bot._trade_log[-1]["total_fee"], 0.14)
            self.assertTrue(state_file.exists())

    def test_live_close_pauses_when_position_not_flat(self):
        class Reader:
            def snapshot(self, price):
                return build_account_snapshot(
                    balance={
                        "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                        "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
                    },
                    positions=[{
                        "symbol": "BTC/USDT:USDT",
                        "side": "short",
                        "contracts": 0.02,
                        "notional": 200.0,
                        "entryPrice": 10000.0,
                    }],
                    open_orders=[],
                    price=price,
                )

        class Executor:
            def submit(self, plan):
                return OrderResult(ok=True, order={"id": "close-1", "status": "open"})

            def wait_for_fill(self, order, symbol):
                return OrderResult(ok=True, order=order, status="closed", filled=0.02)

            def cancel_order(self, order_id, symbol):
                return OrderResult(ok=True, order={"id": order_id, "status": "canceled"})

        bot = ProductionBot(
            api_key="",
            secret="",
            passphrase="",
            portfolio_config=PortfolioConfig(enabled=True, mode="testnet_live"),
        )
        bot.dry_run = False
        bot.account_reader = Reader()
        bot.order_executor = Executor()
        bot.exchange.amount_to_precision = lambda symbol, amount: amount
        bot._position = {
            "side": "short",
            "entry": 10000.0,
            "tp": 9400.0,
            "sl": 10200.0,
            "size": 0.02,
            "notional": 200.0,
            "protective_orders": {
                "take_profit": "tp-1",
                "stop_loss": "sl-1",
            },
        }

        bot._close_position("趋势反转", 9900.0)

        self.assertIsNotNone(bot._position)
        self.assertEqual(bot._paused_reason, "close_position_not_flat")


class OKXAccountSnapshotTests(unittest.TestCase):
    def test_parse_asset_balance(self):
        balance = {
            "USDT": {"free": "10.5", "used": "2", "total": "12.5"},
        }

        parsed = parse_asset_balance(balance, "USDT")

        self.assertEqual(parsed.currency, "USDT")
        self.assertAlmostEqual(parsed.free, 10.5)
        self.assertAlmostEqual(parsed.used, 2.0)
        self.assertAlmostEqual(parsed.total, 12.5)

    def test_parse_swap_position_from_ccxt_shape(self):
        position = parse_swap_position({
            "symbol": "BTC/USDT:USDT",
            "side": "short",
            "contracts": "2",
            "notional": "300",
            "entryPrice": "10000",
            "unrealizedPnl": "-3.5",
            "initialMargin": "150",
        }, "BTC/USDT:USDT")

        self.assertEqual(position.side, "short")
        self.assertAlmostEqual(position.contracts, 2.0)
        self.assertAlmostEqual(position.notional, 300.0)
        self.assertAlmostEqual(position.entry_price, 10000.0)
        self.assertAlmostEqual(position.unrealized_pnl, -3.5)

    def test_build_account_snapshot_computes_exposure(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 900.0, "used": 0.0, "total": 900.0},
                "BTC": {"free": 0.02, "used": 0.0, "total": 0.02},
            },
            positions=[{
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "contracts": 1,
                "notional": 300.0,
                "entryPrice": 10000.0,
                "unrealizedPnl": -10.0,
            }],
            open_orders=[{"id": "1"}],
            price=10000.0,
        )

        self.assertAlmostEqual(snapshot.equity_usdt, 1090.0)
        self.assertAlmostEqual(snapshot.core_notional, 200.0)
        self.assertAlmostEqual(snapshot.swing_notional, 300.0)
        self.assertAlmostEqual(snapshot.net_exposure, -100.0 / 1090.0)
        self.assertAlmostEqual(snapshot.gross_exposure, 500.0 / 1090.0)
        self.assertEqual(len(snapshot.open_orders), 1)

    def test_reconcile_rejects_open_orders(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[],
            open_orders=[{"id": "order-1"}],
            price=10000.0,
        )

        result = reconcile_account_state(snapshot, None)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "open_orders_present")

    def test_reconcile_rejects_exchange_position_without_local_state(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[{
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 1,
                "notional": 100.0,
                "entryPrice": 10000.0,
            }],
            open_orders=[],
            price=10000.0,
        )

        result = reconcile_account_state(snapshot, None)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "exchange_position_without_local_state")

    def test_reconcile_rejects_local_position_without_exchange_position(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[],
            open_orders=[],
            price=10000.0,
        )

        result = reconcile_account_state(snapshot, {"side": "long", "notional": 100.0})

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "local_position_without_exchange_position")

    def test_reconcile_accepts_matching_position(self):
        snapshot = build_account_snapshot(
            balance={
                "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"free": 0.0, "used": 0.0, "total": 0.0},
            },
            positions=[{
                "symbol": "BTC/USDT:USDT",
                "side": "short",
                "contracts": 1,
                "notional": 105.0,
                "entryPrice": 10000.0,
            }],
            open_orders=[],
            price=10000.0,
        )

        result = reconcile_account_state(snapshot, {"side": "short", "notional": 100.0})

        self.assertTrue(result.ok)


class OrderExecutorTests(unittest.TestCase):
    def test_spot_core_plan_buys_when_below_target(self):
        plan = build_spot_core_rebalance_plan(
            symbol="BTC/USDT",
            price=10000.0,
            equity=1000.0,
            current_core_btc=0.01,
            target_weight=0.30,
            threshold=0.05,
            max_notional=50.0,
            min_notional=5.0,
        )

        self.assertEqual(plan.side, "buy")
        self.assertEqual(plan.order_type, "market")
        self.assertAlmostEqual(plan.amount, 50.0)
        self.assertEqual(plan.params["quoteOrderQty"], 50.0)

    def test_spot_core_plan_sells_when_above_target(self):
        plan = build_spot_core_rebalance_plan(
            symbol="BTC/USDT",
            price=10000.0,
            equity=1000.0,
            current_core_btc=0.05,
            target_weight=0.30,
            threshold=0.05,
            max_notional=100.0,
            min_notional=5.0,
        )

        self.assertEqual(plan.side, "sell")
        self.assertAlmostEqual(plan.params["notional"], 100.0)
        self.assertAlmostEqual(plan.amount, 0.01)

    def test_spot_core_plan_holds_below_threshold(self):
        plan = build_spot_core_rebalance_plan(
            symbol="BTC/USDT",
            price=10000.0,
            equity=1000.0,
            current_core_btc=0.029,
            target_weight=0.30,
            threshold=0.05,
            max_notional=100.0,
            min_notional=5.0,
        )

        self.assertIsNone(plan)

    def test_spot_core_plan_holds_below_min_notional(self):
        plan = build_spot_core_rebalance_plan(
            symbol="BTC/USDT",
            price=10000.0,
            equity=1000.0,
            current_core_btc=0.029,
            target_weight=0.36,
            threshold=0.01,
            max_notional=100.0,
            min_notional=100.0,
        )

        self.assertIsNone(plan)

    def test_submit_sends_order_plan_to_exchange(self):
        class Exchange:
            def __init__(self):
                self.calls = []

            def create_order(self, **kwargs):
                self.calls.append(kwargs)
                return {"id": "order-1"}

        exchange = Exchange()
        executor = OrderExecutor(exchange)

        result = executor.submit(OrderPlan(
            symbol="BTC/USDT:USDT",
            order_type="limit",
            side="buy",
            amount=1,
            price=10000,
            params={"tdMode": "isolated"},
        ))

        self.assertTrue(result.ok)
        self.assertEqual(result.order["id"], "order-1")
        self.assertEqual(exchange.calls[0]["symbol"], "BTC/USDT:USDT")
        self.assertEqual(exchange.calls[0]["type"], "limit")

    def test_submit_reports_exchange_failure(self):
        class Exchange:
            def create_order(self, **kwargs):
                raise RuntimeError("rejected")

        result = OrderExecutor(Exchange()).submit(OrderPlan(
            symbol="BTC/USDT:USDT",
            order_type="limit",
            side="buy",
            amount=1,
            price=10000,
        ))

        self.assertFalse(result.ok)
        self.assertIn("rejected", result.reason)

    def test_order_result_parses_fee(self):
        class Exchange:
            def create_order(self, **kwargs):
                return {
                    "id": "order-1",
                    "status": "closed",
                    "filled": "1",
                    "remaining": "0",
                    "fee": {"cost": "0.12", "currency": "USDT"},
                }

        result = OrderExecutor(Exchange()).submit(OrderPlan(
            symbol="BTC/USDT:USDT",
            order_type="market",
            side="buy",
            amount=1,
        ))

        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.fee_cost, 0.12)
        self.assertEqual(result.fee_currency, "USDT")

    def test_submit_bracket_stops_after_take_profit_failure(self):
        class Exchange:
            def __init__(self):
                self.calls = 0

            def create_order(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("tp failed")
                return {"id": str(self.calls)}

            def fetch_order(self, order_id, symbol):
                return {"id": order_id, "status": "closed", "filled": "1", "remaining": "0"}

        exchange = Exchange()
        plan = OrderPlan("BTC/USDT:USDT", "limit", "buy", 1, 10000)

        result = OrderExecutor(exchange).submit_bracket(plan, plan, plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "take_profit_failed")
        self.assertEqual(exchange.calls, 2)

    def test_wait_for_fill_returns_closed_order(self):
        class Exchange:
            def fetch_order(self, order_id, symbol):
                return {
                    "id": order_id,
                    "status": "closed",
                    "filled": "1",
                    "remaining": "0",
                }

        result = OrderExecutor(Exchange(), poll_interval=0, fill_timeout=0).wait_for_fill(
            {"id": "1"},
            "BTC/USDT:USDT",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "closed")
        self.assertAlmostEqual(result.filled, 1.0)

    def test_wait_for_fill_cancels_on_timeout(self):
        class Exchange:
            def __init__(self):
                self.cancelled = False

            def fetch_order(self, order_id, symbol):
                return {
                    "id": order_id,
                    "status": "open",
                    "filled": "0.4",
                    "remaining": "0.6",
                }

            def cancel_order(self, order_id, symbol):
                self.cancelled = True
                return {"id": order_id, "status": "canceled", "filled": "0.4", "remaining": "0.6"}

        exchange = Exchange()
        result = OrderExecutor(exchange, poll_interval=0, fill_timeout=0).wait_for_fill(
            {"id": "1"},
            "BTC/USDT:USDT",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "timeout_cancelled")
        self.assertTrue(exchange.cancelled)
        self.assertAlmostEqual(result.filled, 0.4)

    def test_submit_bracket_waits_for_entry_before_protection_orders(self):
        class Exchange:
            def __init__(self):
                self.created = []

            def create_order(self, **kwargs):
                self.created.append(kwargs)
                return {"id": str(len(self.created)), "status": "open"}

            def fetch_order(self, order_id, symbol):
                if order_id == "1":
                    return {"id": order_id, "status": "closed", "filled": "1", "remaining": "0"}
                return {"id": order_id, "status": "open", "filled": "0", "remaining": "1"}

        exchange = Exchange()
        plan = OrderPlan("BTC/USDT:USDT", "limit", "buy", 1, 10000)
        result = OrderExecutor(exchange, poll_interval=0, fill_timeout=0).submit_bracket(
            plan,
            plan,
            plan,
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(exchange.created), 3)
        self.assertEqual(result.entry.status, "closed")
        self.assertEqual(result.take_profit.status, "open")
        self.assertEqual(result.stop_loss.status, "open")

    def test_submit_bracket_fails_when_take_profit_unconfirmed(self):
        class Exchange:
            def __init__(self):
                self.created = 0

            def create_order(self, **kwargs):
                self.created += 1
                return {"id": str(self.created), "status": "open"}

            def fetch_order(self, order_id, symbol):
                if order_id == "1":
                    return {"id": order_id, "status": "closed", "filled": "1", "remaining": "0"}
                return {"id": order_id, "status": "rejected"}

        plan = OrderPlan("BTC/USDT:USDT", "limit", "buy", 1, 10000)
        result = OrderExecutor(Exchange(), poll_interval=0, fill_timeout=0).submit_bracket(
            plan,
            plan,
            plan,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "take_profit_unconfirmed")

    def test_submit_bracket_fails_when_stop_loss_unconfirmed(self):
        class Exchange:
            def __init__(self):
                self.created = 0

            def create_order(self, **kwargs):
                self.created += 1
                return {"id": str(self.created), "status": "open"}

            def fetch_order(self, order_id, symbol):
                if order_id == "1":
                    return {"id": order_id, "status": "closed", "filled": "1", "remaining": "0"}
                if order_id == "2":
                    return {"id": order_id, "status": "open", "filled": "0", "remaining": "1"}
                return {"id": order_id, "status": "canceled"}

        plan = OrderPlan("BTC/USDT:USDT", "limit", "buy", 1, 10000)
        result = OrderExecutor(Exchange(), poll_interval=0, fill_timeout=0).submit_bracket(
            plan,
            plan,
            plan,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "stop_loss_unconfirmed")

    def test_submit_bracket_stops_when_entry_times_out(self):
        class Exchange:
            def __init__(self):
                self.created = 0
                self.cancelled = False

            def create_order(self, **kwargs):
                self.created += 1
                return {"id": "entry", "status": "open"}

            def fetch_order(self, order_id, symbol):
                return {"id": order_id, "status": "open", "filled": "0", "remaining": "1"}

            def cancel_order(self, order_id, symbol):
                self.cancelled = True
                return {"id": order_id, "status": "canceled"}

        exchange = Exchange()
        plan = OrderPlan("BTC/USDT:USDT", "limit", "buy", 1, 10000)
        result = OrderExecutor(exchange, poll_interval=0, fill_timeout=0).submit_bracket(
            plan,
            plan,
            plan,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "entry_not_filled")
        self.assertEqual(exchange.created, 1)
        self.assertTrue(exchange.cancelled)


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
