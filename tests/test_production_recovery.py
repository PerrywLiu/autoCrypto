"""Regression coverage for polling recovery, restart timing and dashboard health."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ccxt
import numpy as np

from src.live.portfolio import PortfolioConfig
from src.live.production_bot import ProductionBot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
from web import dashboard


BAR = 14_400_000
BASE = 1_790_000_000_000 // BAR * BAR


def bot_at(path, last_ts=BASE, count=299):
    bot = ProductionBot("", "", "", state_file=str(path),
                        portfolio_config=PortfolioConfig(enabled=True, mode="paper"))
    bot._candles = [[last_ts - (count - i - 1) * BAR, 100, 101, 99, 100, 1]
                    for i in range(count)]
    bot._ohlcv = {key: np.array([row[i] for row in bot._candles], dtype=float)
                  for i, key in enumerate(("open", "high", "low", "close", "volume"), 1)}
    bot._update_indicators()
    return bot


class PollingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.bot = bot_at(Path(self.folder.name) / "state.json")

    def test_single_market_failure_does_not_latch_trading_pause(self):
        self.bot._record_api_failure("market_data_failed", ccxt.RequestTimeout("timeout"))
        self.assertIsNone(self.bot._paused_reason)

    def test_market_pause_is_cleared_by_a_valid_poll(self):
        for _ in range(3):
            self.bot._record_api_failure("market_data_failed", ccxt.RequestTimeout("timeout"))
        self.assertEqual(self.bot._paused_reason, "market_data_unavailable")
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=lambda *a, **k: self.bot._candles[-2:],
            milliseconds=lambda: BASE + BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        self.bot._fetch_market_candles()
        self.assertIsNone(self.bot._paused_reason)
        self.assertEqual(self.bot._market_error_count, 0)

    def test_market_failure_never_overwrites_a_hard_pause(self):
        self.bot._paused_reason = "protective_orders_missing"
        self.bot._record_api_failure("market_data_failed", ccxt.RequestTimeout("timeout"))
        self.assertEqual(self.bot._paused_reason, "protective_orders_missing")

    def test_market_recovery_does_not_clear_order_or_state_failures(self):
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=lambda *a, **k: self.bot._candles[-2:],
            milliseconds=lambda: BASE + BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        for reason in ("order_failed:timeout", "state_invalid:version", "protective_orders_missing"):
            self.bot._paused_reason = reason
            self.bot._fetch_market_candles()
            self.assertEqual(self.bot._paused_reason, reason)

    def test_empty_or_stale_poll_is_not_recovery(self):
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        self.bot._paused_reason = "market_data_unavailable"
        for rows in ([], [[BASE - 10 * BAR, 100, 101, 99, 100, 1]],
                     [[BASE, 100, 101, 99, float("nan"), 1]]):
            self.bot.exchange = SimpleNamespace(
                fetch_ohlcv=lambda *a, **k: rows,
                milliseconds=lambda: BASE + BAR + 1000,
                parse_timeframe=lambda tf: 14400,
            )
            self.assertEqual(self.bot._fetch_market_candles(), [])
            self.assertEqual(self.bot._paused_reason, "market_data_unavailable")

    def test_duplicate_candles_cannot_be_treated_as_a_recovered_poll(self):
        self.bot._paused_reason = "market_data_unavailable"
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=lambda *a, **k: self.bot._candles[-1:] * 2,
            milliseconds=lambda: BASE + BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        self.assertEqual(self.bot._fetch_market_candles(), [])
        self.assertEqual(self.bot._paused_reason, "market_data_unavailable")

    def test_paper_start_skips_authenticated_leverage_call(self):
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(args)
        self.bot.exchange = SimpleNamespace(set_leverage=forbidden)
        self.bot.set_leverage()
        self.assertEqual(calls, [])

    def test_main_loop_recovers_after_three_timeouts_and_updates_heartbeat(self):
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        initial = list(self.bot._candles)
        for i, row in enumerate(initial):
            price = 80 + i * 40 / 298
            row[1:5] = [price, price + 1, price - 1, price]
        reads = 0
        def fetch(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                return initial
            if reads <= 4:
                raise ccxt.RequestTimeout("public poll timeout")
            return initial[-1:] + [[BASE + BAR, 120, 121, 119, 120, 1]]
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=fetch, milliseconds=lambda: BASE + 2 * BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        sleeps = 0
        def sleep(_):
            nonlocal sleeps
            sleeps += 1
            if sleeps == 4:
                self.bot._running = False
        with patch("src.live.production_bot.time.sleep", side_effect=sleep):
            self.bot.start()
        self.assertEqual(reads, 5)
        self.assertIsNone(self.bot._paused_reason)
        self.assertEqual(self.bot._candles[-1][0], BASE + BAR)
        self.assertEqual(self.bot._position["side"], "long")
        health = json.loads(self.bot.state_file.with_name("production_health.json").read_text())
        self.assertEqual(health["market_error_count"], 0)
        self.assertEqual(health["last_closed_candle_ts"], BASE + BAR)
        self.assertEqual(health["signal"]["decision"], "opened")

    def test_exchange_recovery_requires_successful_position_reconciliation(self):
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        self.bot.dry_run = False
        self.bot._paused_reason = "market_data_unavailable"
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=lambda *a, **k: self.bot._candles[-2:],
            milliseconds=lambda: BASE + BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        from src.live.okx_account import build_account_snapshot
        snapshot = build_account_snapshot(
            balance={"USDT": {"free": 500, "total": 500}, "BTC": {"total": 0}},
            positions=[], price=100, symbol="BTC/USDT", open_orders=[])
        self.bot.account_reader = SimpleNamespace(snapshot=lambda price: snapshot)
        self.bot._fetch_market_candles()
        self.assertIsNone(self.bot._paused_reason)

    def test_exchange_market_recovery_preserves_reconciliation_failure(self):
        self.assertTrue(hasattr(self.bot, "_fetch_market_candles"))
        self.bot.dry_run = False
        self.bot._paused_reason = "market_data_unavailable"
        self.bot.exchange = SimpleNamespace(
            fetch_ohlcv=lambda *a, **k: self.bot._candles[-2:],
            milliseconds=lambda: BASE + BAR + 1000,
            parse_timeframe=lambda tf: 14400,
        )
        self.bot.account_reader = SimpleNamespace(snapshot=lambda price: (_ for _ in ()).throw(ccxt.RequestTimeout("account timeout")))
        self.bot._fetch_market_candles()
        self.assertEqual(self.bot._paused_reason, "account_snapshot_failed")


class RestartCooldownTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "state.json"

    def test_same_trade_time_has_same_cooldown_after_restart(self):
        original = bot_at(self.path, count=350)
        original._last_trade_bar = 344  # six completed bars since the close
        original._last_trade_candle_ts = BASE - 6 * BAR
        original._consecutive_losses = 1
        original._save_state()
        restored = bot_at(self.path, count=299)
        restored._load_state()
        self.assertTrue(hasattr(restored, "_cooldown_status"))
        self.assertEqual(restored._cooldown_status()["remaining_bars"], 3)
        restored._candles.append([BASE + 3 * BAR, 100, 101, 99, 100, 1])
        self.assertEqual(restored._cooldown_status()["remaining_bars"], 0)

    def test_future_trade_timestamp_rejects_state_instead_of_extending_cooldown(self):
        original = bot_at(self.path)
        state = original._state_payload()
        state["last_trade_candle_ts"] = int(time.time() * 1000) + 10 * BAR
        self.path.write_text(json.dumps(state))
        restored = bot_at(self.path)
        restored._load_state()
        self.assertEqual(restored._paused_reason, "state_invalid:cooldown_timestamp")

    def test_legacy_trade_log_recovers_elapsed_cooldown_without_using_array_index(self):
        original = bot_at(self.path)
        state = original._state_payload()
        state.pop("last_trade_candle_ts", None)
        state.update(last_trade_bar=305, consecutive_losses=1,
                     trade_log=[{"time": "2026-10-07 04:00:07.106057+00:00"}])
        self.path.write_text(json.dumps(state))
        latest = 1_791_590_400_000  # 2026-10-09 20:00 UTC
        restored = bot_at(self.path, last_ts=latest)
        restored._load_state()
        self.assertTrue(hasattr(restored, "_cooldown_status"))
        self.assertEqual(restored._cooldown_status()["remaining_bars"], 0)
        self.assertIsNone(restored._paused_reason)

    def test_unknown_legacy_trade_time_is_visible_and_fail_closed(self):
        original = bot_at(self.path)
        state = original._state_payload()
        state.pop("last_trade_candle_ts", None)
        state.update(last_trade_bar=305, consecutive_losses=1, trade_log=[])
        self.path.write_text(json.dumps(state))
        restored = bot_at(self.path)
        restored._load_state()
        self.assertEqual(restored._paused_reason, "state_invalid:cooldown_timestamp_missing")

    def test_loss_pause_health_matches_actual_release_after_eighteen_bars(self):
        bot = bot_at(self.path)
        self.assertTrue(hasattr(bot, "_cooldown_status"))
        bot._consecutive_losses = 5
        bot._last_trade_candle_ts = BASE - 17 * BAR
        self.assertEqual(bot._cooldown_status()["remaining_bars"], 1)
        bot._last_trade_candle_ts = BASE - 18 * BAR
        self.assertEqual(bot._cooldown_status()["remaining_bars"], 0)

    def test_actual_paper_close_persists_candle_time(self):
        bot = bot_at(self.path)
        bot._position = {"side": "long", "entry": 100, "notional": 100}
        with patch("src.live.production_bot.Path", side_effect=lambda value: (
                self.path.parent / "outputs" if value == "outputs" else Path(value))), \
                patch("builtins.open", side_effect=self._open_trade):
            bot._close_position("SL", 99)
        state = json.loads(self.path.read_text())
        self.assertEqual(state.get("last_trade_candle_ts"), BASE)
        restored = bot_at(self.path, last_ts=BASE + 3 * BAR)
        restored._load_state()
        self.assertEqual(restored._cooldown_status()["remaining_bars"], 6)

    def _open_trade(self, path, *args, **kwargs):
        import io
        if str(path) == "outputs/production_trades.json":
            return (self.path.parent / "trades.json").open(*args, **kwargs)
        return io.open(path, *args, **kwargs)


class HealthAndLedgerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "state.json"
        self.bot = bot_at(self.path)

    def test_heartbeat_is_written_without_a_trade_and_reports_hard_pause(self):
        self.bot._running = True
        self.bot._paused_reason = "order_failed:timeout"
        self.assertTrue(hasattr(self.bot, "_write_health"))
        self.bot._write_health()
        health = json.loads(self.path.with_name("production_health.json").read_text())
        self.assertEqual(health["paused_reason"], "order_failed:timeout")
        self.assertTrue(health["running"])
        self.assertEqual(health["last_closed_candle_ts"], BASE)
        self.assertIn("remaining_bars", health["cooldown"])
        self.assertFalse(self.path.exists())  # heartbeat does not invent a trade snapshot

    def test_core_rebalance_history_survives_restart_with_price_quantity_fee(self):
        self.bot._update_portfolio_plan()
        state = json.loads(self.path.read_text())
        self.assertIn("core_trade_log", state)
        fill = state["core_trade_log"][0]
        self.assertEqual(fill["side"], "buy")
        self.assertEqual(fill["price"], 100)
        self.assertAlmostEqual(fill["size"], 1.25)
        self.assertAlmostEqual(fill["fee"], 0.125)
        self.bot._save_state()
        restored = bot_at(self.path)
        restored._load_state()
        self.assertEqual(restored._core_trade_log, [fill])

    def test_dashboard_fresh_market_does_not_mask_a_stale_bot(self):
        outputs = self.path.parent
        (outputs / "production_state.json").write_text(json.dumps(
            {"dry_run": True, "updated_at": "2026-10-01T00:00:00+00:00"}))
        (outputs / "production_health.json").write_text(json.dumps(
            {"running": True, "updated_at": "2026-10-01T00:00:00+00:00"}))
        with patch.object(dashboard, "OUTPUTS", outputs), patch.object(dashboard, "market_ticker", return_value={
                "price": 100, "updated_at": int(time.time() * 1000), "history": []}):
            result = dashboard.dashboard_data()
        self.assertIn("bot_health", result)
        self.assertTrue(result["bot_health"]["stale"])
        self.assertEqual(result["bot_updated_at"], "2026-10-01T00:00:00+00:00")

    def test_dashboard_reads_core_fills_separately_from_current_holdings(self):
        (self.path.parent / "production_state.json").write_text(json.dumps({
            "dry_run": True, "paper_core_btc": 1,
            "core_trade_log": [{"time": "2026-10-10T00:00:00+00:00", "side": "sell",
                                "price": 100, "size": 0.5, "notional": 50, "fee": 0.05}],
        }))
        with patch.object(dashboard, "OUTPUTS", self.path.parent), patch.object(dashboard, "market_ticker", return_value={
                "price": 100, "updated_at": int(time.time() * 1000), "history": []}):
            result = dashboard.dashboard_data()
        fills = [row for row in result["activity"] if row["kind"] == "spot_trade"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["side"], "sell")
        self.assertEqual(fills[0]["fee"], 0.05)


if __name__ == "__main__":
    unittest.main()
