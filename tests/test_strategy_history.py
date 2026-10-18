import json
import copy
from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_production_recovery import bot_at, BASE, BAR
from web import dashboard


class StrategyHistoryTests(unittest.TestCase):
    def test_paper_open_survives_restart_and_close_without_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            bot = bot_at(path)
            signal = {'side': 'long', 'entry': 100, 'tp': 110, 'sl': 95}
            self.assertTrue(bot.execute_signal(signal))
            opened = bot._position['opened_at']
            self.assertEqual(len(bot._strategy_open_log), 1)
            restored = bot_at(path)
            restored._load_state()
            restored._load_state()
            self.assertEqual(len(restored._strategy_open_log), 1)

            self.assertEqual(restored._position['opened_at'], opened)
            with patch.object(restored, '_save_state'), patch('src.live.production_bot.Path'), patch('builtins.open', unittest.mock.mock_open()):
                restored._close_position('TP', 110)
            self.assertEqual(restored._trade_log[-1]['opened_at'], opened)
            self.assertEqual(len(restored._strategy_open_log), 1)

    def test_legacy_recovery_survives_close_and_restart(self):
        from src.live.trade_history import recover_open_time
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            bot = bot_at(path)
            bot._position = {'side': 'short', 'entry': 100, 'tp': 90, 'sl': 110, 'notional': 50}
            bot._save_state()
            log = path.with_name('production.log')
            fill = '2026-10-15 08:00:41.115 | INFO | x - 📝 [纸上] SELL 0.5张 @100 名义=50.0U SL=110 TP=90\n'
            log.write_text(fill)
            restored = bot_at(path)
            restored._load_state()
            self.assertEqual(restored._position['opened_at_source'], 'production_log_utc')
            self.assertEqual(len(restored._strategy_open_log), 1)
            with patch('src.live.production_bot.Path'), patch('builtins.open', unittest.mock.mock_open()), patch.object(restored, '_save_state'):
                restored._close_position('TP', 90)
            restored._save_state()
            again = bot_at(path)
            again._load_state()
            self.assertIsNone(again._position)
            self.assertEqual(len(again._strategy_open_log), 1)
            self.assertEqual(again._trade_log[-1]['opened_at_source'], 'production_log_utc')
            log.write_text(fill + '2026-10-16 00:00:00.000 | INFO | 平仓: short\n')
            self.assertNotIn('opened_at', recover_open_time(bot._position, log))

    def test_exit_rules_for_both_sides_remain_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            for side, high, low, aligned, reason, price in [
                ('long', 111, 99, True, 'TP', 110),
                ('long', 105, 89, True, 'SL', 90),
                ('long', 105, 95, False, '趋势反转', 100),
                ('short', 105, 89, True, 'TP', 90),
                ('short', 111, 95, True, 'SL', 110),
                ('short', 105, 95, False, '趋势反转', 100),
            ]:
                with self.subTest(side=side, reason=reason):
                    bot = bot_at(Path(folder) / 'state.json')
                    bot._position = {'side': side, 'entry': 100,
                                     'tp': 110 if side=='long' else 90,
                                     'sl': 90 if side=='long' else 110}
                    with patch.object(bot, '_is_bull_align', return_value=aligned), patch.object(bot, '_is_bear_align', return_value=aligned), patch.object(bot, '_close_position') as close:
                        bot._check_exit_conditions([BASE, 100, high, low, 100, 1])
                    close.assert_called_once_with(reason, price)

    def test_holding_evidence_uses_closed_bar_and_exit_behavior_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            bot = bot_at(Path(folder) / 'state.json')
            bot._position = {'side': 'short', 'entry': 100, 'tp': 90, 'sl': 110}
            with patch.object(bot, '_is_bear_align', return_value=True):
                bot._process_closed_candle([BASE + BAR, 100, 105, 95, 99, 1])
                evidence = bot._last_signal['holding_check']
                self.assertEqual(evidence['high'], 105)
                self.assertEqual(evidence['low'], 95)
                self.assertTrue(evidence['alignment_intact'])
                self.assertEqual(evidence['candle_ts'], BASE + BAR)
                with patch.object(bot, '_close_position') as close:
                    bot._process_closed_candle([BASE + 2*BAR, 99, 111, 95, 100, 1])
                    close.assert_called_once_with('SL', 110)

    def test_legacy_open_is_recovered_from_matching_log_only(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            state = {'dry_run': True, 'paper_cash': 500,
                     'position': {'side': 'short', 'entry': 81499.7, 'notional': 48.163, 'tp': 78097, 'sl': 83200}}
            (outputs / 'production_state.json').write_text(json.dumps(state))
            (outputs / 'production.log').write_text('2026-10-15 08:00:41.115 | INFO | x - 📝 [纸上] SELL 0.0006张 @81500 名义=48.2U SL=83201 TP=78098 capped\n')
            with patch.object(dashboard, 'OUTPUTS', outputs), patch.object(dashboard, 'market_ticker', return_value={}):
                data = dashboard.dashboard_data()
                self.assertEqual(data['strategy']['position']['opened_at'], '2026-10-15T08:00:41.115000+00:00')
                self.assertEqual(len([r for r in data['activity'] if r['kind']=='strategy_open']), 1)
                self.assertEqual(data['strategy']['position']['opened_at_source'], 'production_log_utc')
                self.assertIsNone(data['strategy']['holding_check'])
                (outputs / 'production.log').write_text('2026-10-15 08:00:41.115 | INFO | x - 📝 [纸上] BUY 0.0006张 @81500 名义=48.2U\n')
                data = dashboard.dashboard_data()
                self.assertIsNone(data['strategy']['position']['opened_at'])
                self.assertFalse(any(r['kind']=='strategy_open' for r in data['activity']))
            self.assertEqual(json.loads((outputs / 'production_state.json').read_text()), state)

    def test_stale_health_does_not_claim_current_holding_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            position = {'side': 'short', 'entry': 100, 'notional': 50, 'tp': 90, 'sl': 110,
                        'opened_at': '2026-10-15T08:00:41+00:00'}
            (outputs / 'production_state.json').write_text(json.dumps({'dry_run': True, 'position': position}))
            (outputs / 'production_health.json').write_text(json.dumps({
                'updated_at': '2026-01-01T00:00:00+00:00', 'running': True,
                'signal': {'holding_check': {'alignment_intact': True}}}))
            with patch.object(dashboard, 'OUTPUTS', outputs), patch.object(dashboard, 'market_ticker', return_value={}):
                data = dashboard.dashboard_data()
            self.assertIsNone(data['strategy']['holding_check'])
            self.assertEqual(len([r for r in data['activity'] if r['kind']=='strategy_open']), 1)

    def test_optional_history_corruption_never_pauses_trading_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            bot = bot_at(path)
            for value in (None, [None, 'invalid']):
                state = bot._state_payload()
                state['strategy_open_log'] = value
                path.write_text(json.dumps(state))
                bot._load_state()
                self.assertEqual(bot._strategy_open_log, [])
                self.assertIsNone(bot._paused_reason)

    def test_activity_orders_mixed_iso_timestamps_by_actual_time(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            state = {'dry_run': True, 'strategy_open_log': [
                {'time': '2026-10-15T08:00:00+00:00', 'side': 'short', 'entry': 100}],
                'trade_log': [{'time': '2026-10-15 12:00:00+00:00', 'side': 'short', 'entry': 100}],
                'core_trade_log': [{'time': '2026-10-15T22:00:00+08:00', 'side': 'sell'},
                                   {'time': None, 'side': 'buy'}]}
            (outputs / 'production_state.json').write_text(json.dumps(state))
            with patch.object(dashboard, 'OUTPUTS', outputs), patch.object(dashboard, 'market_ticker', return_value={}):
                data = dashboard.dashboard_data()
            self.assertEqual([r['kind'] for r in data['activity']],
                             ['spot_trade', 'strategy_close', 'strategy_open', 'spot_trade'])

    def test_holding_evidence_must_match_current_position_and_processed_bar(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            p = {'side': 'short', 'entry': 100, 'notional': 50,
                 'opened_at': '2026-10-15T08:00:41+00:00'}
            check = {'side': 'short', 'entry': 100, 'opened_at': p['opened_at'], 'candle_ts': BASE}
            health = {'updated_at': datetime.now(timezone.utc).isoformat(), 'running': True,
                      'last_closed_candle_ts': BASE, 'signal': {'holding_check': check}}
            (outputs / 'production_state.json').write_text(json.dumps({'dry_run': True, 'position': p}))
            for mismatch in (None, 'side', 'entry', 'opened_at', 'candle_ts', 'paused', 'market_error', 'stopped'):
                with self.subTest(mismatch=mismatch):
                    h = copy.deepcopy(health)
                    if mismatch in check:
                        h['signal']['holding_check'][mismatch] = 'different'
                    elif mismatch == 'paused':
                        h['paused_reason'] = 'market_data_unavailable'
                    elif mismatch == 'market_error':
                        h['market_error_count'] = 1
                    elif mismatch == 'stopped':
                        h['running'] = False
                    (outputs / 'production_health.json').write_text(json.dumps(h))
                    with patch.object(dashboard, 'OUTPUTS', outputs), patch.object(dashboard, 'market_ticker', return_value={}):
                        data = dashboard.dashboard_data()
                    self.assertEqual(data['strategy']['holding_check'], check if mismatch is None else None)


if __name__ == '__main__':
    unittest.main()
