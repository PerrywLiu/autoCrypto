import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_production_recovery import bot_at
from web.equity import initialize, save, valuation
from web import dashboard


class TradeAnalysisTests(unittest.TestCase):
    def test_summary_uses_shanghai_day_and_never_calls_cash_difference_profit(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'equity.sqlite3'
            initialize(db)
            # Shanghai 2026-10-13 midnight, followed by first available sample.
            start = 1791820800000
            save(db, valuation({'dry_run': True, 'paper_cash': 495}, 100, start + 60000))
            current = {'equity': 490, 'time': start + 3600000, 'strategy_realized_pnl': -2.25}
            with patch.object(dashboard, 'EQUITY_DB', db):
                result = dashboard.performance_summary(current, 500, [{'fee': .17}, {'fee': .16}])
            self.assertEqual(result['total_pnl'], -10)
            self.assertEqual(result['today_pnl'], -5)
            self.assertEqual(result['today_start'], start + 60000)
            self.assertAlmostEqual(result['recorded_spot_fees'], .33)
            self.assertAlmostEqual(result['total_return_pct'], -2)

    def test_missing_history_is_unknown(self):
        with patch.object(dashboard, 'EQUITY_DB', Path('/nonexistent/analysis.sqlite3')):
            result = dashboard.performance_summary(None, 500, [])
        self.assertIsNone(result['today_pnl'])
        self.assertIsNone(result['total_pnl'])

    def test_new_spot_fill_records_reason_and_fee_adjusted_account(self):
        with tempfile.TemporaryDirectory() as folder:
            bot = bot_at(Path(folder) / 'state.json')
            before = bot._portfolio_equity(100)
            bot._update_portfolio_plan()
            fill = bot._core_trade_log[-1]
            self.assertEqual(fill['analysis']['reason'], 'slow_add')
            self.assertAlmostEqual(fill['account_after']['equity'], before - fill['fee'])
            self.assertEqual(fill['analysis']['weight_before'], 0)
            self.assertGreater(fill['analysis']['target_weight'], 0)

    def test_dashboard_retains_trade_evidence_and_does_not_invent_old_evidence(self):
        import json, time
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            (outputs / 'production_state.json').write_text(json.dumps({
                'dry_run': True, 'paper_cash': 200, 'paper_core_btc': 1,
                'core_trade_log': [{'side': 'buy', 'fee': .1, 'analysis': {'reason': 'slow_add'},
                                    'account_after': {'equity': 300}}],
                'trade_log': [{'side': 'long', 'pnl': -2, 'reason': 'SL'}]}))
            with patch.object(dashboard, 'OUTPUTS', outputs), patch.object(dashboard, 'EQUITY_DB', outputs / 'missing.sqlite'), patch.object(dashboard, 'market_ticker', return_value={
                    'price': 100, 'updated_at': int(time.time()*1000), 'history': []}):
                data = dashboard.dashboard_data()
            spot = next(r for r in data['activity'] if r['kind']=='spot_trade')
            closed = next(r for r in data['activity'] if r['kind']=='strategy_close')
            self.assertEqual(spot['analysis']['reason'], 'slow_add')
            self.assertEqual(spot['account_after']['equity'], 300)
            self.assertEqual(closed['reason'], 'SL')
            self.assertIsNone(closed['account_after'])
            self.assertIsNone(closed['entry_analysis'])

    def test_close_snapshot_does_not_double_count_closed_position(self):
        with tempfile.TemporaryDirectory() as folder:
            bot = bot_at(Path(folder) / 'state.json')
            bot._paper_cash = 200
            bot._paper_core_btc = 1
            bot._position = {'side': 'long', 'entry': 100, 'notional': 100,
                             'entry_analysis': {'reason': 'long_conditions_met'}}
            with patch.object(bot, '_save_state'), patch('src.live.production_bot.Path'), patch('builtins.open', unittest.mock.mock_open()):
                bot._close_position('TP', 110)
            trade = bot._trade_log[-1]
            self.assertAlmostEqual(trade['account_after']['equity'], 320)
            self.assertEqual(trade['entry_analysis']['reason'], 'long_conditions_met')
            self.assertEqual(trade['account_after']['total_pnl'], -180)
