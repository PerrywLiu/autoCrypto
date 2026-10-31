"""Restart must not change the strategy's input history or commit partial bars."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.live.production_bot import ProductionBot
from src.live.portfolio import PortfolioConfig, classify_latest_regime

BAR=14_400_000
BASE=1_780_000_000_000//BAR*BAR


def history(count=340):
    # Different starting windows create different EMA200 seeds.
    return [[BASE+i*BAR, 100+i*.03+4*np.sin(i/20),
             106+i*.03+4*np.sin(i/20), 94+i*.03+4*np.sin(i/20),
             100+i*.03+4*np.sin(i/20), 1] for i in range(count)]


def make_bot(path, rows=None):
    b=ProductionBot('', '', '', state_file=str(path),
        portfolio_config=PortfolioConfig(enabled=True,mode='paper'))
    if rows is not None:
        b._candles=copy.deepcopy(rows)
        b._ohlcv={k:np.array([c[i+1] for c in rows]) for i,k in enumerate(('open','high','low','close','volume'))}
        b._update_indicators()
    return b


class IndicatorHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'state.json'

    def test_restart_restores_entire_history_and_exact_indicators(self):
        a=make_bot(self.path,history());a._paper_cash=300;a._paper_core_btc=1;a._save_state()
        b=make_bot(self.path,history()[-299:]);b._load_state()
        self.assertEqual(b._candles,a._candles)
        for k in a._indicators:np.testing.assert_array_equal(b._indicators[k],a._indicators[k])
        self.assertEqual(b._paper_cash,300)
        self.assertEqual(classify_latest_regime(b._ohlcv['close']),classify_latest_regime(a._ohlcv['close']))

    def test_no_trade_bar_is_saved_and_duplicate_is_ignored(self):
        rows=history();b=make_bot(self.path,rows[:-1])
        b._last_trade_candle_ts=rows[-2][0];b._consecutive_losses=1
        b._process_closed_candle(rows[-1]);first=self.path.read_bytes()
        state=json.loads(first)
        self.assertEqual(state['indicator_history']['candles'],rows)
        b._process_closed_candle(rows[-1])
        self.assertEqual(b._candles,rows)
        self.assertEqual(self.path.read_bytes(),first)

    def test_incomplete_bar_never_commits_account_or_history(self):
        rows=history();b=make_bot(self.path,rows[:-1]);b._save_state();before=self.path.read_bytes()
        def interrupted(*args):
            b._paper_cash=1;b._save_state();raise RuntimeError('interrupted')
        with patch.object(b,'_update_portfolio_plan',side_effect=interrupted):
            with self.assertRaises(RuntimeError):b._process_closed_candle(rows[-1])
        self.assertEqual(self.path.read_bytes(),before)
        self.assertIsNotNone(b._paused_reason)

    def test_invalid_checkpoint_pauses_without_restoring_account(self):
        rows=history();a=make_bot(self.path,rows);a._save_state();state=json.loads(self.path.read_text())
        for broken in (rows[:-2]+rows[-1:], rows[:-1]+[rows[-2]],
                       rows[:-1]+[[rows[-1][0],100,101,99,float('nan'),1]],
                       [[c[0]+1000*BAR,*c[1:]] for c in rows]):
            with self.subTest(last=broken[-1][0]):
                bad=copy.deepcopy(state);bad['paper_cash']=1
                bad['indicator_history']={'version':1,'candles':broken}
                self.path.write_text(json.dumps(bad));b=make_bot(self.path)
                b._load_state()
                self.assertIsNotNone(b._paused_reason)
                self.assertEqual(b._paper_cash,500)

    def test_gap_does_not_reset_history_or_trade(self):
        rows=history();b=make_bot(self.path,rows[:-2]);b._save_state();before=self.path.read_bytes()
        with self.assertRaises(ValueError):b._process_closed_candle(rows[-1])
        self.assertEqual(b._candles,rows[:-2]);self.assertEqual(self.path.read_bytes(),before)

    def test_save_failure_stops_processing_and_preserves_previous_file(self):
        b=make_bot(self.path,history());b._save_state();before=self.path.read_bytes();b._running=True
        with patch('pathlib.Path.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):b._save_state()
        self.assertFalse(b._running);self.assertIsNotNone(b._paused_reason)
        self.assertEqual(self.path.read_bytes(),before)

    def test_start_restores_checkpoint_without_refetching_or_rebalancing_same_bar(self):
        rows=history();a=make_bot(self.path,rows);a._paper_cash=320;a._paper_core_btc=1;a._save_state()
        b=make_bot(self.path)
        b.exchange.fetch_ohlcv=lambda *args,**kw: rows[-2:]
        with patch.object(b,'_update_portfolio_plan',side_effect=AssertionError('duplicate rebalance')), \
             patch('src.live.production_bot.time.sleep',side_effect=lambda _:setattr(b,'_running',False)):
            b.start()
        self.assertEqual(b._candles,rows);self.assertEqual(b._paper_cash,320)
        self.assertIsNone(b._paused_reason)

    def test_restart_catches_up_downtime_bars_without_changing_result(self):
        rows=history(342);a=make_bot(self.path,rows[:-2]);a._save_state()
        uninterrupted=make_bot(Path(self.tmp.name)/'control.json',rows[:-2])
        for row in rows[-2:]:uninterrupted._process_closed_candle(row)
        b=make_bot(self.path)
        b.exchange.fetch_ohlcv=lambda *args,**kw:rows[-4:]
        b.exchange.milliseconds=lambda:rows[-1][0]+BAR+1000
        with patch('src.live.production_bot.time.sleep',side_effect=lambda _:setattr(b,'_running',False)):
            b.start()
        self.assertEqual(b._candles,uninterrupted._candles)
        for k in b._indicators:np.testing.assert_array_equal(b._indicators[k],uninterrupted._indicators[k])
        for k in ('_paper_cash','_paper_core_btc','_total_pnl'):
            self.assertEqual(getattr(b,k),getattr(uninterrupted,k))
        self.assertEqual({k:v for k,v in b._position.items() if k!='opened_at'},
                         {k:v for k,v in uninterrupted._position.items() if k!='opened_at'})

    def test_downtime_gap_stops_without_shortening_checkpoint(self):
        rows=history(342);a=make_bot(self.path,rows[:-2]);a._save_state();before=self.path.read_bytes()
        b=make_bot(self.path);b.exchange.fetch_ohlcv=lambda *args,**kw:rows[-1:]
        b.exchange.milliseconds=lambda:rows[-1][0]+BAR+1000
        with patch('src.live.production_bot.time.sleep'):
            b.start()
        self.assertEqual(b._paused_reason,'state_invalid:indicator_history_gap')
        self.assertFalse(b._running);self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(b._candles,rows[:-2])

    def test_backfilled_signal_uses_bar_close_day_for_daily_loss_limit(self):
        from datetime import datetime,timezone
        rows=history();b=make_bot(self.path,rows[:-1])
        b._last_trade_day=datetime.fromtimestamp((rows[-1][0]+BAR)/1000,timezone.utc).day
        b._daily_pnl=-30
        signal={'side':'long','entry':float(rows[-1][4]),'sl':90,'tp':130}
        with patch.object(b,'evaluate',return_value=signal):b._process_closed_candle(rows[-1])
        self.assertIsNone(b._position)
        self.assertEqual(b._daily_pnl,-30)

    def test_cross_utc_midnight_restart_matches_continuous_daily_risk(self):
        from datetime import datetime,timezone
        rows=history(342);day=86_400_000
        offset=rows[-1][0]//day*day+20*3_600_000-rows[-1][0]
        rows=[[r[0]+offset,*r[1:]] for r in rows]
        signal={'side':'long','entry':float(rows[-1][4]),'sl':90,'tp':130}
        a=make_bot(Path(self.tmp.name)/'control.json',rows[:-2])
        a._last_trade_day=datetime.fromtimestamp((rows[-2][0]+BAR)/1000,timezone.utc).day
        a._daily_pnl=-30
        b=make_bot(self.path,rows[:-2]);b._last_trade_day=a._last_trade_day;b._daily_pnl=-30
        for bot in (a,b):
            with patch.object(bot,'evaluate',return_value=signal):bot._process_closed_candle(rows[-2])
            self.assertIsNone(bot._position)
        restored=make_bot(self.path);restored._load_state()
        for bot in (a,restored):
            with patch.object(bot,'evaluate',return_value=signal):bot._process_closed_candle(rows[-1])
            self.assertEqual(bot._position['side'],'long')
            self.assertEqual(bot._daily_pnl,0)
        for key in ('_paper_cash','_paper_core_btc','_total_pnl','_last_trade_day'):
            self.assertEqual(getattr(a,key),getattr(restored,key))
