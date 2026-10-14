import copy
import tempfile
import unittest
from pathlib import Path
from src.live.shadow import ShadowExperiment, validate_seed, buy_buffer_allows


class ShadowTests(unittest.TestCase):
    def test_buffer_strict_boundary_and_first_allocation(self):
        self.assertTrue(buy_buffer_allows(100, 100, sold_before=False))
        self.assertFalse(buy_buffer_allows(101,100,sold_before=True))
        self.assertTrue(buy_buffer_allows(101.01,100,sold_before=True))
        self.assertFalse(buy_buffer_allows(100,float('nan'),sold_before=True))

    def test_rejects_live_or_non_four_hour_seed(self):
        for state,cfg in [({'dry_run':False},{'timeframe':'4h','portfolio':{'mode':'paper'}}),({'dry_run':True},{'timeframe':'1h','portfolio':{'mode':'paper'}})]:
            with self.assertRaises(ValueError):validate_seed(state,cfg)

    def test_duplicate_poll_and_restart_preserve_both_accounts(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg={'symbol':'BTC/USDT','timeframe':'4h','risk':{'initial_capital':500},'strategy':{},'portfolio':{'enabled':True,'mode':'paper'}}
            candles=[[i*14400000,100,101,99,100,1] for i in range(299)]
            source={'dry_run':True,'paper_cash':200,'paper_core_btc':3,'total_pnl':0,'position':None,'core_trade_log':[]}
            before=copy.deepcopy(source)
            p=Path(folder)/'shadow.sqlite3';exp=ShadowExperiment.create(p,cfg,source,candles)
            new=[299*14400000,100,101,99,100,1]
            exp.process([new],now_ms=300*14400000)
            first=exp.report()
            exp.process([new],now_ms=300*14400000)
            self.assertEqual(exp.report(),first)
            restored=ShadowExperiment.load(p)
            self.assertEqual(restored.report(),first)
            self.assertEqual(source,before)
            self.assertEqual(first['processed_candles'],1)
            self.assertEqual(first['baseline']['equity'],first['candidate']['equity'])

    def test_gap_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg={'symbol':'BTC/USDT','timeframe':'4h','risk':{'initial_capital':500},'portfolio':{'mode':'paper','enabled':True}}
            candles=[[i*14400000,100,101,99,100,1] for i in range(299)]
            exp=ShadowExperiment.create(Path(folder)/'db',cfg,{'dry_run':True,'paper_cash':500},candles)
            with self.assertRaises(ValueError):exp.process([[301*14400000,100,101,99,100,1]],302*14400000)
            self.assertEqual(ShadowExperiment.load(exp.path).report()['processed_candles'],0)

class PolicyBehaviorTests(unittest.TestCase):
    def experiment(self,path):
        cfg={'symbol':'BTC/USDT','timeframe':'4h','risk':{'initial_capital':500},'strategy':{},'portfolio':{'enabled':True,'mode':'paper'}}
        candles=[[i*14400000,100,101,99,100,1] for i in range(299)]
        return ShadowExperiment.create(path,cfg,{'dry_run':True,'paper_cash':200,'paper_core_btc':3},candles)

    def test_sell_unaffected_buy_is_blocked_and_then_recovers_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            exp=self.experiment(Path(folder)/'db')
            exp.process([[299*14400000,100,100,97,98,1]],300*14400000)
            a,b=exp.bots
            self.assertEqual(a._paper_core_btc,b._paper_core_btc)
            self.assertEqual(a._core_trade_log[-1]['side'],'sell')
            exp.process([[300*14400000,98,101,98,100.5,1]],301*14400000)
            report=exp.report()
            self.assertEqual(report['candidate']['blocked_buys'],1)
            self.assertEqual(report['baseline']['quick_rebuy_24h'],1)
            self.assertGreater(a._paper_core_btc,b._paper_core_btc)
            exp=ShadowExperiment.load(exp.path)
            exp.process([[301*14400000,100.5,103,100.5,102,1]],302*14400000)
            self.assertGreater(exp.report()['missed_upside_gross'],0)
            self.assertEqual(exp.bots[1]._core_trade_log[-1]['side'],'buy')

    def test_no_production_writes_and_no_order_access(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            exp=self.experiment(Path(folder)/'db')
            bot=exp.bots[0]
            with self.assertRaises(RuntimeError):bot.order_executor.execute_entry({})
            bot._position={'side':'long','entry':100,'notional':100}
            with patch('builtins.open',side_effect=AssertionError('must not write production files')):
                bot._close_position('SL',99)
            self.assertIsNone(bot._position)
            self.assertAlmostEqual(bot._total_pnl,-1)
            bot.dry_run=False
            with self.assertRaises(RuntimeError):bot.execute_signal({})

    def test_failed_candle_rolls_back_both_accounts_and_can_retry(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            exp=self.experiment(Path(folder)/'db');before=exp.report()
            with patch.object(exp.bots[1],'_process_closed_candle',side_effect=RuntimeError('test')):
                with self.assertRaises(RuntimeError):exp.process([[299*14400000,100,101,99,100,1]],300*14400000)
            self.assertEqual(exp.report(),before)
            exp.process([[299*14400000,100,101,99,100,1]],300*14400000)
            self.assertEqual(exp.report()['processed_candles'],1)

class LocalTransportTests(unittest.TestCase):
    def test_server_transport_uses_local_readonly_reader_without_ssh(self):
        import json,time,sys
        from datetime import datetime,timezone
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.run_shadow import source_packet
        now=int(time.time()*1000);stamp=now//14400000*14400000-14400000
        raw={'config_yaml':'mode:\n  testnet: true\n','state':{'dry_run':True},
             'health':{'running':True,'updated_at':str(datetime.now(timezone.utc)),'last_closed_candle_ts':stamp},
             'candles':[[str(stamp),'100','101','99','100','1','1','1','1']], 'now_ms':now}
        with patch('scripts.run_shadow.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(raw))) as call:
            result=source_packet('unused',Path('/source'))
        command=call.call_args.args[0]
        self.assertEqual(command[0],sys.executable)
        self.assertNotIn('ssh',command)
        self.assertIn("root=Path('/source')",command[2])
        self.assertEqual(result['candles'][0][0],stamp)
