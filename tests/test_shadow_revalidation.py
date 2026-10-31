import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from src.live.shadow import ShadowExperiment,BAR


class RevalidationTests(unittest.TestCase):
    def test_revalidation_preserves_original_and_verifies_derived_restart(self):
        from scripts.revalidate_shadow_history import revalidate
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);db=root/'original.sqlite3'
            cfg={'symbol':'BTC/USDT','timeframe':'4h','portfolio':{'enabled':True,'mode':'paper'}}
            rows=[[i*BAR,100,101,99,100,1] for i in range(304)]
            seed={'dry_run':True,'paper_cash':500,'paper_core_btc':0,'total_pnl':0,'position':None}
            exp=ShadowExperiment.create(db,cfg,seed,rows[:299])
            exp.process(rows[299:],304*BAR)
            online=root/'online.json';account=exp.bots[0]._state_payload()
            account['trade_log']=[{'side':'long','entry':99,'exit':100,'time':'1970-01-01T00:00:00+00:00'}]
            online.write_text(json.dumps(account))
            before=hashlib.sha256(db.read_bytes()).hexdigest()
            out=root/'new'
            result=revalidate(db,online,out,301*BAR,299)
            self.assertTrue(result['passed'])
            self.assertTrue(all(result['restart_checks'].values()))
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(),before)
            self.assertTrue((out/'original.sqlite3').exists())
            checkpoint=json.loads((out/'reconstructed_checkpoint.json').read_text())
            for key in ('trade_log','core_trade_log','strategy_open_log'):
                self.assertEqual(checkpoint[key],account[key])
            with self.assertRaises(FileExistsError):revalidate(db,online,out,301*BAR,299)
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(),before)

    def test_equal_balances_at_different_close_must_not_validate(self):
        from scripts.revalidate_shadow_history import revalidate
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);db=root/'original.sqlite3'
            cfg={'symbol':'BTC/USDT','timeframe':'4h','portfolio':{'enabled':True,'mode':'paper'}}
            rows=[[i*BAR,100,101,99,100,1] for i in range(305)]
            exp=ShadowExperiment.create(db,cfg,{'dry_run':True,'paper_cash':500},rows[:299])
            exp.process(rows[299:304],304*BAR)
            account=exp.bots[0]._state_payload()
            account['indicator_history']['candles']=rows
            online=root/'online.json';online.write_text(json.dumps(account))
            with self.assertRaises(ValueError):revalidate(db,online,root/'new',301*BAR,299)
            self.assertFalse((root/'new/reconstructed_checkpoint.json').exists())
