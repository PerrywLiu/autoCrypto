import copy
import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_shadow import source_packet
from src.live.shadow import ShadowExperiment,BAR


def packet_fixture():
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    latest=now//BAR*BAR-BAR
    rows=[[latest-(298-i)*BAR,100,101,99,100,1] for i in range(299)]
    cfg={'symbol':'BTC/USDT','timeframe':'4h','portfolio':{'mode':'paper','enabled':True}}
    source={'dry_run':True,'paper_cash':500,'paper_core_btc':0,'position':None,'total_pnl':0,
            'indicator_history':{'version':1,'candles':rows}}
    packet={'state':source,'candles':rows,'config':cfg,'mark_price':100,'now_ms':now,
            'health':{'running':True,'updated_at':str(datetime.now(timezone.utc)),'last_closed_candle_ts':latest}}
    return packet


class ShadowHistoryInputTests(unittest.TestCase):
    def test_uses_checkpoint_candles_even_if_second_market_read_differs(self):
        p=packet_fixture();raw=copy.deepcopy(p);raw.pop('config');raw.pop('mark_price')
        raw['config_yaml']='mode:\n  testnet: true\n'
        raw['candles']=[[str(p['candles'][-1][0]),'150','151','149','150','1','1','1','1']]
        with patch('scripts.run_shadow.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(raw))):
            actual=source_packet('unused',Path('/source'))
        self.assertEqual(actual['candles'],p['candles'])
        self.assertEqual(actual['mark_price'],100)

    def test_changed_history_is_rejected_before_advancing_either_account(self):
        from scripts.run_shadow import validate_source_history
        p=packet_fixture()
        with tempfile.TemporaryDirectory() as folder:
            exp=ShadowExperiment.create(Path(folder)/'experiment',p['config'],p['state'],p['candles'])
            before=exp.report();changed=copy.deepcopy(p)
            changed['state']['indicator_history']['candles'][0][4]=100.5
            with self.assertRaises(ValueError):validate_source_history(exp,changed)
            self.assertEqual(exp.report(),before)

    def test_local_transport_reads_checkpoint_without_calling_public_market(self):
        import yaml
        p=packet_fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs').mkdir();(root/'config').mkdir()
            (root/'outputs/production_state.json').write_text(json.dumps(p['state']))
            (root/'outputs/production_health.json').write_text(json.dumps(p['health']))
            (root/'config/production.yaml').write_text(yaml.safe_dump({'mode':{'testnet':True}}))
            result=source_packet('unused',root)
            self.assertEqual(result['candles'],p['candles'])

    def test_legacy_account_match_does_not_claim_verified_indicator_history(self):
        from scripts.run_shadow import export
        p=packet_fixture();p['state'].pop('indicator_history')
        with tempfile.TemporaryDirectory() as folder:
            exp=ShadowExperiment.create(Path(folder)/'experiment',p['config'],p['state'],p['candles'])
            report=export(exp,p)
            self.assertTrue(report['online_baseline_comparison']['matched'])
            self.assertFalse(report['comparison_validated'])
