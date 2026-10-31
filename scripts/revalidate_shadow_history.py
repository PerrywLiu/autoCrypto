"""Offline reconstruction only: never overwrite or resume the original experiment."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live.shadow import ShadowExperiment, BAR, validate_seed
from src.live.production_bot import ProductionBot
from src.live.candle_history import checkpoint_candles
from scripts.run_shadow import export


ACCOUNT_KEYS = ('paper_cash', 'paper_core_btc', 'total_pnl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revalidate(source_db, online_state, output_directory, reset_before_ts, history_bars, online_health=None):
    """Replay an explicit historical reset into a new experiment; preserve inputs."""
    source_db = Path(source_db).resolve()
    online_state = Path(online_state).resolve()
    output = Path(output_directory).resolve()
    if history_bars < 221:
        raise ValueError('history_bars must provide at least 221 warmup bars')
    original_hash = digest(source_db)
    online_hash = digest(online_state)
    # Refuse both accidental overwrite and using the live experiment as output.
    output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(output / 'original.sqlite3') as backup:
            source.backup(backup)
    with sqlite3.connect((output / 'original.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        saved = json.loads(db.execute('SELECT payload FROM state WHERE id=1').fetchone()[0])
    online_bytes = online_state.read_bytes()
    online = json.loads(online_bytes)
    validate_seed(online, saved['config'])
    (output / 'original_online_state.json').write_bytes(online_bytes)
    health_hash = None
    if 'indicator_history' in online:
        online_last_ts = checkpoint_candles(online, BAR)[-1][0]
    elif online_health is not None:
        health_bytes = Path(online_health).read_bytes()
        health = json.loads(health_bytes)
        online_last_ts = health['last_closed_candle_ts']
        health_hash = digest(online_health)
        (output / 'original_online_health.json').write_bytes(health_bytes)
    else:
        raise ValueError('legacy online state requires explicit last-close health evidence')
    rows = saved['candles']
    if online_last_ts != rows[-1][0]:
        raise ValueError('online snapshot and experiment end at different closed candles')
    seed_rows = [c for c in rows if c[0] <= saved['start_candle_ts']]
    pending = [c for c in rows if c[0] > saved['start_candle_ts']]
    if reset_before_ts not in [c[0] for c in pending]:
        raise ValueError('reset boundary is outside recorded post-seed closes')
    experiments = {}
    for variant in ('continuous', 'restart_aligned'):
        exp = ShadowExperiment.create(output / variant / 'shadow.sqlite3', saved['config'],
                                      saved['seed'], seed_rows, saved['initial_mark_price'])
        for candle in pending:
            if variant == 'restart_aligned' and candle[0] == reset_before_ts:
                if len(exp.candles) < history_bars:
                    raise ValueError('not enough history at reset boundary')
                # Explicitly reconstruct the OLD online restart, not a new trading rule.
                exp.candles = exp.candles[-history_bars:]
                for bot in exp.bots:
                    bot.set_history(exp.candles)
                with exp.db() as db:
                    exp.persist(db)
            exp.process([candle], candle[0] + BAR)
        export(exp)
        experiments[variant] = exp
    continuous = experiments['continuous']
    aligned = experiments['restart_aligned']
    source_checks = {k: getattr(continuous.bots[0], '_' + k) == saved['bots'][0]['_' + k]
                     for k in ACCOUNT_KEYS}
    source_checks.update({'candidate_' + k: getattr(continuous.bots[1], '_' + k) == saved['bots'][1]['_' + k]
                          for k in ACCOUNT_KEYS})
    online_checks = {k: getattr(aligned.bots[0], '_' + k) == online[k] for k in ACCOUNT_KEYS}
    online_checks['same_last_closed_candle'] = aligned.candles[-1][0] == online_last_ts
    p, q = aligned.bots[0]._position, online.get('position')
    position_keys = ('side', 'entry', 'size', 'notional', 'sl', 'tp')
    online_checks['position'] = (p is None and q is None) or (
        bool(p) and bool(q) and all(p.get(k) == q.get(k) for k in position_keys))
    online_fills = [t for t in online.get('core_trade_log', [])
                    if t.get('candle_time', -1) > saved['start_candle_ts']]
    rebuilt_fills = aligned.bots[0]._core_trade_log
    online_checks['spot_fills'] = len(online_fills) == len(rebuilt_fills) and all(
        all(a.get(k) == b.get(k) for k in ('side', 'price', 'size', 'notional', 'fee', 'candle_time'))
        for a, b in zip(online_fills, rebuilt_fills))
    # A derived experiment must itself survive restart without a new warmup origin.
    restored = ShadowExperiment.load(aligned.path)
    restart_checks = {'report': restored.report() == aligned.report(),
                      'candles': restored.candles == aligned.candles}
    for label, old, new in zip(('baseline', 'candidate'), aligned.bots, restored.bots):
        restart_checks[label + '_account'] = old.dump() == new.dump()
        restart_checks[label + '_indicators'] = all(
            np.array_equal(old._indicators[k], new._indicators[k], equal_nan=True)
            for k in old._indicators)
    # Keep all pre-seed trade records and account metadata from the source state.
    # Shadow logs deliberately contain post-seed events only.
    checkpoint = copy.deepcopy(online)
    checkpoint['indicator_history'] = {'version': 1, 'candles': copy.deepcopy(aligned.candles)}
    checkpoint['last_signal'] = copy.deepcopy(aligned.bots[0]._last_signal)
    checkpoint['history_origin'] = 'reconstructed_20261018_restart'
    candidate = output / 'reconstructed_checkpoint.json'
    if all(source_checks.values()) and all(online_checks.values()) and all(restart_checks.values()):
        candidate.write_text(json.dumps(checkpoint, indent=2, allow_nan=False))
        restored_production = ProductionBot('', '', '', state_file=str(candidate))
        restored_production._load_state()
        restart_checks['production_checkpoint'] = restored_production._paused_reason is None and all(
            np.array_equal(restored_production._indicators[k], aligned.bots[0]._indicators[k], equal_nan=True)
            for k in aligned.bots[0]._indicators)
        restart_checks['complete_original_ledger'] = all(
            getattr(restored_production, '_' + key) == online.get(key, [])
            for key in ('trade_log', 'core_trade_log', 'strategy_open_log'))
    preserved = digest(source_db) == original_hash and digest(online_state) == online_hash
    result = {
        'mode': 'offline_revalidation_only',
        'original_db_sha256': original_hash,
        'online_state_sha256': online_hash,
        'online_health_sha256': health_hash,
        'online_last_closed_candle_ts': online_last_ts,
        'original_inputs_preserved': preserved,
        'historical_reset_before_candle_ms': reset_before_ts,
        'historical_reset_warmup_bars': history_bars,
        'continuous_reproduces_original': source_checks,
        'reconstructed_baseline_matches_online': online_checks,
        'restart_checks': restart_checks,
        'continuous': continuous.report(), 'restart_aligned': aligned.report(),
        'passed': all(source_checks.values()) and all(online_checks.values())
                  and all(restart_checks.values()) and preserved,
        'limitations': ['Historical replay only; no production release or natural-run acceptance.',
                        'Contract fees/funding remain incomplete; this does not validate buffer profitability.',
                        'The historical reset is explicit evidence, never silently injected into the runner.'],
    }
    if not result['passed'] and candidate.exists():
        candidate.unlink()
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-db', required=True, type=Path)
    parser.add_argument('--online-state', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--reset-before', required=True, help='First bar OPEN timestamp after historical restart, with timezone')
    parser.add_argument('--online-health', type=Path, help='Required last-close evidence for legacy state without checkpoint')
    parser.add_argument('--history-bars', required=True, type=int)
    args = parser.parse_args()
    stamp = datetime.fromisoformat(args.reset_before.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        parser.error('--reset-before requires timezone')
    result = revalidate(args.source_db, args.online_state, args.output_directory,
                        int(stamp.timestamp() * 1000), args.history_bars, args.online_health)
    print(json.dumps({'passed': result['passed'], 'output': str(args.output_directory)}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
