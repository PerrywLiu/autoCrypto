#!/usr/bin/env python
"""Run an independent local shadow comparison against read-only jp market/account data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.live.shadow import ShadowExperiment, BAR

# Public data and paper state only. Does not read credentials, write files remotely,
# execute inside the trading container, or call any private trading endpoint.
REMOTE = '''import json, subprocess, time
from pathlib import Path
root=Path('/root/work/autoCrypto')
state_before=(root/'outputs/production_state.json').read_text()
config=(root/'config/production.yaml').read_text()
url='https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=4H&limit=300'
raw=json.loads(subprocess.check_output(['curl','-fsS','--max-time','20','-H','x-simulated-trading: 1',url]))
assert raw['code']=='0',raw.get('msg')
health=json.loads((root/'outputs/production_health.json').read_text())
assert state_before==(root/'outputs/production_state.json').read_text(), 'source state changed during capture; retry'
print(json.dumps({'state':json.loads(state_before),'config_yaml':config,'health':health,'candles':raw['data'],'now_ms':int(time.time()*1000)}))
'''


def source_packet(host, source_directory=None):
    if source_directory is not None:
        code=REMOTE.replace("root=Path('/root/work/autoCrypto')", "root=Path("+repr(str(source_directory))+")")
        command=[sys.executable,'-c',code]
        result=subprocess.run(command,text=True,capture_output=True,timeout=40,check=True)
    else:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -'],input=REMOTE,text=True,capture_output=True,timeout=40,check=True)
    packet=json.loads(result.stdout);packet['config']=yaml.safe_load(packet.pop('config_yaml'))
    if packet['config'].get('mode',{}).get('testnet') is not True:
        raise ValueError('本读取器固定为模拟盘行情；配置变化必须人工核对')
    candles=[]
    for row in packet['candles']:
        candle=[int(row[0])]+[float(v) for v in row[1:6]]
        if row[-1]=='1' and candle[0]+BAR<=packet['now_ms']:candles.append(candle)
    packet['mark_price']=float(max(packet['candles'],key=lambda r:int(r[0]))[4])
    packet['candles']=sorted(candles,key=lambda c:c[0])
    h=packet['health'];stamp=datetime.fromisoformat(h['updated_at'])
    age=packet['now_ms']/1000-stamp.timestamp()
    if not h.get('running') or age>180 or age< -5 or h.get('paused_reason'):
        raise ValueError('源机器人未正常运行；影子暂停推进')
    if not packet['candles'] or h['last_closed_candle_ts']!=packet['candles'][-1][0]:
        raise ValueError('等待源机器人与最新自然收盘对齐')
    return packet


def export(exp,packet=None,error=None):
    report=exp.report();report['checked_at']=datetime.now(timezone.utc).isoformat();report['poll_error']=error
    if packet:
        source=packet['state'];baseline=exp.bots[0]
        checks={k:abs(getattr(baseline,'_'+k)-float(source.get(k,0)))<1e-8 for k in ('paper_cash','paper_core_btc','total_pnl')}
        p=baseline._position;q=source.get('position')
        checks['position']=(p is None and q is None) or (bool(p) and bool(q) and p['side']==q['side'] and all(abs(float(p.get(k,0))-float(q.get(k,0)))<1e-8 for k in ('entry','size','notional','sl','tp')))
        report['online_baseline_comparison']={'checks':checks,'matched':all(checks.values()),
            'note':'同一收盘下核对源账户；不一致仅标记，不覆盖或强行同步'}
    report['comparison_validated']=bool(packet and report['online_baseline_comparison']['matched']) and not error
    folder=exp.path.parent
    temp=folder/'shadow_status.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(folder/'shadow_status.json')
    lines=['# 4小时＋1%买回缓冲影子并跑','',f"检查时间：{report['checked_at']}。已处理自然收盘：{report['processed_candles']} 根。",'','|指标|现有规则基线|1%缓冲候选|','|---|---:|---:|']
    for title,key in [('账户权益','equity'),('启动后收益 USDT','pnl_since_start'),('启动后收益率 %','return_pct'),('最大收盘回撤 %','max_drawdown_pct'),('调仓笔数','rebalance_count'),('现货手续费 USDT','spot_fee'),('卖出后24小时买回','quick_rebuy_24h'),('其中高于卖价买回','quick_rebuy_above_sell'),('被拦截加仓次数','blocked_buys')]:
        lines.append(f"|{title}|{report['baseline'][key]:.6f}|{report['candidate'][key]:.6f}|")
    lines+=['',f"候选减基线权益差：{report['candidate_minus_baseline']:.6f} USDT。",f"上涨期间少持BTC的毛收益差：{report['missed_upside_gross']:.6f} USDT。此值不扣下跌期间防御收益，不能当净损失。",'',report['cost_basis'],'最大回撤按4小时收盘权益计算。旧成交不计入新实验次数或费用。','独立影子进程，只读源行情与账户；没有订单权限，不修改源账户或策略。']
    if error:lines+=['',f'本次读取失败，保留最近已提交比较：{error}']
    if not report['comparison_validated']:lines+=['','本次基线与线上核对未通过或不可用，指标仅作独立模拟保存，不标作已验证线上对比。']
    if 'online_baseline_comparison' in report:lines+=['',f"基线与源账户核对：{report['online_baseline_comparison']}"]
    (folder/'影子并跑比较.md').write_text('\n'.join(lines))
    with exp.db() as db, (folder/'equity_samples.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['candle_open_ms','price','baseline_equity','candidate_equity','missed_upside_gross_interval']);writer.writerows(db.execute('SELECT * FROM samples ORDER BY candle_ts'))
    with exp.db() as db:
        events=[{'variant':v,**json.loads(p)} for v,p in db.execute('SELECT variant,payload FROM events ORDER BY id')]
    (folder/'decision_events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='jp');parser.add_argument('--source-directory',type=Path);parser.add_argument('--directory',default='output/shadow-buffer-1pct');parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=int,default=60);args=parser.parse_args()
    if args.interval<30:parser.error('轮询间隔至少30秒')
    folder=Path(args.directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    # Prevent simultaneous writers to this experiment; released on exit/crash.
    import fcntl
    lock=(folder/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (folder/'runner.pid').write_text(str(os.getpid()))
    exp=None
    while True:
        try:
            packet=source_packet(args.host,args.source_directory);cfg=packet['config'];db=folder/'shadow.sqlite3'
            if exp is None:
                exp=ShadowExperiment.load(db) if db.exists() else ShadowExperiment.create(db,cfg,packet['state'],packet['candles'],mark_price=packet['mark_price'])
            digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            if digest!=exp.config_hash:raise ValueError('源配置改变：冻结实验，不混用新旧策略')
            exp.process(packet['candles'],packet['now_ms']);report=export(exp,packet)
            print(json.dumps({'checked_at':report['checked_at'],'processed':report['processed_candles'],'delta':report['candidate_minus_baseline'],'source_matched':report['online_baseline_comparison']['matched']},ensure_ascii=False),flush=True)
        except KeyboardInterrupt:break
        except Exception as exc:
            message=f'{type(exc).__name__}: {exc}'
            print(message,file=sys.stderr,flush=True)
            if exp:export(exp,error=message)
            if args.once:return 1
        if args.once:return 0
        time.sleep(args.interval)
    return 0

if __name__=='__main__':raise SystemExit(main())
