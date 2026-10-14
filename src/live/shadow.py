"""Independent paper comparison. Never writes production files or sends orders."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.live import production_bot as production
from src.live.production_bot import ProductionBot
from src.live.portfolio import classify_latest_regime, load_portfolio_config

BAR = 14_400_000
BUFFER = .01
ACCOUNT_FIELDS = ('_paper_cash','_paper_core_btc','_position','_total_pnl','_daily_pnl',
                  '_last_trade_day','_consecutive_losses','_last_trade_bar',
                  '_last_trade_candle_ts','_trade_log','_core_trade_log','_last_signal')


def buy_buffer_allows(close, ema200, sold_before):
    return not sold_before or (math.isfinite(ema200) and close > ema200 * (1 + BUFFER))


def validate_seed(state, cfg):
    if state.get('dry_run') is not True or cfg.get('portfolio', {}).get('mode') != 'paper':
        raise ValueError('影子并跑只接受 paper 账户快照')
    if cfg.get('timeframe') != '4h' or cfg.get('symbol', 'BTC/USDT') != 'BTC/USDT':
        raise ValueError('本实验仅支持 BTC/USDT 4h')


def valid_candles(candles):
    if not candles or any(len(c) < 6 or not all(math.isfinite(v) for v in c[:6]) or c[4] <= 0 for c in candles):
        raise ValueError('影子行情字段无效')
    if any(b[0] - a[0] != BAR for a,b in zip(candles,candles[1:])):
        raise ValueError('影子行情有缺口或重复；不会跳过收盘')


class ReplayClock(datetime):
    current = datetime.now(timezone.utc)
    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


class NoOrders:
    def __getattr__(self, name):
        raise RuntimeError(f'影子程序禁止交易所账户/订单操作：{name}')


class ShadowBot(ProductionBot):
    def __init__(self, cfg, candidate=False):
        r=cfg.get('risk',{});s=cfg.get('strategy',{})
        super().__init__('', '', '', symbol=cfg.get('symbol','BTC/USDT'), timeframe='4h',
            initial_capital=r.get('initial_capital',500),leverage=r.get('leverage',2),
            risk_per_trade=r.get('risk_per_trade',.02),max_notional_pct=r.get('max_notional_pct',.95),
            max_daily_loss_pct=r.get('max_daily_loss_pct',.05),max_consecutive_losses=r.get('max_consecutive_losses',5),
            loss_pause_bars=r.get('loss_pause_bars',18),
            ema_fast=s.get('ema_fast',10),ema_mid=s.get('ema_mid',20),ema_slow=s.get('ema_slow',50),ema_trend=s.get('ema_trend',200),
            adx_threshold=s.get('adx_threshold',20),ema_slope_lookback=s.get('ema_slope_lookback',0),
            min_ema_spread=s.get('min_ema_spread',0),min_atr_percentile=s.get('min_atr_percentile',0),
            atr_percentile_window=s.get('atr_percentile_window',180),
            sl_atr=s.get('sl_atr_mult',1.5),tp_atr=s.get('tp_atr_mult',3),
            cooldown_bars=s.get('cooldown_base',6),cooldown_extra_per_loss=s.get('cooldown_extra_per_loss',3),
            portfolio_config=load_portfolio_config(cfg.get('portfolio')))
        self.exchange=SimpleNamespace(parse_timeframe=lambda _:14400,
            amount_to_precision=lambda _,x:str(x),price_to_precision=lambda _,x:str(x))
        self.account_reader=NoOrders();self.order_executor=NoOrders()
        self.candidate=candidate;self.sold_before=False;self.last_sell=None;self.events=[]

    def _save_state(self): pass
    def _write_account_snapshot(self,*args): pass
    def _write_health(self): pass
    def get_position(self): return None
    def set_leverage(self): raise RuntimeError('影子程序禁止设置杠杆')
    def _execute_live_close(self,*args): raise RuntimeError('影子程序禁止真实平仓')
    def execute_signal(self,signal):
        if not self.dry_run: raise RuntimeError('影子程序必须保持 dry_run')
        return super().execute_signal(signal)

    def set_history(self,candles):
        self._candles=copy.deepcopy(candles)
        self._ohlcv={key:np.array([c[i+1] for c in candles],dtype=float)
                     for i,key in enumerate(('open','high','low','close','volume'))}
        self._update_indicators()

    def _update_portfolio_plan(self,*args):
        if not self.dry_run: raise RuntimeError('影子程序必须保持 dry_run')
        close=float(self._ohlcv['close'][-1]);ema200=float(self._indicators['ema200'][-1])
        regime=classify_latest_regime(self._ohlcv['close'])
        equity=self._portfolio_equity(close)
        weight=self._paper_core_btc*close/equity if equity>0 else 0
        target=self.portfolio_config.core_weights[regime]
        # Apply only to an otherwise tradable core BUY, never to reductions or swaps.
        threshold=self.portfolio_config.rebalance_threshold
        if self.candidate and equity>0 and self._paper_cash>0 and target-weight>=threshold and not buy_buffer_allows(close,ema200,self.sold_before):
            self.events.append({'kind':'blocked_buy','candle_ts':self._candles[-1][0],
                'close':close,'ema200':ema200,'required_price':ema200*1.01 if math.isfinite(ema200) else None,
                'current_weight':weight,'target_weight':target,'regime':regime})
            return
        before=len(self._core_trade_log)
        super()._update_portfolio_plan(*args)
        if len(self._core_trade_log)>before:
            fill=self._core_trade_log[-1]
            if fill['side']=='sell':
                self.sold_before=True;self.last_sell={'candle_ts':fill['candle_time'],'price':fill['price']}
            elif self.last_sell:
                elapsed=fill['candle_time']-self.last_sell['candle_ts']
                self.events.append({'kind':'rebuy','candle_ts':fill['candle_time'],
                    'hours_after_sell':elapsed/3_600_000,'within_24h':0<elapsed<=86_400_000,
                    'buy_above_sell_pct':(fill['price']/self.last_sell['price']-1)*100,
                    'price':fill['price'],'sell_price':self.last_sell['price']})

    def _close_position(self,reason,price):
        # Mirrors paper branch only; original method writes production_trades.json.
        if not self.dry_run: raise RuntimeError('影子程序必须保持 dry_run')
        pos=self._position
        if not pos:return
        notional=pos.get('notional',self.initial_capital*self.leverage)
        pnl=notional*((price/pos['entry']-1) if pos['side']=='long' else (pos['entry']/price-1))
        self._total_pnl+=pnl;self._daily_pnl+=pnl
        self._consecutive_losses=self._consecutive_losses+1 if pnl<0 else 0
        self._last_trade_bar=len(self._candles);self._last_trade_candle_ts=self._candles[-1][0]
        self._position=None
        self._trade_log.append({'side':pos['side'],'entry':pos['entry'],'exit':price,
            'notional':round(notional,2),'pnl':round(pnl,2),
            'pnl_pct':round(pnl/max(self.initial_capital,1)*100,2),'reason':reason,
            'time':str(ReplayClock.current),'candle_time':self._last_trade_candle_ts,
            'entry_analysis':pos.get('entry_analysis'),'account_after':self._paper_performance_snapshot(price)})

    def dump(self):
        return {**{k:copy.deepcopy(getattr(self,k)) for k in ACCOUNT_FIELDS},
                'sold_before':self.sold_before,'last_sell':self.last_sell}

    def restore(self,data):
        for k in ACCOUNT_FIELDS:
            if k in data:setattr(self,k,copy.deepcopy(data[k]))
        self.sold_before=data.get('sold_before',False);self.last_sell=copy.deepcopy(data.get('last_sell'))


class ShadowExperiment:
    def __init__(self,path): self.path=Path(path)

    @classmethod
    def create(cls,path,cfg,source,candles,mark_price=None):
        validate_seed(source,cfg);valid_candles(candles)
        if len(candles)<221:raise ValueError('至少需要221根收盘K线预热')
        path=Path(path)
        if path.exists():raise ValueError('实验数据库已存在；请续跑，不重新初始化')
        bots=[]
        for candidate in (False,True):
            bot=ShadowBot(cfg,candidate);bot.set_history(candles)
            for key in ACCOUNT_FIELDS:
                source_key=key[1:]
                if key not in ('_trade_log','_core_trade_log','_last_signal') and source_key in source:
                    setattr(bot,key,copy.deepcopy(source[source_key]))
            previous_sells=[t for t in source.get('core_trade_log',[]) if t.get('side')=='sell']
            bot.sold_before=bool(previous_sells)
            if previous_sells:
                t=previous_sells[-1]
                bot.last_sell={'candle_ts':t.get('candle_time'),'price':t.get('price')}
                if not all(isinstance(v,(int,float)) for v in bot.last_sell.values()):bot.last_sell=None
            bots.append(bot)
        mark_price=candles[-1][4] if mark_price is None else float(mark_price)
        if not math.isfinite(mark_price) or mark_price<=0:raise ValueError('初始化估值价格无效')
        equity=bots[0]._portfolio_equity(mark_price)
        if not math.isfinite(equity) or equity<=0:raise ValueError('起始权益无效')
        path.parent.mkdir(parents=True,exist_ok=True)
        exp=cls(path);exp.cfg=copy.deepcopy(cfg);exp.bots=bots;exp.candles=copy.deepcopy(candles)
        exp.initial_equity=equity;exp.start_candle_ts=candles[-1][0]
        exp.started_at=str(datetime.now(timezone.utc));exp.initial_mark_price=mark_price
        exp.seed=copy.deepcopy(source);exp.config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        with exp.db() as db:
            db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
            db.execute('CREATE TABLE samples (candle_ts INTEGER PRIMARY KEY, price REAL, baseline REAL, candidate REAL, missed_upside REAL)')
            db.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, variant TEXT, candle_ts INTEGER, payload TEXT)')
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',(candles[-1][0],mark_price,equity,equity,0))
            exp.persist(db)
        return exp

    def db(self):return sqlite3.connect(self.path)

    def persist(self,db):
        payload={'version':1,'config':self.cfg,'config_hash':self.config_hash,'candles':self.candles,
                 'initial_equity':self.initial_equity,'start_candle_ts':self.start_candle_ts,
                 'started_at':self.started_at,'initial_mark_price':self.initial_mark_price,
                 'seed':self.seed,'bots':[b.dump() for b in self.bots]}
        db.execute('INSERT OR REPLACE INTO state VALUES (1,?)',(json.dumps(payload,allow_nan=False),))

    @classmethod
    def load(cls,path):
        exp=cls(path)
        with exp.db() as db:p=json.loads(db.execute('SELECT payload FROM state WHERE id=1').fetchone()[0])
        if p['version']!=1:raise ValueError('实验版本不兼容')
        exp.cfg=p['config'];validate_seed(p['seed'],exp.cfg)
        exp.config_hash=p['config_hash'];exp.initial_equity=p['initial_equity'];exp.start_candle_ts=p['start_candle_ts'];exp.seed=p['seed'];exp.started_at=p['started_at'];exp.initial_mark_price=p['initial_mark_price'];exp.candles=p['candles'];exp.bots=[]
        for candidate,data in zip((False,True),p['bots']):
            bot=ShadowBot(exp.cfg,candidate);bot.set_history(exp.candles);bot.restore(data);exp.bots.append(bot)
        return exp

    def process(self,candles,now_ms):
        valid_candles(candles)
        pending=[c for c in candles if c[0]>self.candles[-1][0] and c[0]+BAR<=now_ms]
        if not pending:return
        if pending[0][0]!=self.candles[-1][0]+BAR:raise ValueError('收盘行情缺口；暂停，不跳过')
        # Persist each candle and both accounts in one transaction. Reload after any failure.
        try:
            for candle in pending:
                previous_price=self.candles[-1][4]
                btc_gap=max(self.bots[0]._paper_core_btc-self.bots[1]._paper_core_btc,0)
                missed=btc_gap*max(candle[4]-previous_price,0)
                ReplayClock.current=datetime.fromtimestamp((candle[0]+BAR)/1000,timezone.utc)
                with patch.object(production,'datetime',ReplayClock):
                    for bot in self.bots:bot._process_closed_candle(copy.deepcopy(candle))
                self.candles.append(copy.deepcopy(candle))
                if missed>0:
                    self.bots[1].events.append({'kind':'underexposed_rise','candle_ts':candle[0],
                        'btc_gap_at_interval_start':btc_gap,'price_from':previous_price,'price_to':candle[4],
                        'foregone_gross_pnl':missed})
                with self.db() as db:
                    db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',
                        (candle[0],candle[4],self.bots[0]._portfolio_equity(candle[4]),self.bots[1]._portfolio_equity(candle[4]),missed))
                    for label,bot in zip(('baseline','candidate'),self.bots):
                        for event in bot.events:
                            db.execute('INSERT INTO events(variant,candle_ts,payload) VALUES (?,?,?)',(label,event['candle_ts'],json.dumps(event,allow_nan=False)))
                        bot.events=[]
                    self.persist(db)
        except Exception:
            restored=type(self).load(self.path);self.__dict__.update(restored.__dict__);raise

    def report(self):
        with self.db() as db:
            samples=db.execute('SELECT * FROM samples ORDER BY candle_ts').fetchall()
            events=[(v,json.loads(p)) for v,p in db.execute('SELECT variant,payload FROM events ORDER BY id')]
        result={'mode':'shadow_only','rule':'4h EMA200 + 1% after selling; core BUY only',
                'initial_equity':self.initial_equity,'started_at':self.started_at,'initial_mark_price':self.initial_mark_price,'start_candle_ts':self.start_candle_ts,
                'last_candle_ts':self.candles[-1][0],'processed_candles':len(samples)-1,
                'cost_basis':'与线上paper相同：现货扣0.1%配置费率；合约手续费/资金费未完整计入',
                'missed_upside_gross':sum(s[4] for s in samples),
                'missed_upside_definition':'各上涨4h区间，基线比候选多持有BTC的期初数量×上涨价差；未扣下降期间收益，不能当净损失'}
        for column,(label,bot) in enumerate(zip(('baseline','candidate'),self.bots),2):
            equity=np.array([s[column] for s in samples]);drawdown=equity/np.maximum.accumulate(equity)-1
            rebuys=[e for v,e in events if v==label and e['kind']=='rebuy' and e['within_24h']]
            result[label]={'equity':float(equity[-1]),'pnl_since_start':float(equity[-1]-self.initial_equity),
                'return_pct':float((equity[-1]/self.initial_equity-1)*100),
                'max_drawdown_pct':float(-drawdown.min()*100),'rebalance_count':len(bot._core_trade_log),
                'spot_fee':sum(t['fee'] for t in bot._core_trade_log),'quick_rebuy_24h':len(rebuys),
                'quick_rebuy_above_sell':sum(e['buy_above_sell_pct']>0 for e in rebuys),
                'blocked_buys':sum(v==label and e['kind']=='blocked_buy' for v,e in events),
                'core_btc':bot._paper_core_btc,'cash':bot._paper_cash,'contract_closes':len(bot._trade_log)}
        result['candidate_minus_baseline']=result['candidate']['equity']-result['baseline']['equity']
        return result
