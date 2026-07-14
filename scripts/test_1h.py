#!/usr/bin/env python3
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from datetime import datetime, timezone
from src.data.downloader import OKXDataDownloader
from backtesting import Strategy, lib
import numpy as np, warnings; warnings.filterwarnings('ignore')

def _rsi(a,p=14):
    r=np.full_like(a,np.nan,float)
    if len(a)<p+1: return r
    d=np.diff(a); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.mean(g[:p]); al=np.mean(l[:p]); a_=1/p
    for i in range(p,len(d)):
        r[i+1]=100-(100/(1+ag/al))if al>0 else(100 if ag>0 else 50)
        ag=a_*g[i]+(1-a_)*ag; al=a_*l[i]+(1-a_)*al
    return np.clip(r,0,100)
def _atr(h,l,c,p=14):
    r=np.full_like(c,np.nan,float)
    if len(c)<p+1: return r
    pc=np.roll(c,1); pc[0]=c[0]; tr=np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
    for i in range(p-1,len(tr)): r[i]=np.mean(tr[i-p+1:i+1])
    return r
def _ema(a,s):
    if len(a)<s: return np.full_like(a,np.nan,float)
    alpha=2/(s+1); r=np.full_like(a,np.nan,float); r[s-1]=np.mean(a[:s])
    for i in range(s,len(a)): r[i]=alpha*a[i]+(1-alpha)*r[i-1]
    return r
def _adx(h,l,c,p=14):
    n=len(c); r=np.full(n,np.nan,float)
    if n<p*2: return r
    pc=np.roll(c,1); pc[0]=c[0]; tr=np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
    um=h-np.roll(h,1); dm=np.roll(l,1)-l; um[0]=dm[0]=0
    pdm=np.where((um>dm)&(um>0),um,0); ndm=np.where((dm>um)&(dm>0),dm,0)
    a=1/p; at=np.full(n,np.nan); pt=np.full(n,np.nan); nt=np.full(n,np.nan)
    at[p-1]=np.mean(tr[:p]); pt[p-1]=np.mean(pdm[:p]); nt[p-1]=np.mean(ndm[:p])
    for i in range(p,n):
        at[i]=a*tr[i]+(1-a)*at[i-1]; pt[i]=a*pdm[i]+(1-a)*pt[i-1]; nt[i]=a*ndm[i]+(1-a)*nt[i-1]
    pdi=np.where(at>0,100*pt/at,0); ndi=np.where(at>0,100*nt/at,0)
    ds=pdi+ndi; dx=np.where(ds>0,100*np.abs(pdi-ndi)/ds,0); r[p*2-2]=np.mean(dx[p-1:p*2-1])
    for i in range(p*2-1,n): r[i]=a*dx[i]+(1-a)*r[i-1]
    return r

class H1(Strategy):
    risk=0.08; tp=3.0; sl=2.0; cool=6
    def init(self):
        c,h,l=self.data.Close,self.data.High,self.data.Low
        self.rsi=self.I(_rsi,c,14,name='r')
        self.atr=self.I(_atr,h,l,c,14,name='a')
        self.ef=self.I(_ema,c,20,name='e20')
        self.em=self.I(_ema,c,50,name='e50')
        self.adx=self.I(_adx,h,l,c,14,name='x')
        self._since=999; self._hi=0; self._lo=9e9
    def next(self):
        c=self.data.Close[-1]; h=self.data.High[-1]; l=self.data.Low[-1]
        atr=self.atr[-1]; rsi=self.rsi[-1]; ef=self.ef[-1]; em=self.em[-1]; adx=self.adx[-1]
        if any(np.isnan(x) for x in[atr,rsi,ef,em,adx])or c<=0: return
        self._since+=1
        if self.position:
            if self.position.is_long:
                if h>self._hi: self._hi=h; self._ts=h-atr*self.sl
                if l<=self._ts or (ef<em and rsi>60): self.position.close();self._since=0
            else:
                if l<self._lo: self._lo=l; self._ts=l+atr*self.sl
                if h>=self._ts or (ef>em and rsi<40): self.position.close();self._since=0
            return
        if self._since<self.cool or adx<18: return
        long_ok = ef>em and 25<rsi<45 and c<ef+atr*1.0
        short_ok = ef<em and 55<rsi<75 and c>ef-atr*1.0
        if long_ok:
            sl=c-atr*self.sl; tp=c+atr*self.tp
            f=min(self.risk/(self.sl*atr/c),0.95)
            if f>0: self.buy(size=f,sl=sl,tp=tp); self._hi=h; self._ts=sl; self._since=999
        elif short_ok:
            sl=c+atr*self.sl; tp=c-atr*self.tp
            f=min(self.risk/(self.sl*atr/c),0.95)
            if f>0: self.sell(size=f,sl=sl,tp=tp); self._lo=l; self._ts=sl; self._since=999

d = OKXDataDownloader(data_dir='data/raw', market_type='swap')
df = d.load('BTC/USDT', '1h', datetime(2023,7,1,tzinfo=timezone.utc), datetime(2026,7,1,tzinfo=timezone.utc))
split = int(len(df)*0.7); test = df.iloc[split:]
months = len(test)/(30.44*24)

print(f'BTC 1H: {len(test)}根, {months:.0f}月')
print(f'TP/SL      Ret%     DD%   Win%  Trades   Mon%   FinalU  MoTr')
print('-'*65)
for tp, sl in [(3.0,1.5), (3.5,1.5), (4.0,1.5), (2.5,1.0)]:
    class T(H1): tp=tp; sl=sl
    bt = lib.FractionalBacktest(data=test.set_index('timestamp'), strategy=T, cash=1000, commission=0.0005)
    s = bt.run()
    ret=s['Return [%]']; dd=s['Max. Drawdown [%]']; wr=s.get('Win Rate [%]',0); tr=s['# Trades']
    m=(1+ret/100)**(1/months)-1 if ret>-100 else -1
    mt=tr/months; f=1000*(1+ret/100)
    ok = 'GOOD' if dd>-30 and m*100>=5 else ''
    print(f'TP{tp}/SL{sl:<4} {ret:>7.1f} {dd:>7.1f} {wr:>6.0f} {tr:>6} {m*100:>7.1f} {f:>8.0f} {mt:>4.0f}  {ok}')
