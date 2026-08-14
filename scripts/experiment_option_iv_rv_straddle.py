#!/usr/bin/env python3
"""Weekly 5-session Nikkei-225 ATM straddle IV-RV diagnostic."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from trading.jp_intraday.daily_model import annualized_stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/jp_option_iv_rv_straddle'
HOLD=5; EVAL=pd.Timestamp('2024-01-01')

def load_options():
    d=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT/'data/jp_options/opt225_*.parquet')))],ignore_index=True)
    d['Date']=pd.to_datetime(d.Date); d['Code']=d.Code.astype(str)
    for c in ['Settle','Vo','OI','Strike','UnderPx','IV','dte']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.drop_duplicates(['Date','Code'])

def build_trades(d):
    sessions=pd.Index(sorted(d.Date.unique())); spos=pd.Series(range(len(sessions)),index=sessions)
    spot=d.groupby('Date').UnderPx.median().sort_index(); rv=(spot.pct_change().rolling(20).std()*np.sqrt(252)).shift(1)
    iv=pd.read_parquet(ROOT/'data/jp_options/iv_daily.parquet').set_index('date').atm_iv30.reindex(sessions)
    spread=iv-rv; z=(spread-spread.rolling(252,min_periods=126).mean())/spread.rolling(252,min_periods=126).std()
    rows=[]; last_exit=-1
    for date in sessions:
        pos=spos[date]
        if pos<=last_exit or pos+HOLD>=len(sessions) or date.weekday()>2: continue
        # first available session of ISO week only
        if pos>0 and sessions[pos-1].isocalendar()[:2]==date.isocalendar()[:2]: continue
        signal=z.loc[date]
        if not np.isfinite(signal) or abs(signal)<.5: continue
        x=d[(d.Date==date)&d.dte.between(20,45)&(d.Vo>0)&(d.OI>=100)&(d.Settle>0)].copy()
        if x.empty: continue
        x['atm_dist']=(x.Strike-x.UnderPx).abs()/x.UnderPx
        pairs=x.groupby(['CM','LTD','Strike']).filter(lambda g:g.PCDiv.astype(str).nunique()>=2)
        if pairs.empty: continue
        key=pairs.groupby(['CM','LTD','Strike']).atm_dist.mean().idxmin()
        legs=pairs.set_index(['CM','LTD','Strike']).loc[key]
        if isinstance(legs,pd.Series): continue
        legs=legs.sort_values('PCDiv').drop_duplicates('PCDiv').head(2)
        exit_date=sessions[pos+HOLD]
        ex=d[(d.Date==exit_date)&d.Code.isin(legs.Code)&(d.Vo>0)&(d.Settle>0)]
        if len(ex)!=2: continue
        entry=float(legs.Settle.sum()); exitv=float(ex.Settle.sum())
        side=-1 if signal>=.5 else 1
        ret=side*(exitv-entry)/entry
        rows.append({'date':exit_date,'entry_date':date,'side':'SHORT' if side<0 else 'LONG','z':signal,
                     'entry_premium':entry,'exit_premium':exitv,'gross':ret,'net':ret})
        last_exit=pos+HOLD
    return pd.DataFrame(rows)

def main():
    t=build_trades(load_options()); ev=t[t.date>=EVAL].copy()
    # Returns occur only on exit dates; include zero non-exit sessions for portfolio statistics.
    cal=pd.date_range(EVAL,ev.date.max(),freq='B'); daily=pd.DataFrame({'date':cal}).merge(ev[['date','gross']],how='left').fillna({'gross':0})
    stats=annualized_stats(daily,'gross')
    yearly={str(y):annualized_stats(g,'gross') for y,g in daily.groupby(daily.date.dt.year)}
    summary={'spec':{'hold_sessions':HOLD,'z_threshold':.5,'dte':[20,45]},'evaluation':stats,'yearly':yearly,
             'trades':int(len(ev)),'shorts':int((ev.side=='SHORT').sum()),'longs':int((ev.side=='LONG').sum()),
             'avg_gross_per_trade':float(ev.gross.mean()),'breakeven_roundtrip_cost_pct_of_premium':float(ev.gross.sum()/len(ev)) if len(ev) else np.nan,
             'decision':'PAPER ONLY' if stats['sharpe']>=1 and all(v['ann_return']>0 for v in yearly.values()) else 'NO-GO'}
    OUT.mkdir(parents=True,exist_ok=True); t.to_csv(OUT/'trades.csv',index=False); daily.to_csv(OUT/'daily.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
