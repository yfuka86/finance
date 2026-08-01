#!/usr/bin/env python3
"""Fill canonical yearly O/C/Vo fields by full-market date, without overwrites."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import jquantsapi
import numpy as np
import pandas as pd
from data.collectors.config import JQUANTS_API_KEY

ROOT=Path("data/jp_daily_history"); MIN_ROWS=3000

def retry(fn,tries=8):
    for i in range(tries):
        try:return fn()
        except Exception:
            if i==tries-1:raise
            time.sleep(min(2**i,60))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--start",required=True);ap.add_argument("--end",required=True);args=ap.parse_args()
    years={y:pd.read_parquet(ROOT/f"daily_adj_{y}.parquet") for y in range(pd.Timestamp(args.start).year,pd.Timestamp(args.end).year+1)}
    for f in years.values():
        f["Date"]=pd.to_datetime(f.Date);f["Code"]=f.Code.astype(str)
        for c in ("O","C","Vo"):
            if c not in f:f[c]=np.nan
    dates=[]
    for f in years.values():
        q=f[f.Date.between(args.start,args.end)].groupby("Date")[["O","C","Vo"]].apply(lambda x:x.notna().all(axis=1).mean())
        dates.extend(q[q<.99].index.tolist())
    dates=sorted(set(dates));print(f"dates_pending={len(dates)}",flush=True)
    client=jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    for n,date in enumerate(dates,1):
        raw=retry(lambda date=date:client.get_eq_bars_daily(date_yyyymmdd=date.strftime("%Y%m%d")))
        if len(raw)>=MIN_ROWS:
            raw=raw[["Date","Code","O","C","Vo"]].copy();raw.Date=pd.to_datetime(raw.Date);raw.Code=raw.Code.astype(str)
            y=date.year; idx=years[y].set_index(["Date","Code"]); src=raw.set_index(["Date","Code"])
            for c in ("O","C","Vo"):idx[c]=idx[c].fillna(src[c])
            years[y]=idx.reset_index()
        time.sleep(.12)
        if n%25==0 or n==len(dates):
            for y,f in years.items():f.to_parquet(ROOT/f"daily_adj_{y}.parquet",index=False)
            print(f"{n}/{len(dates)} flushed",flush=True)

if __name__=="__main__":main()
