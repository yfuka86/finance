#!/usr/bin/env python3
"""Backfill raw O/C for value-event symbols into canonical yearly bars.

Runs solo to respect J-Quants rate limits. Existing values are never replaced;
only missing O/C cells are filled. Each batch is flushed to make the job resumable.
"""
from __future__ import annotations

import time
from pathlib import Path

import jquantsapi
import numpy as np
import pandas as pd

from data.collectors.config import JQUANTS_API_KEY
from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.value_event_model import (
    dividend_raise_events, dividend_resumption_events, treasury_cancellation_events,
)

ROOT=Path("data/jp_daily_history")
START,END="20210101","20260801"


def _retry(fn,tries=8):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(min(2**i,60))


def main():
    fins=load_fins()
    events=pd.concat([dividend_raise_events(fins),dividend_resumption_events(fins),
                      treasury_cancellation_events(fins)],ignore_index=True,sort=False)
    # Price-independent preregistered gates first. This cuts API calls without
    # looking at outcomes or approximating PBR with adjusted prices.
    events=events[events.BPS.gt(0)&events.EPS.gt(0)&events.OP.gt(0)].copy()
    daily=load_existing_daily().rename(columns={"Date":"date","Code":"code","Va":"value"})
    daily["date"]=pd.to_datetime(daily.date); daily["symbol"]=daily.code.astype(str).str[:4]
    daily=daily.sort_values(["date","symbol"])[["date","symbol","value"]]
    events=pd.merge_asof(events.sort_values("event_date"),daily.sort_values("date"),
                         left_on="event_date",right_on="date",by="symbol",
                         direction="backward",allow_exact_matches=False)
    symbols=sorted(events.loc[events.value.ge(1e9),"symbol"].unique())
    years={y:pd.read_parquet(ROOT/f"daily_adj_{y}.parquet") for y in range(2021,2027)}
    for frame in years.values():
        for col in ("O","C"):
            if col not in frame: frame[col]=np.nan
        frame["Code"]=frame.Code.astype(str)
        frame["Date"]=pd.to_datetime(frame.Date)
    # A symbol is complete when every canonical row for it has both raw fields.
    pending=[]
    for symbol in symbols:
        code=symbol+"0" if len(symbol)==4 else symbol
        chunks=[f.loc[f.Code.str[:4].eq(symbol),["O","C"]] for f in years.values()]
        present=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
        if present.empty or present[["O","C"]].isna().any(axis=None): pending.append((symbol,code))
    print(f"event_symbols={len(symbols)} pending={len(pending)}",flush=True)
    client=jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    for n,(symbol,code) in enumerate(pending,1):
        raw=_retry(lambda code=code: client.get_eq_bars_daily(
            code=code,from_yyyymmdd=START,to_yyyymmdd=END))
        if not raw.empty:
            raw["Date"]=pd.to_datetime(raw.Date); raw["Code"]=raw.Code.astype(str)
            raw=raw[["Date","Code","O","C"]].drop_duplicates(["Date","Code"])
            for y,frame in years.items():
                part=raw[raw.Date.dt.year.eq(y)].set_index(["Date","Code"])
                if part.empty: continue
                idx=frame.set_index(["Date","Code"])
                for col in ("O","C"):
                    idx[col]=idx[col].fillna(part[col])
                years[y]=idx.reset_index()
        time.sleep(.12)
        if n%50==0 or n==len(pending):
            for y,frame in years.items():
                frame.to_parquet(ROOT/f"daily_adj_{y}.parquet",index=False)
            print(f"{n}/{len(pending)} flushed",flush=True)


if __name__=="__main__": main()
