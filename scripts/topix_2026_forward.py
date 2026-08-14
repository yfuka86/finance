#!/usr/bin/env python3
"""Create append-only raw snapshots for the TOPIX October-2026 forward study."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import pandas as pd
from trading.jp_intraday.daily_gap import load_existing_daily

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/topix_2026_forward/raw'
WEIGHTS=ROOT/'data/jp_intraday_reference/topixweight_current.csv'

def merge_current_weights(day,w):
    day=day.copy();w=w.copy()
    day['symbol4']=day.Code.astype(str).str[:4];w['symbol4']=w.Code.astype(str).str[:4]
    out=day.merge(w[['symbol4','topix_weight','ニューインデックス区分']],on='symbol4',how='left')
    out['current_topix_member']=out.topix_weight.notna()
    return out.drop(columns='symbol4')

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def build(asof: pd.Timestamp):
    daily=load_existing_daily(); daily['Date']=pd.to_datetime(daily.Date)
    day=daily[daily.Date.eq(asof)].copy()
    if day.empty: raise SystemExit(f'{asof.date()} の日次データがありません')
    cols=[c for c in ['Date','Code','AdjO','AdjH','AdjL','AdjC','raw_open','raw_close','raw_volume','Va'] if c in day]
    day=day[cols].copy(); day['Code']=day.Code.astype(str)
    w=pd.read_csv(WEIGHTS,encoding='cp932',dtype={'コード':str})
    w=w.rename(columns={'コード':'Code','TOPIXに占める個別銘柄のウエイト':'topix_weight'})
    w['topix_weight']=pd.to_numeric(w.topix_weight.astype(str).str.rstrip('%'),errors='coerce')/100
    day=merge_current_weights(day,w)
    return day.sort_values('Code')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--asof',required=True); args=ap.parse_args()
    asof=pd.Timestamp(args.asof).normalize(); out=OUT/f'topix2026_{asof:%Y%m%d}.parquet'
    if out.exists(): raise SystemExit(f'append-only: {out} は既に存在します')
    frame=build(asof); OUT.mkdir(parents=True,exist_ok=True); frame.to_parquet(out,index=False)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception: commit='unknown'
    manifest={'asof':str(asof.date()),'rows':len(frame),'file':str(out.relative_to(ROOT)),
              'sha256':sha256(out),'weight_sha256':sha256(WEIGHTS),'git_commit':commit,
              'official_ffw_present':False,
              'current_topix_members':int(frame.current_topix_member.sum()),
              'raw_price_coverage':float(frame.raw_close.notna().mean()) if 'raw_close' in frame else 0.0,
              'raw_volume_coverage':float(frame.raw_volume.notna().mean()) if 'raw_volume' in frame else 0.0,
              'warning':'FFW未取得。選定予測・売買判定には使用禁止。'}
    m=out.with_suffix('.manifest.json'); m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
