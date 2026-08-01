#!/usr/bin/env python3
"""Resumable append-only EDINET type 220/230 collector and parser."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from data.collectors.config import EDINET_API_KEY
from trading.jp_intraday.edinet_buyback import parse_ixbrl_zip

BASE="https://api.edinet-fsa.go.jp/api/v2"; ROOT=Path("data/jp_buybacks/edinet")
RAW=ROOT/"raw"; SCANS=ROOT/"scanned_dates.jsonl"; DOCS=ROOT/"documents.jsonl"

def _get(url,params,tries=8,expect_zip=False):
    for i in range(tries):
        try:
            r=requests.get(url,params=params,timeout=30)
            if r.status_code==200 and (not expect_zip or r.content.startswith(b"PK")):return r
            if r.status_code!=429:r.raise_for_status()
        except requests.RequestException:
            if i==tries-1:raise
        time.sleep(min(2**i,30))
    raise RuntimeError("EDINET retry exhausted")

def _existing(path,key):
    if not path.exists():return set()
    return {json.loads(line)[key] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",required=True); ap.add_argument("--end",required=True); args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True); RAW.mkdir(exist_ok=True)
    scanned=_existing(SCANS,"date"); known=_existing(DOCS,"doc_id")
    dates=[]; cur=dt.date.fromisoformat(args.start); end=dt.date.fromisoformat(args.end)
    while cur<=end:
        if cur.weekday()<5 and cur.isoformat() not in scanned:dates.append(cur.isoformat())
        cur+=dt.timedelta(days=1)
    print(f"dates_pending={len(dates)} known_docs={len(known)}",flush=True)
    for n,ds in enumerate(dates,1):
        listing=_get(f"{BASE}/documents.json",{"date":ds,"type":2,"Subscription-Key":EDINET_API_KEY}).json().get("results",[])
        hits=[d for d in listing if str(d.get("docTypeCode")) in ("220","230")]
        def process(d):
            doc_id=d["docID"]
            if doc_id in known:return None
            raw_path=RAW/f"{doc_id}.zip"
            if raw_path.exists(): content=raw_path.read_bytes()
            else:
                content=_get(f"{BASE}/documents/{doc_id}",{"type":1,"Subscription-Key":EDINET_API_KEY},expect_zip=True).content
                with raw_path.open("xb") as fh:fh.write(content)
            try: terms=parse_ixbrl_zip(content); parse_error=None
            except Exception as exc: terms={}; parse_error=f"{type(exc).__name__}: {exc}"
            return {"doc_id":doc_id,"doc_type":str(d.get("docTypeCode")),"submit_datetime":d.get("submitDateTime"),
                 "sec_code":str(d.get("secCode") or "")[:4],"edinet_code":d.get("edinetCode"),
                 "filer_name":d.get("filerName"),"description":d.get("docDescription"),
                 "withdrawal_status":d.get("withdrawalStatus"),"raw_file":str(raw_path),
                 "raw_sha256":hashlib.sha256(content).hexdigest(),"parse_error":parse_error,"terms":terms}
        with ThreadPoolExecutor(max_workers=3) as pool:
            parsed=list(pool.map(process,hits))
        for row in parsed:
            if row is None:continue
            with DOCS.open("a",encoding="utf-8") as fh:fh.write(json.dumps(row,ensure_ascii=False)+"\n")
            doc_id=row["doc_id"]
            known.add(doc_id)
        with SCANS.open("a",encoding="utf-8") as fh:fh.write(json.dumps({"date":ds,"documents":len(hits)})+"\n")
        if n%50==0 or n==len(dates):print(f"{n}/{len(dates)} dates docs={len(known)}",flush=True)
        time.sleep(.10)

if __name__=="__main__":main()
