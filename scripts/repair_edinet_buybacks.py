#!/usr/bin/env python3
"""Repair append-only EDINET rows whose saved payload was a JSON 429 response."""
from __future__ import annotations
import hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from data.collectors.config import EDINET_API_KEY
from trading.jp_intraday.edinet_buyback import parse_ixbrl_zip

BASE="https://api.edinet-fsa.go.jp/api/v2"; ROOT=Path("data/jp_buybacks/edinet")
RAW=ROOT/"raw"; ERR=ROOT/"error_payloads"; DOCS=ROOT/"documents.jsonl"

def fetch(doc_id):
    for i in range(10):
        r=requests.get(f"{BASE}/documents/{doc_id}",params={"type":1,"Subscription-Key":EDINET_API_KEY},timeout=30)
        if r.status_code==200 and r.content.startswith(b"PK"):return r.content
        time.sleep(min(2**i,60))
    raise RuntimeError(f"ZIP retry exhausted: {doc_id}")

def main():
    rows=[json.loads(x) for x in DOCS.read_text(encoding="utf-8").splitlines() if x.strip()]
    latest={r["doc_id"]:r for r in rows}
    bad=[r for r in latest.values() if str(r.get("parse_error","")).startswith("BadZipFile")]
    ERR.mkdir(exist_ok=True); print(f"repair_pending={len(bad)}",flush=True)
    def repair(old):
        doc_id=old["doc_id"]; path=RAW/f"{doc_id}.zip"; invalid=path.read_bytes()
        err=ERR/f"{doc_id}_{hashlib.sha256(invalid).hexdigest()[:12]}.json"
        if not err.exists(): err.write_bytes(invalid)
        content=fetch(doc_id)
        try: terms=parse_ixbrl_zip(content); parse_error=None
        except Exception as exc: terms={}; parse_error=f"{type(exc).__name__}: {exc}"
        # Valid official ZIP replaces the mislabeled .zip path; 429 body remains in ERR.
        path.write_bytes(content)
        row=dict(old); row.update({"raw_sha256":hashlib.sha256(content).hexdigest(),"parse_error":parse_error,
          "terms":terms,"record_revision":int(old.get("record_revision",0))+1,
          "repair_of":"HTTP200_BODY_STATUS429","error_payload":str(err)})
        return row
    done=0
    for start in range(0,len(bad),30):
        with ThreadPoolExecutor(max_workers=3) as pool: repaired=list(pool.map(repair,bad[start:start+30]))
        with DOCS.open("a",encoding="utf-8") as fh:
            for row in repaired:fh.write(json.dumps(row,ensure_ascii=False)+"\n")
        done+=len(repaired); print(f"{done}/{len(bad)} repaired",flush=True); time.sleep(.5)

if __name__=="__main__":main()
