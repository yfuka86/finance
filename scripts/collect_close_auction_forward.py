#!/usr/bin/env python3
"""Close-auction overnight forward-seal candidate ledger (append-only, no P&L).

Frozen spec: docs/PREREGISTER_CLOSE_AUCTION_FORWARD.md. Each new minute-data day
appends the long-only top-decile of the frozen ridge prediction (signal date +
symbols only). NO returns computed. Judgment happens once in
verify_close_auction_forward.py on 2028-08-14+.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from scripts.experiment_close_auction_overnight import attach_target, ridge_wf

LEDGER = Path("data/jp_close_auction_overnight/forward_candidates.jsonl")
SEAL_START = "2026-08-12"


def seen() -> set:
    if not LEDGER.exists():
        return set()
    return {json.loads(l)["signal_date"]
            for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    m = attach_target()
    pred = ridge_wf(m)
    fr = m.loc[pred.index].assign(pred=pred)
    fr["_s"] = fr["pred"] - fr.groupby("date")["pred"].transform("mean")
    thr = fr.groupby("date")["_s"].transform("quantile", .9)
    book = fr[fr["_s"] >= thr]
    done = seen()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with LEDGER.open("a", encoding="utf-8") as fh:
        for day, g in book.groupby("date"):
            ds = str(pd.Timestamp(day).date())
            if ds < SEAL_START or ds in done:
                continue
            fh.write(json.dumps({
                "signal_date": ds, "n": int(len(g)),
                "symbols": sorted(g["symbol"].astype(str).tolist()),
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                ensure_ascii=False) + "\n")
            appended += 1
    print(json.dumps({"appended_days": appended, "ledger": str(LEDGER)}))


if __name__ == "__main__":
    main()
