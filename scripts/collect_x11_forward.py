#!/usr/bin/env python3
"""X11 sealed-forward candidate ledger (append-only, NO return computation).

Frozen spec: docs/PREREGISTER_OVERSOLD_X11_FORWARD.md. Each business day this
appends the current signal members (5d-return cross-z bottom 20% AND ivol20 at
or below the cross-sectional median, Y1e9 panel) with the signal date only.
Judgment happens once, in verify_x11_forward.py, on 2028-08-12+.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from scripts.oversold_sweep_harness import build, members_for

LEDGER = Path("data/jp_oversold_x11_forward/candidates.jsonl")


def seen_dates() -> set:
    if not LEDGER.exists():
        return set()
    return {json.loads(l)["signal_date"]
            for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    A = build(1e9)
    mem = members_for(A, {"dip": "z20", "ivol": "lo", "market": "none"})
    seen = seen_dates()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with LEDGER.open("a", encoding="utf-8") as fh:
        for day in mem.index[mem.any(axis=1)]:
            ds = str(pd.Timestamp(day).date())
            if ds < "2026-08-12" or ds in seen:      # seal start; catch-up safe
                continue
            syms = mem.columns[mem.loc[day]].tolist()
            fh.write(json.dumps({"signal_date": ds, "n": len(syms),
                                 "symbols": sorted(syms),
                                 "recorded_at": dt.datetime.now(
                                     dt.timezone.utc).isoformat()},
                                ensure_ascii=False) + "\n")
            appended += 1
    print(json.dumps({"appended_days": appended, "ledger": str(LEDGER)}))


if __name__ == "__main__":
    main()
