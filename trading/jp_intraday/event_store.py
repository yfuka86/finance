"""Append-only PIT corporate-event store shared by buyback/dilution/TOB research."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd

REQUIRED = (
    "event_id", "security_code", "event_family", "source", "source_published_at",
    "first_received_at", "effective_at", "revision_no", "document_hash", "state",
    "terms_json", "first_tradable_at",
)

def validate_event(event: dict) -> dict:
    missing=[k for k in REQUIRED if k not in event]
    if missing: raise ValueError(f"missing event fields: {missing}")
    out={k:event[k] for k in REQUIRED}
    out["security_code"]=str(out["security_code"])
    out["revision_no"]=int(out["revision_no"])
    for col in ("source_published_at","first_received_at","effective_at","first_tradable_at"):
        ts=pd.Timestamp(out[col])
        if ts.tzinfo is None: raise ValueError(f"{col} must be timezone-aware")
        out[col]=ts.isoformat()
    if pd.Timestamp(out["first_received_at"]) < pd.Timestamp(out["source_published_at"]):
        raise ValueError("first_received_at precedes publication")
    if pd.Timestamp(out["first_tradable_at"]) < pd.Timestamp(out["first_received_at"]):
        raise ValueError("first_tradable_at precedes receipt")
    if not isinstance(out["terms_json"], (dict,list)): raise ValueError("terms_json must be structured")
    return out

def append_events(path: str|Path, events: Iterable[dict]) -> int:
    """Append revisions without permitting an existing (event_id, revision_no) overwrite."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    existing=set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row=json.loads(line); existing.add((row["event_id"],int(row["revision_no"])))
    clean=[]
    for event in events:
        row=validate_event(event); key=(row["event_id"],row["revision_no"])
        if key in existing: raise ValueError(f"append-only duplicate: {key}")
        existing.add(key); clean.append(row)
    with path.open("a",encoding="utf-8") as fh:
        for row in clean: fh.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    return len(clean)
