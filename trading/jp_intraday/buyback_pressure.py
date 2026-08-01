"""PIT normalization and pressure features for share-buyback programs."""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd

DISCLOSURE_STATE={"start":"announced","status":"active","complete":"completed",
                  "correction":"corrected","alteration":"altered","cancellation":"cancelled"}

def normalize_tdnet_record(row: dict, received_at, first_tradable_at) -> dict:
    published=pd.Timestamp(f"{row['DisclosedDate']} {row.get('DisclosedTime','00:00:00')}",tz="Asia/Tokyo")
    stable=json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)
    event_id=f"TDNET_BUYBACK:{str(row['Code'])}:{row.get('BoardMeetingDate') or row['DisclosedDate']}"
    dtype=str(row.get("DisclosureType","others"))
    terms={k:row.get(k) for k in (
        "PurchasingMethod","BoardMeetingDate","MaximumSharesToBeAcquired",
        "MaximumTotalAcquisitionCost","ApprovedAcquisitionStartDate","ApprovedAcquisitionEndDate",
        "CumulativeNumberOfSharesPurchased","CumulativeTotalPurchasePrice","PurchasePeriodStartDate",
        "PurchasePeriodEndDate","PurchasePricePerShare","PurchaseDate","NumberOfSharesPurchase",
        "TotalPurchasePrice")}
    return {"event_id":event_id,"security_code":str(row["Code"]),"event_family":"buyback",
            "source":"TDnet","source_published_at":published,"first_received_at":pd.Timestamp(received_at),
            "effective_at":published,"revision_no":int(row.get("RevisionNo",0)),
            "document_hash":hashlib.sha256(stable.encode()).hexdigest(),
            "state":DISCLOSURE_STATE.get(dtype,dtype),"terms_json":terms,
            "first_tradable_at":pd.Timestamp(first_tradable_at)}

def pressure_features(programs: pd.DataFrame) -> pd.DataFrame:
    """Compute only preregistered features; caller supplies PIT-safe prices/session counts."""
    p=programs.copy()
    numeric=("max_shares","max_yen","cumulative_shares","cumulative_yen","prior_close",
             "remaining_sessions","elapsed_sessions","total_sessions","adv20_shares")
    for c in numeric: p[c]=pd.to_numeric(p[c],errors="coerce")
    p["remaining_shares"]=(p.max_shares-p.cumulative_shares).clip(lower=0)
    p["remaining_yen"]=(p.max_yen-p.cumulative_yen).clip(lower=0)
    by_yen=p.remaining_yen/p.prior_close.replace(0,np.nan)
    p["remaining_capacity_shares"]=np.minimum(p.remaining_shares,by_yen)
    denom=p.remaining_sessions*p.adv20_shares
    p["remaining_pressure"]=p.remaining_capacity_shares/denom.replace(0,np.nan)
    planned=p.max_shares*p.elapsed_sessions/p.total_sessions.replace(0,np.nan)
    p["pace_surprise"]=p.cumulative_shares/planned.replace(0,np.nan)-1
    return p
