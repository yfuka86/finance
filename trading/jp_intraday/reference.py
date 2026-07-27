from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd


def parse_current_topix(path: str | Path) -> pd.DataFrame:
    """Convert JPX's current TOPIX weight CSV to a dated membership snapshot."""
    raw = pd.read_csv(path, encoding="cp932", dtype={"コード": str})
    as_of = pd.to_datetime(raw["日付"].astype(str), format="%Y%m%d", errors="coerce")
    valid = as_of.notna() & raw["コード"].astype(str).str.fullmatch(r"[0-9A-Z]{4,5}")
    raw, as_of = raw.loc[valid], as_of.loc[valid]
    return pd.DataFrame({
        "symbol": raw["コード"].str.strip(),
        "effective_from": as_of,
        "effective_to": pd.NaT,
    }).drop_duplicates("symbol")


def extract_share_snapshots(cache_dir: str | Path) -> pd.DataFrame:
    """Extract shares using disclosure date as known-at date from cached J-Quants fins."""
    rows = []
    for path in sorted(Path(cache_dir).glob("fins_*.json*")):
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = []
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            for value in payload.values():
                records.extend(value if isinstance(value, list) else [value])
        for row in records:
            if not isinstance(row, dict):
                continue
            issued = pd.to_numeric(row.get("ShOutFY"), errors="coerce")
            treasury = pd.to_numeric(row.get("TrShFY"), errors="coerce")
            if pd.notna(issued) and issued > 0 and row.get("DiscDate") and row.get("Code"):
                rows.append({
                    "symbol": str(row["Code"])[:4],
                    "known_at": row["DiscDate"],
                    "shares": float(issued - (treasury if pd.notna(treasury) else 0)),
                })
    if not rows:
        raise ValueError("no disclosed share counts found in fins cache")
    result = pd.DataFrame(rows)
    result["known_at"] = pd.to_datetime(result["known_at"])
    return result.drop_duplicates(["symbol", "known_at"], keep="last").sort_values(
        ["symbol", "known_at"]
    ).reset_index(drop=True)
