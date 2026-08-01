"""PIT-safe ownership-structure features from 有価証券報告書「大株主の状況」.

Source: J-Quants `/v2/edinet/major-shareholders` (top-10 holders per annual report).
The PIT timestamp is the filing timestamp (SubDate/SubTime), never the fiscal
period end (PerEn), which is roughly three months earlier and not yet public.

The economic quantity is the *fixed* (non-floating) share of ownership. JPX総研
computes FFW as ``1 - 固定株比率`` from the same 有報 disclosure, so a fall in
fixed ownership is a mechanical increase in free float — the "policy-shareholding
unwind" leg of the pre-registered value-unlock family.

Nominee/custodian accounts (信託口, master trust, global custodians) are pooled
vehicles holding on behalf of many beneficial owners. They are *float*, not fixed
ownership, and must be excluded before summing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Pooled nominee vehicles. Matched case-insensitively against the holder name.
# These represent many beneficial owners and are counted as float, not fixed.
CUSTODIAN_PATTERNS = (
    "信託口", "マスタートラスト", "カストディ", "トラスティ",
    "資産管理サービス信託", "STATE STREET", "ステート・ストリート",
    "JPMORGAN", "J.P.MORGAN", "JPモルガン", "モルガン・スタンレー",
    "BANK OF NEW YORK", "BNYM", "ニューヨーク　メロン", "ニューヨークメロン",
    "NORTHERN TRUST", "ノーザン・トラスト", "BNP PARIBAS", "ビー・エヌ・ピー",
    "HSBC", "CITIBANK", "シティバンク", "SSBTC", "GOLDMAN", "ゴールドマン",
    "MERRILL", "メリルリンチ", "UBS", "CREDIT SUISSE", "BARCLAYS",
    "投信口", "年金信託", "退職給付信託",
)


def is_custodian(name: str) -> bool:
    """True when the holder is a pooled nominee/custodian account."""
    if not isinstance(name, str):
        return False
    upper = name.upper()
    return any(p.upper() in upper for p in CUSTODIAN_PATTERNS)


def load_filings(path: str | Path = "data/jp_ownership/filings.jsonl") -> pd.DataFrame:
    """Load raw collected filings, one row per (filing, holder)."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = rec.get("Code")
            if not code:
                continue
            for h in rec.get("Hldrs") or []:
                rows.append({
                    "symbol": str(code)[:4],
                    "doc_id": rec.get("DocId"),
                    "doc_type": str(rec.get("DocTypeCode")),
                    "sub_date": rec.get("SubDate"),
                    "sub_time": rec.get("SubTime"),
                    "period_end": rec.get("PerEn"),
                    "holder": h.get("HldrName"),
                    "rank": h.get("Rank"),
                    "shares": pd.to_numeric(h.get("ShsHeld"), errors="coerce"),
                    "ratio": pd.to_numeric(h.get("ShsRatio"), errors="coerce"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["sub_date"] = pd.to_datetime(df["sub_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df.dropna(subset=["sub_date", "ratio"])


def filing_panel(holders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holder rows into one row per filing with ownership structure."""
    h = holders.copy()
    h["custodian"] = h["holder"].map(is_custodian)
    h["fixed_part"] = np.where(h["custodian"], 0.0, h["ratio"])
    h["cust_part"] = np.where(h["custodian"], h["ratio"], 0.0)
    g = h.groupby(["symbol", "doc_id", "sub_date", "period_end"], sort=False)
    panel = g.agg(
        fixed_ratio=("fixed_part", "sum"),
        custodian_ratio=("cust_part", "sum"),
        top10_ratio=("ratio", "sum"),
        top1_ratio=("ratio", "max"),
        n_holders=("ratio", "size"),
        hhi=("ratio", lambda s: float(np.square(s).sum())),
    ).reset_index()
    # A later-filed report for the same fiscal period supersedes an earlier one
    # (訂正報告書). Keep the newest filing per (symbol, period_end).
    panel = panel.sort_values(["symbol", "period_end", "sub_date"])
    panel = panel.drop_duplicates(["symbol", "period_end"], keep="last")
    return panel.reset_index(drop=True)


def ownership_release_events(panel: pd.DataFrame, min_decline: float = 0.02) -> pd.DataFrame:
    """Year-over-year fall in fixed (non-custodian) top-10 ownership.

    Consecutive annual filings only: the prior fiscal period must end 300-450
    days before the current one, so a restated or skipped year never creates a
    multi-year jump that masquerades as a one-year unwind.
    """
    p = panel.sort_values(["symbol", "period_end"]).copy()
    g = p.groupby("symbol", sort=False)
    for col in ("fixed_ratio", "custodian_ratio", "top1_ratio", "top10_ratio", "hhi"):
        p[f"prev_{col}"] = g[col].shift(1)
    p["prev_period_end"] = g["period_end"].shift(1)
    gap = (p["period_end"] - p["prev_period_end"]).dt.days
    p["delta_fixed"] = p["fixed_ratio"] - p["prev_fixed_ratio"]
    p["delta_custodian"] = p["custodian_ratio"] - p["prev_custodian_ratio"]
    p["delta_top1"] = p["top1_ratio"] - p["prev_top1_ratio"]
    p["delta_hhi"] = p["hhi"] - p["prev_hhi"]
    keep = gap.between(300, 450) & p["delta_fixed"].le(-min_decline + 1e-12)
    events = p.loc[keep].copy()
    events["event_date"] = events["sub_date"]
    return events.reset_index(drop=True)
