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
            # Code "00000" = 非上場の有報提出者（ゴルフ場会員権会社等）。捨てないと
            # 別会社が同一 symbol に潰れ、YoY差分が**企業をまたいで**計算される。
            if not code or str(code)[:4] == "0000":
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


OWNERSHIP_FEATURES_DAILY = [
    "own_fixed_z",        # 固定株（非カストディアン上位10名）比率の断面z
    "own_custodian_z",    # プール型名義比率の断面z（機関投資家の浮動株プロキシ）
    "own_delta_fixed_z",  # 直近開示時点の固定株比率YoY変化の断面z（持合い解消）
    "own_top1_z",         # 筆頭株主比率の断面z（支配集中）
    "own_hhi_z",          # 上位10名HHIの断面z
]


def _cross_sectional_z(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date z-score. 0 then means 'the average stock on that day'."""
    g = frame.groupby("date")
    for c in cols:
        m, s = g[c].transform("mean"), g[c].transform("std")
        frame[c] = ((frame[c] - m) / s.replace(0, np.nan)).clip(-5, 5)
    return frame


def attach_ownership_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach slow-moving ownership structure as PIT-safe daily cross-sectional features.

    有報は年1回なので、開示の**翌営業日**から次の開示まで前方補完する。提出時刻に
    依らず翌営業日起点にするのは保守側。断面zにしてから欠損を0で埋めるので、
    未開示・新規上場は「その日の平均的な銘柄」として扱われ、中核特徴量の
    dropna を緩めずに済む（AGENTS の疎特徴量の罠を回避）。
    """
    p = panel.copy()
    # パネルの symbol は J-Quants 5桁（"13010"）、大株主側は4桁（"1301"）。
    # 揃えずに merge すると重複ゼロで全特徴量が0になる（実際に踏んだ）。
    p["_sym4"] = p["symbol"].astype(str).str[:4]
    try:
        filings = filing_panel(load_filings())
    except (FileNotFoundError, ValueError):
        filings = pd.DataFrame()
    if filings.empty:
        for c in OWNERSHIP_FEATURES_DAILY:
            p[c] = 0.0
        return p

    f = filings.sort_values(["symbol", "period_end"]).copy()
    g = f.groupby("symbol", sort=False)
    prev_fixed = g["fixed_ratio"].shift(1)
    gap = (f["period_end"] - g["period_end"].shift(1)).dt.days
    f["delta_fixed"] = (f["fixed_ratio"] - prev_fixed).where(gap.between(300, 450))

    sessions = pd.Index(sorted(pd.to_datetime(p["date"].unique())))
    pos = sessions.searchsorted(pd.to_datetime(f["sub_date"]), side="right")  # 翌営業日
    ok = pos < len(sessions)
    f = f[ok].copy()
    f["date"] = sessions[pos[ok]]
    f = f.sort_values("date")

    cols = ["fixed_ratio", "custodian_ratio", "delta_fixed", "top1_ratio", "hhi"]
    f = f.rename(columns={"symbol": "_sym4"})
    merged = pd.merge_asof(
        p.sort_values("date"), f[["_sym4", "date", *cols]].sort_values("date"),
        on="date", by="_sym4", direction="backward")
    merged = merged.drop(columns=["_sym4"])
    merged = merged.rename(columns=dict(zip(cols, OWNERSHIP_FEATURES_DAILY)))
    merged = _cross_sectional_z(merged, OWNERSHIP_FEATURES_DAILY)
    for c in OWNERSHIP_FEATURES_DAILY:
        merged[c] = merged[c].fillna(0.0)
    return merged


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
