#!/usr/bin/env python3
"""New 5%-rule filing event study (institutional footprints, stock level).

Frozen in docs/PREREGISTER_JP_LARGE_HOLDINGS.md BEFORE any returns were
computed. New filings only (docDescription starts with 大量保有報告書),
next-business-day open entry, exit at the 20th subsequent session close,
40bps round trip, judged on the EXCESS over the equal-weight market
(entry-1 close -> exit close; the market leg includes the overnight into the
entry morning, which the strategy leg does not = conservative direction).
Selection: events 2021-09-01..2024-12-31; confirmation 2025-01-01..2026-06-30
opened only on a full selection pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily

LEDGER_DIR = Path("data/jp_large_holdings")
COST = .004
HOLD = 20
FLOOR = 1e8
SEL = (pd.Timestamp("2021-09-01"), pd.Timestamp("2024-12-31"))
CONF = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-30"))
OUT = Path("data/jp_large_holdings")


def load_ledger() -> pd.DataFrame:
    rows = []
    for f in sorted(LEDGER_DIR.glob("large_holdings_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("_empty"):
                    rows.append(r)
    d = pd.DataFrame(rows)
    d = d[d["docDescription"].fillna("").str.startswith("大量保有報告書")]
    d = d[d["withdrawalStatus"].astype(str).isin(("0", "None", "nan", ""))]
    d["submit"] = pd.to_datetime(d["submitDateTime"], errors="coerce")
    code = pd.read_csv(LEDGER_DIR / "edinet_code_list.csv", encoding="cp932",
                       skiprows=1, dtype=str)
    m = code.set_index("ＥＤＩＮＥＴコード")["証券コード"].dropna()
    mapped = d["issuerEdinetCode"].map(m)
    d["sec"] = d["secCode"].fillna(mapped)
    d = d.dropna(subset=["submit", "sec"])
    d["symbol"] = d["sec"].astype(str).str[:4]
    return d.sort_values("submit").reset_index(drop=True)


def market_curve(daily: pd.DataFrame) -> pd.Series:
    """Equal-weight close-to-close market with the adjacent-session guard."""
    d = daily.rename(columns={"Date": "date", "Code": "code", "AdjC": "close"})
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"])
    sessions = pd.Index(np.sort(d["date"].unique()))
    sess_no = pd.Series(np.arange(len(sessions)), index=sessions)
    d["sn"] = d["date"].map(sess_no)
    g = d.groupby("code", sort=False)
    ret = d["close"] / g["close"].shift(1) - 1
    ok = d["sn"] - g["sn"].shift(1) == 1
    d["r"] = ret.where(ok)
    mkt = d.groupby("date")["r"].mean()
    return (1 + mkt.fillna(0)).cumprod()


def build_cases(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.rename(columns={"Date": "date", "Code": "code", "AdjO": "open",
                              "AdjC": "close", "Va": "value"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["code"].astype(str).str[:4]
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    mkt = market_curve(daily)
    rows, held_until = [], {}
    for _, e in events.iterrows():
        bars = by_symbol.get(e["symbol"])
        if bars is None:
            continue
        sub_day = pd.Timestamp(e["submit"].date())
        entry_idx = bars["date"].searchsorted(sub_day, side="right")
        exit_idx = entry_idx + HOLD - 1
        prior_idx = entry_idx - 1
        if prior_idx < 0 or exit_idx >= len(bars):
            continue
        if (bars["date"].iloc[entry_idx] - sub_day).days > 5:
            continue                                   # stale symbol (halt/delist)
        hu = held_until.get(e["symbol"])
        if hu is not None and bars["date"].iloc[entry_idx] <= hu:
            continue                                   # overlapping event dropped
        prior, entry, exit_ = bars.iloc[prior_idx], bars.iloc[entry_idx], bars.iloc[exit_idx]
        if not (prior["value"] >= FLOOR and entry["open"] > 0 and exit_["close"] > 0):
            continue
        held_until[e["symbol"]] = exit_["date"]
        raw = exit_["close"] / entry["open"] - 1
        m0, m1 = mkt.asof(prior["date"]), mkt.asof(exit_["date"])
        rows.append({"symbol": e["symbol"], "submit": e["submit"],
                     "entry_date": entry["date"], "exit_date": exit_["date"],
                     "filer": e.get("filerName"),
                     "raw_ret": raw, "mkt_ret": m1 / m0 - 1,
                     "excess_net": raw - COST - (m1 / m0 - 1)})
    return pd.DataFrame(rows)


def report(cases: pd.DataFrame, n_min: int) -> dict:
    r = cases["excess_net"].dropna()
    if r.empty:
        return {"cases": 0}
    total = r.sum()
    top = float(r.max() / total) if total > 0 else None
    top5pct = (float(r.nlargest(max(1, int(len(r) * .05))).sum() / total)
               if total > 0 else None)
    out = {"cases": int(len(r)),
           "excess_median_pct": round(float(r.median() * 100), 3),
           "excess_mean_pct": round(float(r.mean() * 100), 3),
           "win_rate": round(float(r.gt(0).mean()), 3),
           "raw_median_pct": round(float(cases["raw_ret"].median() * 100), 3),
           "top_case_share": None if top is None else round(top, 3),
           "top5pct_share": None if top5pct is None else round(top5pct, 3),
           "es5_pct": round(float(r[r <= r.quantile(.05)].mean() * 100), 2)}
    out["criteria"] = {"n_ok": bool(len(r) >= n_min),
                       "median_positive": bool(out["excess_median_pct"] > 0),
                       "top_share_lt_20": bool((top or 9) < .20),
                       "top5pct_lt_40": bool((top5pct or 9) < .40)}
    return out


def main() -> None:
    events = load_ledger()
    daily = load_existing_daily()
    cases = build_cases(events, daily)
    sel = cases[cases["submit"].between(*SEL)]
    conf = cases[cases["submit"].between(*CONF)]
    summary = {"spec": "docs/PREREGISTER_JP_LARGE_HOLDINGS.md",
               "new_filings_total": int(len(events)),
               "cases_total": int(len(cases)),
               "selection": report(sel, 100)}
    ok = all(summary["selection"].get("criteria", {}).values())
    summary["confirmation"] = report(conf, 40) if ok else "UNOPENED"
    if ok:
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if all(summary["confirmation"]["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
