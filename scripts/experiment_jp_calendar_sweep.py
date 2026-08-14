#!/usr/bin/env python3
"""JP calendar-effect sweep on the TOPIX index. Frozen in PREREGISTER_JP_CALENDAR_SWEEP.md.

Three judged cells with literature-fixed directions (TOM long, pre-holiday long,
Monday short) plus SQ-week as diagnostic-only. Executed via mini/micro TOPIX
futures at 1.0bp per round-trip episode (consecutive in-cell sessions merge into
one episode). Selection 2010-2019, confirmation 2020-2026 opened only on a pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COST_RT = 1.0e-4
SELECTION, CONFIRM = ("2010-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")
OUT = Path("data/jp_calendar_sweep")


def load() -> pd.DataFrame:
    d = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.drop_duplicates("Date").sort_values("Date").set_index("Date")
    d["ret"] = pd.to_numeric(d["C"], errors="coerce").pct_change()
    return d.dropna(subset=["ret"])


def cells(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    cal = pd.read_parquet("data/fx_rates/jp_market_calendar.parquet")
    bdays = pd.DatetimeIndex(sorted(pd.to_datetime(
        cal[cal["HolDiv"].astype(str).isin(("1", "3"))]["Date"]).unique()))
    bset = pd.Series(range(len(bdays)), index=bdays)
    pos = idx.map(bset)

    # TOM: 月末最終営業日〜翌月第3営業日
    month = pd.Series(idx.month, index=idx)
    nxt = pd.Series(idx, index=idx).shift(-1)
    is_last = (month != pd.Series(nxt.dt.month.values, index=idx)).fillna(False)
    tom = pd.Series(False, index=idx)
    last_pos = [int(p) for p, l in zip(pos, is_last) if l and not np.isnan(p)]
    tom_pos = set()
    for p in last_pos:
        tom_pos.update({p, p + 1, p + 2, p + 3})
    tom[:] = [int(p) in tom_pos if not np.isnan(p) else False for p in pos]

    # 休日前: 次の営業日まで暦日差>=2 かつ その間に平日を含む（通常の週末を除く）
    gap = (pd.Series(idx, index=idx).shift(-1) - pd.Series(idx, index=idx)).dt.days
    def has_weekday_holiday(d0, d1):
        if pd.isna(d1):
            return False
        days = pd.date_range(d0 + pd.Timedelta(days=1), d1 - pd.Timedelta(days=1))
        return any(x.weekday() < 5 for x in days)
    pre_hol = pd.Series([has_weekday_holiday(a, b) for a, b in
                         zip(idx, pd.Series(idx, index=idx).shift(-1))], index=idx)

    monday = pd.Series(idx.weekday == 0, index=idx)

    # メジャーSQ週: 3/6/9/12月の第2金曜を含む週（月〜金）
    sq = pd.Series(False, index=idx)
    for y in range(idx.min().year, idx.max().year + 1):
        for m in (3, 6, 9, 12):
            fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
            fridays = [d for d in fridays if d.weekday() == 4 and d.month == m]
            sqd = fridays[1]
            wk_start = sqd - pd.Timedelta(days=sqd.weekday())
            sq |= (pd.Series(idx, index=idx) >= wk_start) & \
                  (pd.Series(idx, index=idx) <= wk_start + pd.Timedelta(days=4))
    return {"TOM_long": tom, "PreHoliday_long": pre_hol,
            "Monday_short": monday, "SQweek_diagnostic": sq}


def episode_costs(mask: pd.Series) -> pd.Series:
    starts = mask & ~mask.shift(1, fill_value=False)
    return starts.astype(float) * COST_RT


def stats(net: pd.Series, mask: pd.Series, lo: str, hi: str) -> dict:
    r = net.loc[lo:hi]
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None}
    m = mask.loc[lo:hi]
    ep_id = (m & ~m.shift(1, fill_value=False)).cumsum()[m]
    ep = r[m].groupby(ep_id).sum()
    pos = ep[ep > 0].sum()
    ex10 = r.drop(ep.nlargest(min(10, len(ep))).index.map(
        lambda i: r[m][ep_id == i].index).map(lambda x: x[0]), errors="ignore") \
        if len(ep) else r
    by = r.groupby(r.index.year).sum()
    d10 = r.copy()
    if len(ep) >= 10:
        top_eps = ep.nlargest(10).index
        d10 = r[~m | ~ep_id.reindex(r.index).isin(top_eps).fillna(False)]
    return {"days_in_market": int(m.sum()), "episodes": int(m.sum() and ep_id.nunique() or 0),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "top5_episode_share": round(float(ep.nlargest(5).sum() / pos), 4) if pos > 0 else None,
            "sharpe_ex_top10": round(float(d10.mean() / d10.std() * 252 ** .5), 3)
            if d10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year_pct": {int(k): round(float(v) * 100, 2) for k, v in by.items()}}


def passes(w: dict) -> list[str]:
    f = []
    if (w.get("sharpe") or -9) < 1.0:
        f.append("sharpe_lt_1.0")
    if w.get("negative_years", 9) > max(1, w.get("years", 1) // 3):
        f.append("too_many_negative_years")
    if (w.get("top5_episode_share") or 1) >= .20:
        f.append("top5_episode_share_ge_20pct")
    if (w.get("sharpe_ex_top10") or -9) < 0.5:
        f.append("sharpe_ex_top10_lt_0.5")
    return f


def main() -> None:
    d = load()
    cs = cells(d.index)
    out = {"note": "3 judged cells (directions fixed from literature) + SQ diagnostic; "
                   "cost 1.0bp per round-trip episode via TOPIX futures"}
    for name, mask in cs.items():
        sign = -1.0 if name.endswith("_short") else 1.0
        net = (sign * d["ret"]).where(mask, 0.0) - episode_costs(mask)
        sel = stats(net, mask, *SELECTION)
        entry = {"selection": sel}
        if name.endswith("_diagnostic"):
            entry["judged"] = False
        else:
            entry["selection_failed"] = passes(sel)
            if not entry["selection_failed"]:
                con = stats(net, mask, *CONFIRM)
                entry["confirmation"] = con
                entry["confirmation_failed"] = passes(con)
                entry["decision"] = ("NO_GO" if entry["confirmation_failed"]
                                     else "PENDING_FULL_RISK_TESTS")
            else:
                entry["confirmation"] = "NOT_OPENED"
                entry["decision"] = "NO_GO"
        out[name] = entry
        print(name, json.dumps(entry.get("selection", {}), ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps({k: v.get("decision", "diagnostic") for k, v in out.items()
                      if isinstance(v, dict)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
