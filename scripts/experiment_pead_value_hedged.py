#!/usr/bin/env python3
"""PEAD-value events: FOP upward revision x low PBR, 1306-short hedge, flow gate.

Frozen in docs/PREREGISTER_PEAD_VALUE_HEDGED.md. Four cells (E1/E2 unhedged
judged on equal-weight market excess; E3/E4 hedged 1:1 with TOPIX judged on
absolute hedged returns). Flow-trajectory diagnostic reported unjudged.
Selection: events 2018-2024; confirmation 2025-01..2026-05 opens once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_value_event_v1 import load_fins
from scripts.experiment_jp_large_holdings import market_curve
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.flow_features import build_flow_features

COST = .004
HEDGE_RT = .0004
LEND_RATE = .0115
HOLD = 60
FLOOR = 1e8
SEL = (pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31"))
CONF = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-05-31"))
OUT = Path("data/jp_pead_value_hedged")


def revision_events() -> pd.DataFrame:
    f = load_fins()
    f["event_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["symbol"] = f["Code"].astype(str).str[:4]
    for c in ("FOP", "BPS", "EPS", "OP"):
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f = f.dropna(subset=["event_date", "symbol", "CurFYEn"])
    order = [c for c in ["symbol", "CurFYEn", "event_date", "DiscTime", "DiscNo"] if c in f]
    f = f.sort_values(order).drop_duplicates(["symbol", "CurFYEn", "event_date"],
                                             keep="last")
    state = ["BPS", "EPS", "OP"]
    f[state] = f.groupby("symbol", sort=False)[state].ffill()
    prev = f.groupby(["symbol", "CurFYEn"], sort=False)["FOP"].shift(1)
    f["op_revision"] = f["FOP"].div(prev) - 1
    return f.loc[prev.gt(0) & f["FOP"].gt(0)
                 & f["op_revision"].ge(.05 - 1e-12)].copy()


def topix_curve() -> pd.Series:
    t = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    t["Date"] = pd.to_datetime(t["Date"])
    return t.set_index("Date")["C"].sort_index()


def build_cases(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.rename(columns={"Date": "date", "Code": "code", "AdjO": "open",
                              "AdjC": "close", "raw_close": "rclose",
                              "Va": "value"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["code"].astype(str).str[:4]
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    mkt = market_curve(daily)
    tpx = topix_curve()
    flows = build_flow_features(lag=2)
    flows["sym4"] = flows["symbol"].astype(str).str[:4]
    fmap = flows.drop_duplicates(["sym4", "date"], keep="last").set_index(
        ["sym4", "date"])["flow_close_z"]
    rows, held_until = [], {}
    for _, e in events.sort_values("event_date").iterrows():
        bars = by_symbol.get(e["symbol"])
        if bars is None:
            continue
        prior_idx = bars["date"].searchsorted(e["event_date"], side="left") - 1
        entry_idx = bars["date"].searchsorted(e["event_date"], side="right")
        exit_idx = entry_idx + HOLD - 1
        if prior_idx < 0 or exit_idx >= len(bars):
            continue
        prior, entry, exit_ = bars.iloc[prior_idx], bars.iloc[entry_idx], bars.iloc[exit_idx]
        if (entry["date"] - e["event_date"]).days > 5:
            continue
        pbr = prior["rclose"] / e["BPS"] if e["BPS"] and e["BPS"] > 0 else np.nan
        if not (0 < pbr <= 1 and e["EPS"] > 0 and e["OP"] > 0
                and prior["value"] >= FLOOR and entry["open"] > 0
                and exit_["close"] > 0):
            continue
        hu = held_until.get(e["symbol"])
        if hu is not None and entry["date"] <= hu:
            continue
        held_until[e["symbol"]] = exit_["date"]
        raw = exit_["close"] / entry["open"] - 1
        m0, m1 = mkt.asof(prior["date"]), mkt.asof(exit_["date"])
        t0, t1 = tpx.asof(prior["date"]), tpx.asof(exit_["date"])
        cal_days = (exit_["date"] - entry["date"]).days
        hedge_cost = HEDGE_RT + LEND_RATE * cal_days / 365
        fz = fmap.get((e["symbol"], entry["date"]), np.nan)
        rows.append({"symbol": e["symbol"], "event_date": e["event_date"],
                     "entry_date": entry["date"], "exit_date": exit_["date"],
                     "op_revision": e["op_revision"], "pbr": pbr,
                     "flow_entry_z": fz,
                     "excess_net": raw - COST - (m1 / m0 - 1),
                     "hedged_net": raw - COST - (t1 / t0 - 1) - hedge_cost})
    return pd.DataFrame(rows)


def report(r: pd.Series, n_min: int) -> dict:
    r = r.dropna()
    if r.empty:
        return {"cases": 0, "criteria": {"n_ok": False}}
    total = r.sum()
    top = float(r.max() / total) if total > 0 else None
    top5pct = (float(r.nlargest(max(1, int(len(r) * .05))).sum() / total)
               if total > 0 else None)
    out = {"cases": int(len(r)), "median_pct": round(float(r.median() * 100), 3),
           "mean_pct": round(float(r.mean() * 100), 3),
           "win_rate": round(float(r.gt(0).mean()), 3),
           "top_case_share": None if top is None else round(top, 3),
           "top5pct_share": None if top5pct is None else round(top5pct, 3),
           "es5_pct": round(float(r[r <= r.quantile(.05)].mean() * 100), 2)}
    out["criteria"] = {"n_ok": bool(len(r) >= n_min),
                       "median_positive": bool(out["median_pct"] > 0),
                       "top_share_lt_20": bool((top or 9) < .20),
                       "top5pct_lt_40": bool((top5pct or 9) < .40)}
    return out


def flow_diagnostic(events: pd.DataFrame, daily: pd.DataFrame) -> dict:
    flows = build_flow_features(lag=2)
    flows["sym4"] = flows["symbol"].astype(str).str[:4]
    piv = flows.pivot_table(index="date", columns="sym4", values="flow_close_z",
                            aggfunc="last")
    sessions = piv.index
    traj = {k: [] for k in range(1, 11)}
    for _, e in events.iterrows():
        if e["symbol"] not in piv.columns:
            continue
        i = sessions.searchsorted(e["event_date"], side="right")
        for k in range(1, 11):
            if i + k - 1 < len(sessions):
                v = piv.iloc[i + k - 1][e["symbol"]]
                if pd.notna(v):
                    traj[k].append(float(v))
    return {"post_event_flow_z_mean_t1_t10":
            {f"t+{k}": round(float(np.mean(v)), 3) if v else None
             for k, v in traj.items()}}


def main() -> None:
    events = revision_events()
    daily = load_existing_daily()
    cases = build_cases(events, daily)
    summary = {"spec": "docs/PREREGISTER_PEAD_VALUE_HEDGED.md",
               "revision_events": int(len(events)), "cases": int(len(cases)),
               "selection": {}, "confirmation": "UNOPENED"}
    sel = cases[cases["event_date"].between(*SEL)]
    conf = cases[cases["event_date"].between(*CONF)]
    gate = sel["flow_entry_z"] > 0
    cells_sel = {"E1_excess": report(sel["excess_net"], 100),
                 "E2_excess_flowgate": report(sel.loc[gate, "excess_net"], 100),
                 "E3_hedged": report(sel["hedged_net"], 100),
                 "E4_hedged_flowgate": report(sel.loc[gate, "hedged_net"], 100)}
    summary["selection"] = cells_sel
    if len(sel):
        corr = sel[["flow_entry_z", "excess_net"]].dropna()
        summary["flow_vs_outcome_corr"] = (round(float(
            corr["flow_entry_z"].corr(corr["excess_net"], method="spearman")), 4)
            if len(corr) > 30 else None)
    summary["flow_trajectory_diagnostic"] = flow_diagnostic(
        events[events["event_date"].between(*SEL)], daily)
    passing = [k for k, v in cells_sel.items() if all(v["criteria"].values())]
    summary["passing_cells"] = passing
    if passing:
        order = ["E3_hedged", "E4_hedged_flowgate", "E1_excess", "E2_excess_flowgate"]
        frozen = next(c for c in order if c in passing)
        summary["frozen_cell"] = frozen
        col = "hedged_net" if frozen.startswith(("E3", "E4")) else "excess_net"
        g = conf["flow_entry_z"] > 0
        sub = conf.loc[g, col] if "flowgate" in frozen else conf[col]
        summary["confirmation"] = {frozen: report(sub, 40)}
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if all(summary["confirmation"][frozen]["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
