#!/usr/bin/env python3
"""Flow x fundamentals, sector-internal L/S, overnight hold (D close -> D+1 open).

Frozen in docs/PREREGISTER_FLOW_FUND_SECTOR_ON.md. Four cells (A/B/B2/C),
production short constraints, close-auction entry + open-auction exit
(2.0bps round trip) + one night of margin interest. Scores use only
information fixed before D 15:24 (flow lag-2, fundamentals from the session
after disclosure, D-1 prices); ret_on_fwd is the schema-guarded target.
Selection 2018-2024; confirmation 2025+ opens only on a full selection pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.flow_features import attach_flow_features
from trading.jp_intraday.strategies import unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
RATE_LONG, RATE_SHORT, SESSIONS = .028, .042, 245
COST_BPS_SIDE = 1.0                       # close 0.5 + next open 1.5 per round trip
OUT = Path("data/jp_flow_fund_sector_on")


def zsec(panel: pd.DataFrame, col: str) -> pd.Series:
    g = panel.groupby(["date", "sector"])[col]
    n = g.transform("count")
    z = (panel[col] - g.transform("mean")) / g.transform("std")
    return z.where(n >= 5)


def build_fund_daily(sessions: pd.Index) -> pd.DataFrame:
    """PIT fundamentals from ALL fins caches (2018 backfill + snapshot + incremental).

    BPS/NP/Eq appear mostly in FY rows; carry the last disclosed state forward
    per issuer (never backward) before computing ratios, so quarterly rows do
    not blank the merge. known_date = the session AFTER disclosure.
    """
    f = load_fins()
    f["disc_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f[f["disc_date"].notna()].copy()
    for c in ("BPS", "NP", "Eq"):
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f["sym4"] = f["Code"].astype(str).str[:4]
    f = f.sort_values(["sym4", "disc_date", "DiscTime"]).drop_duplicates(
        ["sym4", "disc_date"], keep="last")
    f[["BPS", "NP", "Eq"]] = f.groupby("sym4", sort=False)[["BPS", "NP", "Eq"]].ffill()
    f["fn_roe"] = (f["NP"] / f["Eq"].replace(0, np.nan)).clip(-2, 2)
    sessions = pd.Index(sorted(pd.to_datetime(sessions).unique()))
    pos = sessions.searchsorted(f["disc_date"], side="right")
    ok = pos < len(sessions)
    f = f.loc[ok].copy()
    f["known_date"] = sessions[pos[ok]]
    f = f.drop_duplicates(["sym4", "known_date"], keep="last")
    return f[["sym4", "known_date", "BPS", "fn_roe"]].rename(
        columns={"BPS": "_bps"}).sort_values(["known_date", "sym4"])


def build_panel() -> pd.DataFrame:
    p = load_panel_cached(min_value_yen=1e9).copy()
    p["symbol"] = p["symbol"].astype(str)
    p = attach_flow_features(p, lag=2)
    fund = build_fund_daily(pd.Index(sorted(p["date"].unique())))
    # fins codes are 4-char; the panel symbol is 5-char ("9984" vs "99840").
    # Merging on the raw symbol silently matches nothing (known trap, 3rd hit).
    p["sym4"] = p["symbol"].str[:4]
    p = p.sort_values(["date", "sym4"])
    p = pd.merge_asof(p, fund, left_on="date", right_on="known_date", by="sym4",
                      direction="backward")
    p = p.sort_values(["date", "symbol"])
    cov = p.loc[p["date"].ge("2018-07-01"), "_bps"].notna().mean()
    assert cov > .5, f"fundamental merge failed (coverage {cov:.1%})"
    px = p["raw_close"].where(p["raw_close"] > 0)
    p["fn_book_to_price"] = (p["_bps"] / px).clip(-10, 10)
    p["z_flow"] = zsec(p, "flow_close_z")
    p["z_turn"] = zsec(p, "flow_turn_z")
    p["z_bp"] = zsec(p, "fn_book_to_price")
    p["z_roe"] = zsec(p, "fn_roe")
    return p


def cell_score(p: pd.DataFrame, cell: str, flow_col: str = "z_flow") -> pd.Series:
    f = p[flow_col]
    if cell == "A":
        return f
    if cell == "B":
        return .5 * f + .5 * p["z_bp"]
    if cell == "B2":
        return .5 * f + .25 * p["z_bp"] + .25 * p["z_roe"]
    if cell == "C":
        cheap = p["z_bp"] > 0
        long_c = cheap & (f > 0)
        short_c = (~cheap) & (f < 0)
        return f.where(long_c | short_c)
    raise ValueError(cell)


def stats(daily: pd.DataFrame) -> dict:
    if daily.empty or "net" not in daily:
        return {"sharpe": None}
    d = daily.set_index(pd.to_datetime(daily["date"]))
    r = d["net"].dropna()
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return_pct": round(float(r.mean() * 252 * 100), 2),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "days": int(len(r))}


def run_cell(p: pd.DataFrame, cell: str, names: int = 8,
             flow_col: str = "z_flow",
             end: pd.Timestamp | None = SELECTION_END,
             start: pd.Timestamp | None = None) -> dict:
    sub = p if end is None else p[p["date"].le(end)]
    if start is not None:
        sub = sub[sub["date"].ge(start)]
    f = sub.copy()
    f["_score"] = cell_score(f, cell, flow_col)
    f = f.dropna(subset=["_score", "ret_on_fwd"])
    # Alias AFTER scoring: harvest the overnight leg; entry price = D raw close.
    f["_s"] = f["_score"] - f.groupby("date")["_score"].transform("mean")
    f["raw_open"] = f["raw_close"]
    f["open"] = f["raw_close"]
    f["intraday_ret"] = f["ret_on_fwd"]
    daily, blot = unit_lot_backtest(f, capital_yen=2e7, names_per_side=names,
                                    margin_ratio=2.0, cost_bps_side=COST_BPS_SIDE,
                                    construction="magnitude")
    if len(daily):
        carry = (daily["long_yen"] * RATE_LONG
                 + daily["short_yen"] * RATE_SHORT) / SESSIONS
        daily = daily.assign(net_yen=daily["net_yen"] - carry,
                             net=(daily["net_yen"] - carry) / 2e7)
    s = stats(daily)
    if len(daily):
        dep = daily["long_yen"] + daily["short_yen"]
        s["gross_bps_per_day"] = round(float((daily["pnl_yen"] / dep).mean() * 1e4), 2)
        s["all_in_cost_bps_per_day"] = round(float(
            ((daily["cost_yen"] + (daily["long_yen"] * RATE_LONG
                                   + daily["short_yen"] * RATE_SHORT) / SESSIONS)
             / dep).mean() * 1e4), 2)
    return s


def judge(s: dict) -> dict:
    return {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0
                                       and s.get("negative_years", 9) * 3 <= s["years"]),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5),
            "gross_ge_2x_cost": bool((s.get("gross_bps_per_day") or -9)
                                     >= 2 * (s.get("all_in_cost_bps_per_day") or 9))}


def main() -> None:
    p = build_panel()
    summary = {"spec": "docs/PREREGISTER_FLOW_FUND_SECTOR_ON.md", "selection": {},
               "sensitivity_selection": {}, "confirmation": "UNOPENED"}
    passing = []
    for cell in ("A", "B", "B2", "C"):
        s = run_cell(p, cell)
        s["criteria"] = judge(s)
        summary["selection"][cell] = s
        if all(s["criteria"].values()):
            passing.append(cell)
    summary["sensitivity_selection"]["A_turn_z"] = run_cell(p, "A", flow_col="z_turn")
    summary["sensitivity_selection"]["A_15per_side"] = run_cell(p, "A", names=15)
    if passing:
        frozen = passing[0]                     # A > B > B2 > C simplicity order
        summary["frozen_cell"] = frozen
        conf = run_cell(p, frozen, end=None, start=pd.Timestamp("2025-01-01"))
        conf["criteria"] = judge(conf)
        summary["confirmation"] = {frozen: conf}
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if all(conf["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
