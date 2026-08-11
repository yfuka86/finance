#!/usr/bin/env python3
"""Calm-market sector-dip longs, gated by flow and RSI.

Frozen in docs/PREREGISTER_SECTOR_DIP_LONG.md. Trigger: a sector's 5-session
return trails TOPIX by >=5% while TOPIX itself is calm (5d > -3%). Buy the
sector's Y1e9-universe members at the next open, hold 20 sessions, judge the
equal-weight-market EXCESS at episode level. Gates: entry-time flow_close_z>0
(S2), Wilder RSI14<30 (S3), both (S4). Selection 2018-2024; confirmation
2025+ opens once for a passing cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_jp_large_holdings import market_curve
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.flow_features import build_flow_features

SECTOR_BY_CODE = {  # official S33 index-code order (validated by return corr)
    "0040": "水産・農林業", "0041": "鉱業", "0042": "建設業", "0043": "食料品",
    "0044": "繊維製品", "0045": "パルプ・紙", "0046": "化学", "0047": "医薬品",
    "0048": "石油･石炭製品", "0049": "ゴム製品", "004A": "ガラス･土石製品",
    "004B": "鉄鋼", "004C": "非鉄金属", "004D": "金属製品", "004E": "機械",
    "004F": "電気機器", "0050": "輸送用機器", "0051": "精密機器",
    "0052": "その他製品", "0053": "電気･ガス業", "0054": "陸運業",
    "0055": "海運業", "0056": "空運業", "0057": "倉庫･運輸関連業",
    "0058": "情報･通信業", "0059": "卸売業", "005A": "小売業", "005B": "銀行業",
    "005C": "証券･商品先物取引業", "005D": "保険業", "005E": "その他金融業",
    "005F": "不動産業", "0060": "サービス業"}
DIP, CALM, HOLD, COST = -.05, -.03, 20, .001
SEL = (pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31"))
CONF = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-05-31"))
OUT = Path("data/jp_sector_dip_long")


def norm(s: str) -> str:
    return str(s).replace("・", "･").replace(" ", "")


def sector_frames():
    idx = pd.read_parquet("data/jp_derivatives/sector_indices_2008_2026.parquet")
    idx["Date"] = pd.to_datetime(idx["Date"])
    piv = idx.pivot_table(index="Date", columns="Code", values="C", aggfunc="last")
    piv.columns = [norm(SECTOR_BY_CODE[c]) for c in piv.columns]
    tp = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    tp["Date"] = pd.to_datetime(tp["Date"])
    tpx = tp.set_index("Date")["C"].sort_index()
    return piv, tpx


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_episodes() -> pd.DataFrame:
    piv, tpx = sector_frames()
    panel = load_panel_cached(min_value_yen=1e9)
    panel = panel.assign(secn=panel["sector"].map(norm))
    daily = load_existing_daily()
    d = daily.rename(columns={"Date": "date", "Code": "code", "AdjO": "open",
                              "AdjC": "close"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["code"].astype(str)
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    d["rsi"] = d.groupby("symbol", sort=False)["close"].transform(wilder_rsi)
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    mkt = market_curve(daily)
    flows = build_flow_features(lag=2)
    fpiv = flows.pivot_table(index="date", columns="symbol",
                             values="flow_close_z", aggfunc="last").sort_index()

    sec5 = piv.pct_change(5)
    tpx5 = tpx.pct_change(5).reindex(sec5.index)
    trigger = sec5.sub(tpx5, axis=0).le(DIP) & tpx5.gt(CALM).to_numpy()[:, None]
    members = {(dt, sn): g["symbol"].tolist()
               for (dt, sn), g in panel.groupby(["date", "secn"], sort=False)}
    rows, active = [], {}
    for dt in trigger.index:
        for sn in trigger.columns[trigger.loc[dt]]:
            if active.get(sn) is not None and dt <= active[sn]:
                continue
            syms = members.get((dt, sn))
            if not syms:
                continue
            stock_rows = []
            for sym in syms:
                bars = by_symbol.get(sym)
                if bars is None:
                    continue
                i = bars["date"].searchsorted(dt, side="right")
                j = i + HOLD - 1
                if i >= len(bars) or j >= len(bars):
                    continue
                if (bars["date"].iloc[i] - dt).days > 5:
                    continue
                entry, exit_ = bars.iloc[i], bars.iloc[j]
                if not (entry["open"] > 0 and exit_["close"] > 0):
                    continue
                fcol = fpiv.get(sym)
                fz = (fcol.asof(entry["date"]) if fcol is not None else np.nan)
                stock_rows.append({
                    "ret": exit_["close"] / entry["open"] - 1,
                    "rsi": bars["rsi"].iloc[i - 1] if i >= 1 else np.nan,
                    "flow": fz, "exit_date": exit_["date"],
                    "entry_date": entry["date"]})
            if len(stock_rows) < 3:
                continue
            sr = pd.DataFrame(stock_rows)
            active[sn] = sr["exit_date"].max()
            m0 = mkt.asof(dt)
            m1 = mkt.asof(sr["exit_date"].max())
            mret = m1 / m0 - 1
            def ep(sub):
                return float(sub["ret"].mean() - COST - mret) if len(sub) >= 3 else np.nan
            rows.append({"sector": sn, "trigger_date": dt,
                         "n_stocks": len(sr),
                         "S1": ep(sr),
                         "S2": ep(sr[sr["flow"] > 0]),
                         "S3": ep(sr[sr["rsi"] < 30]),
                         "S4": ep(sr[(sr["flow"] > 0) & (sr["rsi"] < 30)])})
    return pd.DataFrame(rows)


def report(r: pd.Series, n_min: int) -> dict:
    r = r.dropna()
    if r.empty:
        return {"episodes": 0, "criteria": {"n_ok": False}}
    total = r.sum()
    top = float(r.max() / total) if total > 0 else None
    top5pct = (float(r.nlargest(max(1, int(len(r) * .05))).sum() / total)
               if total > 0 else None)
    out = {"episodes": int(len(r)),
           "excess_median_pct": round(float(r.median() * 100), 3),
           "excess_mean_pct": round(float(r.mean() * 100), 3),
           "win_rate": round(float(r.gt(0).mean()), 3),
           "top_episode_share": None if top is None else round(top, 3),
           "top5pct_share": None if top5pct is None else round(top5pct, 3),
           "es5_pct": round(float(r[r <= r.quantile(.05)].mean() * 100), 2)}
    out["criteria"] = {"n_ok": bool(len(r) >= n_min),
                       "median_positive": bool(out["excess_median_pct"] > 0),
                       "top_share_lt_20": bool((top or 9) < .20),
                       "top5pct_lt_40": bool((top5pct or 9) < .40)}
    return out


def main() -> None:
    eps = build_episodes()
    summary = {"spec": "docs/PREREGISTER_SECTOR_DIP_LONG.md",
               "episodes_total": int(len(eps)), "selection": {},
               "confirmation": "UNOPENED"}
    sel = eps[eps["trigger_date"].between(*SEL)]
    conf = eps[eps["trigger_date"].between(*CONF)]
    for cell in ("S1", "S2", "S3", "S4"):
        summary["selection"][cell] = report(sel[cell], 60)
    passing = [c for c in ("S1", "S2", "S3", "S4")
               if all(summary["selection"][c]["criteria"].values())]
    summary["passing_cells"] = passing
    if passing:
        frozen = passing[0]
        summary["frozen_cell"] = frozen
        summary["confirmation"] = {frozen: report(conf[frozen], 20)}
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if all(summary["confirmation"][frozen]["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    if len(sel):
        summary["per_sector_episode_counts"] = (
            sel.groupby("sector")["S1"].count().sort_values(ascending=False)
            .head(8).to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
