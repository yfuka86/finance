#!/usr/bin/env python3
"""Walk-forward hour-of-day seasonality in USDJPY/EURUSD with GMO fixed costs.

Frozen in docs/PREREGISTER_FX_INTRADAY_SEASONALITY.md. The selection RULE is
frozen and re-applied each year on an expanding window, so every return in the
2014-2026 series is out-of-sample. Cells are (pair, UTC hour 0-17, direction);
each selected cell pays the full GMO fixed spread every occurrence (no merging —
conservative). Hours 18-23 UTC fall outside GMO's fixed-spread window (3:00-9:00
JST) and are excluded from the tradable set entirely.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {"USDJPY": 0.002, "EURUSD": 0.00003}     # GMO 原則固定スプレッド
T_MIN, MAX_CELLS = 2.5, 6
HOURS = range(0, 18)                              # UTC 0-17 = JST 9:00-翌3:00
START_TRADE, END_TRADE = 2014, 2026
OUT = Path("data/fx_hourly_seasonality")


def load_pair(pair: str) -> pd.DataFrame:
    fs = sorted(glob.glob(f"data/fx_dukascopy_hour/parts/{pair}_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)   # UTC前提・naiveに統一
    d = d.drop_duplicates("ts").sort_values("ts")
    mid_o = (d["open_bid"] + d["open_ask"]) / 2
    d["gross"] = d["mid"] / mid_o - 1.0
    d["cost"] = PAIRS[pair] / mid_o
    d["hour"] = d["ts"].dt.hour
    d["year"] = d["ts"].dt.year
    d["day"] = d["ts"].dt.normalize()
    return d[["ts", "day", "year", "hour", "gross", "cost"]]


def main() -> None:
    data = {p: load_pair(p) for p in PAIRS}
    rows, picks_log = [], {}
    for year in range(START_TRADE, END_TRADE + 1):
        # 拡大窓でセルを選ぶ（凍結された規則）
        cands = []
        for pair, d in data.items():
            tr = d[(d["year"] < year) & d["hour"].isin(HOURS)]
            g = tr.groupby("hour")
            for hour, gg in g:
                for direction in (1, -1):
                    net = direction * gg["gross"] - gg["cost"]
                    if len(net) < 500 or net.std() == 0:
                        continue
                    t = net.mean() / net.std() * np.sqrt(len(net))
                    if t >= T_MIN and net.mean() > 0:
                        cands.append((float(t), pair, int(hour), direction))
        picks = sorted(cands, reverse=True)[:MAX_CELLS]
        picks_log[year] = [{"t": round(t, 2), "pair": p, "utc_hour": h, "dir": s}
                           for t, p, h, s in picks]
        for _, pair, hour, direction in picks:
            d = data[pair]
            yr = d[(d["year"] == year) & (d["hour"] == hour)]
            for _, r in yr.iterrows():
                rows.append({"day": r["day"],
                             "net": direction * r["gross"] - r["cost"]})
    if not rows:
        print(json.dumps({"decision": "NO_GO", "reason": "no cells ever selected"}))
        return
    df = pd.DataFrame(rows).groupby("day")["net"].sum()
    cal = pd.date_range(f"{START_TRADE}-01-01", df.index.max(), freq="D")
    daily = df.reindex(cal).fillna(0.0)

    def sh(v):
        return round(float(v.mean() / v.std() * 252 ** .5), 3) if v.std() else None

    eq = (1 + daily).cumprod()
    by = daily.groupby(daily.index.year).sum()
    pos_days = daily[daily > 0].sum()
    ex10 = daily.drop(daily.nlargest(10).index)
    out = {"walk_forward": {
        "sharpe": sh(daily), "ann_return": round(float(daily.mean() * 252), 4),
        "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
        "trade_days": int((daily != 0).sum()),
        "top5_day_share": round(float(daily.nlargest(5).sum() / pos_days), 4)
        if pos_days > 0 else None,
        "sharpe_ex_top10": sh(ex10),
        "first_half_sharpe": sh(daily.loc["2014":"2019"]),
        "second_half_sharpe": sh(daily.loc["2020":]),
        "negative_years": int((by < 0).sum()), "years": int(len(by)),
        "by_year_pct": {int(k): round(float(v) * 100, 2) for k, v in by.items()}},
        "picks_by_year": picks_log}
    w = out["walk_forward"]
    failed = []
    if (w["sharpe"] or -9) < 1.0:
        failed.append("sharpe_lt_1.0")
    if w["negative_years"] > max(1, w["years"] // 3):
        failed.append("too_many_negative_years")
    if (w["top5_day_share"] or 1) >= .20:
        failed.append("top5_day_share_ge_20pct")
    if (w["sharpe_ex_top10"] or -9) < 0.5:
        failed.append("sharpe_ex_top10_lt_0.5")
    if (w["first_half_sharpe"] or -9) < 0.5 or (w["second_half_sharpe"] or -9) < 0.5:
        failed.append("half_period_sharpe_lt_0.5")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "PENDING_FULL_RISK_TESTS"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
