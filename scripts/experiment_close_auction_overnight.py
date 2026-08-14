#!/usr/bin/env python3
"""Closing-auction reversal x overnight round-trip (auction-to-auction).

Frozen in docs/PREREGISTER_CLOSE_AUCTION_OVERNIGHT.md. Minute-derived close-time
microstructure features (esp. the 15:30-vs-15:24 auction jump = an observable
imbalance proxy) -> predict the residual overnight return (ret_on_fwd, schema
v7/v11 guarded). Enter D close auction, exit D+1 open auction (round trip 2bps),
short leg pays 4.2%/245 borrow. Selection 2024-08..2025-09; confirmation opens
once. Prior is LOW (R7 overnight is tail-dependent) -- honest test of the one
remaining execution path from the Optiver diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.strategies import unit_lot_backtest

SRC = "data/jp_minutes_2y/jp_1m_2024-08-01_2026-07-24.parquet"
SEL = (pd.Timestamp("2024-08-01"), pd.Timestamp("2025-09-30"))
CONF = (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-07-31"))
RT_COST, SHORT_RATE, SESSIONS = 2.0e-4, .042, 245
# ★auction_jump(15:30/15:24) は EXCLUDED: 15:30の値で15:30の板寄せに発注する先読みで、
# かつ target ret_on_fwd=翌寄/当日引 の分母に同じ15:30引値が入る=同一バーノイズ結合
# (オプションカレンダーを棄却した罠と同型)。発注期限は15:24なので使えない。
# 実行可能な特徴量は 15:24 までに確定するもののみ。
FEATS = ["last30", "afternoon", "rvol_aft", "vwap_dev", "day", "late_vol_share"]
OUT = Path("data/jp_close_auction_overnight")


def close_features() -> pd.DataFrame:
    d = pd.read_parquet(SRC, columns=["timestamp", "symbol", "high", "low",
                                       "close", "volume", "open"])
    jt = pd.to_datetime(d["timestamp"]).dt.tz_convert("Asia/Tokyo")
    d["date"] = jt.dt.normalize().dt.tz_localize(None)
    d["hm"] = jt.dt.strftime("%H:%M")
    d["mins"] = jt.dt.hour * 60 + jt.dt.minute
    key = ["date", "symbol"]
    at = lambda hm, col="close": d[d["hm"] == hm].set_index(key)[col]
    o9, c1230, c1454, c1524 = (at("09:00", "open"), at("12:30"),
                               at("14:54"), at("15:24"))
    aft = d[(d["mins"] >= 12 * 60 + 30) & (d["mins"] <= 15 * 60 + 24)].sort_values(
        key + ["mins"])
    volaft = aft.groupby(key)["volume"].sum()
    vlast30 = aft[aft["mins"] >= 14 * 60 + 54].groupby(key)["volume"].sum()
    aft["r1"] = np.log(aft["close"] / aft.groupby(key)["close"].shift(1))
    rvol = aft.groupby(key)["r1"].std()
    aft["tp"] = (aft["high"] + aft["low"] + aft["close"]) / 3
    vwap = (aft.assign(pv=aft["tp"] * aft["volume"]).groupby(key)["pv"].sum()
            / volaft.replace(0, np.nan))
    idx = c1524.index
    f = pd.DataFrame(index=idx)
    f["last30"] = (c1524 / c1454 - 1).reindex(idx)
    f["afternoon"] = (c1524 / c1230 - 1).reindex(idx)
    f["rvol_aft"] = rvol.reindex(idx)
    f["vwap_dev"] = (c1524 / vwap - 1).reindex(idx)
    f["day"] = (c1524 / o9 - 1).reindex(idx)
    f["late_vol_share"] = (vlast30 / volaft.replace(0, np.nan)).reindex(idx)
    f = f.reset_index()
    f["sym4"] = f["symbol"].astype(str).str[:4]
    for c in FEATS:
        f[c] = f[c] - f.groupby("date")[c].transform("mean")
    return f.drop(columns="symbol")


QUALITY = True                 # user-directed 2026-08-14: avoid weird small caps
MCAP_FLOOR = 3e10              # ¥300億
GROWTH_EXCLUDE = {"グロース", "その他", "TOKYO PRO MARKET"}


def _quality_mask(m: pd.DataFrame) -> pd.Series:
    """Exclude weird small caps and flagged names (user request).

    - market cap >= ¥300億 (drops penny/micro like ランド, ソレイジア)
    - established market only (Prime/Standard; excludes Growth where most of the
      speculative IPO names live) -- via current master (segment rarely changes)
    - not under 増担保規制 (xt_alert_flag == 0) = the "注記あり" proxy we have
    """
    ok = m["mktcap_yen"].fillna(0) >= MCAP_FLOOR
    if "MktNm" in m:
        ok &= ~m["MktNm"].isin(GROWTH_EXCLUDE)
    if "xt_alert_flag" in m:
        ok &= m["xt_alert_flag"].fillna(0) <= 0
    return ok


def attach_target(quality: bool = QUALITY) -> pd.DataFrame:
    p = load_panel_cached(min_value_yen=1e9)[
        ["date", "symbol", "ret_on_fwd", "shortable", "short_restricted",
         "raw_close", "prev_value", "ivol", "mktcap_yen"]].copy()
    p["sym4"] = p["symbol"].astype(str).str[:4]
    if quality:
        from trading.jp_intraday.extra_features import attach_extra_features
        p = attach_extra_features(p)
        mst = pd.read_parquet("data/jp_daily_history/master.parquet",
                              columns=["Code", "MktNm"]).drop_duplicates("Code")
        mst["sym4"] = mst["Code"].astype(str).str[:4]
        p = p.merge(mst[["sym4", "MktNm"]].drop_duplicates("sym4"), on="sym4", how="left")
    f = close_features()
    m = f.merge(p, on=["date", "sym4"], how="inner")
    assert m["ret_on_fwd"].notna().mean() > .3, "target merge too sparse"
    if quality:
        m = m[_quality_mask(m)].copy()
    m["target"] = m["ret_on_fwd"] - m.groupby("date")["ret_on_fwd"].transform("mean")
    return m.dropna(subset=["target"])


def ridge_wf(m: pd.DataFrame) -> pd.Series:
    m = m.sort_values("date")
    m["ym"] = m["date"].dt.to_period("M")
    months = sorted(m["ym"].unique())
    out = []
    for i in range(2, len(months)):
        tr = m[m["ym"] < months[i]].dropna(subset=FEATS)
        te = m[m["ym"] == months[i]]
        if len(tr) < 3000 or te.empty:
            continue
        X = tr[FEATS].to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-12
        b = np.linalg.solve(((X - mu) / sd).T @ ((X - mu) / sd)
                            + 30 * np.eye(len(FEATS)), ((X - mu) / sd).T @ tr["target"])
        pr = pd.Series(((te[FEATS].fillna(0).to_numpy() - mu) / sd) @ b, index=te.index)
        out.append(pr)
    return pd.concat(out) if out else pd.Series(dtype=float)


def ls_daily(m: pd.DataFrame, score: pd.Series, lo, hi, long_only=False) -> dict:
    sub = m.loc[score.index].assign(score=score)
    sub = sub[(sub["date"] >= lo) & (sub["date"] <= hi)]
    rows = []
    for dt, g in sub.groupby("date"):
        hi_c = g[g["score"] >= g["score"].quantile(.9)]
        if long_only:
            r = hi_c["target"].mean()             # residual already market-neutral
            rows.append((dt, r - RT_COST / 2))
            continue
        short_ok = g["shortable"].fillna(False) & ~g["short_restricted"].fillna(False)
        lo_c = g[(g["score"] <= g["score"].quantile(.1)) & short_ok]
        if len(hi_c) < 3 or len(lo_c) < 3:
            continue
        gross = hi_c["target"].mean() - lo_c["target"].mean()
        cost = RT_COST + SHORT_RATE / SESSIONS
        rows.append((dt, gross - cost))
    if not rows:
        return {"sharpe": None}
    r = pd.Series(dict(rows)).sort_index()
    r.index = pd.to_datetime(r.index)
    mo = r.groupby(r.index.to_period("M")).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    gross_bps = float(r.mean() * 1e4) + (RT_COST + SHORT_RATE / SESSIONS) * 1e4
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_pct": round(float(r.mean() * 252 * 100), 2),
            "gross_bps_day": round(gross_bps, 2),
            "cost_bps_day": round((RT_COST + SHORT_RATE / SESSIONS) * 1e4, 2),
            "neg_months": int((mo < 0).sum()), "months": int(len(mo)),
            "top5_share": None if top5 is None else round(top5, 3),
            "days": int(len(r))}


def main() -> None:
    m = attach_target()
    res = {"spec": "docs/PREREGISTER_CLOSE_AUCTION_OVERNIGHT.md",
           "n_obs": int(len(m)), "n_days": int(m["date"].nunique()),
           "feature_ic_selection": {}}
    sel = m[(m["date"] >= SEL[0]) & (m["date"] <= SEL[1])]
    for c in FEATS:
        ic = sel.groupby("date").apply(lambda g: g[c].corr(g["target"], "spearman")).dropna()
        res["feature_ic_selection"][c] = {"ic": round(float(ic.mean()), 4),
                                          "t": round(float(ic.mean() / ic.std() * len(ic) ** .5), 2)}
    pred = ridge_wf(m)
    # S1(auction reversal) は先読み判明で除外。実行可能な反転=last30/afternoon を主にする。
    res["executable_note"] = ("auction_jump dropped: look-ahead (uses 15:30 to trade at "
                              "15:30) + same-bar coupling with ret_on_fwd denominator.")
    res["selection"] = {
        "S2_last30_reversal": ls_daily(m, -m["last30"], *SEL),
        "S2b_afternoon_reversal": ls_daily(m, -m["afternoon"], *SEL),
        "S3_ridge": ls_daily(m, pred, *SEL) if len(pred) else {"sharpe": None},
        "S4_last30_longonly": ls_daily(m, -m["last30"], *SEL, long_only=True),
        "S5_ridge_longonly": ls_daily(m, pred, *SEL, long_only=True) if len(pred) else {"sharpe": None},
    }
    # S6 unit-lot of ridge
    if len(pred):
        fr = m.loc[pred.index].assign(_sc=pred)
        fr = fr[(fr["date"] >= SEL[0]) & (fr["date"] <= SEL[1])].copy()
        fr["_s"] = fr["_sc"] - fr.groupby("date")["_sc"].transform("mean")
        fr["raw_open"] = fr["raw_close"]; fr["open"] = fr["raw_close"]
        fr["intraday_ret"] = fr["ret_on_fwd"]
        daily, _ = unit_lot_backtest(fr, capital_yen=2e7, names_per_side=8,
                                     margin_ratio=2.0, cost_bps_side=1.0,
                                     construction="magnitude")
        if len(daily):
            carry = daily["short_yen"] * SHORT_RATE / SESSIONS
            rr = ((daily["net_yen"] - carry) / 2e7)
            res["selection"]["S6_ridge_unitlot"] = {
                "sharpe": round(float(rr.mean() / rr.std() * 252 ** .5), 3),
                "ann_pct": round(float(rr.mean() * 252 * 100), 2), "days": int(len(rr))}
    def passed(s):
        return (s.get("sharpe") or -9) >= 1.0 and \
            (s.get("gross_bps_day") or -9) >= 2 * (s.get("cost_bps_day") or 9) and \
            s.get("months", 0) > 0 and (s.get("neg_months", 9) * 3 <= s.get("months", 0)) and \
            (s.get("top5_share") or 9) < .25
    res["passing_selection"] = [k for k, v in res["selection"].items()
                                if "sharpe" in v and passed(v)]
    res["decision"] = "SEE_CONFIRMATION" if res["passing_selection"] else "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
