#!/usr/bin/env python3
"""Diagnostic: is the afternoon->close residual return predictable? (Optiver TATC)

Inspired by the Optiver "Trading at the Close" Kaggle competition (via @mi_fits/
@gamella). We CANNOT use its order-book features (imbalance/bid-ask/far-near) --
board data is inaccessible for JP retail (real-money NO-GO). We CAN build the
price/volume/vol/cross-sectional subset from 1-minute OHLCV.

This is a DIAGNOSTIC (per-feature cross-sectional IC + pooled-ridge OOS IC +
a decile L/S gross read), NOT a strategy selection. Decision time T=15:00 JST
(30 min before the 15:30 close auction; live orders must be in by 15:24).
Target = residualized close(15:30)/close(T) - 1 (stock minus universe mean).
Data: data/jp_minutes_2y/jp_1m_2024-08-01_2026-07-24.parquet (1022 names).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "data/jp_minutes_2y/jp_1m_2024-08-01_2026-07-24.parquet"
OUT = Path("data/jp_close_prediction")
T_CUT = "15:00"          # decision minute (close of this bar)
CLOSE = "15:30"          # auction print


def load_afternoon() -> pd.DataFrame:
    d = pd.read_parquet(SRC, columns=["timestamp", "symbol", "open", "high",
                                       "low", "close", "volume"])
    ts = pd.to_datetime(d["timestamp"])
    d["date"] = ts.dt.tz_convert("Asia/Tokyo").dt.normalize().dt.tz_localize(None)
    d["hm"] = ts.dt.tz_convert("Asia/Tokyo").dt.strftime("%H:%M")
    d["mins"] = ts.dt.tz_convert("Asia/Tokyo").dt.hour * 60 \
        + ts.dt.tz_convert("Asia/Tokyo").dt.minute
    return d


def px_at(g: pd.DataFrame, hm: str, col: str = "close") -> pd.Series:
    return g[g["hm"] == hm].set_index(["date", "symbol"])[col]


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    tcut = 15 * 60
    aft = d[(d["mins"] >= 12 * 60 + 30) & (d["mins"] <= tcut)]     # 12:30..15:00
    key = ["date", "symbol"]
    c_open9 = d[d["hm"] == "09:00"].set_index(key)["open"]
    c_1130 = px_at(d, "11:30")
    c_1230 = px_at(d, "12:30")
    c_1430 = px_at(d, "14:30")
    c_1450 = px_at(d, "14:50")
    c_T = px_at(d, T_CUT)
    c_close = px_at(d, CLOSE)
    g = aft.groupby(key)
    vol_aft = g["volume"].sum()
    vol_last30 = aft[aft["mins"] >= 14 * 60 + 30].groupby(key)["volume"].sum()
    hi_aft = g["high"].max()
    lo_aft = g["low"].min()
    # 1-min realized vol over the afternoon (log returns of 1-min closes)
    aft = aft.sort_values(key + ["mins"])
    aft["r1"] = np.log(aft["close"] / aft.groupby(key)["close"].shift(1))
    rvol = aft.groupby(key)["r1"].std()
    # VWAP up to T (typical price weighted by volume)
    aft["tp"] = (aft["high"] + aft["low"] + aft["close"]) / 3
    vwap = (aft.assign(pv=aft["tp"] * aft["volume"]).groupby(key)["pv"].sum()
            / vol_aft.replace(0, np.nan))
    f = pd.DataFrame(index=c_T.index)
    f["r_morning"] = (c_1130 / c_open9 - 1).reindex(f.index)
    f["r_aft_sofar"] = (c_T / c_1230 - 1).reindex(f.index)
    f["r_last30"] = (c_T / c_1430 - 1).reindex(f.index)
    f["r_last10"] = (c_T / c_1450 - 1).reindex(f.index)
    f["rvol_aft"] = rvol.reindex(f.index)
    f["vwap_dev"] = (c_T / vwap - 1).reindex(f.index)
    f["hl_range"] = ((hi_aft - lo_aft) / c_T).reindex(f.index)
    f["late_vol_share"] = (vol_last30 / vol_aft.replace(0, np.nan)).reindex(f.index)
    f["target"] = (c_close / c_T - 1).reindex(f.index)
    f = f.reset_index()
    # cross-sectional residualization (Optiver target is index-relative)
    for c in [c for c in f.columns if c not in ("date", "symbol")]:
        f[c] = f[c] - f.groupby("date")[c].transform("mean")
    return f.dropna(subset=["target"])


def diagnostics(f: pd.DataFrame) -> dict:
    feats = [c for c in f.columns if c not in ("date", "symbol", "target")]
    out = {"n_obs": int(len(f)), "n_days": int(f["date"].nunique()),
           "decision": T_CUT, "target": f"{T_CUT}->{CLOSE} residual",
           "target_std_bps": round(float(f["target"].std() * 1e4), 1),
           "feature_ic": {}}
    for c in feats:
        ic = f.groupby("date").apply(
            lambda g: g[c].corr(g["target"], method="spearman"))
        ic = ic.dropna()
        out["feature_ic"][c] = {
            "ic_mean": round(float(ic.mean()), 4),
            "t": round(float(ic.mean() / ic.std() * len(ic) ** .5), 2)}
    # pooled ridge, monthly walk-forward OOS IC + decile L/S gross
    f = f.sort_values("date")
    f["ym"] = f["date"].dt.to_period("M")
    months = sorted(f["ym"].unique())
    preds = []
    for i in range(3, len(months)):
        tr = f[f["ym"] < months[i]]
        te = f[f["ym"] == months[i]]
        if len(tr) < 5000 or te.empty:
            continue
        X = tr[feats].fillna(0).to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xn = (X - mu) / sd
        y = tr["target"].to_numpy()
        beta = np.linalg.solve(Xn.T @ Xn + 30 * np.eye(Xn.shape[1]), Xn.T @ y)
        Xt = (te[feats].fillna(0).to_numpy() - mu) / sd
        p = te[["date", "target"]].copy()
        p["pred"] = Xt @ beta
        preds.append(p)
    P = pd.concat(preds)
    oos_ic = P.groupby("date").apply(
        lambda g: g["pred"].corr(g["target"], method="spearman")).dropna()
    # decile L/S gross (top10% - bottom10% of pred), per day, in bps
    def ls(g):
        hi = g[g["pred"] >= g["pred"].quantile(.9)]["target"].mean()
        lo = g[g["pred"] <= g["pred"].quantile(.1)]["target"].mean()
        return (hi - lo) * 1e4
    ls_bps = P.groupby("date").apply(ls).dropna()
    out["pooled_ridge"] = {
        "oos_ic_mean": round(float(oos_ic.mean()), 4),
        "oos_ic_t": round(float(oos_ic.mean() / oos_ic.std() * len(oos_ic) ** .5), 2),
        "decile_ls_gross_bps_per_day": round(float(ls_bps.mean()), 2),
        "decile_ls_t": round(float(ls_bps.mean() / ls_bps.std() * len(ls_bps) ** .5), 2),
        "oos_days": int(len(oos_ic))}
    out["cost_context"] = {
        "taker_entry_bps_side": 6.66, "auction_exit_bps_side": 0.5,
        "round_trip_bps": 7.16,
        "note": "entry at 15:00 is continuous-session TAKER; exit at 15:30 auction. "
                "gross decile spread must clear ~7bps round trip to be tradable."}
    return out


def main() -> None:
    d = load_afternoon()
    f = build_features(d)
    res = diagnostics(f)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
