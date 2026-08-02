"""未使用だった需給データ4系統から PIT 安全なクロスセクション特徴量を作る.

対象（いずれも当日寄値を必要としない＝気配レス執行と両立する）:
  1. 週次信用残 `margin_interest` … 制度/一般別の買残・売残（週次・記録日ベース）
  2. 空売り残高報告 `short_positions` … 発行済株式数比0.5%以上の大口ショート（開示日ベース）
  3. 増担保規制 `margin_alert` … 日々公表銘柄の残高・信用倍率・規制理由
  4. 業種別空売り比率 `short_ratio` … 業種(S33)単位の空売り比率

これらは過去に**単独戦略として**検証され棄却されている（AGENTS参照）。ここでの目的は
**気配不要MLの特徴量として増分があるか**であり、単独αの再検証ではない。

## PIT（公表ラグ）の扱い

公表時刻を実測できないため、すべて**保守側**に倒す。営業日インデックスでシフトする
（暦日だと連休跨ぎで実際より新しいデータを掴む）:
  - 週次信用残: 記録日(Date, 通常は金曜)から **+5営業日**（公表は3営業日後だがさらに余裕）
  - 空売り残高報告: 開示日(DiscDate)から **+1営業日**
  - 増担保規制: 公表日(PubDate)から **+1営業日**
  - 業種別空売り比率: 集計日(Date)から **+1営業日**

## 注意

信用残の株数は**分割調整されていない**（AGENTS）。したがって水準そのものは使わず、
**比率**（信用倍率・残高/出来高）と**同一銘柄内のzスコア**のみを使う。
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

EXTRA_FEATURES = [
    "xt_margin_ratio_z",    # 信用倍率(買残/売残)のz
    "xt_margin_chg_z",      # 買残の週次変化のz
    "xt_short_bal_chg_z",   # 売残の週次変化のz
    "xt_ssr_to_so",         # 大口空売り残高の発行済比（水準・%）
    "xt_ssr_chg",           # 同 変化幅
    "xt_alert_flag",        # 日々公表/増担保の対象か（0/1）
    "xt_alert_slratio_z",   # 日々公表銘柄の信用倍率のz
    "xt_sector_ssr_z",      # 業種別空売り比率のz（銘柄はその業種の値を継承）
]


def _shift_sessions(df: pd.DataFrame, src_col: str, lag: int,
                    sessions: pd.Index) -> pd.DataFrame:
    """src_col（素材日）を営業日インデックスで lag ずらして `date`（使用可能日）にする."""
    pos = pd.Series(range(len(sessions)), index=sessions)
    # 素材日が営業日でない場合（週末記録日など）は「その日以降で最初の営業日」に丸める
    idx = sessions.searchsorted(pd.to_datetime(df[src_col]), side="left")
    p = pd.Series(idx, index=df.index) + lag
    ok = p < len(sessions)
    out = df[ok].copy()
    out["date"] = sessions[p[ok].astype(int)]
    del pos
    return out


def _z(s: pd.Series, by: pd.Series, win: int = 26) -> pd.Series:
    g = s.groupby(by)
    m = g.rolling(win, min_periods=6).mean().reset_index(level=0, drop=True)
    sd = g.rolling(win, min_periods=6).std().reset_index(level=0, drop=True)
    return ((s - m) / sd.replace(0, np.nan)).clip(-5, 5)


def _read(pat: str) -> pd.DataFrame:
    fs = sorted(glob.glob(pat))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def build_extra_features(sessions: pd.Index) -> pd.DataFrame:
    """(date, symbol, EXTRA_FEATURES...) を返す。sessions はパネルの営業日インデックス."""
    sessions = pd.Index(sorted(pd.to_datetime(pd.Index(sessions).unique())))
    frames = []

    # ── 1. 週次信用残（記録日 +5営業日） ──
    mi = _read("data/jp_flows/margin_interest_*.parquet")
    if not mi.empty:
        mi["symbol"] = mi["Code"].astype(str)
        mi = mi.sort_values(["symbol", "Date"])
        long_, short_ = mi["LongVol"], mi["ShrtVol"].replace(0, np.nan)
        mi["_ratio"] = long_ / short_                       # 信用倍率
        g = mi.groupby("symbol")
        mi["_lchg"] = long_ / g["LongVol"].shift(1) - 1     # 買残の週次変化
        mi["_schg"] = mi["ShrtVol"] / g["ShrtVol"].shift(1) - 1
        mi["xt_margin_ratio_z"] = _z(np.log(mi["_ratio"].replace([np.inf, -np.inf], np.nan)),
                                     mi["symbol"])
        mi["xt_margin_chg_z"] = _z(mi["_lchg"].clip(-2, 2), mi["symbol"])
        mi["xt_short_bal_chg_z"] = _z(mi["_schg"].clip(-2, 2), mi["symbol"])
        frames.append(_shift_sessions(mi, "Date", 5, sessions)[
            ["date", "symbol", "xt_margin_ratio_z", "xt_margin_chg_z", "xt_short_bal_chg_z"]])

    # ── 2. 空売り残高報告（開示日 +1営業日・銘柄内で合算） ──
    sp = _read("data/jp_flows/short_positions_*.parquet")
    if not sp.empty:
        sp["symbol"] = sp["Code"].astype(str)
        agg = (sp.groupby(["DiscDate", "symbol"], as_index=False)["ShrtPosToSO"]
               .sum().rename(columns={"ShrtPosToSO": "xt_ssr_to_so"}))
        agg = agg.sort_values(["symbol", "DiscDate"])
        agg["xt_ssr_chg"] = agg.groupby("symbol")["xt_ssr_to_so"].diff()
        agg["xt_ssr_to_so"] = agg["xt_ssr_to_so"] * 100.0
        agg["xt_ssr_chg"] = agg["xt_ssr_chg"] * 100.0
        frames.append(_shift_sessions(agg, "DiscDate", 1, sessions)[
            ["date", "symbol", "xt_ssr_to_so", "xt_ssr_chg"]])

    # ── 3. 増担保・日々公表（公表日 +1営業日） ──
    ma = _read("data/jp_flows/margin_alert_*.parquet")
    if not ma.empty:
        ma["symbol"] = ma["Code"].astype(str)
        ma = ma.sort_values(["symbol", "PubDate"])
        ma["xt_alert_flag"] = 1.0
        ma["xt_alert_slratio_z"] = _z(np.log(pd.to_numeric(ma["SLRatio"], errors="coerce")
                                             .replace([np.inf, -np.inf], np.nan)
                                             .clip(lower=1e-3)), ma["symbol"])
        frames.append(_shift_sessions(ma, "PubDate", 1, sessions)[
            ["date", "symbol", "xt_alert_flag", "xt_alert_slratio_z"]]
            .drop_duplicates(subset=["date", "symbol"], keep="last"))

    out = None
    for f in frames:
        f = f.drop_duplicates(subset=["date", "symbol"], keep="last")
        out = f if out is None else out.merge(f, on=["date", "symbol"], how="outer")
    if out is None:
        return pd.DataFrame(columns=["date", "symbol"] + EXTRA_FEATURES)
    out["xt_alert_flag"] = out.get("xt_alert_flag", pd.Series(index=out.index)).fillna(0.0)
    return out


def attach_sector_ssr(panel: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    """業種別空売り比率（集計日 +1営業日）をパネルの sector 経由で付ける."""
    sr = _read("data/jp_derivatives/short_ratio_*.parquet")
    if sr.empty or "s33_code" not in panel.columns:
        panel["xt_sector_ssr_z"] = np.nan
        return panel
    tot = (sr["SellExShortVa"].fillna(0) + sr["ShrtWithResVa"].fillna(0)
           + sr["ShrtNoResVa"].fillna(0)).replace(0, np.nan)
    sr["_ssr"] = (sr["ShrtWithResVa"].fillna(0) + sr["ShrtNoResVa"].fillna(0)) / tot
    sr = sr.sort_values(["S33", "Date"])
    sr["xt_sector_ssr_z"] = _z(sr["_ssr"], sr["S33"], win=60)
    sessions = pd.Index(sorted(pd.to_datetime(pd.Index(sessions).unique())))
    sr = _shift_sessions(sr, "Date", 1, sessions)
    sr["s33_code"] = sr["S33"].astype(str)
    panel["s33_code"] = panel["s33_code"].astype(str)
    return panel.merge(sr[["date", "s33_code", "xt_sector_ssr_z"]].drop_duplicates(
        subset=["date", "s33_code"], keep="last"), on=["date", "s33_code"], how="left")


def attach_extra_features(panel: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.Index(sorted(panel["date"].unique()))
    p = panel.copy()
    p["symbol"] = p["symbol"].astype(str)
    e = build_extra_features(sessions)
    if not e.empty:
        p = p.merge(e, on=["date", "symbol"], how="left")
    p = attach_sector_ssr(p, sessions)
    for c in EXTRA_FEATURES:
        if c not in p.columns:
            p[c] = np.nan
    return p
