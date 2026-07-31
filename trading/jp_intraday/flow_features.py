"""売買内訳（投資部門別の売買代金）から PIT 安全なフロー特徴量を作る.

## なぜこれか

2026-07-31 に気配前提が棄却され、**当日寄値をシグナルに使う戦略は全て執行不能**になった。
残るのは「前日までに確定したデータで発注し、寄成→引成で建てて落とす」形だけ。
売買内訳は過去の検証で**唯一グロスαが実在した非価格データ**（4年安定・1272銘柄に分散・
+1日ラグでもリークなし）で、棄却理由は**コストのみ**だった（損益分岐2.47bps/side vs
当時の想定7-10bps）。板寄せコストが 寄り1.5 / 引け0.5 bps/side に訂正されたので、
**損益分岐を跨いだ可能性がある**＝再評価に値する唯一の候補。

## PIT（先読み防止）

J-Quants の売買内訳は取引日 D のデータが翌営業日以降に公表される。公表時刻を実測で
確認できないため、**取引日 D のフローは D+`lag` 営業日の寄付き以降でのみ使用可**とし、
既定は保守側の `lag=2`（D+1 の夕方公表でも間に合う）。lag=1 は感度分析としてのみ見る。

実装は「特徴量を計算してから営業日インデックスで lag だけシフト」する。暦日でなく
**営業日**でシフトするのは、連休を跨いで実際より新しいデータを掴まないため
（schema v7 で修正した ret_on_fwd と同じ罠）。
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

# 売買代金の内訳列（Va = value/代金）。Vo(出来高)は使わない：価格帯の違いで比較不能なため。
_SELL = ["LongSellVa", "ShrtNoMrgnVa", "MrgnSellNewVa", "MrgnSellCloseVa"]
_BUY = ["LongBuyVa", "MrgnBuyNewVa", "MrgnBuyCloseVa"]

# ★注意: 売買内訳は**会計恒等式で 買い代金 ≡ 売り代金**（全約定に買い手と売り手がいる）。
# したがって (buy-sell)/total は定義上つねに 0 で、特徴量にならない（実測で確認済み）。
# 情報は「ネット」ではなく**構成比**＝どの主体が買い、どの主体が売ったかにある。
FLOW_FEATURES = [
    "flow_mrgn_net_z",     # 信用・空売り勢の買い越し（＝現物勢の売り越し）のz（主軸）
    "flow_mrgn_new_z",     # 信用新規の方向性（新規買い−新規売り）のz
    "flow_close_z",        # 信用返済の方向性（返済買い−返済売り）のz
    "flow_short_z",        # 空売り比率（現物空売り＋信用新規売り）のz
    "flow_turn_z",         # 売買代金そのものの異常度（フロー総量のz）
]


def _load_breakdown(years=range(2018, 2027)) -> pd.DataFrame:
    frames = []
    for y in years:
        for f in glob.glob(f"data/jp_flows/breakdown_{y}.parquet"):
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError("data/jp_flows/breakdown_*.parquet が見つかりません")
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["Date"])
    d["symbol"] = d["Code"].astype(str)
    return d


def _zscore(s: pd.Series, by: pd.Series, win: int = 20) -> pd.Series:
    """銘柄内の直近 win 日で標準化（銘柄固有の水準差を消す）.

    transform(lambda rolling) は遅いので groupby.rolling（Cython）を使う（AGENTS の方針）。
    """
    g = s.groupby(by)
    m = g.rolling(win, min_periods=10).mean().reset_index(level=0, drop=True)
    sd = g.rolling(win, min_periods=10).std().reset_index(level=0, drop=True)
    return (s - m) / sd.replace(0, np.nan)


def build_flow_features(lag: int = 2) -> pd.DataFrame:
    """(date, symbol, FLOW_FEATURES...) を返す。date は**その特徴量を使ってよい取引日**.

    lag は営業日単位。lag=2 なら「取引日 D のフローは D+2 の寄付きから使用可」。
    """
    d = _load_breakdown()
    sell = d[_SELL].sum(axis=1)
    buy = d[_BUY].sum(axis=1)
    total = (sell + buy).replace(0, np.nan)

    f = pd.DataFrame({"date": d["date"], "symbol": d["symbol"]})
    # 信用+空売り勢のネット買い ＝ −(現物勢のネット買い)。恒等式の下で唯一意味を持つ「ネット」。
    f["_mrgn_net"] = ((d["MrgnBuyNewVa"] + d["MrgnBuyCloseVa"])
                      - (d["MrgnSellNewVa"] + d["MrgnSellCloseVa"] + d["ShrtNoMrgnVa"])) / total
    f["_mrgn_new"] = (d["MrgnBuyNewVa"] - d["MrgnSellNewVa"]) / total  # 信用新規の方向
    f["_close"] = (d["MrgnBuyCloseVa"] - d["MrgnSellCloseVa"]) / total  # 信用返済の方向
    f["_short"] = (d["ShrtNoMrgnVa"] + d["MrgnSellNewVa"]) / total      # 空売り比率
    f["_turn"] = np.log(total)                                          # フロー総量

    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    for src, dst in (("_mrgn_net", "flow_mrgn_net_z"), ("_mrgn_new", "flow_mrgn_new_z"),
                     ("_close", "flow_close_z"), ("_short", "flow_short_z"),
                     ("_turn", "flow_turn_z")):
        f[dst] = _zscore(f[src], f["symbol"])

    # ── PIT シフト: 営業日インデックスで lag 日ずらす ──
    # 暦日でなく営業日で行う（連休跨ぎで実際より新しいデータを掴まないため）。
    sessions = pd.Index(sorted(f["date"].unique()))
    pos = pd.Series(range(len(sessions)), index=sessions)
    f["_pos"] = f["date"].map(pos) + lag
    usable = f["_pos"] < len(sessions)
    f = f[usable].copy()
    f["date"] = sessions[f["_pos"].astype(int)]        # 使用可能になる取引日に付け替え
    return f[["date", "symbol"] + FLOW_FEATURES].dropna(subset=FLOW_FEATURES, how="all")


def attach_flow_features(panel: pd.DataFrame, lag: int = 2) -> pd.DataFrame:
    """パネルにフロー特徴量を left join する（欠損はNaNのまま＝戦略側で落とす）."""
    f = build_flow_features(lag=lag)
    p = panel.copy()
    p["symbol"] = p["symbol"].astype(str)
    return p.merge(f, on=["date", "symbol"], how="left")
