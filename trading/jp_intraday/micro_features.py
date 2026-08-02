"""前日の日中マイクロストラクチャから、気配不要なクロスセクション特徴量を作る.

## なぜこれか

2026-07-31 に気配前提が棄却され、当日寄値をシグナルに使う戦略は全て執行不能になった。
残るのは「前日引けまでに確定したデータで発注し、寄成→引成で建てて落とす」形だけ。
価格系（日足）と需給系（売買内訳）は既に潰れているが、**分足マイクロストラクチャを
横断面予測に使う軸は未検証**。前日の板の"質"（引け板寄せの偏り、出来高の時間分布、
終盤ドリフト、VWAP乖離）は日足には現れない情報を持ちうる。

全特徴量は**前日の引けまでに確定**するので、当日の寄成発注に間に合う（PIT安全）。

## 実装上の注意

1分足は120M行(1.1GB)あるので row group 単位でストリーム集計する（全読みするとメモリを
数GB食う）。日付は timestamp から JST で取る（tz-aware のまま .dt.date すると
UTC 変換で日付がずれる罠がある）。

出力の `date` は**その特徴量を使ってよい取引日**＝素材となった取引日の翌営業日。
営業日インデックスでシフトする（暦日だと連休跨ぎで新しいデータを掴む）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MICRO_FEATURES = [
    "mic_close_auc_share",   # 引け板寄せの出来高シェア（引けに需給が偏った度合い）
    "mic_open_auc_share",    # 寄付き板寄せの出来高シェア
    "mic_am_pm_spread",      # 前場リターン − 後場リターン（日中の反転度）
    "mic_late_drift",        # 14:00→引け のリターン（終盤の方向性）
    "mic_vwap_dev",          # 引値 / 日中VWAP − 1（引けが高い＝買い上がりで終わった）
    "mic_vol_concentration",  # 出来高の時間集中度（Herfindahl・分単位）
    "mic_rvol",              # 分足実現ボラ（日次レンジでは見えない細かさ）
]

_PATH = "data/jp_minutes_2y/jp_1m_2024-08-01_2026-07-24.parquet"


def _agg_one(df: pd.DataFrame) -> pd.DataFrame:
    """1つの row group（複数銘柄×複数日を含みうる）を (symbol, day) 単位に集計."""
    ts = df["timestamp"].dt.tz_convert("Asia/Tokyo")
    df = df.assign(day=ts.dt.normalize().dt.tz_localize(None),
                   hm=ts.dt.hour * 100 + ts.dt.minute)
    g = df.groupby(["symbol", "day"], sort=False)
    out = g.agg(vol=("volume", "sum"),
                v2=("volume", lambda s: float((s.astype("float64") ** 2).sum())),
                n=("volume", "size"),
                px_open=("open", "first"), px_close=("close", "last"),
                hi=("high", "max"), lo=("low", "min"))
    # 板寄せバー（09:00 と 15:30）と時間帯別の値を別集計して結合
    def _slice(mask, cols):
        sub = df[mask]
        if sub.empty:
            return pd.DataFrame(columns=cols)
        gg = sub.groupby(["symbol", "day"], sort=False)
        return pd.DataFrame({cols[0]: gg["volume"].sum(),
                             cols[1]: gg["close"].last(),
                             cols[2]: gg["open"].first()})
    op = _slice(df["hm"].eq(900), ["ovol", "oclose", "oopen"])
    cl = _slice(df["hm"].eq(1530), ["cvol", "cclose", "copen"])
    am = _slice(df["hm"].between(900, 1130), ["amvol", "amclose", "amopen"])
    pm = _slice(df["hm"].between(1230, 1530), ["pmvol", "pmclose", "pmopen"])
    lt = _slice(df["hm"].between(1400, 1530), ["ltvol", "ltclose", "ltopen"])
    out = out.join([op, cl, am, pm, lt], how="left")
    # VWAP と実現ボラは別途（近似: 分足終値の等加重VWAPでなく出来高加重）
    df["_pv"] = df["close"] * df["volume"]
    out["pv"] = g["_pv"].sum()
    r = np.log(df["close"] / df.groupby(["symbol", "day"], sort=False)["close"].shift(1))
    df["_r2"] = r ** 2
    out["r2"] = df.groupby(["symbol", "day"], sort=False)["_r2"].sum()
    return out.reset_index()


def build_micro_features(path: str = _PATH, lag: int = 1) -> pd.DataFrame:
    """(date, symbol, MICRO_FEATURES...) — date は特徴量を使ってよい取引日."""
    f = pq.ParquetFile(path)
    parts = []
    for i in range(f.num_row_groups):
        parts.append(_agg_one(f.read_row_group(i).to_pandas()))
    a = pd.concat(parts, ignore_index=True)
    # row group 境界で (symbol, day) が分割されるので再集計
    sums = ["vol", "v2", "n", "ovol", "cvol", "amvol", "pmvol", "ltvol", "pv", "r2"]
    firsts = ["px_open", "oopen", "amopen", "pmopen", "ltopen"]
    lasts = ["px_close", "oclose", "cclose", "amclose", "pmclose", "ltclose"]
    agg = {c: "sum" for c in sums if c in a.columns}
    agg.update({c: "first" for c in firsts if c in a.columns})
    agg.update({c: "last" for c in lasts if c in a.columns})
    agg["hi"] = "max"
    agg["lo"] = "min"
    d = a.groupby(["symbol", "day"], as_index=False).agg(agg)

    v = d["vol"].replace(0, np.nan)
    out = pd.DataFrame({"symbol": d["symbol"].astype(str), "day": d["day"]})
    out["mic_close_auc_share"] = d["cvol"] / v
    out["mic_open_auc_share"] = d["ovol"] / v
    out["mic_am_pm_spread"] = (d["amclose"] / d["amopen"] - 1) - (d["pmclose"] / d["pmopen"] - 1)
    out["mic_late_drift"] = d["ltclose"] / d["ltopen"] - 1
    out["mic_vwap_dev"] = d["px_close"] / (d["pv"] / v) - 1
    out["mic_vol_concentration"] = d["v2"] / (v ** 2)          # Herfindahl（分単位シェアの二乗和）
    out["mic_rvol"] = np.sqrt(d["r2"].clip(lower=0))

    # 銘柄内でzスコア化（銘柄固有の水準差を消す。横断面比較のため）
    out = out.sort_values(["symbol", "day"]).reset_index(drop=True)
    for c in MICRO_FEATURES:
        g = out[c].groupby(out["symbol"])
        m = g.rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
        s = g.rolling(60, min_periods=20).std().reset_index(level=0, drop=True)
        out[c] = ((out[c] - m) / s.replace(0, np.nan)).clip(-5, 5)

    # ── PIT: 素材日 day のデータは day+lag 営業日から使用可 ──
    sessions = pd.Index(sorted(out["day"].unique()))
    pos = pd.Series(range(len(sessions)), index=sessions)
    p = out["day"].map(pos) + lag
    ok = p < len(sessions)
    out = out[ok].copy()
    out["date"] = sessions[p[ok].astype(int)]
    return out[["date", "symbol"] + MICRO_FEATURES]


def attach_micro_features(panel: pd.DataFrame, path: str = _PATH, lag: int = 1) -> pd.DataFrame:
    m = build_micro_features(path, lag=lag)
    p = panel.copy()
    p["symbol"] = p["symbol"].astype(str)
    m["symbol"] = m["symbol"].astype(str)
    # 分足は4桁コード、パネルは5桁
    m["symbol"] = m["symbol"].map(lambda x: x if len(x) == 5 else x + "0")
    return p.merge(m, on=["date", "symbol"], how="left")
