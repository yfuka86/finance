"""日中±バリア執行オーバーレイ（5分足・リサーチ専用）.

⚠️ 本番使用禁止（ラウンド3の約定現実性検証で棄却）:
  本シミュの「ブリーチ足の終値で約定」は、TP側でバリアを超えた分（オーバーシュート）
  を利益計上する実装不能な好条件。現実の執行モデルでは名目+4.3 Shが消失する:
    - タッチ約定(OCO相当):      OOS net7 2.92（ベースライン5.48より悪化）
    - バリア価格クランプ下限:    0.93
    - 次バー始値約定(ソフト監視): 6.55（+1.1）だがテイカー5bpで4.36（-1.1）
  → 正直な改善バンドは -2.6〜+1.1 Sh・中心ほぼゼロ。kabu APIはOCO非対応かつ
    HoldQty拘束で同一建玉にTP指値+SL逆指値の同時発注も不可（実装面でも不成立）。
  本モジュールは「執行仮定の感応度分析」用のリサーチツールとして残す。

学び（有効なまま）: アルファは寄付き後15分に集中・遅延エントリーは全滅
→ 寄付きオークション参加が戦略の本体。執行は寄成→引成を維持。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIVE_MIN_PARQUET = "data/jp_minutes_2y/jp_5m_2024-08-01_2026-07-24.parquet"


def _to4(code: pd.Series) -> pd.Series:
    s = code.astype(str)
    return s.where(s.str.len() != 5, s.str[:-1])


def barrier_backtest(blotter: pd.DataFrame, x_pct: float = 1.0,
                     cost_bps_side: float = 7.0,
                     bars_path: str = FIVE_MIN_PARQUET) -> pd.DataFrame:
    """日次ブロッター（date,symbol,weight）に±x%バリア執行を適用した日次リターン。

    コストは基準線と同じ1往復/日（バリアでも往復回数は不変）。
    Returns daily frame: date, gross, net, hit_rate（バリア到達率）.
    """
    bars = pd.read_parquet(bars_path)
    bars["date"] = bars["timestamp"].dt.tz_convert("Asia/Tokyo").dt.normalize().dt.tz_localize(None)
    bars["sym4"] = _to4(bars["symbol"])

    b = blotter[["date", "symbol", "weight"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    b["sym4"] = _to4(b["symbol"])
    b = b[b["weight"] != 0]

    m = bars.merge(b[["date", "sym4", "weight"]], on=["date", "sym4"], how="inner")
    m = m.sort_values(["date", "sym4", "timestamp"])
    g = m.groupby(["date", "sym4"], sort=False)
    m["entry"] = g["open"].transform("first")
    m["ret"] = m["close"] / m["entry"] - 1.0
    x = x_pct / 100.0
    m["breach"] = m["ret"].abs() >= x
    # 最初のブリーチbin（無ければ最終bin）で手仕舞い
    m["bin_no"] = g.cumcount()
    breach_no = m["bin_no"].where(m["breach"]).groupby([m["date"], m["sym4"]]).transform("min")
    last_no = g["bin_no"].transform("max")
    exit_no = breach_no.fillna(last_no)
    exit_rows = m[m["bin_no"].eq(exit_no)].copy()
    exit_rows["pnl"] = exit_rows["weight"] * exit_rows["ret"]
    exit_rows["hit"] = exit_rows["bin_no"].lt(last_no[m["bin_no"].eq(exit_no)])

    daily = exit_rows.groupby("date").agg(
        gross=("pnl", "sum"),
        expo=("weight", lambda s: s.abs().sum()),
        hit_rate=("hit", "mean"),
    ).reset_index()
    daily["net"] = daily["gross"] - daily["expo"] * 2 * cost_bps_side / 10_000
    return daily[["date", "gross", "net", "hit_rate"]]
