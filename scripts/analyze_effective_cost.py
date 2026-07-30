"""実効コストの集計と閾値判定（寄り/引けを分離）.

`run_live cost` が日々書き出す data/live_reports/cost_YYYY-MM-DD.jsonl を読み、
- 脚別（entry=寄成 / exit=引成）・サイド別の実効スリッページ（¥加重・中央値）
- 5日移動平均と、live/README の運用閾値による判定
- J-Quants の確定した公式寄値/引値で ref_px を上書き（当日は速報値のため）
を出力する。

**なぜ脚を分けるか**（R9実測）: 引け板寄せは日次売買代金の13.5%を占め、寄付板寄せ(4.0%)の
3.4倍厚い。同じ建玉でも参加率は寄り側が約4倍で、実効コストは構造的に非対称。
往復合算で持つと入口の悪さが出口に隠れる。

判定閾値（live/README と同一・往復ベース）:
  5日MA ≤14bps(=片道7bps相当)  : 継続
  14〜20bps                    : 継続するが増額凍結
  >20bps が5日継続             : サイズ半減（¥10M）
  >26bps                       : 停止して原因分析（全期間の損益分岐≈28bps）

実行: PYTHONPATH=. python scripts/analyze_effective_cost.py [--days 30]
"""
import argparse
import glob
import json

import pandas as pd


def _load_records() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob("data/live_reports/cost_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _official_prices() -> pd.DataFrame:
    """J-Quants 日次バーの公式寄値/引値（未調整）。翌営業日に確定する."""
    frames = []
    for f in sorted(glob.glob("data/jp_daily_history/daily_adj_202[5-9].parquet")):
        frames.append(pd.read_parquet(f, columns=["Date", "Code", "O", "C"]))
    if not frames:
        return pd.DataFrame(columns=["day", "symbol", "open", "close"])
    d = pd.concat(frames, ignore_index=True)
    d["day"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    # kabu は4桁、J-Quants は5桁。4桁に寄せて突合
    d["symbol"] = d["Code"].astype(str).map(lambda s: s[:-1] if len(s) == 5 else s)
    return d.rename(columns={"O": "open", "C": "close"})[["day", "symbol", "open", "close"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    r = _load_records()
    if r.empty:
        print("実測データがありません（Windows側で `run_live cost` を毎日実行してください）。")
        print("live/README.md の「実効コストの脚別実測」を参照。")
        return

    off = _official_prices()
    if not off.empty:                       # 確定値で ref_px を上書きして再計算
        r = r.merge(off, on=["day", "symbol"], how="left", suffixes=("", "_off"))
        from trading.jp_intraday.live.costs import leg_slippage_bps
        fixed = r["open"].where(r["leg"].eq("entry"), r["close"])
        use = fixed.fillna(r["ref_px"])
        r["slip_bps"] = [leg_slippage_bps(f, p, s, l) for f, p, s, l
                         in zip(r["fill_px"], use, r["side"], r["leg"])]
        r["ref_src"] = fixed.notna().map({True: "official", False: "provisional"})

    r = r[r["slip_bps"].notna()].copy()
    days = sorted(r["day"].unique())[-args.days:]
    r = r[r["day"].isin(days)]
    print(f"対象 {len(days)}営業日 / {len(r)}約定 "
          f"（確定値 {(r.get('ref_src') == 'official').mean() * 100:.0f}%）\n")

    def _wavg(g):
        return (g["slip_bps"] * g["notional"]).sum() / g["notional"].sum()

    print("== 脚別・サイド別の実効コスト（bps・正=コスト・¥加重） ==")
    for leg in ("entry", "exit"):
        sub = r[r["leg"] == leg]
        if sub.empty:
            continue
        lab = "寄成(entry)" if leg == "entry" else "引成(exit)"
        print(f"  {lab}: 全体 {_wavg(sub):+.2f} / 中央値 {sub['slip_bps'].median():+.2f} (n={len(sub)})")
        for side, nm in (("2", "買い"), ("1", "売り")):
            s2 = sub[sub["side"].astype(str) == side]
            if len(s2):
                print(f"      {nm}: {_wavg(s2):+.2f} (n={len(s2)})")

    daily = r.groupby(["day", "leg"]).apply(_wavg).unstack(fill_value=float("nan"))
    daily["roundtrip"] = daily.get("entry", 0) + daily.get("exit", 0)
    print("\n== 日次の往復コスト推移（直近10日） ==")
    print(daily.tail(10).round(2).to_string())

    ma5 = daily["roundtrip"].tail(5).mean()
    print(f"\n== 判定: 5日移動平均の往復コスト = {ma5:.2f} bps ==")
    if ma5 <= 14:
        print("  → ✅ 継続（想定7bps/side以内）")
    elif ma5 <= 20:
        print("  → ⚠️ 継続するが増額凍結（10bps/side水準）")
    elif ma5 <= 26:
        print("  → 🔻 5日継続ならサイズ半減（¥10M）を検討")
    else:
        print("  → 🛑 停止して原因分析（損益分岐≈28bps往復に接近）")
    print("\n参考: 本番前提は片道7bps=往復14bps。実勢が片道3bpsなら年率で最大+17pt相当の上振れ。")


if __name__ == "__main__":
    main()
