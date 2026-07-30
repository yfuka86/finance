"""実効コストの集計と閾値判定（寄り/引けを分離）.

`run_live cost` が日々書き出す data/live_reports/cost_YYYY-MM-DD.jsonl を読み、
- 脚別（entry=寄成 / exit=引成）・サイド別の実効スリッページ（¥加重・中央値）
- 5日移動平均と、live/README の運用閾値による判定
- J-Quants の確定した公式寄値/引値で ref_px を上書き（当日は速報値のため）
を出力する。

**なぜ脚を分けるか**（R9実測）: 引け板寄せは日次売買代金の13.5%を占め、寄付板寄せ(4.0%)の
3.4倍厚い。同じ建玉でも参加率は寄り側が約4倍で、実効コストは構造的に非対称。
往復合算で持つと入口の悪さが出口に隠れる。

**2026-07-30 全面改訂**: 板寄せは単一約定価格なので「fill vs 公式価格」は構造的にゼロで、
実効コストの直接測定にはならない（自分のインパクトは公式価格に内包され、バックテストも
同じ価格を使うため分離不能）。よって監視するのは**コスト前提が崩れる兆候**:

  参加率 p90       : 寄り ≤3% / 引け ≤1.5%   （推定コストの driver。超えるとインパクト増）
  数量約定率       : ≥98%                    （寄らず・部分約定の検知）
  |slip| ¥加重     : ≤0.5bps                 （0が正常。非ゼロは板寄せ外約定=SORのPTS迂回等）
  プレミアム料     : 発生の有無を別建てで記録（上限100bps/日で桁が違う）

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
    """公式寄値/引値。翌営業日に確定する.

    ★罠: daily_adj_*.parquet の**生値列 O/C は 2022-2024 に列自体が無く、2025 も
    非NaNが3.2%しかない**。columns=["O","C"] を決め打ちで読むと古い年で
    ArrowInvalid、新しい年でもデータの3%しか拾えない。列の有無を見て、
    生値が薄い場合は調整値(AdjO/AdjC)にフォールバックする（約定価格との突合は
    調整イベント日以外では一致する）。
    """
    import pyarrow.parquet as pq
    frames = []
    for f in sorted(glob.glob("data/jp_daily_history/daily_adj_202[4-9].parquet")):
        cols = set(pq.ParquetFile(f).schema.names)
        want = ["Date", "Code"] + (["O", "C"] if {"O", "C"} <= cols else ["AdjO", "AdjC"])
        d = pd.read_parquet(f, columns=want).rename(columns={"AdjO": "O", "AdjC": "C"})
        if d["O"].notna().mean() < 0.5 and {"AdjO", "AdjC"} <= cols:   # 生値が薄い年は調整値で補完
            adj = pd.read_parquet(f, columns=["Date", "Code", "AdjO", "AdjC"])
            d["O"] = d["O"].fillna(adj["AdjO"])
            d["C"] = d["C"].fillna(adj["AdjC"])
        frames.append(d)
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

    print("== 参加率（自分の約定 / その板寄せの総約定代金・%） ==")
    if "participation_pct" in r.columns and r["participation_pct"].notna().any():
        for leg, lim in (("entry", 3.0), ("exit", 1.5)):
            sub = r[(r["leg"] == leg) & r["participation_pct"].notna()]
            if len(sub):
                p50, p90 = sub["participation_pct"].median(), sub["participation_pct"].quantile(0.9)
                ok = "✅" if p90 <= lim else "⚠️"
                lab = "寄成" if leg == "entry" else "引成"
                print(f"  {lab}: p50 {p50:.2f}% / p90 {p90:.2f}% (閾値 p90≤{lim}%) {ok}")
    else:
        print("  （板寄せ出来高が未取得。翌日J-Quantsで補完するか board のフィールド名を確認）")

    if "fill_ratio" in r.columns and r["fill_ratio"].notna().any():
        fr = r["fill_ratio"].dropna()
        ok = "✅" if fr.mean() >= 0.98 else "⚠️"
        print(f"\n== 数量約定率: 平均 {fr.mean()*100:.1f}% / 最小 {fr.min()*100:.1f}% "
              f"(閾値 ≥98%) {ok} ==")

    print("\n== |slip|（板寄せ外約定の検知・0が正常） ==")
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

    ma5 = daily["roundtrip"].tail(5).abs().mean()
    print(f"\n== 判定: |slip| 5日平均(往復) = {ma5:.2f} bps ==")
    if ma5 <= 0.5:
        print("  → ✅ 正常（板寄せで約定できている）")
    else:
        print("  → ⚠️ 板寄せ外での約定が疑われる（SORのPTS迂回・寄らず等）。約定明細を確認")
    print("\n参考: オフライン推定の板寄せ実効コストは 寄り1.5bps/引け0.5bps（往復2.0・レンジ0.6-4.5）。")
    print("      現行の計画値は対称2.0bps/side（往復4bps）。verify_baselineの7bps期待値は")
    print("      環境再現アンカーとして据え置き、正本の差し替えはライブ実測40営業日の合格後。")


if __name__ == "__main__":
    main()
