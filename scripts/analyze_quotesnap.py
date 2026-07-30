"""quotesnap（寄前気配3時点）と実寄値の乖離分析 — 戦略成立性の判定材料.

使い方: quotesnapを1朝以上実行後、当日夕方〜翌朝に
  PYTHONPATH=. python scripts/analyze_quotesnap.py
判定の目安（ノイズ予算検証 2026-07-30・AGENTS.md参照）:
  銘柄固有乖離σ ≤25bps → 成績影響ゼロ / σ≈51bps → 単元Sh-10% / σ≥100bps → -30%で要再設計。
  さらに「|気配ギャップ|大のテール銘柄で乖離が系統的に縮む方向か」（勝者の呪いの実測）を必ず見る。
"""
import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("data/live_reports/quotesnap_*.jsonl")):
    day = f.split("_")[-1].replace(".jsonl", "")
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r["day"] = day
        rows.append(r)
if not rows:
    raise SystemExit("quotesnapログがありません（Windowsで run_live quotesnap を実行）")
q = pd.DataFrame(rows)
q["calc"] = pd.to_numeric(q["calc"], errors="coerce")
q = q.dropna(subset=["calc"])

daily = []
for f in sorted(glob.glob("data/jp_daily_history/daily_adj_202[5-9].parquet")):
    d = pd.read_parquet(f, columns=["Date", "Code", "O", "C"])
    daily.append(d)
d = pd.concat(daily, ignore_index=True)
d["day"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
d["symbol"] = d["Code"].astype(str).map(lambda s: s[:-1] if len(s) == 5 else s)
d = d.rename(columns={"O": "open_actual"})
prev_c = d.sort_values("day").groupby("symbol")["C"].shift(1)
d["prev_close"] = prev_c

m = q.merge(d[["day", "symbol", "open_actual", "prev_close"]], on=["day", "symbol"], how="inner")
if m.empty:
    raise SystemExit("実寄値データ未着（collect_jp_daily_history 実行後に再試行）")
m["dev_bps"] = (m["open_actual"] / m["calc"] - 1) * 1e4
m["gap_quote"] = (m["calc"] / m["prev_close"] - 1) * 100
m["gap_actual"] = (m["open_actual"] / m["prev_close"] - 1) * 100

for snap, g in m.groupby("snap"):
    a = g["dev_bps"].abs()
    print(f"\n[{snap}] n={len(g)}  |乖離|bps: 中央値{a.median():.0f} p75 {a.quantile(.75):.0f} "
          f"p90 {a.quantile(.9):.0f} p99 {a.quantile(.99):.0f}  符号付き平均 {g['dev_bps'].mean():+.0f}")
    tail = g[g["gap_quote"].abs() >= 3]
    if len(tail):
        shrink = (tail["gap_actual"].abs() < tail["gap_quote"].abs()).mean()
        print(f"    |気配ギャップ|≥3%のテール({len(tail)}銘柄): |乖離|中央値 {tail['dev_bps'].abs().median():.0f}bps"
              f" / 実ギャップが気配より縮んだ率 {shrink*100:.0f}%（>>50%なら系統的縮小=勝者の呪い実在）")
print("\n判定目安: 08:59時点の中央値≤25bps かつ テール縮小率~50% なら戦略成立。"
      "σ50bps超 or 縮小率70%超なら要再設計（2パス方式=早取得で候補絞り→08:59に候補だけ再取得）。")
