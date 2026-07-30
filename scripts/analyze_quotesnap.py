"""quotesnap（寄前気配3時点）と実寄値の乖離分析 — 戦略成立性の判定材料.

使い方: quotesnapを1朝以上実行後、当日夕方〜翌朝に
  PYTHONPATH=. python scripts/analyze_quotesnap.py
**★2026-07-31: 旧「σ≤25bps→影響ゼロ / 51bps→Sh-10%」という合格基準は撤回済み**。
あれは誤差をiidランダムノイズと仮定した机上値だが、実際の気配誤差は特別気配の
離散更新（更新値幅=制限値幅/10。¥2,000-3,000帯で¥50=205bps/step）に支配され、
機構的下限だけでα加重1,060bpsある＝**旧基準は桁が2つ足りない**。

**このスクリプトは合否を判定しない**。出力するのは判断材料の分解のみ:
  1. **系統成分**: 気配ギャップ→実ギャップの回帰の傾き（=圧縮率λ）と決定係数。
     λが安定していれば**補正可能＝情報は失われていない**（λ=0.3でも順位相関1.000の実測あり）
  2. **ランダム成分**: 回帰の残差σ（銘柄固有）。**これだけが致命的**
     （全銘柄一律σ250bpsでSharpe半減・310bpsで1.0・500bpsでゼロ）
  3. **クリップの有無**: 大ギャップ帯で気配が頭打ちになっているか
     （上限クリップなら床Sharpe1.8で生存するが、svdnスリーブは死ぬ）
合格基準そのものは、この実測が数日分たまってから設計する（AGENTS.md §特別気配の現実 参照）。
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
# ── 系統成分 vs ランダム成分の分解（これが本体） ──
print("\n" + "=" * 70)
print("【誤差の分解】系統成分(補正可能) vs ランダム成分(致命的)")
for snap, g in m.groupby("snap"):
    g = g[g["gap_quote"].abs() > 0.1]          # ギャップがほぼ無い銘柄は分解の意味がない
    if len(g) < 30:
        print(f"  [{snap}] n={len(g)} — サンプル不足（数日分ためてから再実行）")
        continue
    x, y = g["gap_quote"].values, g["gap_actual"].values
    lam = float((x * y).sum() / (x * x).sum())          # 原点通過回帰の傾き=圧縮率
    resid = (y - lam * x) * 100                          # %→bps相当（×100でbps）
    r2 = 1 - ((y - lam * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  [{snap}] n={len(g)}")
    print(f"    系統成分: 圧縮率λ={lam:.3f}（1.0なら気配=寄値・小さいほど気配が過小表示） R²={r2:.3f}")
    print(f"    ランダム成分: 残差σ={resid.std():.0f}bps  中央値|残差|={abs(resid).median():.0f}bps")
    big = g[g["gap_quote"].abs() >= 3]
    if len(big) >= 10:
        lam_big = float((big["gap_quote"] * big["gap_actual"]).sum()
                        / (big["gap_quote"] ** 2).sum())
        print(f"    大ギャップ帯(|気配|≥3%, n={len(big)}): λ={lam_big:.3f}"
              f"（全体λより大きく小さいならクリップ＝テールの順位情報が失われている）")
print("\n※このスクリプトは合否を判定しない。上記の λ / 残差σ / クリップ有無を")
print("　AGENTS.md §特別気配の現実 の歪みモデル別実測と突き合わせて人間が判断する。")
print("　目安: 残差σが250bps級ならSharpe半減。λが安定なら補正可能で情報損失なし。")
