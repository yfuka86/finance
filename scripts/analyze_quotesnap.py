"""quotesnap（寄前気配3時点）と実寄値の乖離分析 — 実弾移行の判断材料.

使い方:
  PYTHONPATH=. python scripts/analyze_quotesnap.py            # 実測ログを分析
  PYTHONPATH=. python scripts/analyze_quotesnap.py --selftest # 計測器自体の妥当性検証

★2026-07-31: 旧「σ≤25bps→影響ゼロ / 51bps→Sh-10%」という合格基準は撤回済み。
あれは誤差をiidランダムノイズと仮定した机上値だが、実際の気配誤差は特別気配の
離散更新（更新値幅=制限値幅/10。¥2,000-3,000帯で¥50=205bps/step）に支配され、
機構的下限だけでα加重1,060bpsある＝**旧基準は桁が2つ足りない**。

★2026-07-31: 旧版の分解ロジック自体にも致命的な誤りがあったので修正した（--selftest で再現可能）:
  旧版は「実ギャップ を 気配ギャップ に回帰」していた。ノイズを含む変数を説明変数に置くと
  誤差変数バイアス（attenuation）で傾きが Var(真)/(Var(真)+Var(誤差)) 倍に縮む。結果:
    - 真の一様圧縮 λ=0.3  → 旧版は λ=3.33 と表示（逆数。docstringの解釈文と正反対）
    - **純ランダムσ250bps → 旧版は λ=0.40 と表示** ＝ 解釈文では「気配が過小表示＝圧縮＝
      補正可能・無害」と読める。**最も危険な状態を最も安全な状態として報告していた**
    - 残差σも真値を回収できない（σ250注入 → 旧版159bps / σ500注入 → 旧版191bps）
  正しい向きは「気配 を 実寄値 に回帰」（測定式 quote = λ·actual + ε の素直な推定）。
  この向きなら λ も σ_ε も真値を回収し、R² が系統/ランダムの判別子として機能する。

出力する判断材料（**このスクリプトは合否を判定しない**）:
  1. **λ（圧縮率）**: 1.0=気配が寄値と等倍。小さいほど気配がギャップを過小表示。
     λがいくつでも**それ自体は無害**（決定論的なら補正で100%復元・λ=0.3でも順位相関1.000）
  2. **R²（判別子・ここが本丸）**: 1.0に近い=決定論的な歪み＝**補正可能**。
     低い=真のランダム成分がある＝**補正不能**。λではなくR²で系統/ランダムを分ける
  3. **σ_ε（ランダム成分の大きさ）**: R²が低いときだけ意味を持つ。
     全銘柄一律σ250bpsでSharpe半減・310bpsで1.0・500bpsでゼロ（AGENTS §特別気配の現実）
  4. **選択一致率（戦略に直結する唯一の量）**: 気配で選んだ上位/下位k銘柄と、
     実寄値で選んだそれの一致率。**順位相関では代替できない**——
     ±3%クリップは順位相関1.000なのに選択一致12%（テールが全部同値に潰れるため）。
     svdnスリーブが死ぬのはこの経路。

合格基準そのものは、実測が数日分たまってから、この4つの実際の形を見て設計する。
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd


# ── 計測器本体 ──────────────────────────────────────────────────
def decompose(gap_quote: np.ndarray, gap_actual: np.ndarray, k: int = 8) -> dict:
    """気配ギャップと実ギャップの誤差を系統成分とランダム成分に分解する.

    測定式は quote = λ·actual + ε（実寄値が真値、気配はその観測）。したがって
    **気配を被説明変数に置く**のが正しい向き。逆向きに回帰すると誤差変数バイアスで
    ランダムノイズが「圧縮」に化けて見える（--selftest 参照）。

    gap_quote / gap_actual は前日終値比のギャップ（単位%）。
    """
    x, y = np.asarray(gap_actual, float), np.asarray(gap_quote, float)
    lam = float((x * y).sum() / (x * x).sum())
    resid = (y - lam * x) * 100.0                      # % → bps
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ((y - lam * x) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    out = {"n": len(x), "lam": lam, "r2": r2,
           "sigma_bps": float(resid.std()), "resid_med_bps": float(np.median(np.abs(resid))),
           "rho": float(pd.Series(y).corr(pd.Series(x), method="spearman"))}
    if len(x) >= 2 * k:                                # 選択一致率（戦略に直結）
        hit = (len(np.intersect1d(np.argsort(y)[-k:], np.argsort(x)[-k:]))
               + len(np.intersect1d(np.argsort(y)[:k], np.argsort(x)[:k])))
        out["overlap"] = hit / (2 * k)
    big = np.abs(y) >= 3.0                             # 大ギャップ帯のλ（クリップ検知）
    if big.sum() >= 10:
        out["lam_big"] = float((x[big] * y[big]).sum() / (x[big] ** 2).sum())
        out["n_big"] = int(big.sum())
    return out


def _fmt(d: dict) -> str:
    s = (f"λ={d['lam']:.3f}  R²={d['r2']:.3f}  σ_ε={d['sigma_bps']:.0f}bps  "
         f"順位ρ={d['rho']:.3f}")
    if "overlap" in d:
        s += f"  選択一致={d['overlap']*100:.0f}%"
    return s


# ── 計測器の自己検証 ────────────────────────────────────────────
def selftest() -> None:
    """既知の歪みモデルを注入し、分解が真値を回収できるかを確認する."""
    rng = np.random.default_rng(0)
    n = 3000
    true_gap = rng.standard_t(3, n) * 1.2              # 実ギャップ%（裾の重い分布）
    cases = [
        ("① 一様圧縮 λ=0.3（決定論・補正可能）", true_gap * 0.3, "λ≈0.30, R²≈1.00, σ≈0"),
        ("② ±3%クリップ（svdnが死ぬ形）", np.clip(true_gap, -3, 3), "R²中位・選択一致が激減"),
        ("③ ランダム σ=250bps（Sharpe半減）", true_gap + rng.normal(0, 2.5, n), "σ≈250, R²低"),
        ("④ ランダム σ=500bps（Sharpeゼロ）", true_gap + rng.normal(0, 5.0, n), "σ≈500, R²最低"),
        ("⑤ 圧縮0.3 ＋ σ250 の混合", true_gap * 0.3 + rng.normal(0, 2.5, n), "λ≈0.30 かつ σ≈250"),
    ]
    print("【計測器の自己検証】既知の歪みを注入して真値を回収できるか\n")
    for name, q, expect in cases:
        print(f"  {name}")
        print(f"    {_fmt(decompose(np.asarray(q), true_gap))}")
        print(f"    期待: {expect}\n")
    print("  読み方:")
    print("    ・λ が小さくても R²≈1 なら決定論的な圧縮＝補正で情報は完全に戻る（①）")
    print("    ・R² が低いときだけ σ_ε が意味を持つ。σ250級で Sharpe 半減（③④）")
    print("    ・**順位ρ は判断に使えない**: ②はρ=1.000（クリップは中間で単調）なのに")
    print("      選択一致は12%まで落ちる。テールが同値に潰れて上位k本が選べないため。")
    print("      戦略が見ているのはテールなので、判断は必ず選択一致率で行う。")
    print("\n  ※旧版（実ギャップを気配に回帰）は③を λ=0.40 と報告していた。")
    print("    旧docstringの解釈では「圧縮＝補正可能・無害」と読める＝最悪ケースを")
    print("    安全と誤判定する向きのバグだった。")


# ── 実測ログの分析 ──────────────────────────────────────────────
def analyze() -> None:
    rows = []
    for f in sorted(glob.glob("data/live_reports/quotesnap_*.jsonl")):
        day = f.split("_")[-1].replace(".jsonl", "")
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            r["day"] = day
            rows.append(r)
    if not rows:
        raise SystemExit("quotesnapログがありません（Windowsで run_live quotesnap を実行）。\n"
                         "計測器の動作確認だけなら --selftest を使う。")
    q = pd.DataFrame(rows)
    q["calc"] = pd.to_numeric(q["calc"], errors="coerce")
    q = q.dropna(subset=["calc"])

    daily = [pd.read_parquet(f, columns=["Date", "Code", "O", "C"])
             for f in sorted(glob.glob("data/jp_daily_history/daily_adj_202[5-9].parquet"))]
    d = pd.concat(daily, ignore_index=True)
    d["day"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    d["symbol"] = d["Code"].astype(str).map(lambda s: s[:-1] if len(s) == 5 else s)
    d = d.rename(columns={"O": "open_actual"})
    d["prev_close"] = d.sort_values("day").groupby("symbol")["C"].shift(1)

    m = q.merge(d[["day", "symbol", "open_actual", "prev_close"]], on=["day", "symbol"], how="inner")
    if m.empty:
        raise SystemExit("実寄値データ未着（collect_jp_daily_history 実行後に再試行）")
    m["dev_bps"] = (m["open_actual"] / m["calc"] - 1) * 1e4
    m["gap_quote"] = (m["calc"] / m["prev_close"] - 1) * 100
    m["gap_actual"] = (m["open_actual"] / m["prev_close"] - 1) * 100

    print(f"対象 {m['day'].nunique()}営業日 / {len(m):,}ペア\n")
    print("【1】乖離の生の分布（時点別の収束カーブ）")
    for snap, g in m.groupby("snap"):
        a = g["dev_bps"].abs()
        print(f"  [{snap}] n={len(g):,}  |乖離|bps: 中央値{a.median():.0f} p75 {a.quantile(.75):.0f} "
              f"p90 {a.quantile(.9):.0f} p99 {a.quantile(.99):.0f}  符号付き平均 {g['dev_bps'].mean():+.0f}")

    print("\n【2】誤差の分解（系統=補正可能 / ランダム=補正不能）")
    for snap, g in m.groupby("snap"):
        g = g[g["gap_quote"].abs() > 0.1]              # ギャップ皆無の銘柄は分解の意味がない
        if len(g) < 30:
            print(f"  [{snap}] n={len(g)} — サンプル不足（数日ためてから再実行）")
            continue
        res = decompose(g["gap_quote"].to_numpy(), g["gap_actual"].to_numpy())
        print(f"  [{snap}] n={res['n']:,}  {_fmt(res)}")
        if "lam_big" in res:
            print(f"        大ギャップ帯(|気配|≥3%, n={res['n_big']}): λ={res['lam_big']:.3f}"
                  f"  ← 全体λより明確に小さければクリップ＝テールの順位情報が失われている")

    print("\n【3】日次のばらつき（安定性 — 1日だけ良くても意味がない）")
    last = sorted(m["snap"].unique())[-1]
    per_day = []
    for day, g in m[m["snap"].eq(last)].groupby("day"):
        g = g[g["gap_quote"].abs() > 0.1]
        if len(g) >= 30:
            r = decompose(g["gap_quote"].to_numpy(), g["gap_actual"].to_numpy())
            per_day.append({"day": day, "n": r["n"], "λ": round(r["lam"], 3),
                            "R²": round(r["r2"], 3), "σ_ε bps": round(r["sigma_bps"]),
                            "選択一致%": round(r.get("overlap", float("nan")) * 100)})
    if per_day:
        print(f"  （{last} 時点）")
        print(pd.DataFrame(per_day).to_string(index=False))
        s = pd.DataFrame(per_day)
        print(f"\n  λの日次ばらつき: 平均{s['λ'].mean():.3f} / 標準偏差{s['λ'].std():.3f}"
              f"  ← 標準偏差が小さいほど「補正可能な系統歪み」に近い")

    print("\n※このスクリプトは合否を判定しない。判断は AGENTS.md §特別気配の現実 の")
    print("　歪みモデル別実測（一様圧縮=100%復元 / クリップ=床Sh1.85 / σ250=半減）と")
    print("　上の λ・R²・σ_ε・選択一致率 を突き合わせて人間が行う。")
    print("　計測器そのものの妥当性は --selftest で確認できる。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="既知の歪みで計測器を検証")
    a = ap.parse_args()
    selftest() if a.selftest else analyze()
