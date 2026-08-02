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
  4. **建玉一致率（★判定の本体・`--book`）**: 気配で組んだ建玉と実寄値で組んだ建玉の一致率。
     これだけが成績に単調に効く（下表）。**順位相関では代替できない**——
     ±3%クリップは順位相関1.000なのに一致率78.6%・Sharpe81%。

## 合格基準（事前登録・2026-07-31。実測を見る前に確定させた）

`scripts/quote_distortion_calibration.py` で、既知の歪みを本番と同じ経路
（ensemble_core ¥20M・8銘柄/側・信用2倍・¥10億フロア・7bps）に注入して作った対応表:

  | 歪み            | 建玉一致率 | OOS24+ Sharpe | 維持率 |
  |-----------------|-----------|---------------|--------|
  | なし（基準）      | 100%      | 3.04          | 100%   |
  | 一様圧縮 λ=0.5   | 100%      | 3.04          | 100%   |
  | 一様圧縮 λ=0.3   | 100%      | 3.04          | 100%   |
  | ±3%クリップ      | 78.6%     | 2.45          |  81%   |
  | ランダム σ=100bps | 53.7%     | 2.21          |  73%   |
  | ランダム σ=250bps | 36.6%     | 1.40          |  46%   |
  | ランダム σ=500bps | 30.2%     | 0.83          |  27%   |

**判定**: `--book` の実測建玉一致率をこの表に当てて Sharpe 維持率を読む。
  - **一致率 ≥75%** → 維持率80%以上。実弾GO可
  - **50-75%**     → 維持率73-81%。減額（¥5M）で開始し実測を積む
  - **<50%**       → 維持率46%以下。**実弾NO-GO**（気配前提の作り直しが必要）
一様圧縮は一致率100%＝**λがいくら小さくても無害**なので、λ単独では絶対に判断しない。
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
    if len(x) >= 2 * k:                                # 生ギャップ順位の一致率（参考値）
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
        s += f"  生ギャップ一致={d['overlap']*100:.0f}%"
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
    print("      生ギャップ一致は12%まで落ちる。テールが同値に潰れて上位k本が選べないため。")
    print("    ・ここの『生ギャップ一致』は素のギャップ順位で測った**参考値**であって、")
    print("      判定に使う『建玉一致率』(--book) とは別物。判定は必ず --book で行う。")
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

    # ★実寄値と前日終値は**パネルから**取る。daily_adj_*.parquet の生値列(O/C)は
    # 2022-2024に列自体が無く、2025も非NaNが3.2%しかない（罠）。ここを直接読むと
    # 気配の3%だけを見て全体を判断することになる。パネルは調整基準を解決済み。
    from trading.jp_intraday.daily_model import load_panel_cached
    panel = load_panel_cached(min_value_yen=1e9)
    pc = panel["raw_close"].fillna(panel["close"]) if "raw_close" in panel else panel["close"]
    ref = pd.DataFrame({
        "day": pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d"),
        # kabuは4桁、パネルは5桁
        "symbol": panel["symbol"].astype(str).map(lambda x: x[:-1] if len(x) == 5 else x),
        "gap_actual_pct": panel["overnight_gap"] * 100,   # 真のギャップ（調整基準解決済み）
        "prev_close_raw": pc,                              # 気配は生値なので生の前日終値と比べる
    })

    n_q = len(q)
    m = q.merge(ref, on=["day", "symbol"], how="inner")
    m = m[(m["prev_close_raw"] > 0) & m["gap_actual_pct"].notna()]
    if m.empty:
        raise SystemExit("実寄値データ未着（collect_jp_daily_history 実行後に再試行）")
    m["gap_quote"] = (m["calc"] / m["prev_close_raw"] - 1) * 100
    m["gap_actual"] = m["gap_actual_pct"]
    m["dev_bps"] = (m["gap_actual"] - m["gap_quote"]) * 100

    cov = len(m) / n_q if n_q else 0
    if cov < 0.8:
        print(f"⚠️ 突合できたのは気配 {n_q:,}行中 {len(m):,}行 ({cov*100:.0f}%)。"
              "欠落が多いと分解が偏るので原因を確認すること\n")
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


# ── 建玉一致率（判定の本体） ────────────────────────────────────
def book_check() -> None:
    """実測の気配で組んだ建玉と、実寄値で組んだ建玉を突き合わせる（判定の本体）.

    事前登録した対応表（docstring）に当てて Sharpe 維持率を読む。
    """
    from trading.jp_intraday.daily_model import load_panel_cached
    from trading.jp_intraday.strategies import run_unit_lot

    from scripts.quote_distortion_calibration import (CAPITAL, COST_BPS, MARGIN,
                                                      NAMES_PER_SIDE, book_overlap,
                                                      recompute_gap_features)

    rows = []
    for f in sorted(glob.glob("data/live_reports/quotesnap_*.jsonl")):
        day = f.split("_")[-1].replace(".jsonl", "")
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            r["day"] = day
            rows.append(r)
    if not rows:
        raise SystemExit("quotesnapログがありません（Windowsで run_live quotesnap を実行）。")
    q = pd.DataFrame(rows)
    q["calc"] = pd.to_numeric(q["calc"], errors="coerce")
    q = q.dropna(subset=["calc"])
    last_snap = sorted(q["snap"].unique())[-1]        # 一番遅い時点＝実運用で使う気配
    q = q[q["snap"].eq(last_snap)]
    q["date"] = pd.to_datetime(q["day"])
    # kabuは4桁、パネルは5桁
    q["symbol"] = q["symbol"].astype(str).map(lambda s: s if len(s) == 5 else s + "0")

    print(f"パネル構築（本番条件）… 気配は {last_snap} 時点・{q['date'].nunique()}営業日")
    panel = load_panel_cached(min_value_yen=1e9)
    days = sorted(set(q["date"]) & set(panel["date"]))
    if not days:
        raise SystemExit("気配の日付がパネルに未反映（collect_jp_daily_history 実行後に再試行）")

    alt = panel.merge(q[["date", "symbol", "calc"]], on=["date", "symbol"], how="left")
    prev_close = alt["raw_close"].fillna(alt["close"]) if "raw_close" in alt else alt["close"]
    hit = alt["calc"].notna() & alt["date"].isin(days) & (prev_close > 0)
    print(f"  気配で置換できた行: {int(hit.sum()):,} / 対象日の全行 "
          f"{int(alt['date'].isin(days).sum()):,}")
    alt.loc[hit, "overnight_gap"] = (alt.loc[hit, "calc"] / prev_close[hit] - 1).to_numpy()
    alt = recompute_gap_features(alt.drop(columns=["calc"]))

    _, blot_true = run_unit_lot(panel, "ensemble_core", capital_yen=CAPITAL,
                                names_per_side=NAMES_PER_SIDE, margin_ratio=MARGIN,
                                cost_bps_side=COST_BPS)
    _, blot_q = run_unit_lot(alt, "ensemble_core", capital_yen=CAPITAL,
                             names_per_side=NAMES_PER_SIDE, margin_ratio=MARGIN,
                             cost_bps_side=COST_BPS)
    d = pd.DatetimeIndex(days)
    bt = blot_true[pd.to_datetime(blot_true["date"]).isin(d)]
    bq = blot_q[pd.to_datetime(blot_q["date"]).isin(d)]
    n_ov, y_ov = book_overlap(bt, bq)

    print(f"\n【判定】建玉一致率: 銘柄ベース {n_ov*100:.1f}% / ¥加重 {y_ov*100:.1f}%"
          f"  （{len(d)}営業日）")
    if n_ov >= 0.75:
        print("  → ✅ 事前登録基準: 一致率≥75% ＝ Sharpe維持率80%以上。実弾GO可")
    elif n_ov >= 0.50:
        print("  → ⚠️ 事前登録基準: 一致率50-75% ＝ 維持率73-81%。減額(¥5M)で開始し実測を積む")
    else:
        print("  → ❌ 事前登録基準: 一致率<50% ＝ 維持率46%以下。**実弾NO-GO**")
    # 実現P&Lの直接比較（日数が少ないうちは参考値）
    for lab, b in (("実寄値で選んだ建玉", bt), ("気配で選んだ建玉", bq)):
        if len(b):
            pnl = b.groupby("date")["pnl_yen"].sum()
            print(f"  {lab}: 実現P&L 合計 ¥{pnl.sum():,.0f} / 日次平均 ¥{pnl.mean():,.0f} "
                  f"（{len(pnl)}日）")
    print("\n※日数が少ないうちP&L差はノイズが支配する。判定は建玉一致率で行うこと"
          "（対応表は docstring 参照）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="既知の歪みで計測器を検証")
    ap.add_argument("--book", action="store_true", help="★判定本体: 建玉一致率を実測")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.book:
        book_check()
    else:
        analyze()
