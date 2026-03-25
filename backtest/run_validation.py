"""
Out-of-sample validation: Temporal CV + Walk-forward + Stability.

検証手法:
1. Temporal CV: Train 3年 → Test 3年 (Trainは必ずTestより過去)
   - 全年から連続3年+連続3年の組み合わせをランダムに100パターン抽出
   - seed固定で再現性確保
2. Walk-Forward: 3年トレーニング → 1年テスト、ローリングで進める
3. 安定性分析: 年ごとの成績分解、パラメータ近傍の感度確認
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.collect import collect
from backtest.strategies.pca_sub import run_pca_sub
from backtest.strategies.pca_plain import run_pca_plain
from backtest.strategies.momentum import run_momentum
from backtest.strategies.double_sort import run_double_sort
from backtest.metrics import compute_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "validation")

# 再現性のためのシード
RANDOM_SEED = 42
# CV分割回数
N_SPLITS = 100
# Train/Test それぞれの年数
N_TRAIN_YEARS = 3
N_TEST_YEARS = 3


# ============================================================
# 1. Temporal Cross-Validation
# ============================================================

def _generate_temporal_splits(all_years, n_train, n_test, n_splits, seed):
    """
    Train 3年 + Test 3年の組み合わせを生成。
    制約: Train年はすべてTest年より過去。
    ランダムに n_splits 個を重複なく抽出。
    """
    rng = np.random.RandomState(seed)
    years = sorted(all_years)

    # 全ての有効な (train_set, test_set) を列挙
    # Train: 任意の3年, Test: 任意の3年, max(Train) < min(Test)
    all_valid = []
    year_combos_train = list(itertools.combinations(years, n_train))
    year_combos_test = list(itertools.combinations(years, n_test))
    for tr in year_combos_train:
        for te in year_combos_test:
            if max(tr) < min(te):
                all_valid.append((list(tr), list(te)))

    print(f"  有効な分割パターン総数: {len(all_valid)}")
    n_actual = min(n_splits, len(all_valid))
    indices = rng.choice(len(all_valid), size=n_actual, replace=False)
    return [all_valid[i] for i in indices], n_actual


def run_temporal_cv(us_cc, jp_oc):
    """
    Train 3年 → Test 3年。Trainは必ずTestより過去。
    ランダムに100パターン抽出して検証。
    """
    print("\n" + "=" * 70)
    print(f"1. TEMPORAL CV (Train {N_TRAIN_YEARS}yr → Test {N_TEST_YEARS}yr, "
          f"seed={RANDOM_SEED}, up to {N_SPLITS} splits)")
    print(f"   Constraint: max(Train year) < min(Test year)")
    print("=" * 70)

    # 全期間で各パラメータのリターンを事前計算
    windows = [30, 60, 120, 250]
    lambdas = [0.5, 0.7, 0.9, 0.95]
    Ks = [2, 3, 5]
    qs = [0.2, 0.3, 0.4]

    print("\n  全パラメータのリターンを事前計算中...")
    param_returns = {}
    total = len(windows) * len(lambdas) * len(Ks) * len(qs)
    count = 0
    for L, lam, K, q in itertools.product(windows, lambdas, Ks, qs):
        count += 1
        if count % 30 == 0:
            print(f"    [{count}/{total}]")
        try:
            df, _ = run_pca_sub(us_cc, jp_oc, window=L, lam=lam, K=K, q=q)
            param_returns[(L, lam, K, q)] = df["strategy_return"]
        except Exception:
            pass

    # MOMも事前計算
    mom_ret = run_momentum(us_cc, jp_oc)["strategy_return"]

    # 利用可能な年を取得
    sample_ret = list(param_returns.values())[0]
    all_years = sorted(sample_ret.index.year.unique())
    print(f"\n  全年: {all_years} ({len(all_years)}年)")

    # 分割パターン生成
    splits, n_actual = _generate_temporal_splits(
        all_years, N_TRAIN_YEARS, N_TEST_YEARS, N_SPLITS, RANDOM_SEED
    )
    print(f"  抽出パターン数: {n_actual}")

    split_summaries = []
    all_cv_results = []

    for split_i, (train_years, test_years) in enumerate(splits):
        train_set = set(train_years)
        test_set = set(test_years)

        if (split_i + 1) % 20 == 0 or split_i == 0:
            print(f"\n  --- Split {split_i + 1}/{n_actual} ---")
            print(f"  Train: {train_years} → Test: {test_years}")

        # 各パラメータのTrain/Test成績を計算
        split_results = []
        for (L, lam, K, q), ret in param_returns.items():
            train_ret = ret[ret.index.year.isin(train_set)]
            test_ret = ret[ret.index.year.isin(test_set)]
            if len(train_ret) < 50 or len(test_ret) < 50:
                continue
            train_m = compute_metrics(train_ret)
            test_m = compute_metrics(test_ret)
            split_results.append({
                "split": split_i + 1,
                "L": L, "lam": lam, "K": K, "q": q,
                "train_AR": train_m["AR (%)"],
                "train_RR": train_m["R/R"],
                "train_MDD": train_m["MDD (%)"],
                "test_AR": test_m["AR (%)"],
                "test_RR": test_m["R/R"],
                "test_MDD": test_m["MDD (%)"],
            })

        if not split_results:
            continue

        split_df = pd.DataFrame(split_results)
        all_cv_results.append(split_df)

        # Trainで最適パラメータを選択 → Test成績
        best_idx = split_df["train_RR"].idxmax()
        best = split_df.loc[best_idx]

        # 論文パラメータの同splitでの成績
        paper = split_df[
            (split_df["L"] == 60) & (split_df["lam"] == 0.9) &
            (split_df["K"] == 3) & (split_df["q"] == 0.3)
        ]
        paper_test_rr = paper.iloc[0]["test_RR"] if not paper.empty else None

        # MOM
        mom_test_ret = mom_ret[mom_ret.index.year.isin(test_set)]
        mom_test_rr = compute_metrics(mom_test_ret)["R/R"] if len(mom_test_ret) > 50 else None

        if (split_i + 1) % 20 == 0 or split_i == 0:
            print(f"  Best: L={int(best['L'])} λ={best['lam']} K={int(best['K'])} "
                  f"q={best['q']}  Train R/R={best['train_RR']:.2f} → "
                  f"Test R/R={best['test_RR']:.2f}, AR={best['test_AR']:.2f}%")

        split_summaries.append({
            "split": split_i + 1,
            "train_years": train_years,
            "test_years": test_years,
            "best_L": int(best["L"]),
            "best_lam": best["lam"],
            "best_K": int(best["K"]),
            "best_q": best["q"],
            "best_train_RR": best["train_RR"],
            "best_test_AR": best["test_AR"],
            "best_test_RR": best["test_RR"],
            "best_test_MDD": best["test_MDD"],
            "paper_test_RR": paper_test_rr,
            "mom_test_RR": mom_test_rr,
        })

    # CV全体のサマリー
    summary_df = pd.DataFrame(split_summaries)
    print("\n  " + "=" * 70)
    print(f"  TEMPORAL CV SUMMARY ({n_actual} splits)")
    print("  " + "=" * 70)

    # 統計サマリー
    print(f"\n  --- Test R/R 統計 ---")
    print(f"  {'':>25s}  {'mean':>6s}  {'std':>5s}  {'min':>6s}  {'25%':>6s}  "
          f"{'50%':>6s}  {'75%':>6s}  {'max':>6s}  {'>0':>5s}")
    print("  " + "-" * 85)

    best_rr = summary_df["best_test_RR"]
    pct_pos = (best_rr > 0).sum() / len(best_rr) * 100
    print(f"  {'Train-selected':>25s}  {best_rr.mean():>6.2f}  {best_rr.std():>5.2f}  "
          f"{best_rr.min():>6.2f}  {best_rr.quantile(0.25):>6.2f}  "
          f"{best_rr.median():>6.2f}  {best_rr.quantile(0.75):>6.2f}  "
          f"{best_rr.max():>6.2f}  {pct_pos:>4.0f}%")

    paper_rr = summary_df["paper_test_RR"].dropna()
    if len(paper_rr) > 0:
        pct_pos = (paper_rr > 0).sum() / len(paper_rr) * 100
        print(f"  {'Paper (L=60 lam=0.9)':>25s}  {paper_rr.mean():>6.2f}  {paper_rr.std():>5.2f}  "
              f"{paper_rr.min():>6.2f}  {paper_rr.quantile(0.25):>6.2f}  "
              f"{paper_rr.median():>6.2f}  {paper_rr.quantile(0.75):>6.2f}  "
              f"{paper_rr.max():>6.2f}  {pct_pos:>4.0f}%")

    mom_rr = summary_df["mom_test_RR"].dropna()
    if len(mom_rr) > 0:
        pct_pos = (mom_rr > 0).sum() / len(mom_rr) * 100
        print(f"  {'MOM':>25s}  {mom_rr.mean():>6.2f}  {mom_rr.std():>5.2f}  "
              f"{mom_rr.min():>6.2f}  {mom_rr.quantile(0.25):>6.2f}  "
              f"{mom_rr.median():>6.2f}  {mom_rr.quantile(0.75):>6.2f}  "
              f"{mom_rr.max():>6.2f}  {pct_pos:>4.0f}%")

    # Test AR 統計
    print(f"\n  --- Test AR (%) 統計 ---")
    print(f"  {'':>25s}  {'mean':>6s}  {'std':>5s}  {'min':>6s}  {'50%':>6s}  {'max':>6s}  {'>0':>5s}")
    print("  " + "-" * 65)

    best_ar = summary_df["best_test_AR"]
    pct_pos = (best_ar > 0).sum() / len(best_ar) * 100
    print(f"  {'Train-selected':>25s}  {best_ar.mean():>6.2f}  {best_ar.std():>5.2f}  "
          f"{best_ar.min():>6.2f}  {best_ar.median():>6.2f}  {best_ar.max():>6.2f}  {pct_pos:>4.0f}%")

    # パラメータの選択頻度
    print("\n  --- 最適パラメータの選択頻度 ---")
    for param in ["best_L", "best_lam", "best_K", "best_q"]:
        counts = summary_df[param].value_counts().sort_index()
        print(f"  {param}: {dict(counts)}")

    # Train R/R vs Test R/R の相関 (全split合算)
    all_cv = pd.concat(all_cv_results, ignore_index=True)
    corr = all_cv[["train_RR", "test_RR"]].corr().iloc[0, 1]
    print(f"\n  Train R/R と Test R/R の相関: {corr:.3f}")

    # Train-selectedがPaperに勝った割合
    both = summary_df[summary_df["paper_test_RR"].notna()]
    if len(both) > 0:
        win = (both["best_test_RR"] > both["paper_test_RR"]).sum()
        print(f"  Train-selected が Paper に勝った割合: {win}/{len(both)} ({100*win/len(both):.0f}%)")

    return summary_df, all_cv


# ============================================================
# 2. Walk-Forward Analysis
# ============================================================

def run_walk_forward(us_cc, jp_oc):
    """
    Walk-Forward: 3年トレーニング → 1年テスト、ローリング
    各期間で最適パラメータを選び、次の1年で検証
    """
    print("\n" + "=" * 70)
    print("2. WALK-FORWARD ANALYSIS (3yr train → 1yr test, rolling)")
    print("=" * 70)

    # 小さめのグリッドで速度確保
    windows = [60, 120]
    lambdas = [0.5, 0.7, 0.9]
    Ks = [3]
    qs = [0.2, 0.3]

    # 期間設定: 2013-2025 (最初の3年はL=250でもwarm-upに足りるよう)
    test_years = list(range(2014, 2026))

    wf_results = []

    for test_year in test_years:
        train_start = test_year - 3
        train_end = test_year - 1
        print(f"\n  Train: {train_start}-{train_end} → Test: {test_year}")

        best_rr = -999
        best_params = None

        for L, lam, K, q in itertools.product(windows, lambdas, Ks, qs):
            try:
                df, _ = run_pca_sub(us_cc, jp_oc, window=L, lam=lam, K=K, q=q)
                ret = df["strategy_return"]
                train_ret = ret[(ret.index.year >= train_start) & (ret.index.year <= train_end)]
                if len(train_ret) < 100:
                    continue
                m = compute_metrics(train_ret)
                if m["R/R"] > best_rr:
                    best_rr = m["R/R"]
                    best_params = {"L": L, "lam": lam, "K": K, "q": q}
            except Exception:
                pass

        if best_params is None:
            print(f"    スキップ (有効な構成なし)")
            continue

        # テスト期間で検証
        df, _ = run_pca_sub(us_cc, jp_oc, **{
            "window": best_params["L"], "lam": best_params["lam"],
            "K": best_params["K"], "q": best_params["q"],
        })
        test_ret = df["strategy_return"][df.index.year == test_year]
        if len(test_ret) < 20:
            print(f"    テストデータ不足 ({len(test_ret)}日)")
            continue

        test_m = compute_metrics(test_ret)
        print(f"    Best: L={best_params['L']}, λ={best_params['lam']}, "
              f"K={best_params['K']}, q={best_params['q']} "
              f"(train R/R={best_rr:.2f})")
        print(f"    Test: AR={test_m['AR (%)']:.2f}%, R/R={test_m['R/R']:.2f}, "
              f"MDD={test_m['MDD (%)']:.2f}%")

        wf_results.append({
            "test_year": test_year,
            **best_params,
            "train_RR": best_rr,
            "test_AR": test_m["AR (%)"],
            "test_RR": test_m["R/R"],
            "test_MDD": test_m["MDD (%)"],
            "test_ret": test_ret,
        })

    # Walk-forward の OOS 結合リターン
    if wf_results:
        oos_ret = pd.concat([r["test_ret"] for r in wf_results])
        oos_m = compute_metrics(oos_ret)
        print(f"\n  --- Walk-Forward OOS 合算成績 ({test_years[0]}-{test_years[-1]}) ---")
        print(f"  AR={oos_m['AR (%)']:.2f}%  R/R={oos_m['R/R']:.2f}  "
              f"MDD={oos_m['MDD (%)']:.2f}%  N={oos_m['N_days']}日")

        # 固定パラメータとの比較
        print("\n  --- 固定パラメータ (論文: L=60 λ=0.9 K=3 q=0.3) 同期間 ---")
        df_fixed, _ = run_pca_sub(us_cc, jp_oc)
        fixed_ret = df_fixed["strategy_return"]
        fixed_same = fixed_ret[(fixed_ret.index.year >= test_years[0]) &
                               (fixed_ret.index.year <= test_years[-1])]
        fixed_m = compute_metrics(fixed_same)
        print(f"  AR={fixed_m['AR (%)']:.2f}%  R/R={fixed_m['R/R']:.2f}  "
              f"MDD={fixed_m['MDD (%)']:.2f}%  N={fixed_m['N_days']}日")

    return wf_results


# ============================================================
# 3. Stability Analysis
# ============================================================

def run_stability_analysis(us_cc, jp_oc):
    """
    年ごとの分解 + パラメータ近傍の安定性確認
    """
    print("\n" + "=" * 70)
    print("3. STABILITY ANALYSIS")
    print("=" * 70)

    # 候補パラメータ
    configs = [
        ("Paper (L=60 lam=0.9 K=3 q=0.3)", {"window": 60, "lam": 0.9, "K": 3, "q": 0.3}),
        ("Best (L=120 lam=0.5 K=3 q=0.2)", {"window": 120, "lam": 0.5, "K": 3, "q": 0.2}),
        ("Mid (L=120 lam=0.7 K=3 q=0.3)", {"window": 120, "lam": 0.7, "K": 3, "q": 0.3}),
    ]

    print("\n  --- 年別パフォーマンス ---")
    yearly_data = {}
    for name, params in configs:
        df, _ = run_pca_sub(us_cc, jp_oc, **params)
        ret = df["strategy_return"]
        yearly_data[name] = ret

        print(f"\n  {name}:")
        print(f"  {'年':>6s}  {'AR':>8s}  {'R/R':>6s}  {'MDD':>8s}")
        print(f"  " + "-" * 35)
        years = sorted(ret.index.year.unique())
        win_years = 0
        for yr in years:
            yr_ret = ret[ret.index.year == yr]
            m = compute_metrics(yr_ret)
            marker = "+" if m["AR (%)"] > 0 else "-"
            if m["AR (%)"] > 0:
                win_years += 1
            print(f"  {yr:>6d}  {m['AR (%)']:>7.2f}%  {m['R/R']:>6.2f}  {m['MDD (%)']:>7.2f}%  {marker}")
        print(f"  勝率: {win_years}/{len(years)} ({100*win_years/len(years):.0f}%)")

    # パラメータ近傍の安定性
    print("\n  --- パラメータ近傍の安定性 (最適構成周辺) ---")
    center = {"window": 120, "lam": 0.5, "K": 3, "q": 0.2}
    perturbations = [
        ("L=90", {"window": 90}),
        ("L=150", {"window": 150}),
        ("λ=0.4", {"lam": 0.4}),
        ("λ=0.6", {"lam": 0.6}),
        ("q=0.15", {"q": 0.15}),
        ("q=0.25", {"q": 0.25}),
    ]

    print(f"  {'構成':>20s}  {'AR':>8s}  {'R/R':>6s}  {'MDD':>8s}")
    print(f"  " + "-" * 50)

    # Center
    df, _ = run_pca_sub(us_cc, jp_oc, **center)
    m = compute_metrics(df["strategy_return"])
    print(f"  {'中心 (L=120 λ=0.5)':>20s}  {m['AR (%)']:>7.2f}%  {m['R/R']:>6.2f}  {m['MDD (%)']:>7.2f}%")

    for label, override in perturbations:
        params = {**center, **override}
        try:
            df, _ = run_pca_sub(us_cc, jp_oc, **params)
            m = compute_metrics(df["strategy_return"])
            print(f"  {label:>20s}  {m['AR (%)']:>7.2f}%  {m['R/R']:>6.2f}  {m['MDD (%)']:>7.2f}%")
        except Exception as e:
            print(f"  {label:>20s}  ERROR: {e}")

    return yearly_data


# ============================================================
# Plots
# ============================================================

def _plot_train_test(all_cv, save_dir):
    """Train R/R vs Test R/R scatter plot (all CV splits combined)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    splits = all_cv["split"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(splits)))
    for split_i, color in zip(splits, colors):
        subset = all_cv[all_cv["split"] == split_i]
        ax.scatter(subset["train_RR"], subset["test_RR"],
                   alpha=0.25, s=15, color=color, label=f"Split {split_i}")

    lim = max(abs(all_cv["train_RR"]).max(), abs(all_cv["test_RR"]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)

    ax.set_xlabel("Train R/R")
    ax.set_ylabel("Test R/R")
    ax.set_title(f"Train vs Test R/R — Temporal CV ({len(splits)} splits, Train<Test)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.savefig(os.path.join(save_dir, "train_vs_test_rr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_walk_forward(wf_results, save_dir):
    """Walk-forward yearly OOS performance"""
    if not wf_results:
        return
    years = [r["test_year"] for r in wf_results]
    ars = [r["test_AR"] for r in wf_results]
    rrs = [r["test_RR"] for r in wf_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = ["green" if v > 0 else "red" for v in ars]
    ax1.bar(years, ars, color=colors, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("AR (%)")
    ax1.set_title("Walk-Forward OOS: Annual Return")
    ax1.grid(True, alpha=0.2)

    colors2 = ["green" if v > 0 else "red" for v in rrs]
    ax2.bar(years, rrs, color=colors2, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("R/R")
    ax2.set_xlabel("Year")
    ax2.set_title("Walk-Forward OOS: Risk-Return Ratio")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "walk_forward_oos.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_yearly_comparison(yearly_data, save_dir):
    """Yearly comparison bar chart"""
    all_years = set()
    for ret in yearly_data.values():
        all_years.update(ret.index.year.unique())
    years = sorted(all_years)

    fig, ax = plt.subplots(figsize=(14, 6))
    n_strats = len(yearly_data)
    width = 0.8 / n_strats
    x = np.arange(len(years))

    for i, (name, ret) in enumerate(yearly_data.items()):
        ars = []
        for yr in years:
            yr_ret = ret[ret.index.year == yr]
            if len(yr_ret) > 0:
                ars.append(compute_metrics(yr_ret)["AR (%)"])
            else:
                ars.append(0)
        ax.bar(x + i * width - 0.4 + width / 2, ars, width, label=name, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("AR (%)")
    ax.set_title("Yearly Performance Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "yearly_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def _plot_cv_summary(summary_df, save_dir):
    """Histogram comparing Test R/R distributions across strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-3, 4, 30)

    ax.hist(summary_df["best_test_RR"], bins=bins, alpha=0.6,
            label=f"Train-selected (mean={summary_df['best_test_RR'].mean():.2f})", edgecolor="black")
    paper_rr = summary_df["paper_test_RR"].dropna()
    if len(paper_rr) > 0:
        ax.hist(paper_rr, bins=bins, alpha=0.6,
                label=f"Paper L=60 lam=0.9 (mean={paper_rr.mean():.2f})", edgecolor="black")
    mom_rr = summary_df["mom_test_RR"].dropna()
    if len(mom_rr) > 0:
        ax.hist(mom_rr, bins=bins, alpha=0.6,
                label=f"MOM (mean={mom_rr.mean():.2f})", edgecolor="black")

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Test R/R")
    ax.set_ylabel("Count")
    ax.set_title(f"Temporal CV: Test R/R Distribution ({len(summary_df)} splits, "
                 f"Train {N_TRAIN_YEARS}yr < Test {N_TEST_YEARS}yr)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cv_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all():
    us_cc, jp_oc, jp_am, jp_pm = collect()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Temporal CV
    summary_df, all_cv = run_temporal_cv(us_cc, jp_oc)
    _plot_train_test(all_cv, RESULTS_DIR)
    _plot_cv_summary(summary_df, RESULTS_DIR)

    # CV結果保存
    save_cols = [c for c in all_cv.columns if not c.startswith("_")]
    all_cv[save_cols].to_csv(
        os.path.join(RESULTS_DIR, "temporal_cv_results.csv"), index=False
    )
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False
    )

    # 2. Walk-Forward
    wf_results = run_walk_forward(us_cc, jp_oc)
    _plot_walk_forward(wf_results, RESULTS_DIR)

    # Walk-Forward結果保存
    if wf_results:
        wf_save = [{k: v for k, v in r.items() if k != "test_ret"} for r in wf_results]
        pd.DataFrame(wf_save).to_csv(
            os.path.join(RESULTS_DIR, "walk_forward_results.csv"), index=False
        )

    # 3. Stability
    yearly_data = run_stability_analysis(us_cc, jp_oc)
    _plot_yearly_comparison(yearly_data, RESULTS_DIR)

    print(f"\n全結果: {RESULTS_DIR}/")


if __name__ == "__main__":
    run_all()
