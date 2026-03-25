"""
Comprehensive strategy sweep: parameter grid search + strategy variants.
Generates comparison tables, heatmaps, and identifies best configurations.
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.collect import collect
from backtest.strategies.pca_sub import run_pca_sub
from backtest.strategies.pca_plain import run_pca_plain
from backtest.strategies.momentum import run_momentum
from backtest.strategies.double_sort import run_double_sort
from backtest.metrics import compute_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "sweep")


# ---- Precomputation helpers ----

def _align_and_standardize(us_ret, jp_ret, us_cols, jp_cols, window):
    """Precompute aligned data and rolling standardization for a given window."""
    us_sorted = sorted(us_ret.index)
    pairs = []
    for jd in jp_ret.index:
        cands = [d for d in us_sorted if d < jd]
        if cands:
            pairs.append((cands[-1], jd))

    us_aligned = us_ret[us_cols].loc[[p[0] for p in pairs]].values
    jp_aligned = jp_ret[jp_cols].loc[[p[1] for p in pairs]].values
    combined = np.nan_to_num(np.hstack([us_aligned, jp_aligned]), nan=0.0)
    T = len(pairs)

    standardized = np.full_like(combined, np.nan)
    for t in range(window, T):
        w = combined[t - window:t]
        mu = w.mean(axis=0)
        sigma = np.where((s := w.std(axis=0)) > 1e-10, s, 1e-10)
        standardized[t] = (combined[t] - mu) / sigma

    return pairs, combined, standardized, T


# ---- Sweep logic ----

def run_sweep():
    us_cc, jp_oc, jp_am, jp_pm = collect()
    print(f"\nData: US={us_cc.shape}, JP_OC={jp_oc.shape}, JP_AM={jp_am.shape}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Parameter grid
    windows = [30, 60, 120, 250]
    lambdas = [0.0, 0.5, 0.7, 0.9, 0.95]
    Ks = [2, 3, 5]
    qs = [0.2, 0.3, 0.4]
    tc_bps_list = [0, 5, 10, 20]

    all_results = []

    # ---- 1. Baseline strategies ----
    print("\n=== Baseline Strategies ===")

    # PCA_SUB defaults (paper params)
    print("  PCA_SUB (paper defaults)...")
    pca_default, _ = run_pca_sub(us_cc, jp_oc)
    all_results.append({
        "strategy": "PCA_SUB", "variant": "Full", "window": 60,
        "lam": 0.9, "K": 3, "q": 0.3, "returns": pca_default["strategy_return"],
    })

    # PCA_SUB AM
    print("  PCA_SUB (AM)...")
    pca_am, _ = run_pca_sub(us_cc, jp_am)
    all_results.append({
        "strategy": "PCA_SUB", "variant": "AM", "window": 60,
        "lam": 0.9, "K": 3, "q": 0.3, "returns": pca_am["strategy_return"],
    })

    # PCA_PLAIN (no regularization)
    print("  PCA_PLAIN...")
    pca_plain = run_pca_plain(us_cc, jp_oc)
    all_results.append({
        "strategy": "PCA_PLAIN", "variant": "Full", "window": 60,
        "lam": 0.0, "K": 3, "q": 0.3, "returns": pca_plain["strategy_return"],
    })

    # MOM
    print("  MOM...")
    mom = run_momentum(us_cc, jp_oc)
    all_results.append({
        "strategy": "MOM", "variant": "Full", "window": 60,
        "lam": None, "K": None, "q": 0.3, "returns": mom["strategy_return"],
    })

    # DOUBLE
    print("  DOUBLE (MOM × PCA_SUB)...")
    double = run_double_sort(us_cc, jp_oc)
    all_results.append({
        "strategy": "DOUBLE", "variant": "Full", "window": 60,
        "lam": 0.9, "K": 3, "q": 0.3, "returns": double["strategy_return"],
    })

    # ---- 2. PCA_SUB parameter sweep ----
    print("\n=== PCA_SUB Parameter Sweep ===")
    total = len(windows) * len(lambdas) * len(Ks) * len(qs)
    # Skip λ=0 (that's PCA_PLAIN) and the default combo (already run)
    count = 0
    for L, lam, K, q in itertools.product(windows, lambdas, Ks, qs):
        if lam == 0.0:
            continue  # PCA_PLAIN handled separately
        if L == 60 and lam == 0.9 and K == 3 and q == 0.3:
            continue  # Already run as default
        count += 1
        if count % 20 == 0:
            print(f"  [{count}/{total}] L={L}, λ={lam}, K={K}, q={q}...")
        try:
            df, _ = run_pca_sub(us_cc, jp_oc, window=L, lam=lam, K=K, q=q)
            all_results.append({
                "strategy": "PCA_SUB", "variant": "Full",
                "window": L, "lam": lam, "K": K, "q": q,
                "returns": df["strategy_return"],
            })
        except Exception as e:
            print(f"  ERROR: L={L}, λ={lam}, K={K}, q={q}: {e}")

    # ---- 3. PCA_PLAIN sweep (window, K, q only) ----
    print("\n=== PCA_PLAIN Sweep ===")
    for L, K, q in itertools.product(windows, Ks, qs):
        if L == 60 and K == 3 and q == 0.3:
            continue
        try:
            df = run_pca_plain(us_cc, jp_oc, window=L, K=K, q=q)
            all_results.append({
                "strategy": "PCA_PLAIN", "variant": "Full",
                "window": L, "lam": 0.0, "K": K, "q": q,
                "returns": df["strategy_return"],
            })
        except Exception as e:
            print(f"  ERROR PCA_PLAIN L={L}, K={K}, q={q}: {e}")

    # ---- 4. MOM sweep (window, q) ----
    print("\n=== MOM Sweep ===")
    for L, q in itertools.product(windows, qs):
        if L == 60 and q == 0.3:
            continue
        try:
            df = run_momentum(us_cc, jp_oc, window=L, q=q)
            all_results.append({
                "strategy": "MOM", "variant": "Full",
                "window": L, "lam": None, "K": None, "q": q,
                "returns": df["strategy_return"],
            })
        except Exception as e:
            print(f"  ERROR MOM L={L}, q={q}: {e}")

    # ---- 5. Compute metrics ----
    print(f"\n=== Computing Metrics ({len(all_results)} configurations) ===")
    metrics_rows = []
    for r in all_results:
        m = compute_metrics(r["returns"])
        metrics_rows.append({
            "strategy": r["strategy"],
            "variant": r["variant"],
            "window": r["window"],
            "lam": r["lam"],
            "K": r["K"],
            "q": r["q"],
            **m,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False)

    # ---- 6. Transaction cost analysis ----
    print("\n=== Transaction Cost Analysis ===")
    tc_rows = []
    # Apply TC to top configurations
    top_configs = metrics_df.nlargest(10, "R/R")
    for _, row in top_configs.iterrows():
        r = next(r for r in all_results
                 if r["strategy"] == row["strategy"]
                 and r["window"] == row["window"]
                 and r["lam"] == row["lam"]
                 and r["K"] == row["K"]
                 and r["q"] == row["q"])
        for tc_bps in tc_bps_list:
            tc = tc_bps / 10000.0
            # Assume full turnover each day (conservative: 2 * q positions change)
            turnover = 2 * (row["q"] if row["q"] else 0.3)
            adj_ret = r["returns"] - tc * turnover
            m = compute_metrics(adj_ret)
            tc_rows.append({
                "strategy": row["strategy"],
                "window": row["window"],
                "lam": row["lam"],
                "K": row["K"],
                "q": row["q"],
                "tc_bps": tc_bps,
                **m,
            })

    tc_df = pd.DataFrame(tc_rows)
    tc_df.to_csv(os.path.join(RESULTS_DIR, "tc_analysis.csv"), index=False)

    # ---- 7. Generate reports ----
    _print_summary(metrics_df)
    _plot_heatmaps(metrics_df)
    _plot_top_cumulative(all_results, metrics_df)
    _plot_tc_impact(tc_df)

    print(f"\nResults saved to {RESULTS_DIR}/")
    return metrics_df


def _print_summary(df):
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)

    # Best by R/R
    print("\n--- Top 10 by R/R ---")
    top = df.nlargest(10, "R/R")
    for _, r in top.iterrows():
        lam_str = f"λ={r['lam']}" if r['lam'] is not None else "    "
        K_str = f"K={int(r['K'])}" if r['K'] is not None else "   "
        print(f"  {r['strategy']:12s} L={int(r['window']):>3d} {lam_str:>6s} {K_str} "
              f"q={r['q']:.1f}  AR={r['AR (%)']:>6.2f}%  R/R={r['R/R']:>5.2f}  "
              f"MDD={r['MDD (%)']:>7.2f}%")

    # Best by AR
    print("\n--- Top 10 by AR ---")
    top = df.nlargest(10, "AR (%)")
    for _, r in top.iterrows():
        lam_str = f"λ={r['lam']}" if r['lam'] is not None else "    "
        K_str = f"K={int(r['K'])}" if r['K'] is not None else "   "
        print(f"  {r['strategy']:12s} L={int(r['window']):>3d} {lam_str:>6s} {K_str} "
              f"q={r['q']:.1f}  AR={r['AR (%)']:>6.2f}%  R/R={r['R/R']:>5.2f}  "
              f"MDD={r['MDD (%)']:>7.2f}%")

    # Strategy type comparison
    print("\n--- Average Metrics by Strategy Type ---")
    grouped = df.groupby("strategy")[["AR (%)", "RISK (%)", "R/R", "MDD (%)"]].agg(["mean", "std"])
    for strat in grouped.index:
        row = grouped.loc[strat]
        print(f"  {strat:12s}  AR={row[('AR (%)', 'mean')]:>6.2f}±{row[('AR (%)', 'std')]:>5.2f}%  "
              f"R/R={row[('R/R', 'mean')]:>5.2f}±{row[('R/R', 'std')]:>4.2f}  "
              f"MDD={row[('MDD (%)', 'mean')]:>7.2f}%")

    # Parameter sensitivity
    print("\n--- PCA_SUB: Sensitivity to λ ---")
    pca_df = df[df["strategy"] == "PCA_SUB"]
    if not pca_df.empty:
        for lam in sorted(pca_df["lam"].dropna().unique()):
            sub = pca_df[pca_df["lam"] == lam]
            print(f"  λ={lam:.2f}:  AR={sub['AR (%)'].mean():>6.2f}%  "
                  f"R/R={sub['R/R'].mean():>5.2f}  MDD={sub['MDD (%)'].mean():>7.2f}%  "
                  f"(n={len(sub)})")

    print("\n--- PCA_SUB: Sensitivity to Window ---")
    if not pca_df.empty:
        for L in sorted(pca_df["window"].unique()):
            sub = pca_df[pca_df["window"] == L]
            print(f"  L={int(L):>3d}:  AR={sub['AR (%)'].mean():>6.2f}%  "
                  f"R/R={sub['R/R'].mean():>5.2f}  MDD={sub['MDD (%)'].mean():>7.2f}%  "
                  f"(n={len(sub)})")

    print("=" * 80)


def _plot_heatmaps(df):
    """Heatmaps: R/R for λ vs window, K vs q."""
    pca_df = df[(df["strategy"] == "PCA_SUB") & (df["K"] == 3) & (df["q"] == 0.3)]
    if pca_df.empty:
        return

    # λ vs Window heatmap
    pivot = pca_df.pivot_table(values="R/R", index="lam", columns="window", aggfunc="mean")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel("Window (L)")
        ax.set_ylabel("λ (regularization)")
        ax.set_title("PCA_SUB R/R: λ vs Window (K=3, q=0.3)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        fig.colorbar(im)
        fig.savefig(os.path.join(RESULTS_DIR, "heatmap_lam_window.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # K vs q heatmap (fixed λ=0.9, L=60)
    pca_kq = df[(df["strategy"] == "PCA_SUB") & (df["lam"] == 0.9) & (df["window"] == 60)]
    pivot2 = pca_kq.pivot_table(values="R/R", index="K", columns="q", aggfunc="mean")
    if not pivot2.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot2.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot2.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot2.columns])
        ax.set_yticks(range(len(pivot2.index)))
        ax.set_yticklabels([int(v) for v in pivot2.index])
        ax.set_xlabel("q (portfolio fraction)")
        ax.set_ylabel("K (factors)")
        ax.set_title("PCA_SUB R/R: K vs q (λ=0.9, L=60)")
        for i in range(len(pivot2.index)):
            for j in range(len(pivot2.columns)):
                ax.text(j, i, f"{pivot2.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        fig.colorbar(im)
        fig.savefig(os.path.join(RESULTS_DIR, "heatmap_K_q.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_top_cumulative(all_results, metrics_df):
    """Cumulative return plot for top 5 + baselines."""
    top5 = metrics_df.nlargest(5, "R/R")
    fig, ax = plt.subplots(figsize=(14, 7))

    plotted = set()
    for _, row in top5.iterrows():
        r = next(r for r in all_results
                 if r["strategy"] == row["strategy"]
                 and r["window"] == row["window"]
                 and r["lam"] == row["lam"]
                 and r["K"] == row["K"]
                 and r["q"] == row["q"])
        label = f"{row['strategy']} L={int(row['window'])}"
        if row["lam"] is not None:
            label += f" λ={row['lam']}"
        if row["K"] is not None:
            label += f" K={int(row['K'])}"
        label += f" q={row['q']}"
        if label not in plotted:
            ax.plot((1 + r["returns"]).cumprod(), label=f"{label} (R/R={row['R/R']:.2f})",
                    linewidth=1.5)
            plotted.add(label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Top 5 Configurations by Risk-Return Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "top5_cumulative.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tc_impact(tc_df):
    """Bar chart showing TC impact on top strategies."""
    if tc_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by strategy config
    configs = tc_df.groupby(["strategy", "window", "lam", "K", "q"])
    for (strat, L, lam, K, q), group in configs:
        label = f"{strat} L={int(L)}"
        if lam is not None:
            label += f" λ={lam}"
        group_sorted = group.sort_values("tc_bps")
        axes[0].plot(group_sorted["tc_bps"], group_sorted["AR (%)"],
                     marker="o", label=label)
        axes[1].plot(group_sorted["tc_bps"], group_sorted["R/R"],
                     marker="o", label=label)

    axes[0].set_xlabel("Transaction Cost (bps)")
    axes[0].set_ylabel("AR (%)")
    axes[0].set_title("Annualized Return vs TC")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Transaction Cost (bps)")
    axes[1].set_ylabel("R/R")
    axes[1].set_title("Risk-Return Ratio vs TC")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "tc_impact.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_sweep()
