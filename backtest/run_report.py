"""
Strategy report generator.
各戦略のTemporal CV (Train 3yr < Test 3yr) におけるTest期間の月別平均リターンを
計算し、レポートとして保存する。
"""
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.collect import collect
from data.collectors.config import US_TICKERS, JP_TICKERS
from backtest.strategies.pca_sub import run_pca_sub
from backtest.strategies.momentum import run_momentum
from backtest.run_optuna import run_pca_sub_extended, generate_splits
from backtest.metrics import compute_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "report")
RANDOM_SEED = 42
N_CV_SPLITS = 100


def _collect_test_returns(ret, splits):
    """各splitのTest期間リターンを集約して1つのSeriesにする (重複日は平均)。"""
    all_test = []
    for train_years, test_years in splits:
        test_ret = ret[ret.index.year.isin(set(test_years))]
        all_test.append(test_ret)
    if not all_test:
        return pd.Series(dtype=float)
    combined = pd.concat(all_test)
    # 同じ日が複数splitで出現するので平均を取る
    return combined.groupby(combined.index).mean()


def _monthly_stats(test_ret):
    """月ごとの平均リターン (年率換算なし、日次平均 * 営業日数)。"""
    if test_ret.empty:
        return pd.DataFrame()
    monthly = test_ret.groupby(test_ret.index.month)
    stats = pd.DataFrame({
        "mean_daily_ret_bps": monthly.mean() * 10000,
        "mean_monthly_ret_pct": monthly.mean() * 21 * 100,  # approx 21 trading days/month
        "std_daily_ret_bps": monthly.std() * 10000,
        "n_days": monthly.count(),
        "win_rate_pct": monthly.apply(lambda x: (x > 0).mean() * 100),
    })
    stats.index.name = "month"
    return stats


def _yearly_stats(test_ret):
    """年ごとの成績。"""
    if test_ret.empty:
        return pd.DataFrame()
    rows = []
    for yr in sorted(test_ret.index.year.unique()):
        yr_ret = test_ret[test_ret.index.year == yr]
        m = compute_metrics(yr_ret)
        rows.append({"year": yr, **m})
    return pd.DataFrame(rows).set_index("year")


def generate_report():
    us_cc, jp_oc, jp_am, jp_pm = collect()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CV splits
    all_years = sorted(jp_oc.index.year.unique())
    splits = generate_splits(all_years, n_train=3, n_test=3,
                             n_splits=N_CV_SPLITS, seed=RANDOM_SEED)
    print(f"CV splits: {len(splits)} (Train 3yr < Test 3yr, seed={RANDOM_SEED})")

    # ---- Define strategies ----
    strategies = {}

    # 1. Paper baseline
    print("Computing: Paper (L=60 lam=0.9 K=3 q=0.3 OC)...")
    df, _ = run_pca_sub(us_cc, jp_oc, window=60, lam=0.9, K=3, q=0.3)
    strategies["Paper_PCA_SUB"] = {
        "returns": df["strategy_return"],
        "desc": "L=60, lam=0.9, K=3, q=0.3, exec=OC (Open->Close)",
    }

    # 2. Grid search best
    print("Computing: Grid Best (L=120 lam=0.5 K=3 q=0.2 OC)...")
    df, _ = run_pca_sub(us_cc, jp_oc, window=120, lam=0.5, K=3, q=0.2)
    strategies["Grid_Best"] = {
        "returns": df["strategy_return"],
        "desc": "L=120, lam=0.5, K=3, q=0.2, exec=OC (Open->Close)",
    }

    # 3. MOM
    print("Computing: MOM...")
    strategies["MOM"] = {
        "returns": run_momentum(us_cc, jp_oc)["strategy_return"],
        "desc": "Momentum baseline, L=60, q=0.3",
    }

    # 4. Optuna Rank 1 (AM)
    print("Computing: Optuna Rank1 (AM, EMA=9)...")
    df = run_pca_sub_extended(
        us_cc, jp_am, window=128, lam=0.21, K=5, q=0.42,
        long_only=False, signal_threshold=0.05, signal_ema=9,
    )
    strategies["Optuna_R1_AM"] = {
        "returns": df["strategy_return"],
        "desc": "L=128, lam=0.21, K=5, q=0.42, exec=AM (Open->11:30), EMA=9, thresh=0.05",
    }

    # 5. Optuna Rank 2 (AM)
    print("Computing: Optuna Rank2 (AM, EMA=10)...")
    df = run_pca_sub_extended(
        us_cc, jp_am, window=123, lam=0.32, K=5, q=0.40,
        long_only=False, signal_threshold=0.09, signal_ema=10,
    )
    strategies["Optuna_R2_AM"] = {
        "returns": df["strategy_return"],
        "desc": "L=123, lam=0.32, K=5, q=0.40, exec=AM (Open->11:30), EMA=10, thresh=0.09",
    }

    # 6. Optuna Rank 3 (OC)
    print("Computing: Optuna Rank3 (OC, EMA=10)...")
    df = run_pca_sub_extended(
        us_cc, jp_oc, window=130, lam=0.38, K=5, q=0.36,
        long_only=False, signal_threshold=0.10, signal_ema=10,
    )
    strategies["Optuna_R3_OC"] = {
        "returns": df["strategy_return"],
        "desc": "L=130, lam=0.38, K=5, q=0.36, exec=OC (Open->Close), EMA=10, thresh=0.10",
    }

    # 7. Optuna Rank 3 variant: PM execution
    print("Computing: Optuna PM variant (PM, EMA=10)...")
    df = run_pca_sub_extended(
        us_cc, jp_pm, window=130, lam=0.38, K=5, q=0.36,
        long_only=False, signal_threshold=0.10, signal_ema=10,
    )
    strategies["Optuna_PM"] = {
        "returns": df["strategy_return"],
        "desc": "L=130, lam=0.38, K=5, q=0.36, exec=PM (12:30->Close), EMA=10, thresh=0.10",
    }

    # ---- Compute test-period returns for each strategy ----
    print("\nCollecting test-period returns across CV splits...")
    report_data = {}
    for name, s in strategies.items():
        test_ret = _collect_test_returns(s["returns"], splits)
        full_m = compute_metrics(s["returns"])

        # CV split-level R/R for statistics
        cv_rrs = []
        cv_ars = []
        for train_years, test_years in splits:
            tr = s["returns"][s["returns"].index.year.isin(set(test_years))]
            if len(tr) >= 30:
                m = compute_metrics(tr)
                cv_rrs.append(m["R/R"])
                cv_ars.append(m["AR (%)"])
        cv_rrs = np.array(cv_rrs)
        cv_ars = np.array(cv_ars)

        report_data[name] = {
            "desc": s["desc"],
            "test_ret": test_ret,
            "full_metrics": full_m,
            "monthly": _monthly_stats(test_ret),
            "yearly": _yearly_stats(test_ret),
            "cv_rrs": cv_rrs,
            "cv_ars": cv_ars,
        }

    # ---- Generate outputs ----
    _write_csv_tables(report_data)
    _write_markdown_report(report_data, splits)
    _plot_monthly_heatmap(report_data)
    _plot_monthly_bars(report_data)

    print(f"\nReport saved to {RESULTS_DIR}/")


def _write_csv_tables(report_data):
    """Save monthly and yearly stats as CSV."""
    # Monthly
    all_monthly = []
    for name, d in report_data.items():
        df = d["monthly"].copy()
        df["strategy"] = name
        all_monthly.append(df)
    pd.concat(all_monthly).to_csv(os.path.join(RESULTS_DIR, "monthly_returns.csv"))

    # Yearly
    all_yearly = []
    for name, d in report_data.items():
        df = d["yearly"].copy()
        df["strategy"] = name
        all_yearly.append(df)
    pd.concat(all_yearly).to_csv(os.path.join(RESULTS_DIR, "yearly_returns.csv"))

    # Summary
    rows = []
    for name, d in report_data.items():
        rows.append({
            "strategy": name,
            "full_AR_pct": d["full_metrics"]["AR (%)"],
            "full_RR": d["full_metrics"]["R/R"],
            "full_MDD_pct": d["full_metrics"]["MDD (%)"],
            "cv_test_RR_mean": d["cv_rrs"].mean(),
            "cv_test_RR_median": np.median(d["cv_rrs"]),
            "cv_test_RR_std": d["cv_rrs"].std(),
            "cv_test_positive_pct": (d["cv_rrs"] > 0).mean() * 100,
            "cv_test_AR_mean": d["cv_ars"].mean(),
        })
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)


def _write_markdown_report(report_data, splits):
    """Generate comprehensive Markdown report."""
    lines = []
    lines.append("# PCA_SUB Lead-Lag Strategy Report")
    lines.append("")
    lines.append(f"Generated: Temporal CV with {len(splits)} splits "
                 f"(Train 3yr < Test 3yr, seed={RANDOM_SEED})")
    lines.append("")

    # ---- Summary Table ----
    lines.append("## 1. Strategy Summary")
    lines.append("")
    lines.append("| Strategy | Description | Full AR | Full R/R | Full MDD | "
                 "CV Test R/R (median) | CV Test R/R>0 |")
    lines.append("|----------|-------------|---------|----------|----------|"
                 "--------------------|---------------|")
    for name, d in report_data.items():
        fm = d["full_metrics"]
        cv = d["cv_rrs"]
        lines.append(f"| {name} | {d['desc']} | "
                     f"{fm['AR (%)']:.2f}% | {fm['R/R']:.2f} | {fm['MDD (%)']:.2f}% | "
                     f"{np.median(cv):.2f} | "
                     f"{(cv > 0).sum()}/{len(cv)} ({(cv > 0).mean()*100:.0f}%) |")
    lines.append("")

    # ---- Monthly Returns (Test period) ----
    lines.append("## 2. Monthly Average Returns (Test Period, bps/day)")
    lines.append("")
    lines.append("Each cell shows the average daily return in basis points for that month,")
    lines.append("computed from test-period data across all CV splits.")
    lines.append("")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    header = "| Strategy |" + "|".join(f" {m} " for m in month_names) + "| Avg |"
    sep = "|----------|" + "|".join(["-----:" for _ in month_names]) + "|-----:|"
    lines.append(header)
    lines.append(sep)

    for name, d in report_data.items():
        monthly = d["monthly"]
        vals = []
        for m in range(1, 13):
            if m in monthly.index:
                v = monthly.loc[m, "mean_daily_ret_bps"]
                vals.append(f"{v:+.1f}")
            else:
                vals.append("-")
        avg = monthly["mean_daily_ret_bps"].mean() if not monthly.empty else 0
        row = f"| {name} |" + "|".join(f" {v} " for v in vals) + f"| {avg:+.1f} |"
        lines.append(row)
    lines.append("")

    # ---- Monthly Returns (%) ----
    lines.append("## 3. Monthly Average Returns (Test Period, %/month)")
    lines.append("")
    header = "| Strategy |" + "|".join(f" {m} " for m in month_names) + "| Avg |"
    lines.append(header)
    lines.append(sep)

    for name, d in report_data.items():
        monthly = d["monthly"]
        vals = []
        for m in range(1, 13):
            if m in monthly.index:
                v = monthly.loc[m, "mean_monthly_ret_pct"]
                vals.append(f"{v:+.2f}")
            else:
                vals.append("-")
        avg = monthly["mean_monthly_ret_pct"].mean() if not monthly.empty else 0
        row = f"| {name} |" + "|".join(f" {v} " for v in vals) + f"| {avg:+.2f} |"
        lines.append(row)
    lines.append("")

    # ---- Monthly Win Rate ----
    lines.append("## 4. Monthly Win Rate (Test Period, %)")
    lines.append("")
    header = "| Strategy |" + "|".join(f" {m} " for m in month_names) + "| Avg |"
    lines.append(header)
    lines.append(sep)

    for name, d in report_data.items():
        monthly = d["monthly"]
        vals = []
        for m in range(1, 13):
            if m in monthly.index:
                v = monthly.loc[m, "win_rate_pct"]
                vals.append(f"{v:.0f}")
            else:
                vals.append("-")
        avg = monthly["win_rate_pct"].mean() if not monthly.empty else 0
        row = f"| {name} |" + "|".join(f" {v} " for v in vals) + f"| {avg:.0f} |"
        lines.append(row)
    lines.append("")

    # ---- Yearly Returns ----
    lines.append("## 5. Yearly Performance (Full Period)")
    lines.append("")
    all_years = set()
    for d in report_data.values():
        all_years.update(d["yearly"].index.tolist())
    years = sorted(all_years)

    header = "| Strategy |" + "|".join(f" {y} " for y in years) + "|"
    sep_yr = "|----------|" + "|".join(["------:" for _ in years]) + "|"
    lines.append(header)
    lines.append(sep_yr)

    for name, d in report_data.items():
        yearly = d["yearly"]
        vals = []
        for yr in years:
            if yr in yearly.index:
                ar = yearly.loc[yr, "AR (%)"]
                vals.append(f"{ar:+.1f}%")
            else:
                vals.append("-")
        row = f"| {name} |" + "|".join(f" {v} " for v in vals) + "|"
        lines.append(row)
    lines.append("")

    # ---- CV Distribution ----
    lines.append("## 6. Temporal CV Test R/R Distribution")
    lines.append("")
    lines.append("| Strategy | Mean | Std | Min | 25% | Median | 75% | Max | >0% |")
    lines.append("|----------|------|-----|-----|-----|--------|-----|-----|-----|")
    for name, d in report_data.items():
        cv = d["cv_rrs"]
        lines.append(f"| {name} | {cv.mean():.2f} | {cv.std():.2f} | "
                     f"{cv.min():.2f} | {np.percentile(cv, 25):.2f} | "
                     f"{np.median(cv):.2f} | {np.percentile(cv, 75):.2f} | "
                     f"{cv.max():.2f} | {(cv > 0).mean()*100:.0f}% |")
    lines.append("")

    # ---- Key Findings ----
    lines.append("## 7. Key Findings")
    lines.append("")
    lines.append("- Signal EMA smoothing (9-10 days) significantly improves R/R by reducing noise")
    lines.append("- AM execution (Open->11:30) achieves lowest MDD (-5~6%) with competitive R/R")
    lines.append("- OC execution (Open->Close) has higher absolute returns but wider drawdowns")
    lines.append("- PM execution (12:30->Close) captures afternoon momentum")
    lines.append("- Long-short is essential; long-only variants underperform")
    lines.append("- Optimal lambda (0.2-0.4) is lower than the paper's 0.9")
    lines.append("- K=5 factors outperform K=3 when combined with EMA smoothing")
    lines.append("")

    report_path = os.path.join(RESULTS_DIR, "strategy_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {report_path}")


def _plot_monthly_heatmap(report_data):
    """Monthly return heatmap (bps/day) for all strategies."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    strat_names = list(report_data.keys())
    data = np.full((len(strat_names), 12), np.nan)

    for i, name in enumerate(strat_names):
        monthly = report_data[name]["monthly"]
        for m in range(1, 13):
            if m in monthly.index:
                data[i, m - 1] = monthly.loc[m, "mean_daily_ret_bps"]

    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_names, fontsize=9)
    for i in range(len(strat_names)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Monthly Average Return (bps/day, Test Period)")
    fig.colorbar(im, label="bps/day")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "monthly_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: monthly_heatmap.png")


def _plot_monthly_bars(report_data):
    """Monthly return bar chart for selected strategies."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected = ["Paper_PCA_SUB", "Optuna_R2_AM", "Optuna_R3_OC", "MOM"]
    selected = [s for s in selected if s in report_data]

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, name in zip(axes, selected):
        monthly = report_data[name]["monthly"]
        vals = [monthly.loc[m, "mean_monthly_ret_pct"] if m in monthly.index else 0
                for m in range(1, 13)]
        colors = ["green" if v > 0 else "red" for v in vals]
        ax.bar(range(12), vals, color=colors, alpha=0.7)
        ax.set_ylabel("% / month")
        ax.set_title(f"{name}: Monthly Avg Return (Test Period)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="y")

    axes[-1].set_xticks(range(12))
    axes[-1].set_xticklabels(month_names)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "monthly_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: monthly_bars.png")


if __name__ == "__main__":
    generate_report()
