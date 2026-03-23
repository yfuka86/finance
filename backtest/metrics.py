"""
Performance metrics: AR, RISK, R/R, MDD.
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_metrics(returns: pd.Series) -> dict:
    ar = returns.mean() * 252 * 100
    risk = returns.std(ddof=1) * np.sqrt(252) * 100
    rr = ar / risk if risk > 0 else 0
    cumret = (1 + returns).cumprod()
    mdd = (cumret / cumret.cummax() - 1).min() * 100
    return {
        "AR (%)": round(ar, 2),
        "RISK (%)": round(risk, 2),
        "R/R": round(rr, 2),
        "MDD (%)": round(mdd, 2),
        "N_days": len(returns),
    }


def plot_cumulative(strategies: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, ret in strategies.items():
        ax.plot((1 + ret).cumprod(), label=name, linewidth=1.2)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns: US-JP Sector Lead-Lag Strategies")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_report(strategies: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    rows = []
    for name, ret in strategies.items():
        m = compute_metrics(ret)
        m["Strategy"] = name
        rows.append(m)
        print(f"\n{name}:")
        for k, v in m.items():
            if k != "Strategy":
                print(f"  {k}: {v}")

    df = pd.DataFrame(rows).set_index("Strategy")
    df.to_csv(os.path.join(results_dir, "metrics.csv"))

    plot_cumulative(strategies, os.path.join(results_dir, "cumulative_returns.png"))
    pd.DataFrame(strategies).to_csv(os.path.join(results_dir, "daily_returns.csv"))

    for name, ret in strategies.items():
        if "PCA" not in name:
            continue
        print(f"\n\nYearly ({name}):")
        print("-" * 65)
        for yr in sorted(ret.index.year.unique()):
            yr_m = compute_metrics(ret[ret.index.year == yr])
            print(f"  {yr}: AR={yr_m['AR (%)']:>7.2f}%  RISK={yr_m['RISK (%)']:>7.2f}%  "
                  f"R/R={yr_m['R/R']:>5.2f}  MDD={yr_m['MDD (%)']:>7.2f}%")

    print("\n" + "=" * 60)
    return df
