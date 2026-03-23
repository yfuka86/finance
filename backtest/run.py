"""
Backtest runner: executes strategies and generates performance reports.
"""
import os
from data.collect import collect
from backtest.strategies import run_pca_sub, run_momentum
from backtest.metrics import print_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_all():
    us_cc, jp_oc, jp_am = collect()

    print(f"\nUS: {us_cc.shape}, JP full: {jp_oc.shape}, JP AM: {jp_am.shape}")

    print("\n--- PCA_SUB (Full Day: Open→Close) ---")
    pca_full, _ = run_pca_sub(us_cc, jp_oc)

    print("\n--- PCA_SUB (AM Only: Open→11:30) ---")
    pca_am, _ = run_pca_sub(us_cc, jp_am)

    print("\n--- MOM Baseline ---")
    mom = run_momentum(us_cc, jp_oc)

    strategies = {
        "PCA_SUB (Full)": pca_full["strategy_return"],
        "PCA_SUB (AM)": pca_am["strategy_return"],
        "MOM": mom["strategy_return"],
    }
    print_report(strategies, RESULTS_DIR)


if __name__ == "__main__":
    run_all()
