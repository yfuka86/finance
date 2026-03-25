"""
Optuna-based strategy optimization with Temporal CV.

探索空間:
- PCA_SUB連続パラメータ (window, λ, K, q)
- 執行タイミング (寄付き→大引け / 寄付き→前引け / 後場寄り→大引け)
- ロングオンリー / ロングショート
- シグナル閾値 (弱いシグナルは無視)
- シグナル平滑化 (複数日のEMA)

目的関数: Temporal CV (Train 3年 < Test 3年) のTest R/R中央値
"""
import os
import warnings
import itertools
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.collect import collect
from data.collectors.config import (
    US_TICKERS, JP_TICKERS,
    US_CYCLICAL, US_DEFENSIVE, JP_CYCLICAL, JP_DEFENSIVE,
)
from backtest.strategies.pca_sub import _build_prior_vectors, _compute_C0
from backtest.metrics import compute_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "optuna")
RANDOM_SEED = 42
N_CV_SPLITS = 30  # CV splits for objective (subset for speed)


# ============================================================
# Flexible PCA_SUB strategy with extended options
# ============================================================

def run_pca_sub_extended(us_ret, jp_ret, *,
                         window, lam, K, q,
                         long_only=False,
                         signal_threshold=0.0,
                         signal_ema=1):
    """
    Extended PCA_SUB with additional parameters.

    Args:
        signal_threshold: minimum |signal| rank-score to enter a position
        signal_ema: EMA span for signal smoothing (1 = no smoothing)
        long_only: if True, only take long positions (no short selling)
    """
    us_cols = [t for t in US_TICKERS if t in us_ret.columns]
    jp_cols = [t for t in JP_TICKERS if t in jp_ret.columns]
    us_ret_df, jp_ret_df = us_ret[us_cols], jp_ret[jp_cols]
    N_U, N_J = len(us_cols), len(jp_cols)
    L = window

    us_sorted = sorted(us_ret_df.index)
    pairs = []
    for jd in jp_ret_df.index:
        cands = [d for d in us_sorted if d < jd]
        if cands:
            pairs.append((cands[-1], jd))

    us_aligned = us_ret_df.loc[[p[0] for p in pairs]].values
    jp_aligned = jp_ret_df.loc[[p[1] for p in pairs]].values
    combined = np.nan_to_num(np.hstack([us_aligned, jp_aligned]), nan=0.0)
    T = len(pairs)

    standardized = np.full_like(combined, np.nan)
    for t in range(L, T):
        w = combined[t - L:t]
        mu = w.mean(axis=0)
        sigma = np.where((s := w.std(axis=0)) > 1e-10, s, 1e-10)
        standardized[t] = (combined[t] - mu) / sigma

    full_start = pd.Timestamp("2010-01-01")
    full_end = pd.Timestamp("2014-12-31")
    full_data = [standardized[t] for t in range(L, T)
                 if full_start <= pd.Timestamp(pairs[t][0]) <= full_end]
    if len(full_data) < 50:
        full_data = [standardized[t] for t in range(L, min(T, L + 600))]
    C_full = np.corrcoef(np.array(full_data).T)
    C_full = np.nan_to_num(C_full, nan=0.0)
    np.fill_diagonal(C_full, 1.0)

    V0 = _build_prior_vectors(us_cols, jp_cols)
    C0 = _compute_C0(V0, C_full)

    # Compute all raw signals first
    raw_signals = np.full((T, N_J), np.nan)
    for t in range(L, T):
        C_t = np.corrcoef(standardized[t - L + 1:t + 1].T)
        C_t = np.nan_to_num(C_t, nan=0.0)
        np.fill_diagonal(C_t, 1.0)

        C_reg = (1 - lam) * C_t + lam * C0
        evals, evecs = np.linalg.eigh(C_reg)
        idx = np.argsort(evals)[::-1]
        V_K = evecs[:, idx[:K]]
        V_U, V_J = V_K[:N_U], V_K[N_U:]

        f_t = V_U.T @ standardized[t, :N_U]
        raw_signals[t] = V_J @ f_t

    # Apply EMA smoothing
    if signal_ema > 1:
        alpha = 2.0 / (signal_ema + 1)
        smoothed = np.full_like(raw_signals, np.nan)
        smoothed[L] = raw_signals[L]
        for t in range(L + 1, T):
            if np.isnan(raw_signals[t, 0]):
                continue
            prev = smoothed[t - 1] if not np.isnan(smoothed[t - 1, 0]) else raw_signals[t]
            smoothed[t] = alpha * raw_signals[t] + (1 - alpha) * prev
        signals = smoothed
    else:
        signals = raw_signals

    # Build portfolio returns
    results = []
    n = max(1, int(np.ceil(N_J * q)))
    for t in range(L, T - 1):
        if np.isnan(signals[t, 0]):
            continue

        sig = signals[t]
        ranked = np.argsort(sig)[::-1]

        w = np.zeros(N_J)

        # Signal threshold: only trade if signal spread is meaningful
        sig_range = sig.max() - sig.min()
        if sig_range < signal_threshold:
            results.append({"date": pairs[t + 1][1], "strategy_return": 0.0})
            continue

        if long_only:
            w[ranked[:n]] = 1.0 / n
        else:
            w[ranked[:n]] = 1.0 / n
            w[ranked[-n:]] = -1.0 / n

        ret = np.dot(w, combined[t + 1, N_U:])
        results.append({"date": pairs[t + 1][1], "strategy_return": ret})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


# ============================================================
# Temporal CV splits
# ============================================================

def generate_splits(all_years, n_train=3, n_test=3, n_splits=30, seed=42):
    rng = np.random.RandomState(seed)
    all_valid = []
    for tr in itertools.combinations(all_years, n_train):
        for te in itertools.combinations(all_years, n_test):
            if max(tr) < min(te):
                all_valid.append((list(tr), list(te)))
    n_actual = min(n_splits, len(all_valid))
    indices = rng.choice(len(all_valid), size=n_actual, replace=False)
    return [all_valid[i] for i in indices]


# ============================================================
# Optuna objective
# ============================================================

def create_objective(us_cc, jp_returns_dict, splits):
    """
    Create Optuna objective function.
    jp_returns_dict: {"oc": jp_oc, "am": jp_am, "pm": jp_pm}
    """
    def objective(trial):
        # --- Sample parameters ---
        window = trial.suggest_int("window", 20, 300)
        lam = trial.suggest_float("lam", 0.0, 0.99)
        K = trial.suggest_int("K", 2, 5)
        q = trial.suggest_float("q", 0.1, 0.5)
        execution = trial.suggest_categorical("execution", ["oc", "am", "pm"])
        long_only = trial.suggest_categorical("long_only", [True, False])
        signal_threshold = trial.suggest_float("signal_threshold", 0.0, 2.0)
        signal_ema = trial.suggest_int("signal_ema", 1, 10)

        jp_ret = jp_returns_dict[execution]

        # Run strategy once on full period
        try:
            df = run_pca_sub_extended(
                us_cc, jp_ret,
                window=window, lam=lam, K=K, q=q,
                long_only=long_only,
                signal_threshold=signal_threshold,
                signal_ema=signal_ema,
            )
        except Exception:
            return -999.0

        ret = df["strategy_return"]
        if len(ret) < 200:
            return -999.0

        # Evaluate on CV splits
        test_rrs = []
        for train_years, test_years in splits:
            train_set = set(train_years)
            test_set = set(test_years)
            train_ret = ret[ret.index.year.isin(train_set)]
            test_ret = ret[ret.index.year.isin(test_set)]
            if len(train_ret) < 30 or len(test_ret) < 30:
                continue
            test_m = compute_metrics(test_ret)
            test_rrs.append(test_m["R/R"])

        if len(test_rrs) < 10:
            return -999.0

        # Objective: median Test R/R (robust to outliers)
        return float(np.median(test_rrs))

    return objective


# ============================================================
# Main
# ============================================================

def run_optuna_optimization():
    us_cc, jp_oc, jp_am, jp_pm = collect()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Data: US={us_cc.shape}, JP_OC={jp_oc.shape}, JP_AM={jp_am.shape}, JP_PM={jp_pm.shape}")

    jp_returns_dict = {"oc": jp_oc, "am": jp_am, "pm": jp_pm}

    # Generate CV splits
    sample_ret_idx = jp_oc.index
    all_years = sorted(sample_ret_idx.year.unique())
    splits = generate_splits(all_years, n_train=3, n_test=3,
                             n_splits=N_CV_SPLITS, seed=RANDOM_SEED)
    print(f"CV splits: {len(splits)} (Train 3yr < Test 3yr)")

    # Create study
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pca_sub_temporal_cv",
    )

    objective = create_objective(us_cc, jp_returns_dict, splits)

    print("\n=== Optuna Optimization (200 trials) ===")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    # ---- Results ----
    print("\n" + "=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)

    best = study.best_trial
    print(f"\nBest Trial #{best.number}:")
    print(f"  Objective (median Test R/R): {best.value:.3f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Top 10 trials
    print(f"\n--- Top 10 Trials ---")
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["value"] > -900]
    top10 = trials_df.nlargest(10, "value")
    print(f"  {'#':>4s}  {'R/R':>6s}  {'window':>6s}  {'lam':>5s}  {'K':>2s}  "
          f"{'q':>4s}  {'exec':>4s}  {'long':>5s}  {'thresh':>6s}  {'ema':>3s}")
    print("  " + "-" * 65)
    for _, r in top10.iterrows():
        print(f"  {int(r['number']):>4d}  {r['value']:>6.3f}  "
              f"{int(r['params_window']):>6d}  {r['params_lam']:>5.2f}  "
              f"{int(r['params_K']):>2d}  {r['params_q']:>4.2f}  "
              f"{r['params_execution']:>4s}  {str(r['params_long_only']):>5s}  "
              f"{r['params_signal_threshold']:>6.2f}  {int(r['params_signal_ema']):>3d}")

    # ---- Detailed evaluation of top 3 ----
    print(f"\n--- Top 3 Detailed Evaluation (100 CV splits) ---")
    splits_100 = generate_splits(all_years, n_train=3, n_test=3,
                                 n_splits=100, seed=RANDOM_SEED)

    top3_results = []
    for rank, (_, r) in enumerate(top10.head(3).iterrows()):
        params = {
            "window": int(r["params_window"]),
            "lam": r["params_lam"],
            "K": int(r["params_K"]),
            "q": r["params_q"],
            "long_only": r["params_long_only"],
            "signal_threshold": r["params_signal_threshold"],
            "signal_ema": int(r["params_signal_ema"]),
        }
        execution = r["params_execution"]
        jp_ret = jp_returns_dict[execution]

        df = run_pca_sub_extended(us_cc, jp_ret, **params)
        ret = df["strategy_return"]
        full_m = compute_metrics(ret)

        test_rrs = []
        test_ars = []
        for train_years, test_years in splits_100:
            test_ret = ret[ret.index.year.isin(set(test_years))]
            if len(test_ret) < 30:
                continue
            m = compute_metrics(test_ret)
            test_rrs.append(m["R/R"])
            test_ars.append(m["AR (%)"])

        test_rrs = np.array(test_rrs)
        test_ars = np.array(test_ars)

        print(f"\n  Rank {rank + 1}: Trial #{int(r['number'])}")
        print(f"    window={params['window']}, lam={params['lam']:.2f}, K={params['K']}, "
              f"q={params['q']:.2f}, exec={execution}, "
              f"long_only={params['long_only']}, thresh={params['signal_threshold']:.2f}, "
              f"ema={params['signal_ema']}")
        print(f"    Full period: AR={full_m['AR (%)']:.2f}%, R/R={full_m['R/R']:.2f}, "
              f"MDD={full_m['MDD (%)']:.2f}%")
        print(f"    100-split CV Test R/R: mean={test_rrs.mean():.2f}, "
              f"median={np.median(test_rrs):.2f}, std={test_rrs.std():.2f}, "
              f">0: {(test_rrs > 0).sum()}/{len(test_rrs)} ({(test_rrs > 0).mean()*100:.0f}%)")
        print(f"    100-split CV Test AR:  mean={test_ars.mean():.2f}%, "
              f"median={np.median(test_ars):.2f}%, "
              f">0: {(test_ars > 0).sum()}/{len(test_ars)} ({(test_ars > 0).mean()*100:.0f}%)")

        # Yearly breakdown
        print(f"    Yearly:")
        years = sorted(ret.index.year.unique())
        win = 0
        for yr in years:
            yr_m = compute_metrics(ret[ret.index.year == yr])
            marker = "+" if yr_m["AR (%)"] > 0 else "-"
            if yr_m["AR (%)"] > 0:
                win += 1
            print(f"      {yr}: AR={yr_m['AR (%)']:>7.2f}%  R/R={yr_m['R/R']:>5.2f}  {marker}")
        print(f"    Win rate: {win}/{len(years)} ({100*win/len(years):.0f}%)")

        top3_results.append({
            "rank": rank + 1,
            "params": params,
            "execution": execution,
            "full_metrics": full_m,
            "cv_test_rrs": test_rrs,
            "returns": ret,
        })

    # ---- Compare with baselines ----
    print(f"\n--- Baseline Comparison (100-split CV) ---")
    from backtest.strategies.pca_sub import run_pca_sub
    from backtest.strategies.momentum import run_momentum

    baselines = {
        "Paper (L=60 lam=0.9 OC)": run_pca_sub(us_cc, jp_oc)[0]["strategy_return"],
        "MOM": run_momentum(us_cc, jp_oc)["strategy_return"],
    }
    for name, ret in baselines.items():
        test_rrs = []
        for train_years, test_years in splits_100:
            test_ret = ret[ret.index.year.isin(set(test_years))]
            if len(test_ret) < 30:
                continue
            test_rrs.append(compute_metrics(test_ret)["R/R"])
        test_rrs = np.array(test_rrs)
        full_m = compute_metrics(ret)
        print(f"  {name}: Full R/R={full_m['R/R']:.2f}, "
              f"CV median={np.median(test_rrs):.2f}, "
              f">0: {(test_rrs > 0).sum()}/{len(test_rrs)} ({(test_rrs > 0).mean()*100:.0f}%)")

    # ---- Plots ----
    _plot_optimization_history(study)
    _plot_param_importances(study)
    _plot_top3_cumulative(top3_results, baselines)

    # Save
    trials_df.to_csv(os.path.join(RESULTS_DIR, "all_trials.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")

    return study, top3_results


def _plot_optimization_history(study):
    fig, ax = plt.subplots(figsize=(12, 5))
    trials = [t for t in study.trials if t.value is not None and t.value > -900]
    values = [t.value for t in trials]
    numbers = [t.number for t in trials]
    ax.scatter(numbers, values, alpha=0.4, s=15)
    # Running best
    best_so_far = []
    current_best = -999
    for v in values:
        current_best = max(current_best, v)
        best_so_far.append(current_best)
    ax.plot(numbers, best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Median Test R/R")
    ax.set_title("Optuna Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "optimization_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_param_importances(study):
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(10, 5))
        params = list(importances.keys())
        vals = list(importances.values())
        ax.barh(params[::-1], vals[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importances (fANOVA)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "param_importances.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _plot_top3_cumulative(top3_results, baselines):
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in top3_results:
        p = r["params"]
        label = (f"Rank {r['rank']}: L={p['window']} lam={p['lam']:.2f} "
                 f"exec={r['execution']} (R/R={r['full_metrics']['R/R']:.2f})")
        ax.plot((1 + r["returns"]).cumprod(), label=label, linewidth=1.5)

    for name, ret in baselines.items():
        ax.plot((1 + ret).cumprod(), label=name, linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Optuna Top 3 vs Baselines")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "top3_cumulative.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_optuna_optimization()
