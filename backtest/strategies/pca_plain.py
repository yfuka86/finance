"""
PCA_PLAIN: Standard PCA without subspace regularization (λ=0).
Baseline to measure the value of prior-guided regularization.
"""
import numpy as np
import pandas as pd
from data.collectors.config import US_TICKERS, JP_TICKERS


def run_pca_plain(us_ret: pd.DataFrame, jp_ret: pd.DataFrame, **kwargs):
    """
    PCA without regularization. Same as PCA_SUB but λ=0 (C_reg = C_t).
    """
    L = kwargs.get("window", 60)
    K = kwargs.get("K", 3)
    q = kwargs.get("q", 0.3)

    us_cols = [t for t in US_TICKERS if t in us_ret.columns]
    jp_cols = [t for t in JP_TICKERS if t in jp_ret.columns]
    us_ret, jp_ret = us_ret[us_cols], jp_ret[jp_cols]
    N_U, N_J = len(us_cols), len(jp_cols)

    us_sorted = sorted(us_ret.index)
    pairs = []
    for jd in jp_ret.index:
        cands = [d for d in us_sorted if d < jd]
        if cands:
            pairs.append((cands[-1], jd))

    us_aligned = us_ret.loc[[p[0] for p in pairs]].values
    jp_aligned = jp_ret.loc[[p[1] for p in pairs]].values
    combined = np.nan_to_num(np.hstack([us_aligned, jp_aligned]), nan=0.0)
    T = len(pairs)

    standardized = np.full_like(combined, np.nan)
    for t in range(L, T):
        w = combined[t - L:t]
        mu = w.mean(axis=0)
        sigma = np.where((s := w.std(axis=0)) > 1e-10, s, 1e-10)
        standardized[t] = (combined[t] - mu) / sigma

    results = []
    for t in range(L, T - 1):
        C_t = np.corrcoef(standardized[t - L + 1:t + 1].T)
        C_t = np.nan_to_num(C_t, nan=0.0)
        np.fill_diagonal(C_t, 1.0)

        evals, evecs = np.linalg.eigh(C_t)
        idx = np.argsort(evals)[::-1]
        V_K = evecs[:, idx[:K]]
        V_U, V_J = V_K[:N_U], V_K[N_U:]

        f_t = V_U.T @ standardized[t, :N_U]
        signal = V_J @ f_t

        n = max(1, int(np.ceil(N_J * q)))
        ranked = np.argsort(signal)[::-1]
        w = np.zeros(N_J)
        w[ranked[:n]] = 1.0 / n
        w[ranked[-n:]] = -1.0 / n

        ret = np.dot(w, combined[t + 1, N_U:])
        results.append({"date": pairs[t + 1][1], "strategy_return": ret})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df
