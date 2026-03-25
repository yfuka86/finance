"""
DOUBLE: 2x2 Double Sort (MOM × PCA_SUB signal).
Intersection of momentum and PCA signal for stronger conviction.
"""
import numpy as np
import pandas as pd
from data.collectors.config import (
    US_TICKERS, JP_TICKERS,
    US_CYCLICAL, US_DEFENSIVE, JP_CYCLICAL, JP_DEFENSIVE,
)
from backtest.strategies.pca_sub import _build_prior_vectors, _compute_C0


def run_double_sort(us_ret: pd.DataFrame, jp_ret: pd.DataFrame, **kwargs):
    """
    Double sort: Long only if both MOM and PCA_SUB agree on direction.
    """
    L = kwargs.get("window", 60)
    lam = kwargs.get("lam", 0.9)
    K = kwargs.get("K", 3)
    q = kwargs.get("q", 0.3)
    full_start = pd.Timestamp(kwargs.get("full_window_start", "2010-01-01"))
    full_end = pd.Timestamp(kwargs.get("full_window_end", "2014-12-31"))

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

    full_data = [standardized[t] for t in range(L, T)
                 if full_start <= pd.Timestamp(pairs[t][0]) <= full_end]
    if len(full_data) < 50:
        full_data = [standardized[t] for t in range(L, min(T, L + 600))]
    C_full = np.corrcoef(np.array(full_data).T)
    C_full = np.nan_to_num(C_full, nan=0.0)
    np.fill_diagonal(C_full, 1.0)

    V0 = _build_prior_vectors(us_cols, jp_cols)
    C0 = _compute_C0(V0, C_full)

    results = []
    for t in range(L, T - 1):
        # PCA signal
        C_t = np.corrcoef(standardized[t - L + 1:t + 1].T)
        C_t = np.nan_to_num(C_t, nan=0.0)
        np.fill_diagonal(C_t, 1.0)
        C_reg = (1 - lam) * C_t + lam * C0
        evals, evecs = np.linalg.eigh(C_reg)
        idx = np.argsort(evals)[::-1]
        V_K = evecs[:, idx[:K]]
        V_U, V_J = V_K[:N_U], V_K[N_U:]
        f_t = V_U.T @ standardized[t, :N_U]
        pca_signal = V_J @ f_t

        # MOM signal
        mom_signal = np.nanmean(combined[t - L:t, N_U:], axis=0)

        # Double sort: rank by each, intersect top/bottom
        n = max(1, int(np.ceil(N_J * q)))
        pca_ranked = np.argsort(pca_signal)[::-1]
        mom_ranked = np.argsort(mom_signal)[::-1]

        pca_long = set(pca_ranked[:n])
        pca_short = set(pca_ranked[-n:])
        mom_long = set(mom_ranked[:n])
        mom_short = set(mom_ranked[-n:])

        long_set = pca_long & mom_long
        short_set = pca_short & mom_short

        w = np.zeros(N_J)
        if long_set:
            for i in long_set:
                w[i] = 1.0 / len(long_set)
        if short_set:
            for i in short_set:
                w[i] = -1.0 / len(short_set)

        ret = np.dot(w, combined[t + 1, N_U:])
        results.append({"date": pairs[t + 1][1], "strategy_return": ret})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df
