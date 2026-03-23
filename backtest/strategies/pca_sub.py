"""
PCA_SUB: Subspace Regularized PCA Lead-Lag Strategy
Nakagawa et al. (2026), SIG-FIN-036-13
"""
import numpy as np
import pandas as pd
from data.collectors.config import (
    US_TICKERS, JP_TICKERS,
    US_CYCLICAL, US_DEFENSIVE, JP_CYCLICAL, JP_DEFENSIVE,
)

# Default parameters from the paper
DEFAULTS = {
    "window": 60,
    "lam": 0.9,
    "K": 3,
    "q": 0.3,
    "full_window_start": "2010-01-01",
    "full_window_end": "2014-12-31",
}


def _build_prior_vectors(us_tickers, jp_tickers):
    N_U, N_J = len(us_tickers), len(jp_tickers)
    N = N_U + N_J
    v1 = np.ones(N); v1 /= np.linalg.norm(v1)

    v2 = np.zeros(N); v2[:N_U] = 1.0; v2[N_U:] = -1.0
    v2 -= np.dot(v2, v1) * v1; v2 /= np.linalg.norm(v2)

    v3 = np.zeros(N)
    for i, t in enumerate(us_tickers + jp_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL: v3[i] = 1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE: v3[i] = -1.0
    v3 -= np.dot(v3, v1) * v1 + np.dot(v3, v2) * v2
    v3 /= np.linalg.norm(v3)

    return np.column_stack([v1, v2, v3])


def _compute_C0(V0, C_full):
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T
    delta = np.maximum(np.diag(C0_raw), 1e-8)
    delta_inv = 1.0 / np.sqrt(delta)
    C0 = C0_raw * np.outer(delta_inv, delta_inv)
    np.fill_diagonal(C0, 1.0)
    return C0


def run_pca_sub(us_ret: pd.DataFrame, jp_ret: pd.DataFrame, **kwargs):
    """
    Run PCA_SUB strategy.

    Args:
        us_ret: US close-to-close returns (columns = US tickers)
        jp_ret: JP open-to-close (or AM) returns (columns = JP tickers)
        **kwargs: override default parameters (window, lam, K, q, ...)

    Returns:
        (results_df, signals_list)
    """
    params = {**DEFAULTS, **kwargs}
    L, lam, K, q = params["window"], params["lam"], params["K"], params["q"]

    us_cols = [t for t in US_TICKERS if t in us_ret.columns]
    jp_cols = [t for t in JP_TICKERS if t in jp_ret.columns]
    us_ret, jp_ret = us_ret[us_cols], jp_ret[jp_cols]
    N_U, N_J = len(us_cols), len(jp_cols)

    # Align dates: US day t → JP day t+1
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

    # Rolling standardization
    standardized = np.full_like(combined, np.nan)
    for t in range(L, T):
        w = combined[t - L:t]
        mu = w.mean(axis=0)
        sigma = np.where((s := w.std(axis=0)) > 1e-10, s, 1e-10)
        standardized[t] = (combined[t] - mu) / sigma

    # C_full from early period
    full_start = pd.Timestamp(params["full_window_start"])
    full_end = pd.Timestamp(params["full_window_end"])
    full_data = [standardized[t] for t in range(L, T)
                 if full_start <= pd.Timestamp(pairs[t][0]) <= full_end]
    if len(full_data) < 50:
        full_data = [standardized[t] for t in range(L, min(T, L + 600))]
    C_full = np.corrcoef(np.array(full_data).T)
    C_full = np.nan_to_num(C_full, nan=0.0)
    np.fill_diagonal(C_full, 1.0)

    V0 = _build_prior_vectors(us_cols, jp_cols)
    C0 = _compute_C0(V0, C_full)

    # Main loop
    results = []
    signals = []
    for t in range(L, T - 1):
        C_t = np.corrcoef(standardized[t - L + 1:t + 1].T)
        C_t = np.nan_to_num(C_t, nan=0.0)
        np.fill_diagonal(C_t, 1.0)

        C_reg = (1 - lam) * C_t + lam * C0
        evals, evecs = np.linalg.eigh(C_reg)
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
        signals.append({"date": pairs[t + 1][1], "signal": signal.copy()})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df, signals
