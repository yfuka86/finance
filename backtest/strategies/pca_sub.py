"""
PCA_SUB: Subspace Regularized PCA Lead-Lag Strategy
Nakagawa et al. (2026), SIG-FIN-036-13

バックテスト (run_pca_sub) と実売買 (compute_signal_latest) は
同じ _prepare / _signal_at を通すので、シグナルの定義が両者でズレない。
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
    # 取引するJPセッションの直前のUSセッションをシグナルに使うか。
    # False (旧実装の挙動):
    #   時点 t のUSから pairs[t+1] のJPセッションを取引する。
    #   pairs[t+1][0] のUSセッション (取引直前の夜) は使われず、1日分古い情報で建てる。
    # True:
    #   相関行列 C_t は時点 t までのまま (未来を見ない) で、
    #   ファクター f だけを pairs[t+1][0] のUSリターンから作る。
    #   取引直前のUSセッションは寄付き前に確定しているので look-ahead にはならない。
    #   論文の z_{U,t} -> z_{J,t+1} に対応するため、こちらを既定とする。
    "fresh_us": True,
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


class _Context:
    """_prepare の結果をまとめて持ち回すだけの入れ物"""
    __slots__ = ("params", "pairs", "combined", "standardized", "C0",
                 "us_cols", "jp_cols", "N_U", "N_J", "T", "used_fallback_C_full")


def _prepare(us_ret: pd.DataFrame, jp_ret: pd.DataFrame, params: dict) -> _Context:
    """US/JPリターンを整列・標準化し、事前分布 C0 まで構築する。"""
    L = params["window"]

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
    if T <= L:
        raise ValueError(
            f"データ不足: 整列後 {T} 日、window={L} 日必要です。"
        )

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
    used_fallback = len(full_data) < 50
    if used_fallback:
        full_data = [standardized[t] for t in range(L, min(T, L + 600))]
    C_full = np.corrcoef(np.array(full_data).T)
    C_full = np.nan_to_num(C_full, nan=0.0)
    np.fill_diagonal(C_full, 1.0)

    V0 = _build_prior_vectors(us_cols, jp_cols)

    ctx = _Context()
    ctx.params = params
    ctx.pairs = pairs
    ctx.combined = combined
    ctx.standardized = standardized
    ctx.C0 = _compute_C0(V0, C_full)
    ctx.us_cols = us_cols
    ctx.jp_cols = jp_cols
    ctx.N_U = N_U
    ctx.N_J = N_J
    ctx.T = T
    ctx.used_fallback_C_full = used_fallback
    return ctx


def _standardize_us(ctx: _Context, t: int, us_raw: np.ndarray) -> np.ndarray:
    """
    時点 t の直前 L 日の統計量で、生のUSリターンベクトルを標準化する。
    _prepare のローリング標準化と同じ式をUS列だけに適用したもの。
    """
    L = ctx.params["window"]
    w = ctx.combined[t - L:t, :ctx.N_U]
    mu = w.mean(axis=0)
    s = w.std(axis=0)
    sigma = np.where(s > 1e-10, s, 1e-10)
    return (us_raw - mu) / sigma


def _signal_at(ctx: _Context, t: int, us_std: np.ndarray = None) -> np.ndarray:
    """
    時点 t の相関構造と、USファクターから JP セクターのシグナルを計算する。
    このシグナルは pairs[t+1][1] のJPセッションを取引するためのもの。

    us_std: 標準化済みUSリターン。省略時は standardized[t] のUS部分
            (元実装の挙動) を使う。fresh_us=True のときは呼び出し側が
            pairs[t+1][0] のUSリターンを標準化して渡す。
    """
    L, lam, K = ctx.params["window"], ctx.params["lam"], ctx.params["K"]

    C_t = np.corrcoef(ctx.standardized[t - L + 1:t + 1].T)
    C_t = np.nan_to_num(C_t, nan=0.0)
    np.fill_diagonal(C_t, 1.0)

    C_reg = (1 - lam) * C_t + lam * ctx.C0
    evals, evecs = np.linalg.eigh(C_reg)
    idx = np.argsort(evals)[::-1]
    V_K = evecs[:, idx[:K]]
    V_U, V_J = V_K[:ctx.N_U], V_K[ctx.N_U:]

    if us_std is None:
        us_std = ctx.standardized[t, :ctx.N_U]
    f_t = V_U.T @ us_std
    return V_J @ f_t


def weights_from_signal(signal: np.ndarray, q: float) -> np.ndarray:
    """シグナル上位q/下位qを等ウェイトでロング/ショートする。"""
    N_J = len(signal)
    n = max(1, int(np.ceil(N_J * q)))
    ranked = np.argsort(signal)[::-1]
    w = np.zeros(N_J)
    w[ranked[:n]] = 1.0 / n
    w[ranked[-n:]] = -1.0 / n
    return w


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
    ctx = _prepare(us_ret, jp_ret, params)
    L, q, fresh = params["window"], params["q"], params["fresh_us"]

    results = []
    signals = []
    for t in range(L, ctx.T - 1):
        # fresh_us: 取引するJPセッション pairs[t+1][1] の直前のUSセッション
        # pairs[t+1][0] を使う。標準化の統計量は時点 t までのものに揃える。
        us_std = _standardize_us(ctx, t, ctx.combined[t + 1, :ctx.N_U]) if fresh else None
        signal = _signal_at(ctx, t, us_std)
        w = weights_from_signal(signal, q)
        ret = np.dot(w, ctx.combined[t + 1, ctx.N_U:])
        results.append({"date": ctx.pairs[t + 1][1], "strategy_return": ret})
        signals.append({"date": ctx.pairs[t + 1][1], "signal": signal.copy()})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df, signals


def compute_signal_latest(us_ret: pd.DataFrame, jp_ret: pd.DataFrame, **kwargs):
    """
    次のJPセッション向けのシグナルを1本だけ計算する (実売買用)。

    バックテストのループは t = T-2 まで進み pairs[T-1][1] を取引する。
    その次の t = T-1 が「まだ取引していない直近のシグナル」であり、
    対応するJPセッションは jp_ret の最終日の次の営業日 = 今日にあたる。

    Returns:
        dict:
          signal   : np.ndarray (JPセクターごとのシグナル)
          weights  : np.ndarray (ロング/ショートのターゲットウェイト)
          jp_tickers: list[str] (signal/weights と同じ並び)
          us_date  : シグナルの元になったUSセッション日
          jp_date  : 相関行列の終端にあたる直近のJPセッション日
          fresh_us : 取引直前のUSセッションを使ったか
          stale_us : fresh_us=True を指定したのに新しいUSセッションが無かったか
          n_days   : 整列後のサンプル数
          used_fallback_C_full: C_full が指定期間から作れず先頭600日で代用されたか
    """
    params = {**DEFAULTS, **kwargs}
    ctx = _prepare(us_ret, jp_ret, params)

    t = ctx.T - 1
    us_date = ctx.pairs[t][0]
    us_std = None
    stale = False

    if params["fresh_us"]:
        # 今日のJPセッションの直前のUSセッション = us_ret の最終行。
        # これは前夜のクローズなので寄付き前に確定している。
        latest = max(us_ret.index)
        if pd.Timestamp(latest) > pd.Timestamp(us_date):
            us_raw = us_ret.loc[latest, ctx.us_cols].values.astype(float)
            us_std = _standardize_us(ctx, t, np.nan_to_num(us_raw, nan=0.0))
            us_date = latest
        else:
            # USが未更新 (祝日など)。1日古いシグナルにフォールバックする。
            stale = True

    signal = _signal_at(ctx, t, us_std)
    return {
        "signal": signal,
        "weights": weights_from_signal(signal, params["q"]),
        "jp_tickers": list(ctx.jp_cols),
        "us_date": us_date,
        "jp_date": ctx.pairs[t][1],
        "fresh_us": bool(params["fresh_us"]) and not stale,
        "stale_us": stale,
        "n_days": ctx.T,
        "used_fallback_C_full": ctx.used_fallback_C_full,
    }
