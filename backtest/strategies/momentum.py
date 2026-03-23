"""
MOM: Simple momentum baseline.
Signal = rolling mean of JP returns over window L.
"""
import numpy as np
import pandas as pd
from data.collectors.config import JP_TICKERS


def run_momentum(us_ret: pd.DataFrame, jp_ret: pd.DataFrame,
                 window: int = 60, q: float = 0.3):
    jp_cols = [t for t in JP_TICKERS if t in jp_ret.columns]
    jp_ret = jp_ret[jp_cols]
    N_J = len(jp_cols)

    us_sorted = sorted(us_ret.index)
    pairs = []
    for jd in jp_ret.index:
        cands = [d for d in us_sorted if d < jd]
        if cands:
            pairs.append((cands[-1], jd))

    jp_aligned = jp_ret.loc[[p[1] for p in pairs]].values
    T = len(pairs)
    results = []

    for t in range(window, T - 1):
        signal = np.nanmean(jp_aligned[t - window:t], axis=0)
        n = max(1, int(np.ceil(N_J * q)))
        ranked = np.argsort(signal)[::-1]
        w = np.zeros(N_J)
        w[ranked[:n]] = 1.0 / n
        w[ranked[-n:]] = -1.0 / n
        ret = np.dot(w, np.nan_to_num(jp_aligned[t + 1], nan=0.0))
        results.append({"date": pairs[t + 1][1], "strategy_return": ret})

    df = pd.DataFrame(results).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df
