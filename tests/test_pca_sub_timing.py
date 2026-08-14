import numpy as np
import pandas as pd

from backtest.strategies.pca_sub import DEFAULTS, run_pca_sub
from data.collectors.config import JP_TICKERS, US_TICKERS


def _returns(columns, dates, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (len(dates), len(columns))),
        index=dates,
        columns=columns,
    )


def test_default_uses_us_session_immediately_before_traded_jp_session():
    dates = pd.bdate_range("2020-01-01", periods=100)
    us = _returns(US_TICKERS, dates, seed=1)
    jp = _returns(JP_TICKERS, dates, seed=2)

    default, _ = run_pca_sub(us, jp, window=20)
    fresh, _ = run_pca_sub(us, jp, window=20, fresh_us=True)
    stale, _ = run_pca_sub(us, jp, window=20, fresh_us=False)

    assert DEFAULTS["fresh_us"] is True
    pd.testing.assert_frame_equal(default, fresh)
    assert not default.equals(stale)
