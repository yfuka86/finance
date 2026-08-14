import pandas as pd

from scripts.experiment_dynamic_cc_lasso import attach_delayed_cc_target
from scripts.experiment_topix500_hierarchical_lasso import attach_next_intraday_target


def _fixtures():
    dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"])
    panel = pd.DataFrame({
        "date": dates.repeat(2),
        "symbol": ["A", "B"] * len(dates),
        "sector": ["x", "x"] * len(dates),
    })
    returns = panel[["date", "symbol"]].copy()
    returns["intraday_full"] = range(1, len(returns) + 1)
    returns["px_ret1"] = [x / 100 for x in range(1, len(returns) + 1)]
    return dates, panel, returns


def test_intraday_target_uses_next_exchange_session_not_calendar_day():
    dates, panel, returns = _fixtures()
    got = attach_next_intraday_target(panel, returns)
    row = got[(got.date == dates[1]) & (got.symbol == "A")].iloc[0]
    assert row.target_date == dates[2]
    assert row.target_raw == 5


def test_dynamic_cc_waits_one_full_close_before_earning_return():
    dates, panel, returns = _fixtures()
    got = attach_delayed_cc_target(panel, returns)
    row = got[(got.date == dates[0]) & (got.symbol == "A")].iloc[0]
    assert row.entry_date == dates[1]
    assert row.target_date == dates[2]
    assert row.target_raw == 0.05
