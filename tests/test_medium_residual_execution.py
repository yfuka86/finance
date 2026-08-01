import pandas as pd

from scripts.experiment_medium_residual_ml import CAPITAL_YEN, _unitize


def test_unitize_uses_100_share_lots_and_rejects_unshortable_names():
    entry = pd.Timestamp("2026-01-06")
    day = pd.DataFrame({"symbol": ["10000", "20000", "30000"]}, index=[0, 1, 2])
    ideal = pd.Series([0.25, -0.15, -0.10], index=day.index)
    index = pd.MultiIndex.from_tuples(
        [(entry, "10000"), (entry, "20000"), (entry, "30000")],
        names=["date", "symbol"],
    )
    returns = pd.DataFrame({"open_full": [1_000.0, 2_000.0, 500.0]}, index=index)
    eligibility = pd.DataFrame(
        {"shortable": [True, False, True], "short_restricted": [False, False, True]},
        index=index,
    )

    actual = _unitize(ideal, day, entry, returns, eligibility)

    assert list(actual.index) == [0]
    yen = actual.iloc[0] * CAPITAL_YEN
    assert round(yen) % 100_000 == 0  # price 1,000 yen * 100 shares
    assert actual.abs().sum() <= 0.25 + 1e-12
