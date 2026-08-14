"""Regression tests for the FX rollover (swap) day conventions."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from trading.fx.swap import rollover_days, swap_return, weekly_days_check


def test_wednesday_pays_three_days():
    """T+2 の受渡日が金→月へ飛ぶので、水曜持ち越しは3日分（GMO 2026-08-04 確認）。"""
    assert rollover_days(dt.date(2026, 8, 5)) == 3      # 水
    for d in (3, 4, 6, 7):                              # 月火木金
        assert rollover_days(dt.date(2026, 8, d)) == 1


def test_weekend_has_no_rollover():
    assert rollover_days(dt.date(2026, 8, 8)) == 0      # 土
    assert rollover_days(dt.date(2026, 8, 9)) == 0      # 日


def test_a_full_week_sums_to_seven_days():
    """1+1+3+1+1 = 7。ここが合わないと規約の解釈が間違っている。"""
    idx = pd.date_range("2026-08-03", "2026-08-09", freq="D")   # 月〜日
    assert weekly_days_check(idx).iloc[0] == 7


def test_holidays_suppress_rollover():
    hol = {dt.date(2026, 8, 5)}
    assert rollover_days(dt.date(2026, 8, 5), holidays=hol) == 0


def test_swap_sign_flips_with_side_and_haircut_always_hurts():
    idx = pd.date_range("2026-08-03", "2026-08-07", freq="D")
    hi = pd.Series(0.05, index=idx)      # base 5%
    lo = pd.Series(0.01, index=idx)      # quote 1%
    long_ = swap_return(hi, lo, idx, side=+1).sum()
    short = swap_return(hi, lo, idx, side=-1).sum()
    assert long_ > 0 > short
    assert long_ == pytest.approx(-short)
    # haircut は受け取り側も支払い側も削る（＝常に自分の不利）
    l2 = swap_return(hi, lo, idx, side=+1, haircut_bp=50).sum()
    s2 = swap_return(hi, lo, idx, side=-1, haircut_bp=50).sum()
    assert l2 < long_ and s2 < short


def test_passthrough_scales_only_the_differential_not_the_haircut():
    idx = pd.date_range("2026-08-03", "2026-08-07", freq="D")
    hi, lo = pd.Series(0.05, index=idx), pd.Series(0.01, index=idx)
    full = swap_return(hi, lo, idx, side=+1, passthrough=1.0).sum()
    half = swap_return(hi, lo, idx, side=+1, passthrough=0.5).sum()
    assert half == pytest.approx(full / 2)


def test_annual_swap_equals_the_average_differential():
    """日数配分が正しければ、年間のスワップ合計は平均金利差にほぼ一致する.

    水曜3日・土日0日という配分は、週合計を7日にするための組み替えでしかない。
    合計が金利差からずれるなら曜日ロジックか日数の数え方が壊れている。
    """
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    base = pd.Series(0.0486, index=idx)
    quote = pd.Series(0.0, index=idx)
    total = swap_return(base, quote, idx, side=+1).sum()
    assert total == pytest.approx(0.0486 * len(idx) / 365, rel=0.01)


def test_short_rates_cover_every_major_to_the_recent_past():
    """政策金利系列は CHF 2024-03 / NZD 2024-12 で止まる。インターバンクを使う理由。"""
    from trading.fx.swap import load_short_rates
    try:
        r = load_short_rates()
    except FileNotFoundError:
        pytest.skip("rates not collected in this environment")
    assert set(r.columns) == {"USD", "JPY", "EUR", "GBP", "CHF", "AUD", "CAD", "NZD"}
    for c in ("CHF", "NZD"):
        assert r[c].dropna().index.max() >= pd.Timestamp("2025-06-01")
