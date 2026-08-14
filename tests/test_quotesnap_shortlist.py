"""Regression tests for the K=50 quote shortlist (2026-08-03)."""
from __future__ import annotations

import pandas as pd

from trading.jp_intraday.live.quotesnap import (
    DEFAULT_SHORTLIST, shortlist_symbols, snapshot_lead_seconds,
)


def _panel():
    old = pd.DataFrame({"date": pd.Timestamp("2026-08-02"),
                        "symbol": [f"{i:04d}0" for i in range(100)],
                        "gap_vol60": 9.9})
    new = pd.DataFrame({"date": pd.Timestamp("2026-08-03"),
                        "symbol": [f"{i:04d}0" for i in range(100)],
                        "gap_vol60": [i / 1000 for i in range(100)]})
    return pd.concat([old, new], ignore_index=True)


def test_shortlist_takes_the_highest_gap_vol_of_the_latest_session_only():
    picks = shortlist_symbols(_panel(), k=10)
    assert len(picks) == 10
    # 直近営業日のみ。前日の行(gap_vol60=9.9)を拾ってはいけない。
    assert picks[0] == "00990"
    assert set(picks) == {f"{i:04d}0" for i in range(90, 100)}


def test_shortlist_none_keeps_the_whole_universe():
    assert len(shortlist_symbols(_panel(), k=None)) == 100


def test_shortlist_ignores_rows_without_the_screen():
    p = _panel()
    p.loc[p["symbol"].eq("00990") & p["date"].eq(pd.Timestamp("2026-08-03")),
          "gap_vol60"] = float("nan")
    assert "00990" not in shortlist_symbols(p, k=10)


def test_lead_time_scales_with_symbol_count():
    """固定70秒のままだと50銘柄でも「70秒前の気配」を撮ることになり、
    絞り込んだ意味が消える（同時性を稼ぐのが目的なので）。"""
    lead50 = snapshot_lead_seconds(DEFAULT_SHORTLIST)
    lead467 = snapshot_lead_seconds(467)
    assert lead50 < 15          # ~5秒で撮れる規模
    assert lead467 > 50         # 従来の全銘柄はやはり1分近く要る
    assert lead50 < lead467
    assert snapshot_lead_seconds(1) >= 10   # 下限は確保する
