"""FX rollover (swap point) model with the day-of-week value-date rules.

Why this matters: for anything held overnight, swap is a first-order P&L term in
FX, not a detail. A basket strategy that ignores it is measuring the wrong thing.

Rules verified against GMO クリック証券 FXネオ (2026-08-04):
  * Swap is credited at the **New York close** (JST 07:00, 06:00 during US DST).
  * A position held through **Wednesday's** NY close earns **3 days** of swap
    (credited Thursday 07:00). Spot value date is T+2, so rolling Wednesday to
    Thursday moves the value date from Friday to Monday — three calendar days.
  * No rollover on Saturday/Sunday (market closed), and none on days where a
    holiday means no rollover happens at all.

The weekly total is therefore 1+1+3+1+1 = 7 days, which is the check that the
convention is self-consistent.

The broker does not pass the raw policy-rate differential through: it keeps a
spread, so the side that receives gets less than theory and the side that pays
gives more. That is modelled explicitly as `passthrough` and `haircut_bp`, which
must be calibrated against the broker's published table — they are NOT free
parameters to fit returns with.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# 週合計が7日になるための配分。水曜だけ3日（T+2の受渡日が金→月へ飛ぶ）。
_WEEKDAY_DAYS = {0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0}


def rollover_days(day: dt.date | pd.Timestamp, holidays: set | None = None) -> int:
    """NYクローズを跨いで持ち越したときに付与されるスワップ日数."""
    d = pd.Timestamp(day).date()
    if holidays and d in holidays:
        return 0
    return _WEEKDAY_DAYS[pd.Timestamp(d).weekday()]


def rollover_days_series(index: pd.DatetimeIndex, holidays: set | None = None) -> pd.Series:
    s = pd.Series([rollover_days(d, holidays) for d in index], index=index, dtype=float)
    return s


def swap_return(rate_base: pd.Series, rate_quote: pd.Series, index: pd.DatetimeIndex,
                side: int = 1, passthrough: float = 1.0, haircut_bp: float = 0.0,
                holidays: set | None = None) -> pd.Series:
    """Daily swap as a **return on notional** (not yen), signed for `side`.

    side = +1 で base 通貨ロング（quote 通貨ショート）。
    `passthrough` は金利差のうちブローカーが渡す割合、`haircut_bp` は
    受払いどちらでも自分の不利に働く年率bp。どちらもブローカーの公表値に
    合わせて較正するものであって、成績を合わせるために動かしてはいけない。
    """
    days = rollover_days_series(index, holidays)
    diff = (rate_base.reindex(index).ffill() - rate_quote.reindex(index).ffill())
    theo = side * diff * passthrough
    return (theo - haircut_bp / 1e4) * days / 365.0


def weekly_days_check(index: pd.DatetimeIndex) -> pd.Series:
    """任意の暦週で日数合計が7になるか（規約の自己整合チェック）."""
    s = rollover_days_series(index)
    return s.groupby([index.isocalendar().year, index.isocalendar().week]).sum()
