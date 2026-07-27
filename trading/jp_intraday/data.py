from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}

# TSE cash-equity trading calendar (JST): 09:00-11:30 am, 12:30-15:00 pm.
AM_OPEN_MINUTE = 9 * 60
AM_CLOSE_MINUTE = 11 * 60 + 29
NOON_MINUTE = 12 * 60
PM_OPEN_MINUTE = 12 * 60 + 30
PM_CLOSE_MINUTE = 15 * 60 + 29
SESSION_MINUTES = 150  # tradable minutes per half-day session


def minute_of_day(ts: pd.Series) -> pd.Series:
    """Minutes since midnight for a timestamp series."""
    return ts.dt.hour * 60 + ts.dt.minute


def am_pm_session(ts: pd.Series) -> np.ndarray:
    """Binary am/pm label split at noon (bars are assumed to be in-session)."""
    return np.where(minute_of_day(ts) < NOON_MINUTE, "am", "pm")


def load_bars(path: str | Path, timezone: str = "Asia/Tokyo") -> pd.DataFrame:
    """Load canonical 1-minute bars from CSV or parquet.

    Timestamps may be timezone-aware or naive (naive values are interpreted as JST).
    Duplicate symbol/timestamps and malformed OHLC bars are rejected rather than hidden.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone)
    else:
        ts = ts.dt.tz_convert(timezone)
    frame = frame.copy()
    frame["timestamp"] = ts
    frame["symbol"] = frame["symbol"].astype(str)
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate timestamp/symbol bars found")
    bad = (
        (frame[numeric[:4]] <= 0).any(axis=1)
        | (frame["volume"] < 0)
        | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if bad.any():
        raise ValueError(f"invalid OHLCV rows: {int(bad.sum())}")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _session(ts: pd.Series) -> pd.Series:
    """Strict 3-valued session label (out-of-session bars become NA)."""
    minutes = minute_of_day(ts)
    return pd.Series(
        pd.NA,
        index=ts.index,
        dtype="string",
    ).mask(minutes.between(AM_OPEN_MINUTE, AM_CLOSE_MINUTE), "am").mask(
        minutes.between(PM_OPEN_MINUTE, PM_CLOSE_MINUTE), "pm"
    )


def resample_bars(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample without ever joining the TSE lunch break or different trading days."""
    if minutes == 1:
        return frame.copy()
    if minutes != 5:
        raise ValueError("only 1-minute and 5-minute bars are supported")
    work = frame.copy()
    work["session"] = _session(work["timestamp"])
    work = work.dropna(subset=["session"])
    work["date"] = work["timestamp"].dt.date
    work = work.set_index("timestamp")
    pieces = []
    for (_, _, _), group in work.groupby(["symbol", "date", "session"], sort=False):
        out = group.resample("5min", origin="start_day", label="left", closed="left").agg(
            symbol=("symbol", "first"),
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
        )
        pieces.append(out.dropna(subset=["open", "close"]))
    if not pieces:
        return pd.DataFrame(columns=sorted(REQUIRED))
    return pd.concat(pieces).reset_index().sort_values(["timestamp", "symbol"]).reset_index(drop=True)
