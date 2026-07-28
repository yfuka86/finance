"""Index-futures overnight context (先物) for JP intraday research.

JPX index futures trade a day session (M*) and a night session (E*, ~16:30 JST
to ~06:00 JST next morning) that spans the entire US session. So for a given
futures trade date D, the night-session return ``EC/MC - 1`` is the overnight
(US-driven) move, realised before the *next* cash open. That makes it a clean,
market-level overnight factor — cleaner than reconstructing it from cash gaps.

DJIAF gives the Dow's move (a US proxy) and NKVIF the Nikkei vol index (regime),
both without any non-JPX data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def front_month(futures: pd.DataFrame, product: str) -> pd.DataFrame:
    """Nearest non-expired contract per date (ties broken by open interest)."""
    f = futures[futures["ProdCat"].astype(str).eq(product)].copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["LTD"] = pd.to_datetime(f["LTD"])
    f = f[f["LTD"] >= f["Date"]]
    f = f.sort_values(["Date", "LTD", "OI"], ascending=[True, True, False])
    return f.groupby("Date", as_index=False).first()


def overnight_factor(futures: pd.DataFrame, product: str = "NK225F") -> pd.DataFrame:
    """Per *next* cash day: the futures night-session (overnight) and day returns.

    Session sequence per futures date D: open O -> day close C (== Settle) ->
    night close EC (~06:00 next morning, spanning the US session). So the
    overnight/US move is ``EC/C - 1`` and the day-session move is ``C/O - 1``.
    The night close is known by ~06:00 on the next trading day, so it is aligned
    to that next day as ``cash_day`` — usable at the cash open without lookahead.
    (This dataset does not populate the M* day-session columns; C/O/EC are used.)
    """
    f = front_month(futures, product).sort_values("Date")
    close = f["C"].replace(0, np.nan)
    open_ = f["O"].replace(0, np.nan)
    ec = f["EC"].replace(0, np.nan)
    out = pd.DataFrame({"fut_date": f["Date"].to_numpy()})
    out["night_ret"] = (ec.div(close).sub(1)).to_numpy()
    out["day_ret"] = (close.div(open_).sub(1)).to_numpy()
    out["settle"] = f["Settle"].to_numpy()
    out["oi"] = f["OI"].to_numpy()
    out["cash_day"] = out["fut_date"].shift(-1)  # night ends the next cash morning
    return out.dropna(subset=["cash_day"]).set_index("cash_day")


def build_overnight_features(futures: pd.DataFrame) -> pd.DataFrame:
    """Combined overnight panel keyed by cash day: Nikkei/TOPIX/Dow/VI."""
    nk = overnight_factor(futures, "NK225F")[["night_ret", "day_ret"]].rename(
        columns={"night_ret": "nk_night", "day_ret": "nk_day"})
    tp = overnight_factor(futures, "TOPIXF")[["night_ret"]].rename(
        columns={"night_ret": "topix_night"})
    dj = overnight_factor(futures, "DJIAF")[["night_ret", "day_ret"]].rename(
        columns={"night_ret": "dow_night", "day_ret": "dow_day"})
    vi = overnight_factor(futures, "NKVIF")[["settle"]].rename(columns={"settle": "nkvi"})
    return nk.join([tp, dj, vi], how="outer").sort_index()
