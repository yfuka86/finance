"""Daily overnight-gap reversal research on split-adjusted daily bars.

Hypothesis: the overnight gap (adj close_{t-1} -> open_t) is where the US session
and overnight index-futures moves get priced into each name. Idiosyncratic gaps
(a name's gap net of the market's) tend to *revert* over the trading day. This
module tests that at daily frequency over years of data, which is the honest way
to establish whether the effect is real before refining it intraday.

Leakage safety: the gap and the residual gap are fully known at open_t; the only
return we ever trade is open_t -> close_t of the SAME day, and the liquidity
filter uses the PRIOR day's traded value. No future information enters a signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_gap_panel(daily: pd.DataFrame, min_value_yen: float = 5e8,
                    max_abs_gap: float = 0.5, max_gap_days: int = 5) -> pd.DataFrame:
    """Per (date, symbol): overnight gap, tradable open->close return, residuals.

    ``max_gap_days`` nulls out "overnight" gaps that actually span a break in the
    data (weekends are fine; a month-long hole is not), so non-contiguous history
    does not manufacture a huge fake gap on the first day after the hole.
    """
    d = daily.rename(columns={
        "Date": "date", "Code": "symbol", "AdjO": "open", "AdjH": "high",
        "AdjL": "low", "AdjC": "close", "AdjVo": "volume", "Va": "value",
    }).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["symbol"].astype(str)
    d = d.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    g = d.groupby("symbol", sort=False)
    d["prev_close"] = g["close"].shift(1)
    d["prev_value"] = g["value"].shift(1)
    d["gap_days"] = d["date"].sub(g["date"].shift(1)).dt.days
    d["overnight_gap"] = d["open"].div(d["prev_close"]).sub(1)
    d["intraday_ret"] = d["close"].div(d["open"]).sub(1)
    # Point-in-time liquidity screen + drop errors, unadjusted events, and gaps
    # that jump across a hole in the data.
    d = d[(d["prev_value"] >= min_value_yen)
          & (d["overnight_gap"].abs() < max_abs_gap)
          & (d["gap_days"].le(max_gap_days))]
    d = d.dropna(subset=["overnight_gap", "intraday_ret"])
    d["residual_gap"] = d["overnight_gap"].sub(
        d.groupby("date")["overnight_gap"].transform("mean")
    )
    return d.reset_index(drop=True)


def load_existing_daily() -> pd.DataFrame:
    """Assemble a de-duplicated adjusted-daily panel from data already on disk.

    Sources (no re-download): screener day snapshots, the intraday reference
    window, and any collected yearly history. Adjusted columns are preferred;
    duplicate (date, symbol) rows across overlapping files are dropped.
    """
    import glob

    frames = []
    sources = (
        sorted(glob.glob("data/cache/bars_day_*.parquet"))
        + ["data/jp_intraday_reference/daily_20260528_20260724.parquet"]
        + sorted(glob.glob("data/jp_daily_history/daily_adj_*.parquet"))
    )
    for path in sources:
        try:
            df = pd.read_parquet(path)
        except (OSError, FileNotFoundError):
            continue
        # Normalise to AdjO/AdjH/AdjL/AdjC/AdjVo/Va; also keep RAW open/close where
        # present (needed for ¥ unit-lot sizing — adjusted prices are not tradable levels).
        if "AdjC" in df.columns:
            keep = ["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va", "O", "C"]
            df = df[[c for c in keep if c in df.columns]].rename(
                columns={"O": "raw_open", "C": "raw_close"})
        elif "AdjClose" in df.columns:  # legacy naming, just in case
            df = df.rename(columns={"AdjOpen": "AdjO", "AdjHigh": "AdjH",
                                    "AdjLow": "AdjL", "AdjClose": "AdjC",
                                    "AdjVolume": "AdjVo", "Value": "Va"})
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no existing daily parquet files found")
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    # Normalise codes to J-Quants 5-digit form: the screener cache uses 4-digit
    # tickers ("1301") while master/reference use 5-digit ("13010" = ticker + "0").
    # Without this the master merge (fund filter, sector) silently fails for the
    # cache window and a stock appears under two identities.
    panel["Code"] = panel["Code"].astype(str)
    four = panel["Code"].str.fullmatch(r"\d{4}")
    panel.loc[four, "Code"] = panel.loc[four, "Code"] + "0"
    return panel.drop_duplicates(["Date", "Code"]).reset_index(drop=True)


def backtest_gap(panel: pd.DataFrame, quantile: float = 0.2, direction: int = -1,
                 cost_bps_side: float = 5.0) -> pd.DataFrame:
    """Cross-sectional dollar-neutral open->close L/S. direction -1 = fade gap."""
    score = direction * panel["residual_gap"]
    rank = score.groupby(panel["date"]).rank(pct=True)
    long = rank.ge(1 - quantile)
    short = rank.le(quantile)
    nl = long.groupby(panel["date"]).transform("sum")
    ns = short.groupby(panel["date"]).transform("sum")
    both = nl.gt(0) & ns.gt(0)
    w = pd.Series(0.0, index=panel.index)
    w.loc[both & long] = 0.5 / nl.loc[both & long]
    w.loc[both & short] = -0.5 / ns.loc[both & short]
    gross = (w * panel["intraday_ret"]).groupby(panel["date"]).sum()
    exposure = w.abs().groupby(panel["date"]).sum()          # ~1.0 gross/day
    cost = exposure * 2 * cost_bps_side / 10_000             # round trip open->close
    out = pd.DataFrame({"gross": gross, "net": gross.sub(cost)})
    out.index.name = "date"
    return out.reset_index()


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def report(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-period and per-year gross/net Sharpe — the regime-stability check."""
    r = returns.copy()
    r["year"] = pd.to_datetime(r["date"]).dt.year
    rows = []
    for year, grp in r.groupby("year"):
        rows.append({"period": str(year), "days": len(grp),
                     "gross_sharpe": _sharpe(grp["gross"]), "net_sharpe": _sharpe(grp["net"]),
                     "gross_sum": float(grp["gross"].sum()), "net_sum": float(grp["net"].sum())})
    rows.append({"period": "ALL", "days": len(r),
                 "gross_sharpe": _sharpe(r["gross"]), "net_sharpe": _sharpe(r["net"]),
                 "gross_sum": float(r["gross"].sum()), "net_sum": float(r["net"].sum())})
    return pd.DataFrame(rows)
