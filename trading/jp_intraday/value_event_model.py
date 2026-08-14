"""PIT-safe value-unlock event model primitives.

V1 is deliberately narrow: a material dividend-forecast increase at a profitable,
low-PBR company.  Buyback reports belong to a different data/model family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["div_raise", "book_to_price", "earnings_yield", "op_margin",
            "equity_ratio", "cash_assets", "roe"]
RECOVERY_FEATURES = ["dividend_yield", "book_to_price", "earnings_yield", "op_margin",
                     "equity_ratio", "cash_assets", "roe"]
CANCEL_FEATURES = ["cancel_fraction", "prior_treasury_ratio", "book_to_price",
                   "earnings_yield", "op_margin", "equity_ratio", "cash_assets", "roe"]
# V3: policy-shareholding unwind. See docs/PREREGISTER_VALUE_EVENT_V3_OWNERSHIP.md.
OWNERSHIP_FEATURES = ["delta_fixed", "fixed_ratio", "custodian_ratio", "delta_top1",
                      "book_to_price", "earnings_yield", "op_margin", "equity_ratio",
                      "cash_assets", "roe"]


def dividend_raise_events(financials: pd.DataFrame) -> pd.DataFrame:
    """Detect V1 events using only disclosures known at the event timestamp."""
    f = financials.copy()
    f["event_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["symbol"] = f["Code"].astype(str).str[:4]
    state_cols = ("BPS", "EPS", "OP", "Sales", "Eq", "TA", "CashEq", "NP")
    for c in ("FDivAnn", *state_cols):
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f = f.dropna(subset=["event_date", "symbol", "CurFYEn"])
    # Multiple documents can share a date. DiscTime/DiscNo provide a stable PIT order.
    order = [c for c in ["symbol", "CurFYEn", "event_date", "DiscTime", "DiscNo"] if c in f]
    f = f.sort_values(order).drop_duplicates(["symbol", "CurFYEn", "event_date"], keep="last")
    # Quarterly disclosure rows often omit balance-sheet/per-share fields. Carry
    # forward only values that this issuer had already disclosed; never bfill.
    f[list(state_cols)] = f.groupby("symbol", sort=False)[list(state_cols)].ffill()
    prior = f.groupby(["symbol", "CurFYEn"], sort=False)["FDivAnn"].shift(1)
    f["div_raise"] = f["FDivAnn"].div(prior).sub(1)
    return f.loc[prior.gt(0) & f["div_raise"].ge(.20 - 1e-12)].copy()


def dividend_resumption_events(financials: pd.DataFrame) -> pd.DataFrame:
    """Explicit forecast dividend 0 -> positive within the same fiscal year."""
    f = financials.copy()
    f["event_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["symbol"] = f["Code"].astype(str).str[:4]
    state_cols=("BPS","EPS","OP","Sales","Eq","TA","CashEq","NP")
    for c in ("FDivAnn",*state_cols): f[c]=pd.to_numeric(f.get(c),errors="coerce")
    f=f.dropna(subset=["event_date","symbol","CurFYEn"])
    order=[c for c in ["symbol","CurFYEn","event_date","DiscTime","DiscNo"] if c in f]
    f=f.sort_values(order).drop_duplicates(["symbol","CurFYEn","event_date"],keep="last")
    f[list(state_cols)]=f.groupby("symbol",sort=False)[list(state_cols)].ffill()
    prior=f.groupby(["symbol","CurFYEn"],sort=False)["FDivAnn"].shift(1)
    return f.loc[prior.eq(0)&f.FDivAnn.gt(0)].copy()


def treasury_cancellation_events(financials: pd.DataFrame) -> pd.DataFrame:
    """Detect disclosed share-count changes consistent with treasury cancellation."""
    f=financials.copy()
    f["event_date"]=pd.to_datetime(f["DiscDate"],errors="coerce")
    f["symbol"]=f["Code"].astype(str).str[:4]
    state_cols=("BPS","EPS","OP","Sales","Eq","TA","CashEq","NP")
    for c in (*state_cols,"ShOutFY","TrShFY"): f[c]=pd.to_numeric(f.get(c),errors="coerce")
    f=f.dropna(subset=["event_date","symbol"]).sort_values(
        [c for c in ["symbol","event_date","DiscTime","DiscNo"] if c in f])
    f=f.drop_duplicates(["symbol","event_date"],keep="last")
    f[list(state_cols)]=f.groupby("symbol",sort=False)[list(state_cols)].ffill()
    obs=f.dropna(subset=["ShOutFY","TrShFY"]).copy()
    g=obs.groupby("symbol",sort=False)
    pi=g.ShOutFY.shift(1); pt=g.TrShFY.shift(1); pf=pi-pt
    cur_float=obs.ShOutFY-obs.TrShFY
    obs["cancel_fraction"]=(pi-obs.ShOutFY)/pi
    obs["treasury_reduction_fraction"]=(pt-obs.TrShFY)/pi
    obs["float_change"]=(cur_float-pf)/pf
    obs["prior_treasury_ratio"]=pt/pi
    keep=(obs.cancel_fraction.ge(.005)&obs.treasury_reduction_fraction.ge(.005)
          &obs.float_change.abs().le(.0025)&pi.gt(0)&pf.gt(0))
    return obs.loc[keep].copy()


def attach_market_and_features(events: pd.DataFrame, daily: pd.DataFrame,
                               min_value_yen: float = 1e9) -> pd.DataFrame:
    """Attach the last pre-disclosure close/value and fixed 60-session outcome."""
    d = daily.rename(columns={"Date": "date", "Code": "code", "AdjO": "open",
                              "AdjC": "close", "raw_close": "valuation_close",
                              "Va": "value"}).copy()
    if "valuation_close" not in d:
        d["valuation_close"] = np.nan
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["code"].astype(str).str[:4]
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    rows = []
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    for _, e in events.iterrows():
        bars = by_symbol.get(e["symbol"])
        if bars is None:
            continue
        prior_idx = bars["date"].searchsorted(e["event_date"], side="left") - 1
        entry_idx = bars["date"].searchsorted(e["event_date"], side="right")
        exit_idx = entry_idx + 59
        if prior_idx < 0 or exit_idx >= len(bars):
            continue
        prior, entry, exit_ = bars.iloc[prior_idx], bars.iloc[entry_idx], bars.iloc[exit_idx]
        r = e.to_dict()
        r.update({"prior_close": prior.valuation_close, "prior_value": prior.value,
                  "entry_date": entry.date, "entry_price": entry.open,
                  "exit_date": exit_.date, "exit_price": exit_.close})
        rows.append(r)
    x = pd.DataFrame(rows)
    if x.empty:
        return x
    # BPS is stated on the share basis at disclosure. AdjC may be restated after
    # a later split and must never be mixed with it; missing raw close => reject.
    x["pbr"] = x["prior_close"] / x["BPS"]
    x["book_to_price"] = x["BPS"] / x["prior_close"]
    x["earnings_yield"] = x["EPS"] / x["prior_close"]
    x["op_margin"] = x["OP"] / x["Sales"].replace(0, np.nan)
    x["equity_ratio"] = x["Eq"] / x["TA"].replace(0, np.nan)
    x["cash_assets"] = x["CashEq"] / x["TA"].replace(0, np.nan)
    x["roe"] = x["NP"] / x["Eq"].replace(0, np.nan)
    if "FDivAnn" in x:
        x["dividend_yield"] = pd.to_numeric(x["FDivAnn"],errors="coerce") / x["prior_close"]
    x["forward_return"] = x["exit_price"] / x["entry_price"] - 1
    eligible = (x["pbr"].gt(0) & x["pbr"].le(1) & x["EPS"].gt(0)
                & x["OP"].gt(0) & x["prior_value"].ge(min_value_yen)
                & x["entry_price"].gt(0) & x["exit_price"].gt(0))
    return x.loc[eligible].reset_index(drop=True)


def fit_and_select_oos(cases: pd.DataFrame, cutoff="2024-01-01",
                       max_positions: int = 10, features=None,
                       cost: float = .002) -> tuple[object, pd.DataFrame]:
    """Fit once before cutoff, then apply the frozen positive-score/cap rule."""
    cut = pd.Timestamp(cutoff)
    train = cases[cases["exit_date"].lt(cut)].dropna(subset=["forward_return"])
    oos = cases[cases["event_date"].ge(cut)].copy().sort_values("entry_date")
    if len(train) < 10:
        raise ValueError(f"insufficient completed training cases: {len(train)}")
    features=FEATURES if features is None else list(features)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    model.fit(train[features], train["forward_return"])
    oos["prediction"] = model.predict(oos[features])
    active_exits: list[pd.Timestamp] = []
    selected = []
    for i, row in oos.iterrows():
        active_exits = [d for d in active_exits if d >= row.entry_date]
        take = row.prediction > 0 and len(active_exits) < max_positions
        selected.append(take)
        if take:
            active_exits.append(row.exit_date)
    oos["selected"] = selected
    oos["net_case_return"] = oos["forward_return"] - cost
    return model, oos


def case_report(oos: pd.DataFrame) -> dict:
    s = oos[oos["selected"]].copy()
    r = s["net_case_return"].dropna()
    if r.empty:
        return {"cases": 0}
    losses = r[r <= r.quantile(.05)]
    total = r.sum()
    contribution = float(r.max() / total) if total > 0 else np.nan
    return {"cases": int(len(r)), "mean": float(r.mean()), "median": float(r.median()),
            "win_rate": float(r.gt(0).mean()), "expected_shortfall_5pct": float(losses.mean()),
            "max_loss": float(r.min()), "top_case_profit_share": contribution,
            "sum": float(total)}
