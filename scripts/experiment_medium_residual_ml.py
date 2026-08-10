#!/usr/bin/env python3
"""One-shot medium-horizon, sector-neutral residual ML L/S experiment.

The frozen specification is documented in
docs/PREREGISTER_medium_residual_ml_ls.md.  This script deliberately exposes no
hyper-parameter CLI: seeing the result and changing a knob would consume the
evaluation window a second time.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import annualized_stats, load_panel_cached
from trading.jp_intraday.extra_features import EXTRA_FEATURES, attach_extra_features
from trading.jp_intraday.flow_features import FLOW_FEATURES, attach_flow_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jp_medium_residual_results"
HOLD_DAYS = 10
REBALANCE_DAYS = 5
ALPHA = 30.0
QUANTILE = 0.20
COST_BPS = 1.0
BORROW_RATE = 0.042
CAPITAL_YEN = 20_000_000.0
EVAL_START = pd.Timestamp("2024-01-01")

PRICE_FEATURES = [
    "px_ret1", "px_mom5", "px_mom20", "px_mom60", "px_ivol20",
    "beta", "amihud20", "log_mktcap",
]
FUND_FEATURES = [
    "fn_earn_yield", "fn_book_to_price", "fn_op_margin", "fn_roe",
    "fn_equity_ratio", "fn_cfo_to_np", "fn_sales_revision", "fn_op_revision",
]
ALL_FEATURES = PRICE_FEATURES + FUND_FEATURES + FLOW_FEATURES + EXTRA_FEATURES


def _fins_records() -> pd.DataFrame:
    # 2026-08-11 fix: read ALL fins caches (incl. the 2018-2021 backfill) and key
    # by 4-char code. The original part-file loader also produced 4-char symbols
    # while the panel uses 5-char ones, so the fundamental merge silently matched
    # NOTHING (fn_* were 100% NaN through the recorded judgment). See AGENTS.
    from scripts.run_value_event_v1 import load_fins
    f = load_fins().copy()
    f["symbol"] = f["Code"].astype(str).str[:4]
    return f


def build_fundamentals(sessions: pd.Index) -> pd.DataFrame:
    """Disclosure-PIT fundamental ratios, available from the next session."""
    f = _fins_records()
    f["disc_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f[f["disc_date"].notna()].copy()
    numeric = ["FEPS", "BPS", "OP", "Sales", "NP", "Eq", "TA", "CFO", "FSales", "FOP"]
    for col in numeric:
        f[col] = pd.to_numeric(f.get(col), errors="coerce")
    f = f.sort_values(["symbol", "disc_date", "DiscTime"]).drop_duplicates(
        ["symbol", "disc_date"], keep="last")
    # Quarterly rows omit BPS/FEPS and parts of the balance sheet. Carry the
    # last DISCLOSED state forward per issuer (never backward) before ratios,
    # exactly as value_event_model does -- otherwise merge_asof lands on a
    # quarterly row and blanks the ratio (coverage 38.7% -> ~97%).
    f[numeric] = f.groupby("symbol", sort=False)[numeric].ffill()

    # Revisions are only comparable inside the same forecast fiscal year.
    fy = f.get("CurFYEn", pd.Series("", index=f.index)).astype(str)
    for src, dst in (("FSales", "fn_sales_revision"), ("FOP", "fn_op_revision")):
        prev = f.groupby(["symbol", fy], sort=False)[src].shift(1)
        f[dst] = ((f[src] - prev) / prev.abs().replace(0, np.nan)).clip(-2, 2)

    f["fn_op_margin"] = (f["OP"] / f["Sales"].replace(0, np.nan)).clip(-1, 1)
    f["fn_roe"] = (f["NP"] / f["Eq"].replace(0, np.nan)).clip(-2, 2)
    f["fn_equity_ratio"] = (f["Eq"] / f["TA"].replace(0, np.nan)).clip(-1, 1)
    f["fn_cfo_to_np"] = (f["CFO"] / f["NP"].abs().replace(0, np.nan)).clip(-10, 10)

    # Per-share valuation needs the decision-date price, so carry raw values.
    f["_feps"] = f["FEPS"]
    f["_bps"] = f["BPS"]

    sessions = pd.Index(sorted(pd.to_datetime(sessions).unique()))
    pos = sessions.searchsorted(f["disc_date"], side="right")  # strictly next session
    ok = pos < len(sessions)
    f = f.loc[ok].copy()
    f["known_date"] = sessions[pos[ok]]
    cols = ["symbol", "known_date", "_feps", "_bps"] + FUND_FEATURES[2:]
    return f[cols].sort_values(["known_date", "symbol"])


def _price_and_target(daily: pd.DataFrame, symbols: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = daily[daily["Code"].astype(str).isin(symbols)].copy()
    d = d.rename(columns={"Date": "date", "Code": "symbol", "AdjO": "open_full",
                          "AdjC": "close_full", "Va": "value_full"})
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["symbol"].astype(str)
    d = d.sort_values(["symbol", "date"])
    g = d.groupby("symbol", sort=False)
    d["px_ret1"] = g["close_full"].pct_change(fill_method=None)
    d["intraday_full"] = d["close_full"] / d["open_full"] - 1
    for h in (5, 20, 60):
        d[f"px_mom{h}"] = g["close_full"].pct_change(h, fill_method=None)
    d["px_ivol20"] = (d["px_ret1"].groupby(d["symbol"])
                       .rolling(20, min_periods=10).std().reset_index(level=0, drop=True))

    sessions = pd.Index(sorted(d["date"].unique()))
    session_no = pd.Series(np.arange(len(sessions)), index=sessions)
    cur = d["date"].map(session_no)
    entry_open = g["open_full"].shift(-1)
    exit_close = g["close_full"].shift(-HOLD_DAYS)
    exit_date = g["date"].shift(-HOLD_DAYS)
    valid = exit_date.map(session_no).eq(cur + HOLD_DAYS)
    d["target_raw"] = (exit_close / entry_open - 1).where(valid)
    d["target_end_date"] = exit_date.where(valid)
    features = d[["date", "symbol", "px_ret1", "px_mom5", "px_mom20", "px_mom60",
                  "px_ivol20", "target_raw", "target_end_date"]]
    returns = d[["date", "symbol", "open_full", "px_ret1", "intraday_full"]]
    return features, returns


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_panel_cached(min_value_yen=1e9).copy()
    # UKI article's execution-first universe: TOPIX Core30 + Large70 + Mid400.
    panel = panel[panel["scale_ord"] >= 3].copy()
    panel["symbol"] = panel["symbol"].astype(str)
    panel["log_mktcap"] = np.log(panel["mktcap_yen"].where(panel["mktcap_yen"] > 0))
    sessions = pd.Index(sorted(panel["date"].unique()))

    daily = load_existing_daily()
    px, asset_returns = _price_and_target(daily, set(panel["symbol"].unique()))
    p = panel.merge(px, on=["date", "symbol"], how="left")
    del daily, px

    fund = build_fundamentals(sessions)
    # merge_asof requires the ``on`` key to be globally monotonic even with ``by``.
    # Fins codes are 4-char; the panel symbol is 5-char -> merge on sym4.
    p["sym4"] = p["symbol"].str[:4]
    fund = fund.rename(columns={"symbol": "sym4"})
    p = p.sort_values(["date", "sym4"])
    p = pd.merge_asof(p, fund, left_on="date", right_on="known_date", by="sym4",
                      direction="backward")
    p = p.sort_values(["date", "symbol"])
    cov = p["_bps"].notna().mean()
    assert cov > .5, f"fundamental merge failed (coverage {cov:.1%})"
    p["fn_earn_yield"] = (p["_feps"] / p["close"]).clip(-1, 1)
    p["fn_book_to_price"] = (p["_bps"] / p["close"]).clip(-10, 10)
    p = p.drop(columns=["known_date", "_feps", "_bps"], errors="ignore")

    p = attach_flow_features(p, lag=2)
    p = attach_extra_features(p)
    p["target"] = p["target_raw"] - p.groupby(["date", "sector"])["target_raw"].transform("mean")
    return p.sort_values(["date", "symbol"]).reset_index(drop=True), asset_returns


def _rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in ALL_FEATURES:
        out[col] = frame.groupby("date")[col].rank(pct=True) - 0.5
    return out


def _ridge_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["year"] = p["date"].dt.year
    core = PRICE_FEATURES
    p = p[p[core].notna().all(axis=1)].copy()
    ranked = _rank_features(p)
    for col in ALL_FEATURES:
        p[col] = ranked[col]
        if col not in core:
            p[f"{col}__missing"] = p[col].isna().astype(float)
    model_cols = ALL_FEATURES + [f"{c}__missing" for c in ALL_FEATURES if c not in core]

    outputs = []
    for year in sorted(y for y in p["year"].unique() if y >= 2023):
        cutoff = pd.Timestamp(f"{year}-01-01")
        # Purge labels whose holding interval reaches into the test year.
        train = p[(p["year"] < year) & (p["target_end_date"] < cutoff) & p["target"].notna()].copy()
        test = p[p["year"].eq(year)].copy()
        if len(train) < 20_000 or test.empty:
            continue
        med = train[model_cols].median().fillna(0.0)
        xtr = train[model_cols].fillna(med)
        xte = test[model_cols].fillna(med)
        mean = xtr.mean()
        std = xtr.std().replace(0, 1).fillna(1)
        a = ((xtr - mean) / std).to_numpy(dtype=float)
        y = train["target"].to_numpy(dtype=float)
        beta = np.linalg.solve(a.T @ a + np.eye(a.shape[1]) * ALPHA, a.T @ y)
        test["pred"] = ((xte - mean) / std).to_numpy(dtype=float) @ beta
        outputs.append(test[["date", "symbol", "sector", "pred", "shortable",
                             "short_restricted", "intraday_ret", "ret"]])
    if not outputs:
        raise RuntimeError("walk-forward predictionを作れませんでした")
    return pd.concat(outputs, ignore_index=True)


def _sector_neutral_weights(day: pd.DataFrame) -> pd.Series:
    """Sector-balanced book; sector budgets scale with sqrt(number of names)."""
    w = pd.Series(0.0, index=day.index)
    selected = []
    for sector, g in day.groupby("sector"):
        nlong = max(1, int(np.ceil(len(g) * QUANTILE)))
        longs = g.nlargest(nlong, "pred").index
        eligible = g[g["shortable"] & ~g["short_restricted"]]
        if eligible.empty:
            continue
        nshort = max(1, int(np.ceil(len(eligible) * QUANTILE)))
        shorts = eligible.nsmallest(nshort, "pred").index
        shorts = shorts.difference(longs)
        if len(shorts):
            selected.append((sector, longs, shorts, np.sqrt(len(g))))
    if not selected:
        return w
    budget_sum = sum(x[3] for x in selected)
    for _, longs, shorts, size in selected:
        side = 0.5 * size / budget_sum
        w.loc[longs] += side / len(longs)
        w.loc[shorts] -= side / len(shorts)
    return w


def _unitize(weights: pd.Series, day: pd.DataFrame, entry: pd.Timestamp,
             returns: pd.DataFrame, eligibility: pd.DataFrame) -> pd.Series:
    """Convert fractional targets to affordable 100-share lots at the entry open."""
    symbols = day.loc[weights.index, "symbol"]
    idx = pd.MultiIndex.from_arrays([[entry] * len(symbols), symbols])
    prices = returns.reindex(idx)["open_full"].to_numpy(dtype=float)
    unit_yen = pd.Series(prices * 100.0, index=weights.index)

    elig = eligibility.reindex(idx)
    short_ok = (elig["shortable"].eq(True).to_numpy()
                & elig["short_restricted"].eq(False).to_numpy())
    valid = unit_yen.notna() & unit_yen.gt(0) & unit_yen.le(CAPITAL_YEN * 0.25)
    valid &= weights.ge(0) | pd.Series(short_ok, index=weights.index)

    out = pd.Series(0.0, index=weights.index)
    for sign in (1.0, -1.0):
        candidates = weights[(weights * sign > 0) & valid].copy()
        if candidates.empty:
            continue
        budget = CAPITAL_YEN * 0.25  # one tranche: ±25% capital; two tranches => ±50%
        target = candidates.abs() * CAPITAL_YEN
        lots = np.floor(target / unit_yen.loc[candidates.index]).astype(int)
        spent = float((lots * unit_yen.loc[candidates.index]).sum())
        # Whole-lot Hamilton-style fill: strongest desired names get residual budget.
        priority = candidates.abs().sort_values(ascending=False).index.tolist()
        while True:
            added = False
            for ix in priority:
                price = float(unit_yen.at[ix])
                if spent + price <= budget + 1e-6:
                    lots.at[ix] += 1
                    spent += price
                    added = True
            if not added:
                break
        out.loc[candidates.index] = sign * lots * unit_yen.loc[candidates.index] / CAPITAL_YEN
    return out[out.ne(0)]


def simulate(predictions: pd.DataFrame, asset_returns: pd.DataFrame,
             eligibility_frame: pd.DataFrame,
             sessions: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two staggered 10-day tranches, including entry/exit and borrow costs."""
    sessions = pd.Index(sorted(pd.to_datetime(sessions).unique()))
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    returns = asset_returns.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
    eligibility = (eligibility_frame[["date", "symbol", "shortable", "short_restricted"]]
                   .drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"]))
    pnl: dict[pd.Timestamp, float] = {}
    costs: dict[pd.Timestamp, float] = {}
    borrow: dict[pd.Timestamp, float] = {}
    blotter = []

    decision_dates = sorted(predictions["date"].unique())
    # Global cadence, fixed from the first available prediction date.
    decision_dates = [d for i, d in enumerate(decision_dates) if i % REBALANCE_DAYS == 0]
    for decision in decision_dates:
        i = int(pos.get(pd.Timestamp(decision), -1))
        if i < 0 or i + HOLD_DAYS >= len(sessions):
            continue
        day = predictions[predictions["date"].eq(decision)].copy()
        entry = sessions[i + 1]
        exit_ = sessions[i + HOLD_DAYS]
        ideal = _sector_neutral_weights(day) / 2.0  # two overlapping tranches
        weights = _unitize(ideal[ideal.ne(0)], day, entry, returns, eligibility)
        if weights.empty:
            continue
        costs[entry] = costs.get(entry, 0.0) + float(weights.abs().sum()) * COST_BPS / 10_000
        costs[exit_] = costs.get(exit_, 0.0) + float(weights.abs().sum()) * COST_BPS / 10_000
        for k in range(1, HOLD_DAYS + 1):
            date = sessions[i + k]
            idx = pd.MultiIndex.from_arrays([[date] * len(weights), day.loc[weights.index, "symbol"]])
            rr = returns.reindex(idx)
            asset_ret = rr["intraday_full"] if k == 1 else rr["px_ret1"]
            contribution = float(np.nansum(weights.to_numpy() * asset_ret.to_numpy()))
            pnl[date] = pnl.get(date, 0.0) + contribution
            short_expo = float((-weights.clip(upper=0)).sum())
            borrow[date] = borrow.get(date, 0.0) + short_expo * BORROW_RATE / 252
        for ix, weight in weights.items():
            blotter.append({"decision_date": decision, "entry_date": entry, "exit_date": exit_,
                            "symbol": day.at[ix, "symbol"], "sector": day.at[ix, "sector"],
                            "weight": float(weight), "pred": float(day.at[ix, "pred"])})

    idx = pd.Index(sorted(set(pnl) | set(costs) | set(borrow)), name="date")
    out = pd.DataFrame(index=idx)
    out["gross"] = pd.Series(pnl).reindex(idx).fillna(0.0)
    out["turnover_cost"] = pd.Series(costs).reindex(idx).fillna(0.0)
    out["borrow_cost"] = pd.Series(borrow).reindex(idx).fillna(0.0)
    out["net"] = out["gross"] - out["turnover_cost"] - out["borrow_cost"]
    return out.reset_index(), pd.DataFrame(blotter)


def main() -> None:
    panel, asset_returns = build_dataset()
    predictions = _ridge_predictions(panel)
    daily, blotter = simulate(predictions, asset_returns, panel,
                              pd.Index(sorted(panel["date"].unique())))
    evaluation = daily[daily["date"] >= EVAL_START].copy()

    summary = {
        "spec": {"hold_days": HOLD_DAYS, "rebalance_days": REBALANCE_DAYS,
                 "alpha": ALPHA, "quantile": QUANTILE, "cost_bps": COST_BPS,
                 "borrow_rate": BORROW_RATE, "capital_yen": CAPITAL_YEN,
                 "unit_shares": 100, "universe": "TOPIX500(scale_ord>=3)",
                 "features": ALL_FEATURES},
        "evaluation": annualized_stats(evaluation, "net"),
        "gross": annualized_stats(evaluation, "gross"),
        "yearly": {},
        "n_predictions": int(len(predictions)),
        "n_trades": int(len(blotter)),
    }
    if not blotter.empty:
        books = blotter.groupby("decision_date")["weight"]
        summary["execution"] = {
            "avg_positions_per_tranche": float(books.size().mean()),
            "avg_long_exposure": float(books.apply(lambda x: x.clip(lower=0).sum()).mean()),
            "avg_short_exposure": float(books.apply(lambda x: (-x.clip(upper=0)).sum()).mean()),
            "max_abs_net_exposure": float(books.sum().abs().max()),
        }
    for year, group in evaluation.groupby(evaluation["date"].dt.year):
        summary["yearly"][str(year)] = annualized_stats(group, "net")
    ev = summary["evaluation"]
    summary["decision"] = "GO" if (
        ev["sharpe"] >= 1.0 and ev["max_drawdown"] > -0.20
        and all(x["ann_return"] > 0 for x in summary["yearly"].values())
    ) else "NO-GO"

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "daily_returns.csv", index=False)
    blotter.to_parquet(OUT / "blotter.parquet", index=False)
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
