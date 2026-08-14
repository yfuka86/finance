#!/usr/bin/env python3
"""Dynamic close-to-close L/S with daily score updates and hysteresis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_medium_residual_ml import ALL_FEATURES, CAPITAL_YEN, build_dataset
from scripts.experiment_topix500_hierarchical_lasso import (
    hierarchical_features, walk_forward_predictions,
)
from trading.jp_intraday.daily_model import annualized_stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jp_dynamic_cc_results"
ENTRY_Q = 0.10
KEEP_Q = 0.20
MAX_NAMES_SIDE = 40
SIDE_BUDGET_YEN = 20_000_000.0
COST_BPS_CLOSE = 0.5
SHORT_FINANCE_RATE = 0.042
EVAL_START = pd.Timestamp("2024-01-01")


def attach_delayed_cc_target(panel: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Feature at D -> rebalance close D+1 -> earn close D+1 to close D+2."""
    p = panel.drop(columns=["target", "target_raw", "target_end_date"], errors="ignore")
    sessions = pd.Index(sorted(p["date"].unique()))
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    r = asset_returns.copy()
    cur = r["date"].map(pos)
    ok = cur.ge(2)
    r = r[ok].copy()
    cur = cur[ok].astype(int)
    r["decision_date"] = sessions[cur - 2]
    r["entry_date"] = sessions[cur - 1]
    r = r.rename(columns={"date": "target_date", "px_ret1": "target_raw"})
    p = p.merge(r[["decision_date", "entry_date", "target_date", "symbol", "target_raw"]],
                left_on=["date", "symbol"], right_on=["decision_date", "symbol"], how="left")
    p["target"] = p["target_raw"] - p.groupby(["target_date", "sector"])["target_raw"].transform("mean")
    return p.drop(columns=["decision_date"])


def _whole_lot_book(selected: pd.DataFrame, sign: float) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    mag = selected["pred"].abs().clip(lower=1e-12)
    target = mag / mag.sum() * SIDE_BUDGET_YEN
    unit = selected["close_full"] * 100.0
    valid = unit.gt(0) & unit.le(SIDE_BUDGET_YEN)
    selected, target, unit = selected[valid], target[valid], unit[valid]
    lots = np.floor(target / unit).astype(int)
    spent = float((lots * unit).sum())
    for ix in selected.sort_values("pred", ascending=sign < 0).index:
        price = float(unit.at[ix])
        if spent + price <= SIDE_BUDGET_YEN + 1e-6:
            lots.at[ix] += 1
            spent += price
    keep = lots.gt(0)
    return sign * lots[keep] * unit[keep] / CAPITAL_YEN


def _select_with_buffer(day: pd.DataFrame, previous: pd.Series) -> tuple[pd.Index, pd.Index]:
    day = day.copy()
    day["long_rank"] = day["pred"].rank(pct=True, ascending=False)
    short_pool = day[day["short_ok"]].copy()
    short_pool["short_rank"] = short_pool["pred"].rank(pct=True, ascending=True)

    prev_long = set(previous[previous > 0].index)
    prev_short = set(previous[previous < 0].index)
    retain_long = day[day["symbol"].isin(prev_long) & day["long_rank"].le(KEEP_Q)]
    retain_short = short_pool[short_pool["symbol"].isin(prev_short)
                              & short_pool["short_rank"].le(KEEP_Q)]
    enter_long = day[day["long_rank"].le(ENTRY_Q)]
    enter_short = short_pool[short_pool["short_rank"].le(ENTRY_Q)]
    longs = (pd.concat([retain_long, enter_long]).drop_duplicates("symbol")
             .nsmallest(MAX_NAMES_SIDE, "long_rank").index)
    shorts = (pd.concat([retain_short, enter_short]).drop_duplicates("symbol")
              .nsmallest(MAX_NAMES_SIDE, "short_rank").index)
    return longs, shorts


def simulate(predictions: pd.DataFrame, panel: pd.DataFrame,
             asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = asset_returns.set_index(["date", "symbol"])
    eligibility = (panel[["date", "symbol", "shortable", "short_restricted"]]
                   .drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"]))
    previous = pd.Series(dtype=float)  # indexed by symbol, weights after the prior rebalance
    daily, blotter = [], []
    for target_date, raw_day in predictions.groupby("target_date"):
        if pd.isna(target_date):
            continue
        decision = raw_day["date"].iloc[0]
        sessions = pd.Index(sorted(panel["date"].unique()))
        tpos = sessions.get_indexer([target_date])[0]
        if tpos <= 0:
            continue
        entry = sessions[tpos - 1]
        day = raw_day.reset_index(drop=True)
        entry_idx = pd.MultiIndex.from_arrays([[entry] * len(day), day["symbol"]])
        target_idx = pd.MultiIndex.from_arrays([[target_date] * len(day), day["symbol"]])
        em = market.reindex(entry_idx).reset_index(drop=True)
        tm = market.reindex(target_idx).reset_index(drop=True)
        el = eligibility.reindex(entry_idx).reset_index(drop=True)
        day["close_full"] = em["open_full"] * (1 + em["intraday_full"])
        day["cc_return"] = tm["px_ret1"]
        day["short_ok"] = el["shortable"].eq(True) & el["short_restricted"].eq(False)
        day = day.dropna(subset=["close_full", "cc_return", "pred"])

        if day["pred"].std() < 1e-12:
            desired = pd.Series(dtype=float)
        else:
            # Previous is symbol-indexed; selection routine only uses its signs.
            longs, shorts = _select_with_buffer(day, previous)
            lw = _whole_lot_book(day.loc[longs], 1.0)
            sw = _whole_lot_book(day.loc[shorts], -1.0)
            desired_idx = pd.concat([lw, sw]).groupby(level=0).sum()
            desired = pd.Series(desired_idx.to_numpy(), index=day.loc[desired_idx.index, "symbol"])

        union = previous.index.union(desired.index)
        turnover = float((desired.reindex(union, fill_value=0.0)
                          - previous.reindex(union, fill_value=0.0)).abs().sum())
        returns = day.set_index("symbol")["cc_return"]
        held = desired.index.intersection(returns.index)
        gross = float((desired.loc[held] * returns.loc[held]).sum())
        cost = turnover * COST_BPS_CLOSE / 10_000
        finance = float((-desired.clip(upper=0)).sum()) * SHORT_FINANCE_RATE / 252
        net = gross - cost - finance
        daily.append({"date": target_date, "decision_date": decision, "entry_date": entry,
                      "gross": gross, "cost": cost, "finance": finance, "net": net,
                      "turnover": turnover, "gross_exposure": float(desired.abs().sum()),
                      "net_exposure": float(desired.sum()), "positions": int(len(desired))})
        for symbol, weight in desired.items():
            blotter.append({"date": target_date, "symbol": symbol, "weight": float(weight),
                            "pnl": float(weight * returns.get(symbol, np.nan))})
        previous = desired
    return pd.DataFrame(daily), pd.DataFrame(blotter)


def main() -> None:
    panel, asset_returns = build_dataset()
    panel = attach_delayed_cc_target(panel, asset_returns)
    panel, cols = hierarchical_features(panel)
    predictions, choices = walk_forward_predictions(panel, cols)
    daily, blotter = simulate(predictions, panel, asset_returns)
    evd = daily[daily["date"] >= EVAL_START].copy()
    abs_pnl = blotter.groupby("symbol")["pnl"].sum().abs().sort_values(ascending=False)
    concentration = float(abs_pnl.head(10).sum() / abs_pnl.sum()) if abs_pnl.sum() else 1.0
    summary = {
        "spec": {"entry_q": ENTRY_Q, "keep_q": KEEP_Q, "max_names_side": MAX_NAMES_SIDE,
                 "capital_yen": CAPITAL_YEN, "gross_leverage": 2.0,
                 "cost_bps_close": COST_BPS_CLOSE, "short_finance_rate": SHORT_FINANCE_RATE,
                 "features": ALL_FEATURES},
        "alpha_choices": choices,
        "evaluation": annualized_stats(evd, "net"),
        "gross": annualized_stats(evd, "gross"),
        "yearly": {str(y): annualized_stats(g, "net") for y, g in evd.groupby(evd["date"].dt.year)},
        "execution": {"avg_turnover": float(evd["turnover"].mean()),
                      "avg_positions": float(evd["positions"].mean()),
                      "avg_gross_exposure": float(evd["gross_exposure"].mean()),
                      "max_abs_net_exposure": float(evd["net_exposure"].abs().max()),
                      "annualized_cost_drag": float(evd["cost"].mean() * 252),
                      "annualized_finance_drag": float(evd["finance"].mean() * 252),
                      "top10_abs_pnl_concentration": concentration},
    }
    e = summary["evaluation"]
    summary["decision"] = "GO" if (e["sharpe"] >= 1.0 and e["max_drawdown"] > -0.20
        and all(v["ann_return"] > 0 for v in summary["yearly"].values())
        and summary["execution"]["avg_turnover"] < 0.20 and concentration < 0.30) else "NO-GO"
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "daily_returns.csv", index=False)
    blotter.to_parquet(OUT / "blotter.parquet", index=False)
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
