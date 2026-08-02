#!/usr/bin/env python3
"""Where does the quote-free ML signal die between the ideal book and ¥unit lots?

The signal is real on the selection window (IC 0.0323, t=10.15), yet the
fractional "ideal" book scores Sharpe ~3.4 while the tradable ¥20M unit-lot book
scores ~0.46. This peels the gap apart one constraint at a time so we know which
step destroys it — and therefore whether any of it is recoverable.

Ladder (each step adds one constraint to the one above):
  A ideal book, no constraints at all          (= daily_model.portfolio_returns)
  B ideal book + 貸借/規制 のショート適格性
  C unit lot, short constraints OFF, 資本を大きく(=単元粒度を無視できる)
  D unit lot, short constraints OFF, ¥20M      (単元粒度・1単元>予算・保証金拘束が効く)
  E unit lot, 本番制約, ¥20M                    (= the number we actually trade)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import (
    BASE_FEATURES, annualized_stats, load_panel_cached, portfolio_returns,
    walk_forward_predictions,
)
from trading.jp_intraday.extra_features import EXTRA_FEATURES, attach_extra_features
from trading.jp_intraday.flow_features import FLOW_FEATURES, attach_flow_features
from trading.jp_intraday.strategies import TODAY_OPEN_COLS, unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
NAMES, COST = 8, 1.0
OUT = Path("data/jp_ideal_vs_unitlot")


def stats(daily: pd.DataFrame) -> dict:
    r = daily["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4)}


def constrained_ideal(preds: pd.DataFrame, names: int, cost: float) -> dict:
    """Equal-weight top/bottom `names`, but shorts must actually be borrowable."""
    p = preds.copy()
    long_rank = p.groupby("date")["pred"].rank(ascending=False, method="first")
    eligible = p["shortable"].fillna(True) & ~p["short_restricted"].fillna(False)
    sp = p[eligible].copy()
    short_rank = sp.groupby("date")["pred"].rank(ascending=True, method="first")
    p["w"] = 0.0
    p.loc[long_rank.le(names), "w"] = .5 / names
    p.loc[sp.index[short_rank.le(names)], "w"] = -.5 / names
    gross = (p["w"] * p["intraday_ret"]).groupby(p["date"]).sum()
    expo = p["w"].abs().groupby(p["date"]).sum()
    net = gross.sub(expo * 2 * cost / 10_000)
    return stats(pd.DataFrame({"net": net.values}))


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    panel = attach_extra_features(attach_flow_features(panel))
    panel = panel[panel["date"].le(SELECTION_END)].copy()
    sparse = [c for c in EXTRA_FEATURES if c in panel.columns]
    panel[sparse] = panel[sparse].fillna(0.0)
    feats = [f for f in [f for f in BASE_FEATURES if f not in TODAY_OPEN_COLS]
             + FLOW_FEATURES + ["prev_value"] + EXTRA_FEATURES if f in panel.columns]
    preds = walk_forward_predictions(panel, feats, alpha=30.0)

    cols = [c for c in ["date", "symbol", "shortable", "short_restricted", "prev_value",
                        "prev_close", "raw_open", "open", "ivol"] if c in panel.columns]
    frame = preds.merge(panel[cols], on=["date", "symbol"], how="left")
    frame = frame.assign(_s=frame["pred"] - frame.groupby("date")["pred"].transform("mean"))
    per_day = frame.groupby("date").size().median()

    out = {"names_per_side": NAMES, "cost_bps_side": COST,
           "median_names_per_day": int(per_day), "ladder": {}}
    q = NAMES / per_day
    out["ladder"]["A_ideal_no_constraints"] = annualized_stats(
        portfolio_returns(preds, quantile=q, gross_leverage=1.0, cost_bps_side=COST))
    out["ladder"]["B_ideal_plus_borrow_filter"] = constrained_ideal(frame, NAMES, COST)

    off = dict(require_shortable=False, short_min_value_yen=0.0,
               apply_short_reg_cap=False, exclude_short_restricted=False)
    for cap, label in ((2e9, "C_unitlot_noshortconstraints_capital2000M"),
                       (2e7, "D_unitlot_noshortconstraints_capital20M")):
        daily, _ = unit_lot_backtest(frame, capital_yen=cap, names_per_side=NAMES,
                                     margin_ratio=2.0, cost_bps_side=COST,
                                     construction="magnitude", **off)
        out["ladder"][label] = stats(daily)
    daily, blot = unit_lot_backtest(frame, capital_yen=2e7, names_per_side=NAMES,
                                    margin_ratio=2.0, cost_bps_side=COST,
                                    construction="magnitude")
    out["ladder"]["E_unitlot_production_capital20M"] = stats(daily)

    # How often can a side not even be filled? 1 lot must fit the per-name budget.
    px = frame["raw_open"].fillna(frame["open"])
    budget = 2e7 * 2.0 / 2 / NAMES
    out["one_lot_exceeds_budget_share"] = round(float((px * 100 > budget).mean()), 4)
    if len(blot):
        filled = blot.groupby("date").size()
        out["median_positions_filled_of_16"] = int(filled.median())
        out["days_with_fewer_than_16"] = round(float(filled.lt(2 * NAMES).mean()), 4)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
