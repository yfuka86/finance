#!/usr/bin/env python3
"""Reproduce the frozen quote-free ML v2 configuration on its selection window.

`docs/PREREGISTER_quotefree_ml_v2.md` records 選択窓 Sharpe 1.43 / DD −36.8% at
1.0bps/side, but the script that produced it was never committed. This rebuilds
the frozen spec from committed code and sweeps every construction axis the
pre-registration leaves ambiguous, so the baseline stops being an unverifiable
number.

Run `scripts/verify_baseline.py` first: it must print 4/4 PASS. That proves the
panel and the unit-lot machinery in this environment reproduce the documented
ensemble_core numbers, so any gap found here belongs to the v2 record, not to
the harness.

This touches the selection window (<=2024-12-31) only. It does not open the
confirmation window and is not the 2027-08-02 evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.jp_intraday.daily_model import (
    BASE_FEATURES, annualized_stats, load_panel_cached, portfolio_returns,
    walk_forward_predictions,
)
from trading.jp_intraday.extra_features import EXTRA_FEATURES, attach_extra_features
from trading.jp_intraday.flow_features import FLOW_FEATURES, attach_flow_features
from trading.jp_intraday.strategies import TODAY_OPEN_COLS, unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
RECORDED = {"sharpe": 1.43, "max_drawdown": -0.368, "cost_bps_side": 1.0}
PASS_BAR = 1.0
OUT = Path("data/jp_quotefree_v2_verify")


def stats(daily: pd.DataFrame) -> dict:
    if daily.empty or "net" not in daily:
        return {"sharpe": None}
    r = daily["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "days": int(len(r))}


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    panel = attach_extra_features(attach_flow_features(panel))
    panel = panel[panel["date"].le(SELECTION_END)].copy()
    # v2 の欠損規約: 中核が欠けた行は除外、疎な追加特徴量だけ0埋め。
    sparse = [c for c in EXTRA_FEATURES if c in panel.columns]
    panel[sparse] = panel[sparse].fillna(0.0)

    feats = [f for f in [f for f in BASE_FEATURES if f not in TODAY_OPEN_COLS]
             + FLOW_FEATURES + ["prev_value"] + EXTRA_FEATURES if f in panel.columns]
    summary = {"recorded": RECORDED, "pass_bar_net_sharpe": PASS_BAR,
               "features": feats, "n_features": len(feats),
               "selection_window_end": str(SELECTION_END.date()), "unit_lot": {},
               "ideal_book": {}}

    for target in ("demeaned", "rank"):
        preds = walk_forward_predictions(panel, feats, alpha=30.0, target=target)
        if target == "demeaned":
            ic = preds.groupby("date").apply(
                lambda g: g["pred"].corr(g["intraday_ret"], method="spearman"))
            summary["ic_mean"] = round(float(ic.mean()), 4)
            summary["ic_t"] = round(float(ic.mean() / ic.std() * len(ic) ** .5), 2)
        cols = [c for c in ["date", "symbol", "shortable", "short_restricted",
                            "prev_value", "prev_close", "raw_open", "open", "ivol"]
                if c in panel.columns]
        frame = preds.merge(panel[cols], on=["date", "symbol"], how="left")
        frame = frame.assign(
            _s=frame["pred"] - frame.groupby("date")["pred"].transform("mean"))
        frame = frame[frame["_s"].notna()]
        for constr in ("magnitude", "dollar_neutral", "risk_parity"):
            # risk_parity は 1/ivol でサイズするので ivol 欠損行は単元がNaNになる。
            sub = frame.dropna(subset=["ivol"]) if constr == "risk_parity" else frame
            for names, lev in ((8, 2.0), (8, 1.0), (15, 2.0), (30, 2.0)):
                daily, _ = unit_lot_backtest(sub, capital_yen=2e7, names_per_side=names,
                                             margin_ratio=lev, cost_bps_side=1.0,
                                             construction=constr)
                summary["unit_lot"][f"{target}/{constr}/{names}per side/{lev}x"] = stats(daily)
        if target == "demeaned":
            for q in (0.017, 0.05, 0.10):
                summary["ideal_book"][f"q={q}"] = annualized_stats(
                    portfolio_returns(preds, quantile=q, gross_leverage=1.0,
                                      cost_bps_side=1.0))

    best = max((v["sharpe"] for v in summary["unit_lot"].values()
                if v.get("sharpe") is not None), default=None)
    summary["best_unit_lot_sharpe"] = best
    summary["reproduces_recorded_1_43"] = bool(best is not None and best >= 1.30)
    summary["clears_pass_bar_on_selection_window"] = bool(best is not None and best >= PASS_BAR)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
