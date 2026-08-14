#!/usr/bin/env python3
"""Does ownership structure add anything to the quote-free daily ML?

This is a **selection-window design measurement**, not an evaluation. The
confirmation window (2025-01+) was consumed by quote-free ML v1 and must not be
opened here. We only ask: on <=2024-12-31, does adding the ownership feature
family move the frozen v2 configuration?

Frequency note: unlike the value-event family (annual filings, ~27 trades in
2.6 years), this lane rebalances **every session** — 8 names per side, full
turnover at the open and the close. The ownership data is annual, but it enters
as a slow-moving cross-sectional feature, not as a trade trigger.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.jp_intraday.daily_model import (
    BASE_FEATURES, load_panel_cached, walk_forward_predictions,
)
from trading.jp_intraday.extra_features import EXTRA_FEATURES, attach_extra_features
from trading.jp_intraday.flow_features import FLOW_FEATURES, attach_flow_features
from trading.jp_intraday.ownership_features import (
    OWNERSHIP_FEATURES_DAILY, attach_ownership_features,
)
from trading.jp_intraday.strategies import TODAY_OPEN_COLS, unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
ALPHA, COSTS = 30.0, (1.0, 2.0)
OUT = Path("data/jp_ownership_ml_increment")


def stats(daily: pd.DataFrame) -> dict:
    if daily.empty or "net" not in daily:
        return {"sharpe": None}
    r = daily["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": float(r.mean() / r.std() * (252 ** .5)),
            "ann_return": float(r.mean() * 252),
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "days": int(len(r))}


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    panel = attach_flow_features(panel)
    panel = attach_extra_features(panel)
    panel = attach_ownership_features(panel)
    panel = panel[panel["date"].le(SELECTION_END)].copy()
    # v2 の欠損規約: 中核(BASE/FLOW/prev_value)が欠けた行は除外、疎な追加特徴量だけ0埋め。
    # これを守らないと流動性境界の断続銘柄が混入して成績が壊れる（AGENTS の実踏事例）。
    # xt_* は z か「該当なし=0」の量なので0が平均相当。ownership も断面z済み。
    sparse = [c for c in EXTRA_FEATURES + OWNERSHIP_FEATURES_DAILY if c in panel.columns]
    panel[sparse] = panel[sparse].fillna(0.0)

    # Frozen v2 feature set: quote-free only (no today's open), flows, extras.
    quotefree_base = [f for f in BASE_FEATURES if f not in TODAY_OPEN_COLS]
    v2 = [f for f in quotefree_base + FLOW_FEATURES + ["prev_value"] + EXTRA_FEATURES
          if f in panel.columns]
    configs = {"A_v2_frozen": v2,
               "B_v2_plus_ownership": v2 + [f for f in OWNERSHIP_FEATURES_DAILY
                                            if f in panel.columns]}

    summary = {"selection_window_end": str(SELECTION_END.date()), "alpha": ALPHA,
               "rows": len(panel), "results": {}}
    for name, feats in configs.items():
        preds = walk_forward_predictions(panel, feats, alpha=ALPHA)
        cols = [c for c in ["date", "symbol", "shortable", "short_restricted",
                            "prev_value", "prev_close", "raw_open", "open"]
                if c in panel.columns]
        frame = preds.merge(panel[cols], on=["date", "symbol"], how="left")
        # v2 仕様: スコアは日次でクロスセクション・デミーンする。magnitude 構築は
        # |score| 比例ウェイトなので、デミーンしないと恒常的な片張りになる。
        frame = frame.assign(_s=frame["pred"] - frame.groupby("date")["pred"].transform("mean"))
        entry = {"n_features": len(feats), "pred_rows": len(frame)}
        if not frame.empty:
            ic = frame.groupby("date").apply(
                lambda g: g["pred"].corr(g["intraday_ret"], method="spearman"))
            entry["ic_mean"] = float(ic.mean())
            entry["ic_t"] = float(ic.mean() / ic.std() * (len(ic) ** .5))
        for cost in COSTS:
            daily, _ = unit_lot_backtest(frame, capital_yen=2e7, names_per_side=8,
                                         margin_ratio=2.0, cost_bps_side=cost,
                                         construction="magnitude")
            entry[f"cost_{cost}bps"] = stats(daily)
        summary["results"][name] = entry
        print(name, json.dumps(entry, ensure_ascii=False, default=str), flush=True)

    a = summary["results"]["A_v2_frozen"].get("cost_1.0bps", {}).get("sharpe")
    b = summary["results"]["B_v2_plus_ownership"].get("cost_1.0bps", {}).get("sharpe")
    summary["delta_sharpe_1bps"] = (b - a) if (a is not None and b is not None) else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
