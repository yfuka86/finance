#!/usr/bin/env python3
"""Frozen PIT futures-night interaction hypotheses for intraday JP L/S."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from scripts.experiment_medium_residual_ml import build_dataset
from scripts.experiment_topix500_hierarchical_lasso import (
    EVAL_START, attach_next_intraday_target, hierarchical_features, simulate,
    walk_forward_predictions,
)
from trading.jp_intraday.daily_model import annualized_stats
from trading.jp_intraday.futures_context import build_overnight_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jp_futures_night_interactions"


def attach_interactions(panel: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    files = sorted(glob.glob(str(ROOT / "data/jp_derivatives/futures_*.parquet")))
    if not files:
        raise FileNotFoundError("futures parquetがありません")
    futures = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    futures = futures.drop_duplicates(["Date", "Code"])
    night = build_overnight_features(futures).reset_index()
    date_col = "cash_day" if "cash_day" in night else night.columns[0]
    night = night.rename(columns={date_col: "target_date"})
    keep = ["target_date", "nk_night", "topix_night", "dow_night"]
    # load_panel_cached may already carry futures keyed to the feature date.
    # They are one session stale for this experiment; use target-day 06:00 values only.
    p = panel.drop(columns=keep[1:], errors="ignore").merge(
        night[keep], on="target_date", how="left")

    definitions = {
        "sensitivity": {
            "fx_beta_nk": p["hr_beta"] * p["nk_night"],
            "fx_beta_topix": p["hr_beta"] * p["topix_night"],
            "fx_beta_dow": p["hr_beta"] * p["dow_night"],
        },
        "stress": {
            "fx_ivol_absnk": p["hr_px_ivol20"] * p["nk_night"].abs(),
            "fx_amihud_absnk": p["hr_amihud20"] * p["nk_night"].abs(),
        },
        "state": {
            "fx_ret1_nk": p["hr_px_ret1"] * p["nk_night"],
            "fx_mom5_nk": p["hr_px_mom5"] * p["nk_night"],
        },
    }
    variants = {}
    for name, cols in definitions.items():
        for col, values in cols.items():
            p[col] = values
        variants[name] = base_cols + list(cols)
    return p, variants


def main() -> None:
    base_summary = json.loads((ROOT / "data/jp_hierarchical_lasso_results/summary.json").read_text())
    base_sharpe = float(base_summary["evaluation"]["sharpe"])
    panel, returns = build_dataset()
    panel = attach_next_intraday_target(panel, returns)
    panel, base_cols = hierarchical_features(panel)
    panel, variants = attach_interactions(panel, base_cols)
    result = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cols in variants.items():
        predictions, choices = walk_forward_predictions(panel, cols)
        daily, blotter = simulate(predictions, panel, returns)
        ev = daily[daily["date"] >= EVAL_START]
        stats = annualized_stats(ev, "net")
        gross = annualized_stats(ev, "gross")
        yearly = {str(y): annualized_stats(g, "net") for y, g in ev.groupby(ev.date.dt.year)}
        abs_pnl = blotter.groupby("symbol")["pnl"].sum().abs().sort_values(ascending=False)
        concentration = float(abs_pnl.head(10).sum() / abs_pnl.sum()) if abs_pnl.sum() else 1.0
        delta = stats["sharpe"] - base_sharpe
        decision = "GO" if (stats["sharpe"] >= 1 and delta >= .2
            and stats["max_drawdown"] > -.2 and concentration < .3
            and all(v["ann_return"] > 0 for v in yearly.values())) else "NO-GO"
        result[name] = {"evaluation": stats, "gross": gross, "delta_sharpe_vs_base": delta,
                        "yearly": yearly, "top10_abs_pnl_concentration": concentration,
                        "alpha_choices": choices, "decision": decision}
        daily.to_csv(OUT / f"daily_{name}.csv", index=False)
        blotter.to_parquet(OUT / f"blotter_{name}.parquet", index=False)
        print(name, json.dumps(result[name], ensure_ascii=False, indent=2))
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
