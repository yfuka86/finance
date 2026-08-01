#!/usr/bin/env python3
"""Frozen TOPIX500 hierarchical-residual LASSO intraday L/S experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from scripts.experiment_medium_residual_ml import (
    ALL_FEATURES, CAPITAL_YEN, build_dataset,
)
from trading.jp_intraday.daily_model import annualized_stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jp_hierarchical_lasso_results"
ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
MAX_NAMES_SIDE = 80
SIDE_BUDGET_YEN = 20_000_000.0
COST_BPS_ROUNDTRIP = 2.0
EVAL_START = pd.Timestamp("2024-01-01")


def attach_next_intraday_target(panel: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    panel = panel.drop(columns=["target", "target_raw", "target_end_date"], errors="ignore")
    sessions = pd.Index(sorted(panel["date"].unique()))
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    r = asset_returns.copy()
    current = r["date"].map(pos)
    valid = current.gt(0)
    r = r[valid].copy()
    r["decision_date"] = sessions[(current[valid] - 1).astype(int)]
    r = r.rename(columns={"date": "target_date", "intraday_full": "target_raw"})
    p = panel.merge(r[["decision_date", "target_date", "symbol", "target_raw"]],
                    left_on=["date", "symbol"], right_on=["decision_date", "symbol"], how="left")
    p["target"] = p["target_raw"] - p.groupby(["target_date", "sector"])["target_raw"].transform("mean")
    return p.drop(columns=["decision_date"])


def hierarchical_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rank -> sector demean -> residualise on beta and size, date by date."""
    ranked = pd.DataFrame(index=panel.index)
    for col in ALL_FEATURES:
        ranked[col] = panel.groupby("date")[col].rank(pct=True) - 0.5
        ranked[col] -= ranked[col].groupby([panel["date"], panel["sector"]]).transform("mean")

    controls = ranked[["beta", "log_mktcap"]].fillna(0.0)
    residual = pd.DataFrame(index=panel.index, columns=ALL_FEATURES, dtype=float)
    for _, idx in panel.groupby("date", sort=False).groups.items():
        loc = np.asarray(list(idx))
        x = np.column_stack([np.ones(len(loc)), controls.loc[loc].to_numpy(dtype=float)])
        y = ranked.loc[loc, ALL_FEATURES].to_numpy(dtype=float)
        missing = ~np.isfinite(y)
        y0 = np.nan_to_num(y, nan=0.0)
        coef = np.linalg.lstsq(x, y0, rcond=None)[0]
        z = y0 - x @ coef
        z[missing] = np.nan
        residual.loc[loc, ALL_FEATURES] = z

    # Controls remain economically meaningful; do not annihilate them by regressing on themselves.
    residual["beta"] = ranked["beta"]
    residual["log_mktcap"] = ranked["log_mktcap"]
    model_cols = []
    for col in ALL_FEATURES:
        panel[f"hr_{col}"] = residual[col]
        model_cols.append(f"hr_{col}")
        if col not in ("px_ret1", "px_mom5", "px_mom20", "px_mom60",
                       "px_ivol20", "beta", "amihud20", "log_mktcap"):
            miss = f"hr_{col}__missing"
            panel[miss] = residual[col].isna().astype(float)
            model_cols.append(miss)
    return panel, model_cols


def _fit_lasso(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], alpha: float):
    med = train[cols].median().fillna(0.0)
    xtr = train[cols].fillna(med)
    mean = xtr.mean()
    std = xtr.std().replace(0, 1).fillna(1)
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=20_000, tol=1e-6,
                  selection="cyclic")
    model.fit((xtr - mean) / std, train["target"])
    pred = model.predict((test[cols].fillna(med) - mean) / std)
    return pred, model


def walk_forward_predictions(panel: pd.DataFrame, cols: list[str]):
    p = panel.copy()
    p["year"] = p["date"].dt.year
    outputs, choices = [], {}
    for year in (2024, 2025, 2026):
        cutoff = pd.Timestamp(f"{year}-01-01")
        val_year = year - 1
        fit = p[(p["year"] < val_year) & (p["target_date"] < pd.Timestamp(f"{val_year}-01-01"))
                & p["target"].notna()].copy()
        val = p[(p["year"] == val_year) & (p["target_date"] < cutoff)
                & p["target"].notna()].copy()
        test = p[p["year"].eq(year)].copy()
        if len(fit) < 20_000 or len(val) < 10_000 or test.empty:
            continue
        mse = {}
        for alpha in ALPHAS:
            pred, _ = _fit_lasso(fit, val, cols, alpha)
            mse[alpha] = float(np.mean((val["target"].to_numpy() - pred) ** 2))
        alpha = min(mse, key=mse.get)
        train = p[(p["year"] < year) & (p["target_date"] < cutoff)
                  & p["target"].notna()].copy()
        pred, model = _fit_lasso(train, test, cols, alpha)
        test["pred"] = pred
        outputs.append(test[["date", "target_date", "symbol", "sector", "pred"]])
        choices[str(year)] = {"alpha": alpha, "validation_mse": mse[alpha],
                              "nonzero": int(np.count_nonzero(model.coef_)),
                              "fit_rows": int(len(train))}
    if not outputs:
        raise RuntimeError("LASSO predictionを作れませんでした")
    return pd.concat(outputs, ignore_index=True), choices


def _allocate(candidates: pd.DataFrame, sign: float) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float)
    candidates = candidates.head(MAX_NAMES_SIDE).copy()
    mag = candidates["pred"].abs().clip(lower=1e-12)
    target = mag / mag.sum() * SIDE_BUDGET_YEN
    unit = candidates["open_full"] * 100.0
    valid = unit.gt(0) & unit.le(SIDE_BUDGET_YEN)
    candidates, target, unit = candidates[valid], target[valid], unit[valid]
    lots = np.floor(target / unit).astype(int)
    spent = float((lots * unit).sum())
    order = candidates.index.tolist()
    while True:
        added = False
        for ix in order:
            price = float(unit.at[ix])
            if spent + price <= SIDE_BUDGET_YEN + 1e-6:
                lots.at[ix] += 1
                spent += price
                added = True
        if not added:
            break
    return sign * lots[lots.gt(0)] * unit.loc[lots.gt(0)] / CAPITAL_YEN


def simulate(predictions: pd.DataFrame, panel: pd.DataFrame,
             asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret = asset_returns.set_index(["date", "symbol"])
    eligibility = (panel[["date", "symbol", "shortable", "short_restricted"]]
                   .drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"]))
    daily, blotter = [], []
    for entry, day in predictions.groupby("target_date"):
        if pd.isna(entry):
            continue
        if not np.isfinite(day["pred"].std()) or day["pred"].std() < 1e-12:
            daily.append({"date": entry, "gross": 0.0, "cost": 0.0, "net": 0.0,
                          "gross_exposure": 0.0, "net_exposure": 0.0})
            continue
        syms = day["symbol"]
        idx = pd.MultiIndex.from_arrays([[entry] * len(syms), syms])
        market = ret.reindex(idx).reset_index(drop=True)
        decision = day["date"].iloc[0]
        decision_idx = pd.MultiIndex.from_arrays([[decision] * len(syms), syms])
        known_market = ret.reindex(decision_idx).reset_index(drop=True)
        elig = eligibility.reindex(idx).reset_index(drop=True)
        day = day.reset_index(drop=True)
        # Quantity must be fixed before the opening auction.  Use the last known
        # close, never the still-unknown entry open, for whole-lot budgeting.
        day["open_full"] = known_market["open_full"] * (1 + known_market["intraday_full"])
        day["intraday_full"] = market["intraday_full"]
        day["short_ok"] = elig["shortable"].eq(True) & elig["short_restricted"].eq(False)
        day = day.dropna(subset=["open_full", "intraday_full", "pred"])
        longs = day.sort_values("pred", ascending=False)
        shorts = day[day["short_ok"]].sort_values("pred", ascending=True)
        lw = _allocate(longs, 1.0)
        sw = _allocate(shorts, -1.0)
        # Preserve dollar neutrality after integer rounding by scaling the larger side down notionally.
        if lw.empty or sw.empty:
            continue
        common = min(float(lw.sum()), float(-sw.sum()))
        lw *= common / float(lw.sum())
        sw *= common / float(-sw.sum())
        weights = pd.concat([lw, sw]).groupby(level=0).sum()
        gross = float((weights * day.loc[weights.index, "intraday_full"]).sum())
        cost = float(weights.abs().sum()) * COST_BPS_ROUNDTRIP / 10_000
        daily.append({"date": entry, "gross": gross, "cost": cost, "net": gross - cost,
                      "gross_exposure": float(weights.abs().sum()), "net_exposure": float(weights.sum())})
        for ix, weight in weights.items():
            blotter.append({"date": entry, "symbol": day.at[ix, "symbol"],
                            "sector": day.at[ix, "sector"], "weight": float(weight),
                            "pred": float(day.at[ix, "pred"]),
                            "pnl": float(weight * day.at[ix, "intraday_full"])})
    return pd.DataFrame(daily), pd.DataFrame(blotter)


def main() -> None:
    panel, asset_returns = build_dataset()
    panel = attach_next_intraday_target(panel, asset_returns)
    panel, cols = hierarchical_features(panel)
    predictions, choices = walk_forward_predictions(panel, cols)
    daily, blotter = simulate(predictions, panel, asset_returns)
    evaluation = daily[daily["date"] >= EVAL_START].copy()
    abs_pnl = blotter.groupby("symbol")["pnl"].sum().abs().sort_values(ascending=False)
    concentration = float(abs_pnl.head(10).sum() / abs_pnl.sum()) if abs_pnl.sum() else 1.0
    summary = {
        "spec": {"alphas": ALPHAS, "max_names_side": MAX_NAMES_SIDE,
                 "capital_yen": CAPITAL_YEN, "gross_leverage": 2.0,
                 "cost_bps_roundtrip": COST_BPS_ROUNDTRIP,
                 "universe": "TOPIX500(scale_ord>=3)", "features": ALL_FEATURES},
        "alpha_choices": choices,
        "evaluation": annualized_stats(evaluation, "net"),
        "gross": annualized_stats(evaluation, "gross"),
        "yearly": {str(y): annualized_stats(g, "net")
                   for y, g in evaluation.groupby(evaluation["date"].dt.year)},
        "execution": {"avg_positions": float(blotter.groupby("date").size().mean()),
                      "avg_gross_exposure": float(evaluation["gross_exposure"].mean()),
                      "max_abs_net_exposure": float(evaluation["net_exposure"].abs().max()),
                      "top10_abs_pnl_concentration": concentration},
    }
    ev = summary["evaluation"]
    summary["decision"] = "GO" if (ev["sharpe"] >= 1.0 and ev["max_drawdown"] > -0.20
        and all(x["ann_return"] > 0 for x in summary["yearly"].values())
        and concentration < 0.30) else "NO-GO"
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "daily_returns.csv", index=False)
    blotter.to_parquet(OUT / "blotter.parquet", index=False)
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
