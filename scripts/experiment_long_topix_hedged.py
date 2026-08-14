#!/usr/bin/env python3
"""Frozen long-only / rounded mini-TOPIX-hedged executable experiment."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_medium_residual_ml import CAPITAL_YEN, build_dataset
from scripts.experiment_topix500_hierarchical_lasso import (
    EVAL_START, attach_next_intraday_target, hierarchical_features, walk_forward_predictions,
)
from trading.jp_intraday.daily_model import annualized_stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/jp_long_topix_hedged"
NAMES = 40
EQUITY_COST_BPS = 2.0
FUTURES_COST_BPS = 2.0
MINI_MULTIPLIER = 1_000.0


def load_topix() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data/jp_derivatives/indices_*.parquet")))
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    d = d[d["Code"].astype(str).eq("0000")].copy()
    d["date"] = pd.to_datetime(d["Date"])
    d["topix_ret"] = pd.to_numeric(d["C"], errors="coerce") / pd.to_numeric(d["O"], errors="coerce") - 1
    return d[["date", "O", "topix_ret"]].drop_duplicates("date").set_index("date")


def simulate(pred: pd.DataFrame, panel: pd.DataFrame, returns: pd.DataFrame, hedged: bool):
    market = returns.set_index(["date", "symbol"])
    known = panel[["date", "symbol", "beta"]].drop_duplicates(["date", "symbol"])
    known = known.set_index(["date", "symbol"])
    topix = load_topix()
    daily, blotter = [], []
    for entry, raw in pred.groupby("target_date"):
        if pd.isna(entry) or entry not in topix.index or raw.pred.std() < 1e-12:
            daily.append({"date": entry, "gross": 0., "net": 0., "topix_ret": 0., "hedge_notional": 0.})
            continue
        decision = raw.date.iloc[0]
        day = raw.nlargest(NAMES, "pred").reset_index(drop=True)
        didx = pd.MultiIndex.from_arrays([[decision] * len(day), day.symbol])
        eidx = pd.MultiIndex.from_arrays([[entry] * len(day), day.symbol])
        dm = market.reindex(didx).reset_index(drop=True)
        em = market.reindex(eidx).reset_index(drop=True)
        k = known.reindex(didx).reset_index(drop=True)
        day["known_close"] = dm.open_full * (1 + dm.intraday_full)
        day["ret"] = em.intraday_full
        day["beta"] = k.beta.fillna(1.0)
        day = day.dropna(subset=["known_close", "ret", "pred"])
        mag = day.pred.clip(lower=0)
        target = mag / mag.sum() * CAPITAL_YEN
        units = np.floor(target / (day.known_close * 100)).astype(int)
        day = day[units > 0].copy(); units = units[units > 0]
        day["position"] = units.to_numpy() * 100 * day.known_close
        weights = day.position / CAPITAL_YEN
        equity = float((weights * day.ret).sum())
        equity_cost = float(weights.sum()) * EQUITY_COST_BPS / 10_000
        beta_yen = float((day.position * day.beta).sum())
        contract_notional = float(topix.at[entry, "O"]) * MINI_MULTIPLIER
        contracts = int(np.rint(beta_yen / contract_notional)) if hedged else 0
        hedge_notional = contracts * contract_notional / CAPITAL_YEN
        hedge = -hedge_notional * float(topix.at[entry, "topix_ret"])
        hedge_cost = abs(hedge_notional) * FUTURES_COST_BPS / 10_000
        gross = equity + hedge
        daily.append({"date": entry, "gross": gross, "net": gross-equity_cost-hedge_cost,
                      "topix_ret": float(topix.at[entry, "topix_ret"]),
                      "hedge_notional": hedge_notional, "contracts": contracts,
                      "long_exposure": float(weights.sum())})
        for i, row in day.iterrows():
            blotter.append({"date": entry, "symbol": row.symbol, "weight": float(row.position/CAPITAL_YEN),
                            "pnl": float(row.position/CAPITAL_YEN*row.ret)})
    return pd.DataFrame(daily), pd.DataFrame(blotter)


def main():
    panel, returns = build_dataset()
    panel = attach_next_intraday_target(panel, returns)
    panel, cols = hierarchical_features(panel)
    pred, choices = walk_forward_predictions(panel, cols)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, hedged in (("long_only", False), ("mini_topix_hedged", True)):
        daily, blotter = simulate(pred, panel, returns, hedged)
        ev = daily[daily.date >= EVAL_START]
        stats = annualized_stats(ev, "net")
        yearly = {str(y): annualized_stats(g, "net") for y,g in ev.groupby(ev.date.dt.year)}
        pnl = blotter.groupby("symbol").pnl.sum().abs().sort_values(ascending=False)
        concentration = float(pnl.head(10).sum()/pnl.sum()) if pnl.sum() else 1.
        beta = float(ev[["net","topix_ret"]].cov().iloc[0,1] / ev.topix_ret.var()) if ev.topix_ret.var() else np.nan
        go = (stats["sharpe"] >= 1 and stats["max_drawdown"] > -.2 and concentration < .3
              and all(x["ann_return"] > 0 for x in yearly.values()) and (not hedged or abs(beta) < .2))
        summary[name] = {"evaluation": stats, "yearly": yearly, "realized_topix_beta": beta,
                         "top10_abs_pnl_concentration": concentration,
                         "avg_contracts": float(ev.get("contracts", pd.Series(0,index=ev.index)).abs().mean()),
                         "decision": "GO" if go else "NO-GO"}
        daily.to_csv(OUT/f"daily_{name}.csv",index=False); blotter.to_parquet(OUT/f"blotter_{name}.parquet",index=False)
    summary["alpha_choices"] = choices
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
