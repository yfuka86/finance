#!/usr/bin/env python3
"""How much of ensemble_core survives if only K names' quotes can be fetched?

Frozen in docs/PREREGISTER_QUOTE_SHORTLIST.md.

kabu reads ~9.5 symbols/sec, so the production 467-name universe takes 49-66s and
each name is observed at a different moment. AGENTS records that the user's
"quotes do not predict the open" measurement cannot be separated from that 66s
smear. Shortlisting to ~50 names costs ~5s, which is the same simultaneity the
"95 symbols/sec" spec was asking another broker for — on kabu, unchanged.

The prior finding that "shrinking the universe kills the alpha" was measured by
raising the *liquidity floor*. Shortlisting by a quote-free volatility signal is
a different cut and has never been tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.strategies import run_unit_lot

SCREEN, KS, OOS = "gap_vol60", (30, 50, 100, 200, None), "2024-01-01"
COST_BPS, CAPITAL, NAMES = 7.0, 2e7, 8
PASS_SHARPE, PASS_RETENTION = 1.0, .50
OUT = Path("data/jp_quote_shortlist")


def stats(daily: pd.DataFrame) -> dict:
    if daily.empty or "net" not in daily:
        return {"sharpe": None}
    r = daily[daily["date"].ge(OOS)]["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "days": int(len(r))}


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    rank = panel.groupby("date")[SCREEN].rank(ascending=False, method="first")
    out = {"screen": SCREEN, "oos_start": OOS, "cost_bps_side": COST_BPS,
           "capital": CAPITAL, "names_per_side": NAMES,
           "median_universe_per_day": int(panel.groupby("date").size().median()),
           "results": {}}
    for k in KS:
        sub = panel if k is None else panel[rank.le(k)]
        daily, blot = run_unit_lot(sub, "ensemble_core", capital_yen=CAPITAL,
                                   names_per_side=NAMES, cost_bps_side=COST_BPS)
        s = stats(daily)
        if len(blot):
            b = blot[blot["date"].ge(OOS)]
            s["median_positions"] = int(b.groupby("date").size().median()) if len(b) else 0
        out["results"]["all" if k is None else f"K={k}"] = s
        print("all" if k is None else f"K={k}", json.dumps(s, ensure_ascii=False), flush=True)

    base = out["results"]["all"].get("sharpe")
    for key, s in out["results"].items():
        if base and s.get("sharpe") is not None:
            s["retention"] = round(s["sharpe"] / base, 3)
    k50 = out["results"].get("K=50", {})
    failed = []
    if (k50.get("sharpe") or -9) < PASS_SHARPE:
        failed.append("sharpe_lt_1.0")
    if (k50.get("retention") or 0) < PASS_RETENTION:
        failed.append("retention_lt_50pct")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "PENDING_QUOTE_MEASUREMENT"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
