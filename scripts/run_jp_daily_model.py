"""Train the daily cross-sectional model walk-forward (gap + 需給 + 先物).

Uses only on-disk data (no collection). Adds index-futures overnight features
automatically once data/jp_derivatives/futures_*.parquet exist.
"""
import glob
import json
from pathlib import Path

import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import (
    BASE_FEATURES, FUT_FEATURES, build_daily_features, walk_forward,
)
from trading.jp_intraday.futures_context import build_overnight_features


def _load_futures():
    files = sorted(glob.glob("data/jp_derivatives/futures_*.parquet"))
    if not files:
        return None
    fut = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return fut.drop_duplicates(["Date", "Code"])


def main() -> None:
    out = Path("data/jp_daily_model_results")
    out.mkdir(parents=True, exist_ok=True)
    daily = load_existing_daily()
    fut = _load_futures()
    overnight = build_overnight_features(fut) if fut is not None else None
    # Only trust futures features once coverage is (near) complete, else the
    # partial download would zero-fill most years and understate their value.
    fut_ready = overnight is not None and pd.to_datetime(
        pd.Series(overnight.index)).dt.year.nunique() >= 5
    panel = build_daily_features(daily, min_value_yen=5e8,
                                 futures_overnight=overnight if fut_ready else None)

    configs = {"base": BASE_FEATURES}
    if fut_ready:
        configs["base+futures"] = BASE_FEATURES + FUT_FEATURES
        print(f"futures overnight rows={len(overnight)} years={pd.to_datetime(pd.Series(overnight.index)).dt.year.nunique()} (先物/US enabled)")
    elif overnight is not None:
        print(f"futures partial ({len(overnight)} days) — base only for now; re-run when collection completes")

    summary = {}
    for name, feats in configs.items():
        feats = [f for f in feats if f in panel.columns]
        for cost in (3.0, 5.0):
            wf = walk_forward(panel, feats, quantile=0.05, alpha=10.0, cost_bps_side=cost)
            tag = f"{name}_q05_{int(cost)}bps"
            wf.to_csv(out / f"walkforward_{tag}.csv", index=False)
            mean = wf[wf["test_year"] == "MEAN"].iloc[0]
            summary[tag] = {"gross_sharpe": round(mean["gross_sharpe"], 3),
                            "net_sharpe": round(mean["net_sharpe"], 3)}
            print(f"\n=== {tag} ===")
            print(wf.to_string(index=False))
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
