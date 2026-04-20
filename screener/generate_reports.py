"""
Generate stock reports for filtered screener results.

Usage:
    python -m screener.generate_reports [--top N] [--mcap-min 100] [--va-min 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screener.report_format import evaluate_stock, FORMAT_VERSION
from screener.ir_fetcher import fetch_ir_links_batch


REPORT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "reports.json"
)


def _load_report_cache() -> dict:
    if os.path.exists(REPORT_CACHE):
        with open(REPORT_CACHE, "r") as f:
            return json.load(f)
    return {}


def _save_report_cache(data: dict):
    os.makedirs(os.path.dirname(REPORT_CACHE), exist_ok=True)
    with open(REPORT_CACHE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def generate_summary(row: dict, eval_result: dict) -> str:
    """定量データから簡潔なサマリーを生成。"""
    code = row.get("Code", "")
    name = row.get("NameJP") or row.get("Name", "")
    sector = row.get("SectorJP") or row.get("Sector", "")
    per = row.get("PER")
    fper = row.get("fPER")
    pbr = row.get("PBR")
    mix = row.get("MIX")
    mcap = row.get("MarketCap_B")
    cash = row.get("CashRatio")
    rsi = row.get("RSI")
    tech = row.get("TechScore")
    vr = row.get("VolRatio")
    earn = row.get("EarningsDate")

    bd = eval_result["breakdown"]
    total = eval_result["total_score"]
    verdict = eval_result["verdict"]

    parts = []

    # バリュエーション
    val_parts = []
    if _v(per):
        val_parts.append(f"PER {per:.1f}")
    if _v(fper):
        val_parts.append(f"予想PER {fper:.1f}")
    if _v(pbr):
        val_parts.append(f"PBR {pbr:.2f}")
    if _v(mix):
        val_parts.append(f"MIX {mix:.1f}")
    if val_parts:
        parts.append("【バリュエーション】" + ", ".join(val_parts))
        if _v(per) and _v(fper) and per > 0 and fper > 0:
            ratio = fper / per
            if ratio < 0.7:
                parts.append(f"  → 会社予想ベースでは大幅増益見通し (fPER/PER={ratio:.1%})")
            elif ratio < 0.9:
                parts.append(f"  → 増益見通し (fPER/PER={ratio:.1%})")
        if _v(mix) and mix < 22.5:
            parts.append("  → Graham基準(MIX<22.5)を充足 — 割安水準")

    # 財務
    if _v(cash):
        parts.append(f"【財務】現金比率 {cash:.1f}%")
        if cash > 30:
            parts.append("  → 豊富なキャッシュポジション")

    # テクニカル
    tech_tags = []
    if _v(rsi) and rsi < 35:
        tech_tags.append(f"RSI={rsi:.0f}(売られ過ぎ)")
    elif _v(rsi) and rsi > 65:
        tech_tags.append(f"RSI={rsi:.0f}(買われ過ぎ)")
    if _v(tech):
        tech_tags.append(f"Tech={tech:.2f}")
    if _v(vr) and vr > 1.3:
        tech_tags.append(f"出来高急増(x{vr:.1f})")
    if tech_tags:
        parts.append("【テクニカル】" + ", ".join(tech_tags))

    # カタリスト
    if earn and str(earn) != "nan":
        parts.append(f"【カタリスト】次回決算予定: {earn}")

    # リスク
    risks = []
    if _v(per) and per > 100:
        risks.append("超高PER(利益僅少/赤字転落リスク)")
    if _v(pbr) and pbr < 0.5 and _v(cash) and cash < 5:
        risks.append("バリュートラップ懸念(低PBR+低キャッシュ)")
    if risks:
        parts.append("【リスク】" + "; ".join(risks))

    return "\n".join(parts)


def _v(val) -> bool:
    if val is None:
        return False
    try:
        return not math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def generate_all_reports(csv_path: str, mcap_min: float = 100,
                         va_min: float = 1, top_n: int = 0) -> dict:
    """全レポートを生成して返す。"""
    df = pd.read_csv(csv_path, index_col=0)

    # Filter
    mask = pd.Series(True, index=df.index)
    if "MarketCap_B" in df.columns:
        mask &= df["MarketCap_B"] >= mcap_min
    if "Va_avg5" in df.columns:
        mask &= df["Va_avg5"] >= va_min
    df = df[mask].sort_values("Score", ascending=False)

    if top_n > 0:
        df = df.head(top_n)

    print(f"レポート生成: {len(df)} 銘柄 (mcap>={mcap_min}億, va>={va_min}億)")

    codes = df["Code"].astype(str).tolist()

    # IR links
    ir_links = fetch_ir_links_batch(codes)

    # Evaluate each stock
    cache = _load_report_cache()
    reports = {}
    for i, (_, row) in enumerate(df.iterrows()):
        code = str(row["Code"])
        row_dict = row.to_dict()

        # Evaluate
        eval_result = evaluate_stock(row_dict)
        summary = generate_summary(row_dict, eval_result)
        ir = ir_links.get(code, {})

        report = {
            "code": code,
            "name": row.get("NameJP") or row.get("Name", ""),
            "sector": row.get("SectorJP") or row.get("Sector", ""),
            "screener_score": row.get("Score", 0),
            "eval": eval_result,
            "summary": summary,
            "ir_url": ir.get("ir_url", ""),
            "presentation_url": ir.get("presentation_url", ""),
            "pdf_url": ir.get("pdf_url"),
            "tdnet_url": ir.get("tdnet_url", ""),
            "shikiho_url": ir.get("shikiho_url", ""),
        }
        reports[code] = report

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")

    # Save
    cache.update(reports)
    _save_report_cache(cache)
    print(f"  → {len(reports)} レポート生成完了")
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--mcap-min", type=float, default=100)
    ap.add_argument("--va-min", type=float, default=1)
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    import glob
    if args.input:
        csv_path = args.input
    else:
        pattern = os.path.join(
            os.path.dirname(__file__), "..", "data", "screener_results",
            "value_reversal_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print("No CSV found")
            return
        csv_path = files[-1]

    reports = generate_all_reports(
        csv_path, mcap_min=args.mcap_min,
        va_min=args.va_min, top_n=args.top)

    print(f"\nFormat: {FORMAT_VERSION}")
    # Show top 10
    sorted_reports = sorted(
        reports.values(),
        key=lambda r: r["eval"]["total_score"],
        reverse=True)
    for r in sorted_reports[:10]:
        e = r["eval"]
        print(f"  {r['code']} {r['name'][:12]:12s} "
              f"{'★' * e['stars']}{'☆' * (5 - e['stars'])} "
              f"{e['total_score']:5.1f}pt {e['verdict']}")


if __name__ == "__main__":
    main()
