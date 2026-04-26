"""
Value-Reversal Screener — CLI entry point.

Usage:
    python -m screener.run [options]

Examples:
    # デフォルト設定で実行
    python -m screener.run

    # グレアム基準を厳しく設定
    python -m screener.run --mix-max 15 --pbr-max 1.0

    # 小型株含めて50銘柄
    python -m screener.run --market-cap-min 5e9 --top 50
"""
import argparse
import datetime as dt
import os
import sys

# ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screener.value_screener import ValueReversalScreener, format_results
from screener.dashboard import generate_dashboard


def main():
    ap = argparse.ArgumentParser(
        description="バリュー×テクニカル反転スクリーナー (日本株)")
    ap.add_argument("--per-max",  type=float, default=15.0,  help="PER 上限 (default: 15)")
    ap.add_argument("--pbr-max",  type=float, default=2.5,   help="PBR 上限 (default: 2.5)")
    ap.add_argument("--mix-max",  type=float, default=30.0,  help="MIX 上限 (default: 30)")
    ap.add_argument("--market-cap-min", type=float, default=10e9,
                    help="最低時価総額 (円, default: 100億)")
    ap.add_argument("--turnover-min", type=float, default=1e7,
                    help="最低売買代金 (円/日, default: 1000万)")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="テクニカル分析対象の最大銘柄数 (0=全銘柄, default: 0)")
    ap.add_argument("--top",      type=int,   default=0,     help="出力銘柄数 (0=全銘柄, default: 0)")
    ap.add_argument("--output",   type=str,   default=None,  help="CSV出力先パス")
    args = ap.parse_args()

    screener = ValueReversalScreener(
        per_max=args.per_max,
        pbr_max=args.pbr_max,
        mix_max=args.mix_max,
        turnover_min=args.turnover_min,
        max_scan=args.max_scan,
        top_n=args.top,
    )

    results = screener.screen()
    print(format_results(results))

    if results.empty:
        return

    # save dated CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "screener_results")
    os.makedirs(out_dir, exist_ok=True)
    date_str = dt.date.today().strftime("%Y%m%d")
    csv_path = args.output or os.path.join(out_dir, f"value_reversal_{date_str}.csv")
    results.to_csv(csv_path)
    print(f"結果保存: {csv_path}")

    # generate HTML dashboard
    html = generate_dashboard(csv_path)
    # 日付付きファイル
    html_path = csv_path.replace(".csv", ".html").replace("value_reversal_", "dashboard_")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    # 常に最新を指す固定ファイル
    latest_html = os.path.join(out_dir, "dashboard.html")
    with open(latest_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"ダッシュボード: {latest_html}")


if __name__ == "__main__":
    main()
