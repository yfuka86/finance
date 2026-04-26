"""
Generate an interactive HTML dashboard from screener results.

Usage:
    python -m screener.dashboard [--input path/to/csv] [--output path/to/html]
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _latest_csv() -> str:
    pattern = os.path.join(
        os.path.dirname(__file__), "..", "data", "screener_results",
        "value_reversal_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No screener result CSVs found")
    return files[-1]


def _score_class(v: float, thresholds=(0.3, 0.5)) -> str:
    if pd.isna(v):
        return "na"
    if v >= thresholds[1]:
        return "high"
    if v >= thresholds[0]:
        return "mid"
    return "low"



def _parse_list_col(val):
    """CSV から読み込んだリスト文字列をパース."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or not val:
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _load_reports() -> dict:
    """レポートキャッシュを読み込み."""
    rpath = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "reports.json")
    if os.path.exists(rpath):
        with open(rpath, "r") as f:
            return json.load(f)
    return {}


def generate_dashboard(csv_path: str) -> str:
    df = pd.read_csv(csv_path, index_col=0)
    date_str = os.path.basename(csv_path).replace("value_reversal_", "").replace(".csv", "")
    if len(date_str) == 8:
        display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    else:
        display_date = date_str

    # Load reports
    reports = _load_reports()

    # Prepare table rows
    rows_html = []
    for _, r in df.iterrows():
        code = str(r.get("Code", ""))
        name_jp = str(r.get("NameJP", "")) if pd.notna(r.get("NameJP")) else ""
        name_en = str(r.get("Name", "")) if pd.notna(r.get("Name")) else ""
        name = name_jp if name_jp else name_en[:16]
        sector_jp = str(r.get("SectorJP", "")) if pd.notna(r.get("SectorJP")) else ""
        sector_en = str(r.get("Sector", "")) if pd.notna(r.get("Sector")) else ""
        sector = sector_jp if sector_jp else sector_en

        per = r.get("PER")
        pbr = r.get("PBR")
        mix = r.get("MIX")
        mcap = r.get("MarketCap_B")
        cash = r.get("CashRatio")
        net_cash = r.get("NetCashRatio")
        score = r.get("FundaScore", 0) or 0
        earn = r.get("EarningsDate", "")
        fper = r.get("fPER")
        close = r.get("Close")

        # 売買代金
        va_latest = r.get("Va_latest")
        va_avg5 = r.get("Va_avg5")
        va_avg20 = r.get("Va_avg20")
        va_spark = _parse_list_col(r.get("Va_spark"))
        va_dates = _parse_list_col(r.get("Va_dates"))

        def fmt(v, d=1):
            if pd.isna(v):
                return '<span class="na">-</span>'
            return f"{v:.{d}f}"

        mix_cls = "good" if (not pd.isna(mix) and mix < 15) else (
            "ok" if (not pd.isna(mix) and mix < 22.5) else "na")

        # 決算日までの日数
        earn_days = 999
        earn_html = ""
        if pd.notna(earn) and earn and str(earn) != "nan":
            try:
                earn_dt = dt.datetime.strptime(str(earn)[:10], "%Y-%m-%d").date()
                earn_days = (earn_dt - dt.date.today()).days
                if earn_days < 0:
                    label = f"{earn} (済)"
                    cls = "catalyst-past"
                elif earn_days <= 14:
                    label = f"{earn} ({earn_days}日後)"
                    cls = "catalyst-soon"
                elif earn_days <= 30:
                    label = f"{earn} ({earn_days}日後)"
                    cls = "catalyst-badge"
                else:
                    label = f"{earn} ({earn_days}日後)"
                    cls = "catalyst-far"
                earn_html = f'<span class="{cls}">{label}</span>'
            except Exception:
                earn_html = f'<span class="catalyst-badge">{earn}</span>'

        # sparkline SVG for turnover
        spark_svg = ""
        if va_spark and len(va_spark) >= 2:
            vals = [float(v) for v in va_spark]
            mx = max(vals) if max(vals) > 0 else 1
            w, h = 80, 24
            points = []
            for i, v in enumerate(vals):
                x = i / (len(vals) - 1) * w
                y = h - (v / mx) * (h - 2) - 1
                points.append(f"{x:.1f},{y:.1f}")
            poly = " ".join(points)
            # area fill
            area = f"0,{h} " + poly + f" {w},{h}"
            spark_svg = (
                f'<svg width="{w}" height="{h}" class="spark">'
                f'<polygon points="{area}" fill="rgba(59,130,246,0.12)" />'
                f'<polyline points="{poly}" fill="none" '
                f'stroke="#3b82f6" stroke-width="1.5" stroke-linejoin="round" />'
                f'</svg>'
            )

        # va info text
        va_html = ""
        if pd.notna(va_latest):
            va_html = f'<span class="va-num">{va_latest:.1f}</span>'
        va_avg_html = ""
        if pd.notna(va_avg5):
            va_avg_html = f'<span class="va-sub">5d:{va_avg5:.1f} / 20d:{va_avg20:.1f}</span>'

        # Report data
        rpt = reports.get(code, {})
        rpt_eval = rpt.get("eval", {})
        rpt_score = rpt_eval.get("total_score", "")
        rpt_verdict = rpt_eval.get("verdict", "")
        rpt_stars = rpt_eval.get("stars", 0)
        rpt_ver = rpt_eval.get("format_version", "")
        rpt_summary = rpt.get("summary", "")
        rpt_ir = rpt.get("ir_url", "")
        rpt_pdf = rpt.get("pdf_url", "")
        rpt_tdnet = rpt.get("tdnet_url", "")
        rpt_shikiho = rpt.get("shikiho_url", "")
        rpt_presentation = rpt.get("presentation_url", "")
        rpt_bd = rpt_eval.get("breakdown", {})

        # Verdict HTML — FundaScore (70点満点) ベースで判定
        funda_score = r.get("FundaScore", 0) or 0
        if funda_score >= 45:
            verdict_label, verdict_cls, stars = "Strong Buy", "verdict-sbuy", 5
        elif funda_score >= 40:
            verdict_label, verdict_cls, stars = "Buy", "verdict-buy", 4
        elif funda_score >= 35:
            verdict_label, verdict_cls, stars = "Hold", "verdict-hold", 3
        elif funda_score >= 30:
            verdict_label, verdict_cls, stars = "Sell", "verdict-sell", 2
        else:
            verdict_label, verdict_cls, stars = "Strong Sell", "verdict-ssell", 1
        star_str = "★" * stars + "☆" * (5 - stars)
        verdict_html = f'<span class="{verdict_cls}">{star_str} {verdict_label}</span>'

        # Detail row (hidden by default)
        detail_html = ""
        if rpt_summary:
            ir_links_html = ""
            if rpt_ir:
                ir_links_html += f'<a href="{rpt_ir}" target="_blank" class="ir-link">IR情報</a>'
            if rpt_presentation:
                ir_links_html += f' <a href="{rpt_presentation}" target="_blank" class="ir-link">決算説明資料</a>'
            if rpt_pdf:
                ir_links_html += f' <a href="{rpt_pdf}" target="_blank" class="ir-link pdf-link">決算説明PDF</a>'
            if rpt_tdnet:
                ir_links_html += f' <a href="{rpt_tdnet}" target="_blank" class="ir-link">TDnet</a>'
            if rpt_shikiho:
                ir_links_html += f' <a href="{rpt_shikiho}" target="_blank" class="ir-link">四季報</a>'

            bd_html = ""
            if rpt_bd:
                bd_html = '<div class="bd-bar">'
                bd_labels = {"valuation": "割安 (PER/fPER/PBR/MIX)", "financial": "財務 (現金比率+資産裏付け)",
                             "growth": "成長 (fPER/PER比+推定ROE)", "catalyst": "触媒 (決算近接度)", "risk": "リスク (減点式)"}
                bd_maxes = {"valuation": 20, "financial": 15, "growth": 15,
                            "catalyst": 10, "risk": 10}
                for k, label in bd_labels.items():
                    v = rpt_bd.get(k, 0)
                    mx = bd_maxes.get(k, 10)
                    pct = min(100, v / mx * 100) if mx > 0 else 0
                    bd_html += f'<div class="bd-item"><span class="bd-label">{label}</span><div class="bd-track"><div class="bd-fill" style="width:{pct:.0f}%"></div></div><span class="bd-val">{v:.1f}/{mx}</span></div>'
                bd_html += '</div>'

            summary_escaped = rpt_summary.replace("\n", "<br>")
            detail_html = f"""
        <tr class="detail-row" data-parent="{code}" style="display:none">
          <td colspan="16" class="detail-cell">
            <div class="detail-content">
              <div class="detail-header">
                <span class="detail-verdict {verdict_cls}">{verdict_label} ({funda_score:.1f}/70)</span>
                <span class="detail-ver">{rpt_ver}</span>
                {ir_links_html}
              </div>
              {bd_html}
              <div class="detail-summary">{summary_escaped}</div>
            </div>
          </td>
        </tr>"""

        # ファンダメンタルスコア (70点満点)
        funda_score = r.get("FundaScore", 0) or 0
        # スコアに応じたクラス
        fs_cls = "fs-high" if funda_score >= 45 else ("fs-mid" if funda_score >= 38 else "fs-low")

        rows_html.append(f"""
        <tr class="main-row" data-sector="{sector}" data-score="{funda_score}" data-earn-days="{earn_days}" data-mcap="{mcap if pd.notna(mcap) else 0}" data-va-avg5="{va_avg5 if pd.notna(va_avg5) else 0}" data-code="{code}" onclick="toggleDetail(this)">
          <td class="code"><a href="https://irbank.net/{code}" target="_blank" onclick="event.stopPropagation()">{code}</a></td>
          <td class="name" title="{name_en}">{name}</td>
          <td class="sector">{sector}</td>
          <td class="num funda-score {fs_cls}">{funda_score:.1f}</td>
          <td class="verdict-cell">{verdict_html}</td>
          <td class="num">{fmt(close, 0)}</td>
          <td class="num per">{fmt(per)}</td>
          <td class="num">{fmt(fper)}</td>
          <td class="num pbr">{fmt(pbr, 2)}</td>
          <td class="num mix {mix_cls}">{fmt(mix)}</td>
          <td class="num">{fmt(mcap, 0)}</td>
          <td class="num cash">{fmt(cash)}</td>
          <td class="num {"net-cash-neg" if pd.notna(net_cash) and net_cash < 0 else "net-cash"}">{fmt(net_cash)}</td>
          <td class="catalyst">{earn_html}</td>
          <td class="va-cell">{va_html}{va_avg_html}</td>
          <td class="spark-cell">{spark_svg}</td>
        </tr>{detail_html}""")

    table_body = "\n".join(rows_html)

    # Summary stats
    n = len(df)
    avg_per = df["PER"].mean() if "PER" in df else 0
    avg_pbr = df["PBR"].mean() if "PBR" in df else 0
    avg_mix = df["MIX"].mean() if "MIX" in df else 0
    avg_funda = df["FundaScore"].mean() if "FundaScore" in df.columns else 0
    n_catalyst = int(df["EarningsDate"].notna().sum()) if "EarningsDate" in df else 0
    top_score = df["FundaScore"].max() if "FundaScore" in df.columns else 0

    # Sector distribution (Japanese)
    sec_col = "SectorJP" if "SectorJP" in df.columns else "Sector"
    sector_counts = df[sec_col].value_counts().to_dict() if sec_col in df else {}
    sector_counts = {k: v for k, v in sector_counts.items() if k and str(k) != "nan"}
    sector_json = json.dumps(sector_counts, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Value-Reversal Screener | {display_date}</title>
<style>
  :root {{
    --bg: #ffffff;
    --card: #f8f9fb;
    --border: #e2e5ea;
    --text: #1a1d27;
    --dim: #6b7280;
    --accent: #2563eb;
    --green: #16a34a;
    --red: #dc2626;
    --orange: #d97706;
    --yellow: #ca8a04;
    --light-blue: #eff6ff;
    --light-green: #f0fdf4;
    --light-red: #fef2f2;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Hiragino Sans', 'Hiragino Kaku Gothic ProN',
                 'Noto Sans JP', 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.6;
  }}
  .container {{ max-width: 2400px; margin: 0 auto; padding: 24px; }}

  /* Header */
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 2px solid var(--border);
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }}
  .header h1 span {{ color: var(--accent); }}
  .header .date {{
    font-size: 14px;
    color: var(--dim);
    font-weight: 500;
  }}

  /* Summary cards */
  .summary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }}
  .card .label {{
    font-size: 11px;
    color: var(--dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}
  .card .value {{
    font-size: 24px;
    font-weight: 800;
    margin-top: 4px;
  }}
  .card .value.green {{ color: var(--green); }}
  .card .value.accent {{ color: var(--accent); }}
  .card .value.orange {{ color: var(--orange); }}

  /* Sector badges */
  .sector-bar {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .sector-badge {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: var(--dim);
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
  }}
  .sector-badge:hover {{ border-color: var(--accent); color: var(--accent); }}
  .sector-badge.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }}
  .sector-badge .cnt {{
    color: inherit;
    font-weight: 700;
    margin-left: 4px;
    opacity: 0.7;
  }}

  /* Filters */
  .filters {{
    display: flex;
    gap: 14px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    align-items: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
  }}
  .filters label {{
    font-size: 12px;
    color: var(--dim);
    font-weight: 600;
  }}
  .filters input[type="range"] {{
    width: 100px;
    accent-color: var(--accent);
  }}
  .filters span {{
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
    min-width: 28px;
  }}
  .filters .val-input {{
    width: 52px;
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 2px 6px;
    text-align: center;
    background: #fff;
    font-family: inherit;
  }}
  .filters .val-input:focus {{
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
  }}
  .filters .unit {{
    font-size: 11px;
    color: var(--dim);
    font-weight: 400;
    min-width: 0;
  }}

  /* Table */
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    white-space: nowrap;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: #f1f3f8;
    color: var(--dim);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 11px 8px;
    text-align: left;
    border-bottom: 2px solid var(--border);
    cursor: pointer;
    user-select: none;
  }}
  thead th:hover {{ color: var(--accent); }}
  tbody tr {{
    border-bottom: 1px solid #f0f1f4;
    transition: background 0.1s;
  }}
  tbody tr:hover {{ background: var(--light-blue); }}
  td {{ padding: 10px 8px; vertical-align: middle; }}
  .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
  }}
  .code a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 700;
    font-size: 13px;
  }}
  .code a:hover {{ text-decoration: underline; }}
  .name {{
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 600;
    font-size: 13px;
  }}
  .sector {{
    color: var(--dim);
    font-size: 12px;
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .na {{ color: #ccc; }}

  /* Score coloring */
  .score.high {{ color: #fff; background: var(--green); border-radius: 4px; padding: 3px 8px; font-weight: 700; }}
  .score.mid {{ color: var(--orange); font-weight: 700; }}
  .score.low {{ color: var(--dim); }}
  .mix.good {{ color: var(--green); font-weight: 700; }}
  .mix.ok {{ color: var(--text); }}
  .cash {{ color: var(--accent); }}
  .net-cash {{ color: var(--accent); }}
  .net-cash-neg {{ color: var(--red); }}

  /* Volume / turnover */
  .va-cell {{
    text-align: right;
    font-size: 12px;
    font-family: 'SF Mono', monospace;
    line-height: 1.3;
  }}
  .va-num {{
    font-weight: 700;
    color: var(--text);
    font-size: 13px;
  }}
  .va-sub {{
    display: block;
    font-size: 10px;
    color: var(--dim);
  }}
  .spark-cell {{ padding: 4px 4px; }}
  .spark {{ display: block; }}

  /* Score column */
  .funda-score {{
    font-size: 13px;
    font-weight: 700;
  }}
  .fs-high {{ color: var(--green); }}
  .fs-mid {{ color: var(--accent); }}
  .fs-low {{ color: var(--dim); }}
  .catalyst-soon {{
    background: #fef2f2;
    color: var(--red);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid rgba(220,38,38,0.25);
  }}
  .catalyst-badge {{
    background: var(--light-green);
    color: var(--green);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(22,163,74,0.2);
  }}
  .catalyst-far {{
    background: #f8f9fb;
    color: var(--dim);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid var(--border);
  }}
  .catalyst-past {{
    background: #f8f9fb;
    color: #bbb;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 400;
    border: 1px solid #eee;
    text-decoration: line-through;
  }}

  /* Verdict badges */
  .verdict-sbuy {{ color: #fff; background: #16a34a; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .verdict-buy {{ color: #16a34a; font-weight: 700; font-size: 11px; }}
  .verdict-hold {{ color: var(--orange); font-size: 11px; }}
  .verdict-sell {{ color: var(--red); font-size: 11px; }}
  .verdict-ssell {{ color: #fff; background: var(--red); padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .verdict-cell {{ text-align: center; white-space: nowrap; }}
  .rpt-score {{ text-align: right; font-size: 12px; }}

  /* Expandable detail row */
  .main-row {{ cursor: pointer; }}
  .main-row:hover {{ background: var(--light-blue) !important; }}
  .main-row.expanded {{ background: #f0f4ff; }}
  .detail-row td {{ padding: 0 !important; border-bottom: 2px solid var(--accent); }}
  .detail-cell {{ padding: 0 !important; }}
  .detail-content {{
    padding: 16px 24px;
    background: linear-gradient(135deg, #fafbff, #f0f4ff);
    font-size: 13px;
    line-height: 1.7;
  }}
  .detail-header {{
    display: flex; gap: 12px; align-items: center;
    margin-bottom: 12px; flex-wrap: wrap;
  }}
  .detail-verdict {{
    font-size: 15px; font-weight: 800; padding: 4px 12px; border-radius: 6px;
  }}
  .detail-verdict.verdict-sbuy {{ color: #fff; background: #16a34a; }}
  .detail-verdict.verdict-buy {{ color: #fff; background: #22c55e; }}
  .detail-verdict.verdict-hold {{ color: #fff; background: var(--orange); }}
  .detail-verdict.verdict-sell {{ color: #fff; background: #ef4444; }}
  .detail-verdict.verdict-ssell {{ color: #fff; background: #991b1b; }}
  .detail-ver {{
    font-size: 10px; color: var(--dim); background: var(--card);
    border: 1px solid var(--border); border-radius: 10px; padding: 2px 8px;
  }}
  .ir-link {{
    font-size: 12px; color: var(--accent); text-decoration: none;
    border: 1px solid var(--accent); border-radius: 6px; padding: 3px 10px;
    transition: all 0.15s;
  }}
  .ir-link:hover {{ background: var(--accent); color: #fff; }}
  .pdf-link {{ border-color: var(--red); color: var(--red); }}
  .pdf-link:hover {{ background: var(--red); color: #fff; }}

  /* Breakdown bar */
  .bd-bar {{
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .bd-item {{
    display: flex; align-items: center; gap: 4px; font-size: 11px;
  }}
  .bd-label {{ color: var(--dim); min-width: 56px; font-weight: 600; }}
  .bd-track {{
    width: 60px; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;
  }}
  .bd-fill {{
    height: 100%; background: var(--accent); border-radius: 4px;
    transition: width 0.3s;
  }}
  .bd-val {{ color: var(--dim); font-family: 'SF Mono', monospace; min-width: 40px; }}

  .detail-summary {{
    white-space: pre-wrap;
    font-size: 12.5px;
    color: var(--text);
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    line-height: 1.8;
  }}

  /* Legend */
  .legend {{
    margin-top: 24px;
    padding: 18px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 12px;
    color: var(--dim);
    line-height: 1.9;
  }}
  .legend h3 {{
    color: var(--text);
    font-size: 14px;
    margin-bottom: 8px;
  }}
  .legend .formula {{
    color: var(--accent);
    font-weight: 600;
    font-family: 'SF Mono', monospace;
    font-size: 12px;
  }}
  .legend-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 12px;
    font-size: 12px;
  }}
  .legend-table th {{
    text-align: left;
    padding: 4px 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    font-weight: 600;
    color: var(--text);
  }}
  .legend-table td {{
    padding: 4px 8px;
    border: 1px solid var(--border);
    vertical-align: top;
    line-height: 1.6;
  }}
  .legend-table td:first-child {{
    white-space: nowrap;
    font-weight: 600;
    color: var(--accent);
  }}
  .legend-table td:nth-child(2) {{
    text-align: center;
    white-space: nowrap;
    width: 50px;
  }}

  @media (max-width: 960px) {{
    .summary {{ grid-template-columns: repeat(3, 1fr); }}
    .name {{ max-width: 120px; }}
    .container {{ padding: 12px; }}
  }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1><span>Value-Reversal</span> Screener</h1>
  <div class="date">{display_date} | <span id="visibleCount">{n}</span> / {n} 銘柄</div>
</div>

<div class="summary">
  <div class="card">
    <div class="label">候補銘柄数</div>
    <div class="value accent">{n}</div>
  </div>
  <div class="card">
    <div class="label">最高スコア</div>
    <div class="value green">{top_score:.1f}/70</div>
  </div>
  <div class="card">
    <div class="label">平均 PER</div>
    <div class="value">{avg_per:.1f}</div>
  </div>
  <div class="card">
    <div class="label">平均 PBR</div>
    <div class="value">{avg_pbr:.2f}</div>
  </div>
  <div class="card">
    <div class="label">平均 MIX</div>
    <div class="value orange">{avg_mix:.1f}</div>
  </div>
  <div class="card">
    <div class="label">平均スコア</div>
    <div class="value">{avg_funda:.1f}/70</div>
  </div>
  <div class="card">
    <div class="label">決算予定あり</div>
    <div class="value green">{n_catalyst}</div>
  </div>
</div>

<div class="sector-bar" id="sectorBar">
  <span class="sector-badge active" onclick="filterSector(this, '')">ALL<span class="cnt">{n}</span></span>
</div>

<div class="filters">
  <label>PER</label>
  <input type="text" id="perMinVal" class="val-input" value="0"
    onchange="applyFilters()" onkeydown="if(event.key==='Enter')this.blur()">
  <span class="unit">~</span>
  <input type="text" id="perMaxVal" class="val-input" value="999"
    onchange="applyFilters()" onkeydown="if(event.key==='Enter')this.blur()">

  <label>PBR上限</label>
  <input type="range" id="pbrSlider" min="0.1" max="10.0" step="0.05" value="10.0"
    oninput="syncFromSlider('pbr')">
  <input type="text" id="pbrVal" class="val-input" value="10.0"
    onchange="syncFromInput('pbr')" onkeydown="if(event.key==='Enter')this.blur()">

  <label>MIX上限</label>
  <input type="range" id="mixSlider" min="5" max="200" step="1" value="200"
    oninput="syncFromSlider('mix')">
  <input type="text" id="mixVal" class="val-input" value="200"
    onchange="syncFromInput('mix')" onkeydown="if(event.key==='Enter')this.blur()">

  <label>時価総額下限</label>
  <input type="range" id="mcapSlider" min="0" max="100" step="1" value="50"
    oninput="syncLogSlider('mcap')">
  <input type="text" id="mcapVal" class="val-input" value="100"
    onchange="syncLogInput('mcap')" onkeydown="if(event.key==='Enter')this.blur()">
  <span class="unit">億円~</span>

  <label>売買代金(5d平均)</label>
  <input type="range" id="vaSlider" min="0" max="100" step="1" value="43"
    oninput="syncLogSlider('va')">
  <input type="text" id="vaVal" class="val-input" value="1"
    onchange="syncLogInput('va')" onkeydown="if(event.key==='Enter')this.blur()">
  <span class="unit">億円~</span>

  <label style="margin-left:8px">決算まで</label>
  <select id="earnFilter" onchange="applyFilters()" style="padding:5px 8px">
    <option value="all">全て</option>
    <option value="14">14日以内</option>
    <option value="30">30日以内</option>
    <option value="60">60日以内</option>
    <option value="none">決算予定なし除外</option>
  </select>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
  <th onclick="sortTable(0)">コード</th>
  <th onclick="sortTable(1)">銘柄名</th>
  <th onclick="sortTable(2)">業種</th>
  <th onclick="sortTable(3)" title="割安(20)+財務(15)+成長(15)+触媒(10)+リスク(10)=70点満点 ※行クリックで内訳表示">スコア</th>
  <th onclick="sortTable(4)">判定</th>
  <th onclick="sortTable(5)">株価</th>
  <th onclick="sortTable(6)">PER</th>
  <th onclick="sortTable(7)">fPER</th>
  <th onclick="sortTable(8)">PBR</th>
  <th onclick="sortTable(9)">MIX</th>
  <th onclick="sortTable(10)">時価総額(億)</th>
  <th onclick="sortTable(11)">現金比率%</th>
  <th onclick="sortTable(12)">ネット現金%</th>
  <th>決算予定</th>
  <th onclick="sortTable(14)">売買代金(億)</th>
  <th>推移</th>
</tr>
</thead>
<tbody id="tableBody">
{table_body}
</tbody>
</table>
</div>

<div class="legend">
  <h3>バリュースコア (70点満点)</h3>
  <p style="margin:4px 0 8px;color:var(--text)">ファンダメンタルのみで銘柄を評価。5つの軸の合計点がスコア。行をクリックすると内訳バーが表示されます。</p>
  <table class="legend-table">
    <tr><th>軸</th><th>満点</th><th>算出方法</th></tr>
    <tr><td><strong>割安</strong></td><td>20</td><td>PER(5) + fPER(5) + PBR(5) + MIX(5)。各指標が低いほど高得点<br><span style="color:#888">PER: &lt;8→5, &lt;12→4, &lt;15→3, &lt;20→2, &lt;30→1 ／ PBR: &lt;0.5→5, &lt;0.8→4, &lt;1.0→3.5, &lt;1.5→2.5 ／ MIX: &lt;10→5, &lt;15→4, &lt;22.5→3</span></td></tr>
    <tr><td><strong>財務</strong></td><td>15</td><td>現金比率(10) + PBR資産裏付け(5)<br><span style="color:#888">現金比率: &gt;50%→10, &gt;30%→8, &gt;20%→6, &gt;10%→4 ／ 資産: min(5, 5÷PBR)</span></td></tr>
    <tr><td><strong>成長</strong></td><td>15</td><td>fPER/PER比(10) + 推定ROE(5)<br><span style="color:#888">fPER/PER: &lt;0.5→10(大幅増益), &lt;0.7→8, &lt;0.9→5 ／ 推定ROE(=PBR/PER×100): &gt;15%→5, &gt;10%→3</span></td></tr>
    <tr><td><strong>触媒</strong></td><td>10</td><td>次回決算までの日数<br><span style="color:#888">14日以内→8, 30日以内→5, 60日以内→3, 60日超→1</span></td></tr>
    <tr><td><strong>リスク</strong></td><td>10</td><td>10点から減点<br><span style="color:#888">バリュートラップ(PBR&lt;0.5+現金比率&lt;5%): −3 ／ 超高PER(&gt;100): −4, (&gt;50): −2 ／ 低流動性(代金&lt;1億): −3, (&lt;3億): −1</span></td></tr>
  </table>
  <strong>判定</strong>: ★★★★★ Strong Buy(≥45) / ★★★★ Buy(≥40) / ★★★ Hold(≥35) / ★★ Sell(≥30) / ★ Strong Sell(&lt;30)<br>
  <br>
  <span style="color:#888">
  MIX = PER × PBR (Graham基準 &lt; 22.5) &nbsp;|&nbsp;
  fPER = 会社予想ベースのPER &nbsp;|&nbsp;
  現金比率 = 現金同等物 ÷ 時価総額 × 100 &nbsp;|&nbsp;
  ネット現金 = (現金同等物 − 負債) ÷ 時価総額 × 100
  </span>
</div>

</div>

<script>
const sectorCounts = {sector_json};
const bar = document.getElementById('sectorBar');
Object.entries(sectorCounts).forEach(([sec, cnt]) => {{
  if (!sec || sec === 'nan' || sec === '') return;
  const b = document.createElement('span');
  b.className = 'sector-badge';
  b.innerHTML = sec + '<span class="cnt">' + cnt + '</span>';
  b.onclick = function() {{ filterSector(this, sec); }};
  bar.appendChild(b);
}});

let activeSector = '';
function filterSector(el, sec) {{
  document.querySelectorAll('.sector-badge').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  activeSector = sec;
  applyFilters();
}}

const sliderMap = {{
  mix:  {{ slider: 'mixSlider',  input: 'mixVal' }},
  pbr:  {{ slider: 'pbrSlider',  input: 'pbrVal' }},
}};
function syncFromSlider(key) {{
  const s = sliderMap[key];
  document.getElementById(s.input).value = document.getElementById(s.slider).value;
  applyFilters();
}}
function syncFromInput(key) {{
  const s = sliderMap[key];
  const inp = document.getElementById(s.input);
  const sl = document.getElementById(s.slider);
  let v = parseFloat(inp.value);
  if (isNaN(v)) v = parseFloat(sl.min);
  v = Math.max(parseFloat(sl.min), Math.min(parseFloat(sl.max), v));
  sl.value = v;
  inp.value = v;
  applyFilters();
}}

// Log-scale sliders: slider 0-100 → real value via log mapping
const logConfig = {{
  mcap: {{ minVal: 0, maxVal: 10000 }},  // 0~10000億円
  va:   {{ minVal: 0, maxVal: 200 }},     // 0~200億円
}};
const logIds = {{
  mcap: {{ slider: 'mcapSlider', input: 'mcapVal' }},
  va:   {{ slider: 'vaSlider',   input: 'vaVal' }},
}};
function sliderToReal(key, pos) {{
  // pos: 0-100 linear → real value (log scale)
  if (pos <= 0) return 0;
  const cfg = logConfig[key];
  return Math.round(cfg.maxVal ** (pos / 100));
}}
function realToSlider(key, val) {{
  if (val <= 0) return 0;
  const cfg = logConfig[key];
  const v = Math.min(val, cfg.maxVal);
  return Math.round(Math.log(v) / Math.log(cfg.maxVal) * 100);
}}
function syncLogSlider(key) {{
  const ids = logIds[key];
  const pos = parseFloat(document.getElementById(ids.slider).value);
  const real = sliderToReal(key, pos);
  document.getElementById(ids.input).value = real;
  applyFilters();
}}
function syncLogInput(key) {{
  const ids = logIds[key];
  const inp = document.getElementById(ids.input);
  let v = parseFloat(inp.value);
  if (isNaN(v) || v < 0) v = 0;
  inp.value = Math.round(v);
  document.getElementById(ids.slider).value = realToSlider(key, v);
  applyFilters();
}}

function toggleDetail(mainRow) {{
  const code = mainRow.dataset.code;
  const detailRow = document.querySelector(`.detail-row[data-parent="${{code}}"]`);
  if (!detailRow) return;
  const isOpen = detailRow.style.display !== 'none';
  detailRow.style.display = isOpen ? 'none' : '';
  mainRow.classList.toggle('expanded', !isOpen);
}}

function applyFilters() {{
  const perMin = parseFloat(document.getElementById('perMinVal').value) || 0;
  const perMax = parseFloat(document.getElementById('perMaxVal').value) || 999;
  const pbrMax = parseFloat(document.getElementById('pbrVal').value);
  const mixMax = parseFloat(document.getElementById('mixVal').value);
  const mcapMin = parseFloat(document.getElementById('mcapVal').value) || 0;
  const vaMin = parseFloat(document.getElementById('vaVal').value) || 0;
  const earnVal = document.getElementById('earnFilter').value;
  let visCount = 0;
  document.querySelectorAll('#tableBody tr.main-row').forEach(row => {{
    const sector = row.dataset.sector;
    const ed = parseInt(row.dataset.earnDays);
    const mcap = parseFloat(row.dataset.mcap);
    const vaAvg5 = parseFloat(row.dataset.vaAvg5);
    const c = row.cells;
    const g = t => {{ const v = c[t].textContent.trim(); return v === '-' ? NaN : parseFloat(v); }};
    const per = g(6), pbr = g(8), mix = g(9);
    let show = true;
    if (activeSector && sector !== activeSector) show = false;
    if (!isNaN(per) && (per < perMin || per > perMax)) show = false;
    if (!isNaN(pbr) && pbr > pbrMax) show = false;
    if (!isNaN(mix) && mix > mixMax) show = false;
    if (mcapMin > 0 && mcap < mcapMin) show = false;
    if (vaMin > 0 && vaAvg5 < vaMin) show = false;
    if (earnVal === 'none' && ed >= 999) show = false;
    else if (earnVal !== 'all' && earnVal !== 'none') {{
      const maxDays = parseInt(earnVal);
      if (ed > maxDays || ed < 0) show = false;
    }}
    row.style.display = show ? '' : 'none';
    // Also hide detail row if main row is hidden
    const code = row.dataset.code;
    const detail = document.querySelector(`.detail-row[data-parent="${{code}}"]`);
    if (detail && !show) {{
      detail.style.display = 'none';
      row.classList.remove('expanded');
    }}
    if (show) visCount++;
  }});
  document.getElementById('visibleCount').textContent = visCount;
}}

let sortCol = -1, sortAsc = true;
function sortTable(col) {{
  const tb = document.getElementById('tableBody');
  // Collect main rows with their following detail rows
  const pairs = [];
  const allRows = Array.from(tb.rows);
  for (let i = 0; i < allRows.length; i++) {{
    if (allRows[i].classList.contains('main-row')) {{
      const pair = [allRows[i]];
      if (i + 1 < allRows.length && allRows[i+1].classList.contains('detail-row')) {{
        pair.push(allRows[i+1]);
      }}
      pairs.push(pair);
    }}
  }}
  if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = false; }}
  pairs.sort((a, b) => {{
    let va = a[0].cells[col].textContent.trim();
    let vb = b[0].cells[col].textContent.trim();
    if (va === '-' || va === '') va = sortAsc ? '99999' : '-99999';
    if (vb === '-' || vb === '') vb = sortAsc ? '99999' : '-99999';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return sortAsc ? na - nb : nb - na;
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  pairs.forEach(pair => pair.forEach(r => tb.appendChild(r)));
}}

// Apply default filters on load
document.addEventListener('DOMContentLoaded', applyFilters);
</script>
</body>
</html>"""
    return html


def main():
    ap = argparse.ArgumentParser(description="Generate screening dashboard")
    ap.add_argument("--input", type=str, default=None, help="Input CSV path")
    ap.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = ap.parse_args()

    csv_path = args.input or _latest_csv()
    html = generate_dashboard(csv_path)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "screener_results")
    os.makedirs(out_dir, exist_ok=True)
    date_str = os.path.basename(csv_path).replace("value_reversal_", "").replace(".csv", "")
    out_path = args.output or os.path.join(out_dir, f"dashboard_{date_str}.html")
    latest_path = os.path.join(out_dir, "dashboard.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard: {latest_path}")
    return latest_path


if __name__ == "__main__":
    main()
