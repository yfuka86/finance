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


def _signal_label(rsi_sc, macd_sc, ma_sc, bottom_sc) -> str:
    parts = []
    if rsi_sc >= 0.5:
        parts.append("RSI反転")
    if macd_sc >= 0.5:
        parts.append("MACDクロス")
    if ma_sc >= 0.5:
        parts.append("MA転換")
    if bottom_sc >= 0.5:
        parts.append("底値反発")
    return " / ".join(parts) if parts else "-"


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


def generate_dashboard(csv_path: str) -> str:
    df = pd.read_csv(csv_path, index_col=0)
    date_str = os.path.basename(csv_path).replace("value_reversal_", "").replace(".csv", "")
    if len(date_str) == 8:
        display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    else:
        display_date = date_str

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
        rsi = r.get("RSI")
        tech = r.get("TechScore", 0)
        vol_ratio = r.get("VolRatio")
        val = r.get("ValueScore", 0)
        score = r.get("Score", 0)
        earn = r.get("EarningsDate", "")
        fper = r.get("fPER")
        close = r.get("Close")

        # 売買代金
        va_latest = r.get("Va_latest")
        va_avg5 = r.get("Va_avg5")
        va_avg20 = r.get("Va_avg20")
        va_spark = _parse_list_col(r.get("Va_spark"))
        va_dates = _parse_list_col(r.get("Va_dates"))

        # signal tags
        sig = _signal_label(
            r.get("RSI_sc", 0), r.get("MACD_sc", 0),
            r.get("MA_sc", 0), r.get("Bottom_sc", 0))

        def fmt(v, d=1):
            if pd.isna(v):
                return '<span class="na">-</span>'
            return f"{v:.{d}f}"

        score_cls = _score_class(score, (0.25, 0.35))
        tech_cls = _score_class(tech, (0.3, 0.5))
        val_cls = _score_class(val, (0.3, 0.5))
        rsi_cls = "oversold" if (not pd.isna(rsi) and rsi < 35) else (
            "overbought" if (not pd.isna(rsi) and rsi > 70) else "neutral")
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
            va_html = f'<span class="va-num">{va_latest:.0f}</span>'
        va_avg_html = ""
        if pd.notna(va_avg5):
            va_avg_html = f'<span class="va-sub">5d:{va_avg5:.0f} / 20d:{va_avg20:.0f}</span>'

        rows_html.append(f"""
        <tr data-sector="{sector}" data-score="{score}" data-earn-days="{earn_days}" data-mcap="{mcap if pd.notna(mcap) else 0}">
          <td class="code"><a href="https://finance.yahoo.co.jp/quote/{code}.T" target="_blank">{code}</a></td>
          <td class="name" title="{name_en}">{name}</td>
          <td class="sector">{sector}</td>
          <td class="num">{fmt(close, 0)}</td>
          <td class="num per">{fmt(per)}</td>
          <td class="num">{fmt(fper)}</td>
          <td class="num pbr">{fmt(pbr, 2)}</td>
          <td class="num mix {mix_cls}">{fmt(mix)}</td>
          <td class="num">{fmt(mcap, 0)}</td>
          <td class="num cash">{fmt(cash)}</td>
          <td class="num rsi {rsi_cls}">{fmt(rsi)}</td>
          <td class="num tech {tech_cls}">{fmt(tech, 2)}</td>
          <td class="num vol">{fmt(vol_ratio, 2)}</td>
          <td class="num val {val_cls}">{fmt(val, 2)}</td>
          <td class="num score {score_cls}">{fmt(score, 3)}</td>
          <td class="va-cell">{va_html}{va_avg_html}</td>
          <td class="spark-cell">{spark_svg}</td>
          <td class="signals">{sig}</td>
          <td class="catalyst">{earn_html}</td>
        </tr>""")

    table_body = "\n".join(rows_html)

    # Summary stats
    n = len(df)
    avg_per = df["PER"].mean() if "PER" in df else 0
    avg_pbr = df["PBR"].mean() if "PBR" in df else 0
    avg_mix = df["MIX"].mean() if "MIX" in df else 0
    avg_tech = df["TechScore"].mean() if "TechScore" in df else 0
    n_catalyst = int(df["EarningsDate"].notna().sum()) if "EarningsDate" in df else 0
    top_score = df["Score"].max() if "Score" in df else 0

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
  .container {{ max-width: 1680px; margin: 0 auto; padding: 24px; }}

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
  .tech.high {{ color: var(--green); font-weight: 700; }}
  .tech.mid {{ color: var(--orange); font-weight: 600; }}
  .val.high {{ color: var(--green); font-weight: 700; }}
  .val.mid {{ color: var(--orange); font-weight: 600; }}
  .rsi.oversold {{ color: var(--green); font-weight: 700; }}
  .rsi.overbought {{ color: var(--red); }}
  .mix.good {{ color: var(--green); font-weight: 700; }}
  .mix.ok {{ color: var(--text); }}
  .cash {{ color: var(--accent); }}

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

  /* Signals & Catalyst */
  .signals {{
    font-size: 11px;
    color: var(--orange);
    font-weight: 500;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
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
  <div class="date">{display_date} | {n} candidates</div>
</div>

<div class="summary">
  <div class="card">
    <div class="label">候補銘柄数</div>
    <div class="value accent">{n}</div>
  </div>
  <div class="card">
    <div class="label">最高スコア</div>
    <div class="value green">{top_score:.3f}</div>
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
    <div class="label">平均 Tech</div>
    <div class="value">{avg_tech:.2f}</div>
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
  <label>MIX上限</label>
  <input type="range" id="mixSlider" min="5" max="40" step="0.5" value="30"
    oninput="document.getElementById('mixVal').textContent=this.value; applyFilters()">
  <span id="mixVal">30</span>

  <label>PBR上限</label>
  <input type="range" id="pbrSlider" min="0.1" max="3.0" step="0.05" value="2.5"
    oninput="document.getElementById('pbrVal').textContent=this.value; applyFilters()">
  <span id="pbrVal">2.5</span>

  <label>Tech下限</label>
  <input type="range" id="techSlider" min="0" max="1" step="0.05" value="0"
    oninput="document.getElementById('techVal').textContent=this.value; applyFilters()">
  <span id="techVal">0</span>

  <label>時価総額</label>
  <select id="mcapFilter" onchange="applyFilters()" style="padding:5px 8px">
    <option value="0">全て</option>
    <option value="50">50億+</option>
    <option value="100">100億+</option>
    <option value="500">500億+</option>
    <option value="1000">1000億+</option>
    <option value="5000">5000億+</option>
    <option value="-500">500億以下</option>
    <option value="-100">100億以下</option>
  </select>

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
  <th onclick="sortTable(3)">株価</th>
  <th onclick="sortTable(4)">PER</th>
  <th onclick="sortTable(5)">fPER</th>
  <th onclick="sortTable(6)">PBR</th>
  <th onclick="sortTable(7)">MIX</th>
  <th onclick="sortTable(8)">時価総額(億)</th>
  <th onclick="sortTable(9)">現金比率%</th>
  <th onclick="sortTable(10)">RSI</th>
  <th onclick="sortTable(11)">Tech</th>
  <th onclick="sortTable(12)">出来高比</th>
  <th onclick="sortTable(13)">Value</th>
  <th onclick="sortTable(14)">総合</th>
  <th onclick="sortTable(15)">売買代金(億)</th>
  <th>推移</th>
  <th>シグナル</th>
  <th>決算予定</th>
</tr>
</thead>
<tbody id="tableBody">
{table_body}
</tbody>
</table>
</div>

<div class="legend">
  <h3>スコアリング方法</h3>
  <span class="formula">MIX = PER x PBR</span> &mdash; Benjamin Graham 基準 &lt; 22.5<br>
  <span class="formula">Tech = RSI反転(25%) + MACDクロス(25%) + MA転換(25%) + 底値反発(25%)</span><br>
  <span class="formula">総合 = Value(40%) + Tech(40%) + 出来高(15%) + カタリスト(5%)</span><br>
  <br>
  fPER = 予想PER &nbsp;|&nbsp; 現金比率 = 総現預金 / 時価総額 x 100 &nbsp;|&nbsp;
  出来高比 = 直近5日平均 / 20日平均 &nbsp;|&nbsp; 売買代金推移 = 直近10営業日
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

function applyFilters() {{
  const mixMax = parseFloat(document.getElementById('mixSlider').value);
  const pbrMax = parseFloat(document.getElementById('pbrSlider').value);
  const techMin = parseFloat(document.getElementById('techSlider').value);
  const mcapVal = parseInt(document.getElementById('mcapFilter').value);
  const earnVal = document.getElementById('earnFilter').value;
  document.querySelectorAll('#tableBody tr').forEach(row => {{
    const sector = row.dataset.sector;
    const ed = parseInt(row.dataset.earnDays);
    const mcap = parseFloat(row.dataset.mcap);
    const c = row.cells;
    const g = t => {{ const v = c[t].textContent.trim(); return v === '-' ? NaN : parseFloat(v); }};
    const mix = g(7), pbr = g(6), tech = g(11);
    let show = true;
    if (activeSector && sector !== activeSector) show = false;
    if (!isNaN(mix) && mix > mixMax) show = false;
    if (!isNaN(pbr) && pbr > pbrMax) show = false;
    if (!isNaN(tech) && tech < techMin) show = false;
    if (mcapVal > 0 && mcap < mcapVal) show = false;
    if (mcapVal < 0 && mcap > Math.abs(mcapVal)) show = false;
    if (earnVal === 'none' && ed >= 999) show = false;
    else if (earnVal !== 'all' && earnVal !== 'none') {{
      const maxDays = parseInt(earnVal);
      if (ed > maxDays || ed < 0) show = false;
    }}
    row.style.display = show ? '' : 'none';
  }});
}}

let sortCol = -1, sortAsc = true;
function sortTable(col) {{
  const tb = document.getElementById('tableBody');
  const rows = Array.from(tb.rows);
  if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = false; }}
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent.trim();
    let vb = b.cells[col].textContent.trim();
    if (va === '-') va = sortAsc ? '99999' : '-99999';
    if (vb === '-') vb = sortAsc ? '99999' : '-99999';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return sortAsc ? na - nb : nb - na;
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tb.appendChild(r));
}}
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

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
