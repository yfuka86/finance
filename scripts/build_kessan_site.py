#!/usr/bin/env python3
"""決算発表予定ダッシュボード (kessan.a-tokyo.jp) の静的サイトを生成する。

デザインは work/tc の cards.a-tokyo.jp と同一のシェル（CSS変数・ダーク対応・topnav）。

データ:
  1. 本日の確定発表: J-Quants /v2/equities/earnings-calendar（当日分のみ返るAPI）。
     取得のたびに data/jp_earnings_calendar/ へ append-only 保存＝実績履歴が今日から貯まる
  2. 予想発表日: fins の前年同期実績 +364日→翌営業日（実測精度: 中央値誤差1日・±5日92.4%）
出力: data/kessan_site/index.html
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "data/kessan_site"
SNAP = ROOT / "data/jp_earnings_calendar"
WINDOW_DAYS = 45

SHELL = '''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 64 64%22><rect width=%2264%22 height=%2264%22 rx=%2213%22 fill=%22%23151a24%22/><rect x=%2212%22 y=%2214%22 width=%2240%22 height=%2236%22 rx=%225%22 fill=%22%23f7f8fb%22/><rect x=%2212%22 y=%2214%22 width=%2240%22 height=%2210%22 rx=%225%22 fill=%22%233b5bdb%22/><rect x=%2218%22 y=%2230%22 width=%228%22 height=%2214%22 fill=%22%230f9d58%22/><rect x=%2230%22 y=%2234%22 width=%228%22 height=%2210%22 fill=%22%233b5bdb%22/><rect x=%2242%22 y=%2226%22 width=%224%22 height=%2218%22 fill=%22%23c9302c%22/></svg>">
<title>__TITLE__ ｜ A-Tokyo Kessan</title>
<style>
:root{--bg:#f5f6fa;--panel:#fff;--line:#e4e7ef;--ink:#161a22;--ink2:#4c5566;--ink3:#7b8496;--accent:#3b5bdb;
--shadow:0 1px 3px rgba(18,24,40,.06);
--sans:"Hiragino Kaku Gothic ProN","Hiragino Sans","Yu Gothic",Meiryo,system-ui,-apple-system,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1017;--panel:#161a24;--line:#252b39;--ink:#eef1f7;--ink2:#a7b0c2;--ink3:#727c92;--accent:#7d96ff;}}
:root[data-theme=light]{--bg:#f5f6fa;--panel:#fff;--line:#e4e7ef;--ink:#161a22;--ink2:#4c5566;--ink3:#7b8496;--accent:#3b5bdb;}
:root[data-theme=dark]{--bg:#0e1017;--panel:#161a24;--line:#252b39;--ink:#eef1f7;--ink2:#a7b0c2;--ink3:#727c92;--accent:#7d96ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased}
.topnav{position:sticky;top:0;z-index:50;background:var(--panel);border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:4px;padding:0 16px;height:52px;overflow-x:auto;white-space:nowrap;
box-shadow:0 1px 3px rgba(18,24,40,.06)}
.brand{font-weight:900;font-size:15px;letter-spacing:-.01em;margin-right:14px;flex:none;color:var(--ink);text-decoration:none}
.brand span{color:var(--accent)}
.topnav a.nl{font-size:13.5px;font-weight:700;color:var(--ink2);text-decoration:none;padding:8px 13px;border-radius:9px;flex:none}
.topnav a.nl:hover{background:var(--bg);color:var(--ink)}
.topnav a.nl[aria-current=page]{background:var(--accent);color:#fff}
.topnav .sp{margin-left:auto;flex:none}
.topnav .stamp{font-size:11.5px;color:var(--ink3);font-weight:700;flex:none;padding-left:10px}
.dw{max-width:1120px;margin:0 auto;padding:22px 16px 60px}
.dw h1{font-size:22px;margin:0;font-weight:800}
.dw h1 .s{display:block;font-size:12.5px;font-weight:600;color:var(--ink2);margin-top:6px}
.dgrp{margin:22px 0 0}
.dgrp>h2{font-size:13px;font-weight:800;color:var(--ink3);margin:0 0 9px;letter-spacing:.04em}
.dcards{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:12px}
.dcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 15px;
box-shadow:var(--shadow);position:relative}
.dcard .nm{font-size:13.5px;font-weight:800;margin-bottom:3px}
.dcard .n{font-size:23px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.2}
.dcard .n span{font-size:12px;font-weight:700;color:var(--ink3);margin-left:3px}
.dcard .note{font-size:11.5px;color:var(--ink2);margin-top:5px;line-height:1.6}
.bars{display:flex;align-items:flex-end;gap:3px;height:88px;padding:8px 2px 0}
.bars .b{flex:1;min-width:7px;background:var(--accent);opacity:.75;border-radius:3px 3px 0 0;position:relative}
.bars .b.we{background:var(--ink3);opacity:.25}
.bars .b .t{position:absolute;top:-17px;left:50%;transform:translateX(-50%);font-size:9.5px;color:var(--ink3);font-weight:700}
.blab{display:flex;gap:3px;padding:4px 2px 0}
.blab span{flex:1;min-width:7px;text-align:center;font-size:9px;color:var(--ink3);font-weight:700}
table.k{width:100%;border-collapse:separate;border-spacing:0;background:var(--panel);
border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);font-size:12.5px}
table.k th{background:var(--panel);
background-image:linear-gradient(rgba(127,127,127,.09),rgba(127,127,127,.09));
color:var(--ink2);font-size:11px;letter-spacing:.03em;font-weight:800;
text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
position:sticky;top:52px;z-index:5}
table.k thead th:first-child{border-top-left-radius:13px}
table.k thead th:last-child{border-top-right-radius:13px}
table.k tr:last-child td:first-child{border-bottom-left-radius:13px}
table.k tr:last-child td:last-child{border-bottom-right-radius:13px}
table.k td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
table.k tr:last-child td{border-bottom:none}
table.k td.num{font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:10px;font-weight:800;border-radius:6px;padding:2px 7px}
.badge.fix{background:rgba(15,157,88,.14);color:#0f9d58}
.badge.est{background:rgba(59,91,219,.12);color:var(--accent)}
.badge.done{background:rgba(127,127,127,.15);color:var(--ink3)}
.q{font-size:10.5px;font-weight:800;color:var(--ink3)}
.search{width:100%;max-width:420px;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:9px 13px;font-size:13px;color:var(--ink);margin-bottom:10px;font-family:var(--sans)}
.search:focus{outline:2px solid var(--accent);border-color:transparent}
.dnote{font-size:12px;color:var(--ink3);margin-top:22px;line-height:1.7}
.dayhead td{background:rgba(127,127,127,.055);font-weight:800;font-size:12px;color:var(--ink2)}
</style>
</head>
<body>
<nav class="topnav">
<a class="brand" href="/">A-Tokyo <span>Kessan</span></a>
<a class="nl" href="/" aria-current="page">決算予定</a>
<span class="sp"></span>
<span class="stamp">更新 __STAMP__</span>
</nav>
__BODY__
<script>
const q=document.getElementById('q');
if(q){q.addEventListener('input',()=>{const v=q.value.trim().toLowerCase();
document.querySelectorAll('tr.r').forEach(tr=>{tr.style.display=!v||tr.dataset.k.includes(v)?'':'none'});
document.querySelectorAll('tr.dayhead').forEach(h=>{let n=h.nextElementSibling,any=false;
while(n&&!n.classList.contains('dayhead')){if(n.style.display!=='none')any=true;n=n.nextElementSibling}
h.style.display=any?'':'none'});});}
</script>
</body>
</html>'''


def jp_business_days() -> pd.DatetimeIndex:
    cal = pd.read_parquet(ROOT / "data/fx_rates/jp_market_calendar.parquet")
    ok = cal[cal["HolDiv"].astype(str).isin(("1", "3"))]
    return pd.DatetimeIndex(sorted(pd.to_datetime(ok["Date"]).unique()))


def fetch_today_confirmed() -> pd.DataFrame:
    """当日の確定分を取得し、append-only スナップショットに保存する。"""
    import requests
    from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE
    r = requests.get(f"{JQUANTS_BASE}/v2/equities/earnings-calendar",
                     headers={"x-api-key": JQUANTS_API_KEY}, timeout=30)
    rows = r.json().get("data", []) if r.status_code == 200 else []
    d = pd.DataFrame(rows)
    if len(d):
        SNAP.mkdir(parents=True, exist_ok=True)
        day = str(d["Date"].iloc[0])
        out = SNAP / f"{day}.json"
        if not out.exists():
            out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return d


def expected() -> pd.DataFrame:
    """前年同期実績 +364日 → 翌営業日。今後 WINDOW_DAYS 日ぶん。"""
    from scripts.experiment_earnings_timing import announcements
    from trading.jp_intraday.daily_model import load_master
    ann = announcements()                                  # symbol(4桁), q, fy, date
    sessions = jp_business_days()
    a = ann.sort_values(["symbol", "q", "fy"]).groupby(["symbol", "q"]).tail(1)
    raw = a["date"] + pd.Timedelta(days=364)
    idx = sessions.searchsorted(raw, side="left")
    ok = idx < len(sessions)
    a = a[ok].copy()
    a["expected"] = sessions[idx[ok]]
    today = pd.Timestamp(dt.date.today())
    a = a[a["expected"].between(today, today + pd.Timedelta(days=WINDOW_DAYS))]
    m = load_master()
    m = m.assign(sym4=m["symbol"].astype(str).str[:4]).drop_duplicates("sym4")
    a = a.merge(m[["sym4", "name", "sector", "market"]],
                left_on="symbol", right_on="sym4", how="left")
    return a.sort_values(["expected", "symbol"])


def build() -> None:
    today = pd.Timestamp(dt.date.today())
    conf = fetch_today_confirmed()
    exp = expected()
    conf_syms = set(conf["Code"].astype(str).str[:4]) if len(conf) else set()

    # サマリカード
    wk = exp[exp["expected"] <= today + pd.Timedelta(days=7)]
    fins_max = exp["date"].max()
    cards = f'''
<div class="dgrp"><h2>サマリ</h2><div class="dcards">
<div class="dcard"><div class="nm">本日の発表（確定）</div><div class="n">{len(conf)}<span>件</span></div>
<div class="note">J-Quants 決算発表予定（当日分）。取得毎に履歴へ保存</div></div>
<div class="dcard"><div class="nm">今後7日の予想</div><div class="n">{len(wk)}<span>件</span></div>
<div class="note">前年同期の実績発表日 +364日 → 翌営業日</div></div>
<div class="dcard"><div class="nm">今後{WINDOW_DAYS}日の予想</div><div class="n">{len(exp)}<span>件</span></div>
<div class="note">対象 {exp["symbol"].nunique():,} 銘柄</div></div>
<div class="dcard"><div class="nm">予想モデルの実測精度</div><div class="n">±1<span>日（中央値）</span></div>
<div class="note">実績との差: 中央値1日・92.4%が±5日以内（2022-2026・42,263件で検証）</div></div>
</div></div>'''

    # 日別バー
    days = pd.date_range(today, today + pd.Timedelta(days=WINDOW_DAYS - 1))
    cnt = exp.groupby(exp["expected"].dt.normalize()).size().reindex(days).fillna(0).astype(int)
    mx = max(1, cnt.max())
    bars = "".join(
        f'<div class="b{" we" if d.weekday() >= 5 else ""}" style="height:{max(3, int(84 * c / mx))}px">'
        + (f'<span class="t">{c}</span>' if c and c >= mx * .5 else "") + "</div>"
        for d, c in cnt.items())
    labs = "".join(f'<span>{d.day if (d.day % 5 == 0 or i == 0) else ""}</span>'
                   for i, d in enumerate(days))
    chart = f'''
<div class="dgrp"><h2>日別の予想件数（今後{WINDOW_DAYS}日）</h2>
<div class="dcard" style="padding:18px 16px 12px"><div class="bars">{bars}</div>
<div class="blab">{labs}</div></div></div>'''

    # テーブル
    rows = []
    for day, g in exp.groupby(exp["expected"].dt.normalize()):
        wd = "月火水木金土日"[day.weekday()]
        rows.append(f'<tr class="dayhead"><td colspan="6">{day.strftime("%-m/%-d")}（{wd}）'
                    f'　<span class="q">{len(g)}件</span></td></tr>')
        for _, r in g.iterrows():
            def _s(x):
                return "" if pd.isna(x) else str(x)
            nm, sec = _s(r.get("name")), _s(r.get("sector"))
            mkt = _s(r.get("market")).replace("市場", "")
            badge = ('<span class="badge fix">本日確定</span>' if r["symbol"] in conf_syms
                     and day == today else '<span class="badge est">予想</span>')
            key = f'{r["symbol"]} {nm} {sec}'.lower()
            rows.append(
                f'<tr class="r" data-k="{key}"><td class="num"><b>{r["symbol"]}</b></td>'
                f'<td>{nm}</td><td>{sec}</td><td>{mkt}</td>'
                f'<td><span class="q">{r["q"]}</span>　昨年 {r["date"].strftime("%-m/%-d")}</td>'
                f'<td>{badge}</td></tr>')
    table = f'''
<div class="dgrp"><h2>予定一覧</h2>
<input id="q" class="search" placeholder="コード・社名・業種で絞り込み…">
<table class="k"><thead><tr><th>コード</th><th>社名</th><th>業種</th><th>市場</th>
<th>四半期・昨年実績</th><th>状態</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'''

    note = f'''<div class="dnote">予想 = 各銘柄の前年同期の実績発表日 +364日を翌営業日に丸めたもの。
確定情報は J-Quants 決算発表予定（当日分のみ提供）で毎朝上書きされ、スナップショットは
data/jp_earnings_calendar/ に append-only 保存。fins 実績の最終日: {fins_max.strftime("%Y-%m-%d")}。
研究メモ: 発表前ドリフト戦略は利益集中で棄却済み（詳細は研究台帳）。本画面は観測用。</div>'''

    body = f'<div class="dw"><h1>決算発表予定<span class="s">前年実績ベースの予想 ＋ J-Quants当日確定分</span></h1>{cards}{chart}{table}{note}</div>'
    SITE.mkdir(parents=True, exist_ok=True)
    html = (SHELL.replace("__TITLE__", "決算発表予定")
                 .replace("__STAMP__", dt.datetime.now().strftime("%-m/%-d %H:%M"))
                 .replace("__BODY__", body))
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"built: {SITE/'index.html'} ({len(html):,} bytes) / 確定{len(conf)}件 予想{len(exp)}件")


if __name__ == "__main__":
    build()
