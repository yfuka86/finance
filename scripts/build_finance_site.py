#!/usr/bin/env python3
"""ファイナンス統合ダッシュボード (fin.a-tokyo.jp) の静的サイト生成。

work/tc の cards.a-tokyo.jp と同じ思想: 各ページは本文HTML断片を作って PAGES に足すだけ。
ページ: 研究台帳(index) / 決算予定 / フォワード監視 / データ資産
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "data/kessan_site"

# 共通シェル（tc と同一のデザイントークン）。__NAV__/__TITLE__/__STAMP__/__BODY__ を差し込む。
from scripts.build_kessan_site import SHELL as _KSHELL  # noqa: E402

SHELL = _KSHELL.replace(
    '<a class="brand" href="/">A-Tokyo <span>Kessan</span></a>\n'
    '<a class="nl" href="/" aria-current="page">決算予定</a>',
    '<a class="brand" href="/">A-Tokyo <span>Fin</span></a>\n__NAV__')

# 検索 × 時価総額しきい値の合成フィルタ（時価総額ボタンが無いページでは検索のみ）
_OLD_JS = SHELL[SHELL.index('<script>'):SHELL.index('</script>') + 9]
SHELL = SHELL.replace(_OLD_JS, '''<script>
const q=document.getElementById('q');
let MC=+(document.querySelector('.mcbtn[aria-pressed="true"]')?.dataset.mc||0);
function applyF(){const v=q?q.value.trim().toLowerCase():'';
document.querySelectorAll('tr.r').forEach(tr=>{
  const okq=!v||tr.dataset.k.includes(v);
  const okm=!MC||(+(tr.dataset.mc||0)>=MC);
  tr.style.display=okq&&okm?'':'none'});
document.querySelectorAll('tr.dayhead').forEach(h=>{let n=h.nextElementSibling,c=0;
  while(n&&!n.classList.contains('dayhead')){if(n.style.display!=='none')c++;n=n.nextElementSibling}
  h.style.display=c?'':'none';
  const b=h.querySelector('.q'); if(b&&b.dataset.t) b.textContent=c+'件';});}
if(q)q.addEventListener('input',applyF);
document.querySelectorAll('.mcbtn').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.mcbtn').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');MC=+b.dataset.mc;applyF();}));
applyF();
</script>''')


def badge(status: str) -> str:
    s = str(status)
    if "SEALED" in s or "PENDING" in s:
        cls, txt = "est", s
    elif "FORWARD" in s:
        cls, txt = "fix", s
    elif "INVALIDATED" in s:
        cls, txt = "done", s
    elif "DIAGNOSTIC" in s or "BLOCKED" in s:
        cls, txt = "done", s
    else:
        cls, txt = "done", s
    return f'<span class="badge {cls}">{txt}</span>'


def page_registry() -> str:
    from trading.jp_intraday.research_registry import research_rows
    df = research_rows()
    t = df[df["OOS Sharpe"].notna()]
    n_fwd = int(df["状態"].astype(str).str.contains("FORWARD|SEALED|PENDING").sum())
    cards = f'''
<div class="dgrp"><h2>サマリ</h2><div class="dcards">
<div class="dcard"><div class="nm">記録戦略</div><div class="n">{len(df)}<span>本</span></div>
<div class="note">事前登録→1回検証→記録、の累積</div></div>
<div class="dcard"><div class="nm">実弾GO</div><div class="n">0<span>本</span></div>
<div class="note">OOS規律を通過して実弾承認された戦略はまだ無い</div></div>
<div class="dcard"><div class="nm">Sharpe ≥ 1 の記録</div><div class="n">{int(t["OOS Sharpe"].ge(1).sum())}<span>本</span></div>
<div class="note">数値はあるが全て条件付き（PAPER/実測待ち/棄却済み）</div></div>
<div class="dcard"><div class="nm">フォワード/封印 監視中</div><div class="n">{n_fwd}<span>本</span></div>
<div class="note">判定日まで成績を開かない運用</div></div>
</div></div>'''
    rows = []
    for _, r in df.iterrows():
        sh = "" if pd.isna(r["OOS Sharpe"]) else f'{r["OOS Sharpe"]:.2f}'
        ann = "" if pd.isna(r.get("年率%")) else f'{r["年率%"]:.1f}%'
        dd = "" if pd.isna(r.get("最大DD%")) else f'{r["最大DD%"]:.1f}%'
        n = "" if pd.isna(r.get("案件/日数")) else f'{int(r["案件/日数"])}'
        note = str(r.get("注記") or "")
        key = f'{r["戦略"]} {r["ファミリー"]} {r["状態"]}'.lower()
        rows.append(
            f'<tr class="r" data-k="{key}"><td><b>{r["戦略"]}</b></td>'
            f'<td><span class="q">{r["ファミリー"]}</span></td>'
            f'<td>{badge(r["状態"])}</td>'
            f'<td class="num">{sh}</td><td class="num">{ann}</td><td class="num">{dd}</td>'
            f'<td class="num">{n}</td><td style="font-size:11.5px;color:var(--ink2)">{note}</td></tr>')
    return f'''<div class="dw">
<h1>研究台帳<span class="s">OOS・フォワード結果のみ（IS成績は載せない）</span></h1>
{cards}
<div class="dgrp"><h2>全記録</h2>
<input id="q" class="search" placeholder="戦略・ファミリー・状態で絞り込み…">
<table class="k"><thead><tr><th>戦略</th><th>ファミリー</th><th>状態</th><th>OOS Sh</th>
<th>年率</th><th>最大DD</th><th>n</th><th>注記</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<div class="dnote">正本: trading/jp_intraday/research_registry.py（結果ファイルから毎回再構築）。
確定知見の詳細は AGENTS.md。</div></div>'''


def page_forward() -> str:
    items = []
    # ゴトー封印
    scans = ROOT / "data/fx_ticks_fix/forward/scanned.jsonl"
    n_scan = len(scans.read_text().splitlines()) if scans.exists() else 0
    n_parts = len(list((ROOT / "data/fx_ticks_fix/forward").glob("*.parquet")))
    left = (dt.date(2027, 8, 6) - dt.date.today()).days
    items.append(("ゴトー日ティック形（USDJPY 9:00→9:55）", "SEALED",
                  f"判定日 2027-08-06（あと{left}日）",
                  f"封印: 2020-2026は未閲覧のまま。フォワード収集 {n_scan}日 / {n_parts}パーツ"
                  "（launchd 毎日12:00・no-peek＝生ティックのみ保存）。"
                  "選択窓 Sh0.856・Sharpe以外の全基準通過→約560取引で最終判定。落ちたら恒久クローズ"))
    # 気配不要ML v2
    left2 = (dt.date(2027, 8, 2) - dt.date.today()).days
    items.append(("気配不要ML v2（凍結フォワード）", "FROZEN",
                  f"主評価日 2027-08-02（あと{left2}日）",
                  "PnLは判定日まで開かない。※選択窓基準値1.43は再現不能と判明済み"
                  "（再現値0.43-0.46）。判定前に基準の扱いを要決定"))
    # TOPIX2026
    rd = ROOT / "data/topix_2026_forward/readiness_20260801.json"
    if rd.exists():
        r = json.loads(rd.read_text())
        items.append(("TOPIX 2026 第2段階見直し", "INPUT BLOCKED",
                      "基準日 2026-08-31 / 公表 10月第5営業日",
                      f"公式FFW未入手（J-QuantsにAPI無し・個人経路はDataCube有料のみ）。"
                      f"weight {r.get('current_weight_asof','—')} / raw収集は継続"))
    # 自社株買いフォワード
    fw = list((ROOT / "data/jp_buybacks/forward").glob("*.parquet"))
    items.append(("自社株買い実行圧力 v3.3 / 企業Put v1", "FORWARD",
                  "2026-08-03 観測開始・主評価 2027-08-03",
                  f"実行適格シグナル保存 {len(fw)}ファイル。閾値はリターン不参照で凍結済み"))
    # 決算カレンダー蓄積
    snaps = len(list((ROOT / "data/jp_earnings_calendar").glob("*.json")))
    items.append(("決算発表予定スナップショット", "ACCUMULATING",
                  "毎朝07:40 自動取得",
                  f"当日確定分の append-only 保存 {snaps}日分（APIは当日分しか返さないため、"
                  "履歴はここで自作している）"))
    cards = "".join(
        f'<div class="dcard"><div class="nm">{n}　{badge(s)}</div>'
        f'<div class="note" style="font-size:12px;font-weight:700;color:var(--ink)">{w}</div>'
        f'<div class="note">{d}</div></div>'
        for n, s, w, d in items)
    return f'''<div class="dw">
<h1>フォワード監視<span class="s">判定日まで成績を開かない案件の状態だけを見る</span></h1>
<div class="dgrp"><div class="dcards" style="grid-template-columns:repeat(auto-fill,minmax(340px,1fr))">{cards}</div></div>
<div class="dnote">規律: 封印・凍結案件のPnL/勝率など成績に類する数値は判定日まで一切計算・表示しない。
ここに出すのはデータ完全性（収集日数・欠損）のみ。</div></div>'''


def page_data() -> str:
    def stat(pat, label, cmd, note=""):
        fs = glob.glob(str(ROOT / pat))
        if not fs:
            return {"label": label, "n": 0, "size": 0, "ts": None, "cmd": cmd, "note": note}
        size = sum(os.path.getsize(f) for f in fs)
        ts = max(os.path.getmtime(f) for f in fs)
        return {"label": label, "n": len(fs), "size": size, "ts": ts, "cmd": cmd, "note": note}

    groups = [
        ("日本株", [
            stat("data/jp_daily_history/daily_adj_*.parquet", "日次バー(調整+生値) 2018-2026",
                 "collect_jp_daily_history.py", "生値O/Cは全年95-97%に補完済み"),
            stat("data/jp_derivatives/sector_indices_2008_2026.parquet", "業種指数33本 2008-2026",
                 "collect_jp_sector_indices.py", ""),
            stat("data/jp_derivatives/topix_index_2008_2026.parquet", "TOPIX指数 2008-2026", "", ""),
            stat("data/jp_derivatives/futures_*.parquet", "先物(NK/TOPIX/VI/ダウ) 2018-2026",
                 "collect_jp_derivatives.py", ""),
            stat("data/jp_options/opt225_*.parquet", "日経225オプション清算値 2019-2026",
                 "", "bid/askなし=PAPER ONLY用途"),
            stat("data/jp_ownership/filings.jsonl", "有報 大株主 2016-2026",
                 "collect_edinet_major_shareholders.py", "44,382件・4,589銘柄"),
            stat("data/jp_buybacks/edinet/documents.jsonl", "自社株買い開示(EDINET)",
                 "collect_edinet_buybacks.py", ""),
            stat("data/jp_earnings_calendar/*.json", "決算予定スナップショット",
                 "build_finance_site.py(毎朝)", "当日分の履歴を自作蓄積"),
        ]),
        ("FX", [
            stat("data/fx_oanda_min/parts/*.parquet", "分足 7ペア 2011-2026 (OANDA)",
                 "collect_oanda_fx_minute.py", "3,872万行・bid/ask・863MB"),
            stat("data/fx_dukascopy/*_day.parquet", "日足 10ペア 2011-2025 (Dukascopy)",
                 "collect_dukascopy_fx.py", "クロス3ペア含む・実測スプレッド"),
            stat("data/fx_dukascopy_hour/parts/*.parquet", "時間足 USDJPY/EURUSD 2011-2026",
                 "collect_dukascopy_fx_hourly.py", ""),
            stat("data/fx_ticks_fix/*.parquet", "ゴトー日ティック 2011-2026",
                 "collect_dukascopy_fix_ticks.py", "1,080万ティック・封印判定用"),
            stat("data/fx_ticks_fix/forward/*.parquet", "ゴトー・フォワードティック",
                 "collect_gotobi_forward.py(毎日12:00)", "no-peek"),
            stat("data/fx_rates/short_rates_monthly.parquet", "3Mインターバンク金利 8通貨",
                 "collect_fred_rates.py", "スワップ推定の正本"),
            stat("data/fx_ecb/eurofxref_hist.parquet", "ECB仲値 1999-2026", "collect_ecb_fx.py", ""),
        ]),
        ("米国", [
            stat("data/fx_oanda_us/parts/*_D.parquet", "米指数CFD日足 2005-2026",
                 "collect_oanda_fx_minute.py --root data/fx_oanda_us --granularity D",
                 "SPX500/NAS100/US30/US2000・bid/ask"),
            stat("data/fx_oanda_us/parts/*[0-9].parquet", "米指数CFD分足 2011-2026",
                 "collect_oanda_fx_minute.py --root data/fx_oanda_us", ""),
            stat("data/raw/us_cc_returns.csv", "米セクターETF 2010-2025",
                 "(Stooq・現在IPブロック中)", "更新停止中"),
        ]),
    ]
    now = dt.datetime.now().timestamp()
    out = []
    for gname, items in groups:
        cards = []
        for it in items:
            if it["ts"]:
                age = (now - it["ts"]) / 86400
                dot = "fresh" if age < 3 else ("stale" if age < 21 else "old")
                when = dt.datetime.fromtimestamp(it["ts"]).strftime("%-m/%-d %H:%M")
            else:
                dot, when = "old", "なし"
            sz = it["size"] / 1e6
            szs = f"{sz/1000:.1f}GB" if sz > 1000 else f"{sz:.0f}MB"
            cmd = (f'<div class="note"><span class="dcmd">{it["cmd"]}</span></div>'
                   if it.get("cmd") else "")
            cards.append(
                f'<div class="dcard"><div class="nm">{it["label"]}</div>'
                f'<div class="n">{it["n"]}<span>ファイル・{szs}</span></div>'
                f'<div class="note"><span class="dot {dot}"></span>{when}　{it.get("note","")}</div>{cmd}</div>')
        out.append(f'<div class="dgrp"><h2>{gname}</h2><div class="dcards">{"".join(cards)}</div></div>')
    extra = '''<style>.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle}
.fresh{background:#22a05a}.stale{background:#e0a020}.old{background:#d0454c}
.dcmd{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;color:var(--ink3);
background:rgba(127,127,127,.09);border-radius:5px;padding:2px 6px}</style>'''
    return (f'<div class="dw"><h1>データ資産<span class="s">収集済みデータの鮮度と再取得コマンド</span></h1>'
            f'{extra}{"".join(out)}'
            f'<div class="dnote">鮮度: 緑&lt;3日 / 黄&lt;21日 / 赤=停滞。巨大な生データはgit外'
            f'（コレクタで再生成可能）。</div></div>')


def _mktcap_map() -> dict:
    """パネルの分割補正済み時価総額（億円）。symbol4桁 -> 億円。"""
    from trading.jp_intraday.daily_model import load_panel_cached
    p = load_panel_cached(min_value_yen=5e8)
    last = p.sort_values("date").groupby("sym4").tail(1)
    return {str(r["sym4"]): float(r["mktcap_yen"]) / 1e8
            for _, r in last.iterrows() if pd.notna(r.get("mktcap_yen"))}


def _fmt_oku(v: float | None) -> str:
    if not v:
        return "—"
    return f"{v/1e4:.2f}兆" if v >= 1e4 else f"{v:,.0f}億"


def page_kessan() -> str:
    import scripts.build_kessan_site as K
    today = pd.Timestamp(dt.date.today())
    conf = K.fetch_today_confirmed()
    exp = K.expected()
    mc = _mktcap_map()
    exp["mc_oku"] = exp["symbol"].map(mc)
    sched_date = pd.to_datetime(conf["Date"].iloc[0]) if len(conf) else None
    conf_syms = set(conf["Code"].astype(str).str[:4]) if len(conf) else set()
    wk = exp[exp["expected"] <= today + pd.Timedelta(days=7)]
    # 公式予定 vs 自前予想モデルの一致率（公式予定日の±5営業日以内に予想があるか）
    hit = 0
    if sched_date is not None and conf_syms:
        near = exp[(exp["expected"] - sched_date).abs() <= pd.Timedelta(days=7)]
        hit = len(conf_syms & set(near["symbol"]))
    sched_lab = sched_date.strftime("%-m/%-d") if sched_date is not None else "—"
    cards = f'''
<div class="dgrp"><h2>サマリ</h2><div class="dcards">
<div class="dcard"><div class="nm">公式予定（{sched_lab} 発表分）</div><div class="n">{len(conf)}<span>件</span></div>
<div class="note">J-Quants 翌営業日分（夕方更新・履歴なし）→ 毎朝append-only保存で履歴を自作。
自前予想と±5営業日一致 {hit}/{len(conf_syms) or 1}件</div></div>
<div class="dcard"><div class="nm">今後7日の予想</div><div class="n">{len(wk)}<span>件</span></div>
<div class="note">前年同期実績 +364日 → 翌営業日</div></div>
<div class="dcard"><div class="nm">今後{K.WINDOW_DAYS}日の予想</div><div class="n">{len(exp)}<span>件</span></div>
<div class="note">対象 {exp["symbol"].nunique():,} 銘柄</div></div>
<div class="dcard"><div class="nm">予想の実測精度</div><div class="n">±1<span>日（中央値）</span></div>
<div class="note">92.4%が±5日以内（42,263件で検証）</div></div>
</div></div>'''
    days = pd.date_range(today, today + pd.Timedelta(days=K.WINDOW_DAYS - 1))
    cnt = exp.groupby(exp["expected"].dt.normalize()).size().reindex(days).fillna(0).astype(int)
    mx = max(1, cnt.max())
    bars = "".join(
        f'<div class="b{" we" if d.weekday() >= 5 else ""}" style="height:{max(3, int(84 * c / mx))}px">'
        + (f'<span class="t">{c}</span>' if c and c >= mx * .5 else "") + "</div>"
        for d, c in cnt.items())
    labs = "".join(f'<span>{d.day if (d.day % 5 == 0 or i == 0) else ""}</span>'
                   for i, d in enumerate(days))
    chart = (f'<div class="dgrp"><h2>日別の予想件数（今後{K.WINDOW_DAYS}日）</h2>'
             f'<div class="dcard" style="padding:18px 16px 12px"><div class="bars">{bars}</div>'
             f'<div class="blab">{labs}</div></div></div>')
    rows = []
    for day, g in exp.groupby(exp["expected"].dt.normalize()):
        wd = "月火水木金土日"[day.weekday()]
        g = g.sort_values("mc_oku", ascending=False, na_position="last")
        rows.append(f'<tr class="dayhead"><td colspan="7">{day.strftime("%-m/%-d")}（{wd}）'
                    f'　<span class="q" data-t="1">{len(g)}件</span></td></tr>')
        for _, r in g.iterrows():
            def _s(x):
                return "" if pd.isna(x) else str(x)
            nm, sec = _s(r.get("name")), _s(r.get("sector"))
            mkt = _s(r.get("market")).replace("市場", "")
            b = ('<span class="badge fix">公式確定</span>'
                 if sched_date is not None and r["symbol"] in conf_syms
                 and day == sched_date.normalize()
                 else '<span class="badge est">予想</span>')
            key = f'{r["symbol"]} {nm} {sec}'.lower()
            mco = r.get("mc_oku")
            mcv = 0 if pd.isna(mco) else int(mco)
            rows.append(
                f'<tr class="r" data-k="{key}" data-mc="{mcv}">'
                f'<td class="num"><b>{r["symbol"]}</b></td>'
                f'<td>{nm}</td><td class="num">{_fmt_oku(mco if pd.notna(mco) else None)}</td>'
                f'<td>{sec}</td><td>{mkt}</td>'
                f'<td><span class="q">{r["q"]}</span>　昨年 {r["date"].strftime("%-m/%-d")}</td>'
                f'<td>{b}</td></tr>')
    mcbtns = "".join(
        f'<button class="mcbtn" data-mc="{v}" aria-pressed="{str(v==3000).lower()}">{lab}</button>'
        for v, lab in [(0, "全て"), (1000, "1000億+"), (3000, "3000億+"), (10000, "1兆+")])
    table = (f'<div class="dgrp"><h2>予定一覧</h2>'
             f'<style>.mcbtn{{font-family:var(--sans);font-size:12px;font-weight:800;'
             f'border:1px solid var(--line);background:var(--panel);color:var(--ink2);'
             f'border-radius:9px;padding:8px 13px;cursor:pointer;margin-right:6px}}'
             f'.mcbtn[aria-pressed="true"]{{background:var(--accent);color:#fff;border-color:transparent}}</style>'
             f'<div style="margin-bottom:10px">{mcbtns}</div>'
             f'<input id="q" class="search" placeholder="コード・社名・業種で絞り込み…">'
             f'<table class="k"><thead><tr><th>コード</th><th>社名</th><th>時価総額</th><th>業種</th><th>市場</th>'
             f'<th>四半期・昨年実績</th><th>状態</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return (f'<div class="dw"><h1>決算発表予定<span class="s">前年実績ベースの予想 ＋ 当日確定分</span></h1>'
            f'{cards}{chart}{table}</div>')


def main() -> None:
    pages = [
        {"file": "index.html", "href": "/", "nav": "研究台帳",
         "title": "研究台帳", "body": page_registry()},
        {"file": "kessan.html", "href": "/kessan.html", "nav": "決算予定",
         "title": "決算発表予定", "body": page_kessan()},
        {"file": "forward.html", "href": "/forward.html", "nav": "フォワード監視",
         "title": "フォワード監視", "body": page_forward()},
        {"file": "data.html", "href": "/data.html", "nav": "データ資産",
         "title": "データ資産", "body": page_data()},
    ]
    SITE.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%-m/%-d %H:%M")
    for p in pages:
        nav = "".join(
            f'<a class="nl" href="{q["href"]}"'
            + (' aria-current="page"' if q is p else "")
            + f'>{q["nav"]}</a>' for q in pages)
        html = (SHELL.replace("__TITLE__", p["title"]).replace("__NAV__", nav)
                     .replace("__STAMP__", stamp).replace("__BODY__", p["body"]))
        (SITE / p["file"]).write_text(html, encoding="utf-8")
        print(f'  {p["file"]}  ({len(html):,} bytes)  {p["title"]}')


if __name__ == "__main__":
    main()
