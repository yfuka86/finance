#!/usr/bin/env python3
"""
Bybit Strategy Dashboard — 戦略ごとに TF×通貨 を透過的に表示。

Usage:
    streamlit run trading/bybit/dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.presets import STRATEGY_PRESETS, ALL_SYMBOLS

st.set_page_config(page_title="Bybit Strategy Dashboard", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;600;700&display=swap');
:root{--card:#1a1d27;--card-border:#2a2d3a;--text:#e6e8ec;--text2:#9da3ae;--text3:#6b7280;--green:#22c55e;--red:#ef4444;--gold:#f59e0b;}
.stApp{font-family:'Inter','Noto Sans JP',sans-serif;}
header[data-testid="stHeader"]{display:none!important;}
div[data-testid="stDecoration"],.stDeployButton{display:none;}
.block-container{max-width:100%!important;padding:1rem 2rem 3rem!important;}
.title{font-size:22px;font-weight:700;color:var(--text);margin:0 0 4px;}
.sub{font-size:13px;color:var(--text3);margin:0 0 20px;}
.sec{font-size:16px;font-weight:700;color:var(--text);margin:28px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--card-border);}
.kpi-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.kpi{background:var(--card);border:1px solid var(--card-border);border-radius:10px;padding:14px 18px;min-width:110px;flex:1;}
.kpi-label{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin:0 0 4px;font-weight:500;}
.kpi-val{font-size:22px;font-weight:700;margin:0;line-height:1.2;}
.up{color:var(--green)!important;}.dn{color:var(--red)!important;}.nt{color:var(--text2)!important;}.gold{color:var(--gold)!important;}
.desc-card{background:#161b22;border:1px solid #2a2d3a;border-radius:10px;padding:16px 20px;margin-bottom:16px;line-height:1.75;font-size:13px;color:#c9d1d9;white-space:pre-wrap;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────
IV = {"15": "15m", "60": "1H", "240": "4H", "D": "1D"}
SLAB = {"momentum":"Momentum","donchian_breakout":"Donchian","macd_adx":"MACD+ADX",
        "mean_reversion_filtered":"MRF","dual_regime":"Dual Regime","mtf_rsi2":"MTF RSI2"}

def _cc(v): return "up" if v > 0 else ("dn" if v < 0 else "nt")
def _zero(d): return d.get("3yr",0)==0 and d.get("sr",0)==0 and all(d.get(y,0)==0 for y in("2023","2024","2025"))
def _sr(rets):
    if len(rets)<2: return 0.0
    a=np.array(rets); s=a.std()
    return float(a.mean()/s) if s>0 else 0.0


def _build_tf_sym_rows(preset):
    """戦略の全 TF×通貨 の行を生成。"""
    iv = preset.get("recommended_interval", "")
    tf_rec = IV.get(iv, iv)
    rows = []
    for vkey, tf in [("validated", tf_rec), ("validated_daily", "1D")]:
        v = preset.get(vkey, {})
        for sym in ALL_SYMBOLS:
            d = v.get(sym, {})
            if vkey == "validated_daily" and _zero(d):
                continue
            rows.append({
                "tf": tf, "sym": sym,
                "y23": d.get("2023", 0), "y24": d.get("2024", 0), "y25": d.get("2025", 0),
                "t3": d.get("3yr", 0), "sr": d.get("sr", 0),
            })
    return rows


def _strat_unified_sr(preset):
    rets = []
    for vkey in ("validated", "validated_daily"):
        v = preset.get(vkey, {})
        for d in v.values():
            if vkey == "validated_daily" and _zero(d): continue
            for y in ("2023","2024","2025"):
                rets.append(d.get(y, 0))
    return _sr(rets), len(rets)


# ── State ────────────────────────────────────────────────────────
if "detail" not in st.session_state:
    st.session_state["detail"] = None

detail_pk = st.session_state["detail"]


# ══════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════
if detail_pk is None:
    st.markdown('<p class="title">Bybit Strategy Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">2023-2025 &middot; 10通貨 &middot; 推奨TF+日足 統合 &middot; '
                '統合Sharpe = mean(全TF×通貨 年次リターン) / std</p>', unsafe_allow_html=True)

    # Build overview
    ov = []
    for pk, p in STRATEGY_PRESETS.items():
        u_sr, n_obs = _strat_unified_sr(p)
        tfrows = _build_tf_sym_rows(p)
        n_pos = sum(1 for r in tfrows if r["t3"] > 0)
        n_con = sum(1 for r in tfrows if r["y23"]>0 and r["y24"]>0 and r["y25"]>0)
        ov.append({"_key": pk, "name": p["name"],
                   "type": SLAB.get(p["strategy"], p["strategy"]),
                   "u_sr": round(u_sr, 3), "n_obs": n_obs,
                   "n_combos": len(tfrows), "n_pos": n_pos, "n_con": n_con,
                   "avg": round(np.mean([r["t3"] for r in tfrows]), 1) if tfrows else 0})
    ov.sort(key=lambda r: r["u_sr"], reverse=True)

    # KPIs
    best = ov[0] if ov else None
    kpi = '<div class="kpi-row">'
    kpi += f'<div class="kpi"><p class="kpi-label">戦略数</p><p class="kpi-val nt">{len(ov)}</p></div>'
    if best:
        kpi += f'<div class="kpi"><p class="kpi-label">最高統合SR</p><p class="kpi-val up">{best["name"]} ({best["u_sr"]:.3f})</p></div>'
    kpi += f'<div class="kpi"><p class="kpi-label">総TF×通貨</p><p class="kpi-val nt">{sum(r["n_combos"] for r in ov)}</p></div>'
    kpi += '</div>'
    st.markdown(kpi, unsafe_allow_html=True)

    # Overview table
    st.markdown('<p class="sec">戦略一覧 (統合Sharpe順)</p>', unsafe_allow_html=True)

    th = 'padding:8px 12px;font-size:11px;font-weight:600;color:#6b7280;border-bottom:2px solid #2a2d3a;white-space:nowrap;'
    td = 'padding:8px 12px;font-size:13px;border-bottom:1px solid #2a2d3a;'

    html = '<div style="overflow-x:auto;background:var(--card);border:1px solid var(--card-border);border-radius:10px;">'
    html += '<table style="width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;">'
    html += f'<thead><tr style="background:#1e2230;">'
    for h in ["戦略","種別","統合SR","TF×通貨数","3yr+","3年連続+","平均3yr%"]:
        html += f'<th style="{th}">{h}</th>'
    html += '</tr></thead><tbody>'
    for i, r in enumerate(ov):
        bg = "#1a1d27" if i%2==0 else "#1e2130"
        c_sr = "#22c55e" if r["u_sr"]>0 else "#ef4444"
        html += f'<tr style="background:{bg};">'
        html += f'<td style="{td}font-weight:700;color:#e6e8ec;">{r["name"]}</td>'
        html += f'<td style="{td}color:#9da3ae;font-size:12px;">{r["type"]}</td>'
        html += f'<td style="{td}font-weight:700;font-size:15px;color:{c_sr};">{r["u_sr"]:.3f}</td>'
        html += f'<td style="{td}color:#e6e8ec;">{r["n_combos"]}</td>'
        html += f'<td style="{td}color:#e6e8ec;">{r["n_pos"]}/{r["n_combos"]}</td>'
        html += f'<td style="{td}color:{"#22c55e" if r["n_con"]>0 else "#6b7280"};">{r["n_con"]}</td>'
        c_a = "#22c55e" if r["avg"]>0 else "#ef4444"
        html += f'<td style="{td}font-weight:600;color:{c_a};">{r["avg"]:+.1f}%</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)

    # Buttons
    st.markdown("")
    cols = st.columns(min(len(ov), 5))
    for i, r in enumerate(ov):
        with cols[i % len(cols)]:
            if st.button(r["name"], key=f"go_{r['_key']}", use_container_width=True):
                st.session_state["detail"] = r["_key"]
                st.rerun()

    # ── Global heatmap ───────────────────────────────────────────
    st.markdown('<p class="sec">統合Sharpe ヒートマップ (戦略 × 通貨, TF混合)</p>', unsafe_allow_html=True)

    hm = {}
    for r in ov:
        p = STRATEGY_PRESETS[r["_key"]]
        row = {}
        for sym in ALL_SYMBOLS:
            rets = []
            for vk in ("validated","validated_daily"):
                d = p.get(vk,{}).get(sym,{})
                if vk=="validated_daily" and _zero(d): continue
                for y in ("2023","2024","2025"):
                    rets.append(d.get(y,0))
            row[sym.replace("USDT","")] = round(_sr(rets),2) if len(rets)>=2 else None
        hm[p["name"]] = row

    hm_df = pd.DataFrame(hm).T.reindex(columns=[s.replace("USDT","") for s in ALL_SYMBOLS])
    st.dataframe(
        hm_df.style.format("{:.2f}", na_rep="-").background_gradient(
            cmap="RdYlGn", vmin=-1.0, vmax=1.5, axis=None,
        ).set_properties(**{"font-size":"12px","font-weight":"600","text-align":"center"}),
        use_container_width=True, height=min(len(hm_df)*38+45,500))


# ══════════════════════════════════════════════════════════════════
#  DETAIL — TF×通貨 一覧
# ══════════════════════════════════════════════════════════════════
else:
    if detail_pk not in STRATEGY_PRESETS:
        st.session_state["detail"] = None; st.rerun()

    preset = STRATEGY_PRESETS[detail_pk]
    pname = preset["name"]
    stype = SLAB.get(preset["strategy"], preset["strategy"])
    u_sr, n_obs = _strat_unified_sr(preset)
    tfrows = _build_tf_sym_rows(preset)

    if st.button("< 一覧に戻る"):
        st.session_state["detail"] = None; st.rerun()

    st.markdown(f'<p class="title">{pname}</p>'
                f'<p class="sub">{stype} &middot; {len(tfrows)} TF×通貨 &middot; 2023-2025</p>',
                unsafe_allow_html=True)

    # Description
    desc = preset.get("description_ja","")
    if desc:
        st.markdown(f'<div class="desc-card">{desc}</div>', unsafe_allow_html=True)

    # KPIs
    n_pos = sum(1 for r in tfrows if r["t3"]>0)
    n_con = sum(1 for r in tfrows if r["y23"]>0 and r["y24"]>0 and r["y25"]>0)
    avg_ret = np.mean([r["t3"] for r in tfrows]) if tfrows else 0

    kpi = '<div class="kpi-row">'
    kpi += f'<div class="kpi"><p class="kpi-label">統合Sharpe</p><p class="kpi-val {_cc(u_sr)}" style="font-size:28px;">{u_sr:.3f}</p></div>'
    kpi += f'<div class="kpi"><p class="kpi-label">TF×通貨</p><p class="kpi-val nt">{len(tfrows)}</p></div>'
    kpi += f'<div class="kpi"><p class="kpi-label">3yr+</p><p class="kpi-val {_cc(n_pos-len(tfrows)//2)}">{n_pos}/{len(tfrows)}</p></div>'
    kpi += f'<div class="kpi"><p class="kpi-label">3年連続+</p><p class="kpi-val {_cc(n_con)}">{n_con}</p></div>'
    kpi += f'<div class="kpi"><p class="kpi-label">平均3yr</p><p class="kpi-val {_cc(avg_ret)}">{avg_ret:+.1f}%</p></div>'
    kpi += '</div>'
    st.markdown(kpi, unsafe_allow_html=True)

    # ── TF×通貨 テーブル (Sharpe順) ──────────────────────────────
    st.markdown('<p class="sec">TF × 通貨 一覧 (個別Sharpe順)</p>', unsafe_allow_html=True)

    th = 'padding:8px 10px;font-size:11px;font-weight:600;color:#6b7280;border-bottom:2px solid #2a2d3a;white-space:nowrap;text-align:center;'
    td = 'padding:7px 10px;font-size:13px;border-bottom:1px solid #2a2d3a;text-align:center;font-variant-numeric:tabular-nums;'

    # Sort by sr descending
    tfrows_sorted = sorted(tfrows, key=lambda r: r["sr"], reverse=True)

    html = '<div style="overflow-x:auto;background:var(--card);border:1px solid var(--card-border);border-radius:10px;">'
    html += '<table style="width:100%;border-collapse:collapse;">'
    html += f'<thead><tr style="background:#1e2230;">'
    for h in ["#","TF","通貨","2023 %","2024 %","2025 %","3yr %","Sharpe","3年連続+"]:
        align = "text-align:left;" if h in ("#","TF","通貨") else ""
        html += f'<th style="{th}{align}">{h}</th>'
    html += '</tr></thead><tbody>'

    for i, r in enumerate(tfrows_sorted):
        bg = "#1a1d27" if i%2==0 else "#1e2130"
        short = r["sym"].replace("USDT","")
        is_rec = r["sym"] in preset.get("recommended_symbols", [])
        tf_color = "#60a5fa" if r["tf"] != "1D" else "#c084fc"
        tf_bg = "#1e3a5f" if r["tf"] != "1D" else "#3b1f5c"
        con = r["y23"]>0 and r["y24"]>0 and r["y25"]>0

        html += f'<tr style="background:{bg};">'
        html += f'<td style="{td}text-align:left;color:#6b7280;font-size:11px;">{i+1}</td>'
        html += f'<td style="{td}text-align:left;"><span style="background:{tf_bg};color:{tf_color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">{r["tf"]}</span></td>'
        star = '<span style="color:#f59e0b;"> *</span>' if is_rec else ''
        html += f'<td style="{td}text-align:left;font-weight:600;color:#e6e8ec;">{short}{star}</td>'

        for v in [r["y23"], r["y24"], r["y25"]]:
            cv = "#22c55e" if v>0 else "#ef4444" if v<0 else "#6b7280"
            html += f'<td style="{td}color:{cv};font-weight:600;">{v:+.1f}</td>'

        ct = "#22c55e" if r["t3"]>0 else "#ef4444" if r["t3"]<0 else "#6b7280"
        html += f'<td style="{td}color:{ct};font-weight:700;">{r["t3"]:+.1f}</td>'

        cs = "#22c55e" if r["sr"]>0 else "#ef4444" if r["sr"]<0 else "#6b7280"
        html += f'<td style="{td}color:{cs};font-weight:700;font-size:14px;">{r["sr"]:.2f}</td>'

        if con:
            html += f'<td style="{td}color:#22c55e;font-weight:600;">Yes</td>'
        else:
            html += f'<td style="{td}color:#4b5563;">-</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    html += '<p style="font-size:11px;color:#6b7280;margin-top:4px;">* = 推奨通貨 &middot; <span style="color:#60a5fa;">青</span> = 推奨TF &middot; <span style="color:#c084fc;">紫</span> = 日足</p>'
    st.markdown(html, unsafe_allow_html=True)

    # ── TF別集計 ─────────────────────────────────────────────────
    st.markdown('<p class="sec">TF別サマリー</p>', unsafe_allow_html=True)

    tf_groups = {}
    for r in tfrows:
        tf_groups.setdefault(r["tf"], []).append(r)

    tf_summary = []
    for tf, rs in tf_groups.items():
        rets = []
        for r in rs:
            rets.extend([r["y23"], r["y24"], r["y25"]])
        tf_summary.append({
            "TF": tf,
            "通貨数": len(rs),
            "平均3yr %": round(np.mean([r["t3"] for r in rs]), 1),
            "平均SR": round(np.mean([r["sr"] for r in rs]), 2),
            "3yr+数": sum(1 for r in rs if r["t3"]>0),
            "3年連続+": sum(1 for r in rs if r["y23"]>0 and r["y24"]>0 and r["y25"]>0),
            "Sharpe(全体)": round(_sr(rets), 3),
        })

    tf_df = pd.DataFrame(tf_summary)
    st.dataframe(tf_df.style.format({
        "平均3yr %": "{:+.1f}", "平均SR": "{:.2f}", "Sharpe(全体)": "{:.3f}",
    }).background_gradient(
        subset=["Sharpe(全体)"], cmap="RdYlGn", vmin=-0.5, vmax=1.0,
    ), use_container_width=True, hide_index=True)

    # ── 通貨別集計 ───────────────────────────────────────────────
    st.markdown('<p class="sec">通貨別サマリー (全TF統合)</p>', unsafe_allow_html=True)

    sym_summary = []
    for sym in ALL_SYMBOLS:
        sym_rows = [r for r in tfrows if r["sym"] == sym]
        if not sym_rows: continue
        rets = []
        for r in sym_rows:
            rets.extend([r["y23"], r["y24"], r["y25"]])
        tfs = ", ".join(sorted(set(r["tf"] for r in sym_rows)))
        sym_summary.append({
            "通貨": sym.replace("USDT",""),
            "TF": tfs,
            "平均3yr %": round(np.mean([r["t3"] for r in sym_rows]), 1),
            "平均SR": round(np.mean([r["sr"] for r in sym_rows]), 2),
            "Sharpe(統合)": round(_sr(rets), 3),
        })

    sym_summary.sort(key=lambda r: r["Sharpe(統合)"], reverse=True)
    sym_df = pd.DataFrame(sym_summary)
    st.dataframe(sym_df.style.format({
        "平均3yr %": "{:+.1f}", "平均SR": "{:.2f}", "Sharpe(統合)": "{:.3f}",
    }).background_gradient(
        subset=["Sharpe(統合)"], cmap="RdYlGn", vmin=-0.5, vmax=1.0,
    ), use_container_width=True, hide_index=True)

    # ── Bar chart ────────────────────────────────────────────────
    st.markdown('<p class="sec">TF×通貨 Sharpe分布</p>', unsafe_allow_html=True)

    chart_data = pd.DataFrame([
        {"label": f"{r['tf']} {r['sym'].replace('USDT','')}", "Sharpe": r["sr"]}
        for r in tfrows_sorted
    ]).set_index("label")
    st.bar_chart(chart_data, height=350)

    # ── Parameters ───────────────────────────────────────────────
    st.markdown('<p class="sec">パラメータ</p>', unsafe_allow_html=True)
    params = preset.get("params", {})
    items = list(params.items())
    mid = (len(items)+1)//2
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame([{"パラメータ":k,"値":v} for k,v in items[:mid]]),
                      use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(pd.DataFrame([{"パラメータ":k,"値":v} for k,v in items[mid:]]),
                      use_container_width=True, hide_index=True)
