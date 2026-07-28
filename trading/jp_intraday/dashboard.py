"""Local management dashboard for intraday (flat-overnight) JP equity strategies.

    streamlit run trading/jp_intraday/dashboard.py

Controls sit in a top bar; the strategy INDEX is a dense list (click 詳細 → SHOW).
All positions open at the open and close at the close — nothing held overnight.
Individual stocks only. Walk-forward ML is strictly train-past / test-future.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import annualized_stats, build_daily_features
from trading.jp_intraday.strategies import (
    STRATEGIES, book_from_scores, score_frame, unit_lot_backtest, walk_forward_folds,
)

st.set_page_config(page_title="JP場中戦略 管理画面", layout="wide")
_KIND = {"xs": "ルール", "ml": "ML"}
ss = st.session_state
ss.setdefault("strat", list(STRATEGIES)[0])

# ── Design system (scoped CSS: palette, surfaces, typography, pill controls) ──
st.markdown("""<style>
:root{
  --primary:#4f46e5; --primary-weak:#eef2ff; --surface:#ffffff; --bg:#f5f6f8;
  --border:#e6e8ec; --text:#0f172a; --muted:#64748b; --pos:#059669; --neg:#e11d48;
  --radius:12px; --shadow:0 1px 3px rgba(15,23,42,.06);
}
.stApp{background:var(--bg);}
.block-container{padding-top:3.2rem;padding-bottom:1.5rem;max-width:1440px;}
h1,h2,h3,h4{color:var(--text);font-weight:700;letter-spacing:-.01em;}
h2,h3{font-size:1.05rem;margin:.4rem 0 .2rem;}
[data-testid="stVerticalBlock"]{gap:.25rem;}
div[data-testid="stMarkdownContainer"] p{margin-bottom:.05rem;}
hr{margin:0;border:none;border-top:1px solid var(--border);}
/* grouped panels / bordered containers → cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:.25rem .5rem;}
/* form labels */
label p, .stSlider label, .stSelectbox label{font-size:11px!important;color:var(--muted)!important;
  font-weight:600!important;text-transform:uppercase;letter-spacing:.03em;}
/* radio OPTIONS → segmented pills (group label stays a plain label) */
div[data-testid="stRadio"] [role=radiogroup]{gap:6px;flex-wrap:wrap;}
div[data-testid="stRadio"] [role=radiogroup] label{border:1px solid var(--border);
  border-radius:999px;padding:3px 14px;background:var(--surface);transition:all .12s;}
div[data-testid="stRadio"] [role=radiogroup] label:hover{border-color:var(--primary);
  background:var(--primary-weak);}
div[data-testid="stRadio"] > label{border:none;background:none;padding:0;}
/* buttons → pill, primary accent */
.stButton button{border-radius:999px;border:1px solid var(--primary);color:var(--primary);
  background:var(--surface);font-weight:600;padding:.05rem .7rem;font-size:13px;min-height:1.9rem;}
.stButton button:hover{background:var(--primary);color:#fff;border-color:var(--primary);}
/* metric cards */
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:8px 12px;box-shadow:var(--shadow);}
div[data-testid="stMetricValue"]{font-size:1.3rem;font-weight:700;}
div[data-testid="stMetricLabel"] p{font-size:11px;color:var(--muted);}
/* dataframe / tables */
div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px;overflow:hidden;}
/* sliders in brand color */
div[data-testid="stSlider"] [data-baseweb=slider] div[role=slider]{background:var(--primary);}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="日次データ読込＋特徴量生成中…")
def _panel(min_value_yen: float, markets: tuple | None = None,
           min_cap: float | None = None, max_cap: float | None = None):
    from trading.jp_intraday.daily_model import load_panel_cached
    return load_panel_cached(min_value_yen=min_value_yen, markets=markets,
                             min_mktcap_yen=min_cap, max_mktcap_yen=max_cap)


@st.cache_data(show_spinner="シグナル計算中…")
def _scores(min_value_yen: float, strategy: str, markets: tuple | None = None,
            min_cap: float | None = None, max_cap: float | None = None):
    return score_frame(_panel(min_value_yen, markets, min_cap, max_cap), strategy)


@st.cache_data(show_spinner="全戦略を評価中…")
def _summaries(min_value_yen, mode, p, MKT=None, MINCAP=None, MAXCAP=None):
    out = {}
    for k in STRATEGIES:
        if not mode.startswith("理想") and not _is_flat(k):
            out[k] = {"unit_na": True}          # ¥単元モード非対応（非フラット）
            continue
        daily, _ = _book_any(min_value_yen, k, mode, p)
        out[k] = annualized_stats(daily, "net")
    return out


def _book(frame, mode, p, con):
    if mode.startswith("理想"):
        return book_from_scores(frame, quantile=p["q"], gross_leverage=p["lev"],
                                cost_bps_side=p["cost"], construction=con)
    return unit_lot_backtest(frame, capital_yen=p["capital"], names_per_side=p["nps"],
                             margin_ratio=p["lev"], cost_bps_side=p["cost"], construction=con)


def _is_flat(k):
    """場中フラット（一日信用でシミュレート可能）か。アンサンブルは全メンバーが条件."""
    spec = STRATEGIES[k]
    if spec["kind"] == "ensemble":
        return all(_is_flat(m) for m, _ in spec["members"])
    return spec.get("holding", "intraday") == "intraday"


def _book_any(liq, strat, mode, p):
    """Book for one strategy OR an ensemble (capital split across member sleeves)."""
    from trading.jp_intraday.strategies import _combine_sleeves
    spec = STRATEGIES[strat]
    if not mode.startswith("理想") and not _is_flat(strat):
        # ¥単元モードは一日信用（寄成建て/引成返済・保証金即日回転・金利0）前提で、
        # オーバーナイト/翌日跨ぎには適用不能 → 空を返し UI 側で「単元非対応」と表示。
        return unit_lot_backtest(pd.DataFrame())
    if spec["kind"] != "ensemble":
        return _book(_scores(liq, strat, MKT, MINCAP, MAXCAP), mode, p, spec.get("construction", "dollar_neutral"))
    sleeves = []
    for member, w in spec["members"]:
        con = STRATEGIES[member].get("construction", "dollar_neutral")
        if mode.startswith("理想"):
            d, b = _book(_scores(liq, member, MKT, MINCAP, MAXCAP), mode, p, con)
            sleeves.append((w, d, b))
        else:
            pm = dict(p); pm["capital"] = p["capital"] * w
            d, b = _book(_scores(liq, member, MKT, MINCAP, MAXCAP), mode, pm, con)
            d = d.copy(); d[["gross", "net"]] = d[["gross", "net"]] * w
            sleeves.append((1.0, d, b))
    return _combine_sleeves(sleeves)


def _open(k):
    ss.strat = k
    ss["nav"] = "🔍 詳細"


# ── TOP CONTROL BAR ─────────────────────────────────────────────────
st.markdown("## 📈 JP場中フラット戦略 管理画面")
st.markdown("<span style='font-size:12px;color:var(--muted)'>寄付き建て・引け手仕舞い（オーバーナイトなし）／"
            "個別株のみ／ML=過去学習・翌年OOS</span>", unsafe_allow_html=True)
def _yen_label(v_million: int) -> str:
    return f"¥{v_million/100:g}億" if v_million >= 100 else f"¥{v_million*100:,}万"


with st.container(border=True):
    t = st.columns([1.7, 2.4, 1.5])
    nav = t[0].radio("画面", ["📊 一覧", "🔍 詳細"], horizontal=True, key="nav")
    mode = t[1].radio("モード", ["💰 単元取引（予算反映・現実）", "理想バックテスト(5年)"],
                      horizontal=True, key="mode")
    liq = t[2].select_slider("流動性 (前日売買代金≥)", options=[3e8, 5e8, 1e9, 2e9, 5e9], value=5e8,
                             format_func=lambda v: f"¥{v/1e8:.0f}億")
    u = st.columns([1.6, 2.6])
    seg = u[0].radio("市場区分", ["全市場", "プライムのみ", "プライム+スタンダード"],
                     horizontal=True, key="seg")
    MKT = {"全市場": None, "プライムのみ": ("プライム",),
           "プライム+スタンダード": ("プライム", "スタンダード")}[seg]
    # 注意: float("inf") はウィジェット値の往復（JSON化）で壊れて反対側のつまみに
    # 伝染し、時価総額≥∞→空パネル→全戦略0.0 になる。有限の番兵 1e15(¥1000兆) を使う。
    _NOCAP = 1e15
    _CAPS = [0, 100e8, 300e8, 1000e8, 3000e8, 1e12, 3e12, _NOCAP]
    def _cap_lab(v):
        return "制限なし" if v in (0, _NOCAP) else (f"¥{v/1e12:g}兆" if v >= 1e12 else f"¥{v/1e8:,.0f}億")
    cap_lo, cap_hi = u[1].select_slider(
        "時価総額バンド（PIT株数×前日終値）", options=_CAPS, value=(0, _NOCAP),
        format_func=_cap_lab)
    MINCAP = None if cap_lo in (0, _NOCAP) else cap_lo
    MAXCAP = None if cap_hi >= _NOCAP else cap_hi
    v = st.columns([1.4, 2.8])
    MIN_SH = v[0].select_slider("表示Sharpe下限（一覧の省略）",
                                options=[None, 0.0, 0.5, 1.0, 2.0], value=0.5,
                                format_func=lambda x: "全表示" if x is None else f"Sh≥{x:g}")
    from trading.jp_intraday.strategies import HOLDING_LABEL
    _ALL_HOLD = list(HOLDING_LABEL.values())
    HOLD_SEL = v[1].multiselect("保有区分タグ", _ALL_HOLD, default=_ALL_HOLD)
    if mode.startswith("理想"):
        pc = st.columns(3)
        p = {"q": pc[0].select_slider("集中度(分位)", options=[0.02, 0.03, 0.05, 0.10, 0.15, 0.20], value=0.05),
             "lev": pc[1].slider("グロスレバレッジ", 1.0, 4.0, 1.0, 0.5),
             "cost": pc[2].slider("片道コスト(bps)", 1.0, 8.0, 3.0, 0.5)}
    else:
        pc = st.columns(4)
        cap_m = pc[0].select_slider("元本＝委託保証金（¥1000万単位）", options=list(range(10, 310, 10)),
                                    value=20, format_func=_yen_label)
        p = {"capital": cap_m * 1e6, "nps": pc[1].slider("片側銘柄数", 6, 25, 10),
             "lev": pc[2].select_slider("信用倍率（保証金率30%・上限3.3x）",
                                        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.3], value=2.0,
                                        format_func=lambda v: f"{v}倍"),
             "cost": pc[3].slider("片道bps(一日信用:手数料0)", 3.0, 25.0, 7.0, 1.0)}
        st.caption("発注時の保証金拘束は実務どおり**ストップ高価格×30%**（値幅制限テーブル準拠）で計算し、"
                   "保証金超過日は単元数を自動縮小。ショートは貸借銘柄・売買代金≥¥10億・50単元以内（価格規制回避）。")
        st.caption("非・場中フラット戦略（オーバーナイト/翌日跨ぎ）は一日信用の前提が成り立たないため "
                   "**単元モード非対応**（一覧では「単元非対応」表示）。理想バックテストで評価してください。")

is_show = nav.startswith("🔍")

# =====================================================================
# INDEX — dense list
# =====================================================================
if not is_show:
    if _panel(liq, MKT, MINCAP, MAXCAP).empty:
        st.warning("この制約（市場区分・時価総額・流動性）では対象銘柄が0件です。制約を緩めてください。")
        st.stop()
    summ = _summaries(liq, mode, p, MKT, MINCAP, MAXCAP)
    from trading.jp_intraday.strategies import HOLDING_LABEL
    def _hold_label(k):
        return HOLDING_LABEL.get(STRATEGIES[k].get("holding", "intraday"))
    ranked_all = sorted(STRATEGIES, key=lambda k: summ[k].get("sharpe", 0), reverse=True)
    ranked = [k for k in ranked_all
              if (_hold_label(k) in HOLD_SEL)
              and (summ[k].get("unit_na")            # 単元非対応は数値なし→Sh下限の対象外
                   or MIN_SH is None or summ[k].get("sharpe", 0) >= MIN_SH)]
    hidden_keys = [k for k in ranked_all if k not in ranked]
    if hidden_keys:
        st.caption(f"表示 {len(ranked)}件 ／ フィルタで非表示 {len(hidden_keys)}件"
                   "（Sharpe下限・保有区分タグ。下部の折りたたみから開けます）")
    with st.expander("📊 成績サマリ表（並べ替え可）", expanded=False):
        rows = [{"戦略": STRATEGIES[k]["title"], "種別": _KIND.get(STRATEGIES[k]["kind"]),
                 "構築": STRATEGIES[k].get("construction", "dollar_neutral"),
                 "年率%": round(summ[k].get("ann_return", 0) * 100, 1),
                 "Sharpe": round(summ[k].get("sharpe", 0), 2),
                 "勝率%": round(summ[k].get("win_rate", 0) * 100, 1),
                 "最大DD%": round(summ[k].get("max_drawdown", 0) * 100, 1)}
                for k in ranked if not summ[k].get("unit_na")]
        if rows:
            tbl = pd.DataFrame(rows)
            st.dataframe(tbl.style.background_gradient(cmap="Greens", subset=["年率%"])
                         .background_gradient(cmap="Blues", subset=["Sharpe"]), width="stretch", hide_index=True)
        else:
            st.caption("数値のある戦略がありません（フィルタ対象がすべて単元非対応、または0件）。")

    W = [0.42, 0.10, 0.09, 0.09, 0.10, 0.11]
    h = st.columns(W, vertical_alignment="center")
    for col, txt in zip(h, ["戦略 / 概要", "年率%", "Sharpe", "勝率%", "最大DD%", ""]):
        col.markdown(f"<span style='font-size:12px;color:gray'>{txt}</span>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    def _clr(v):
        return "var(--pos)" if v >= 0 else "var(--neg)"
    kind_of = {"ensemble": "合成"}

    def _row(k):
        spec, s = STRATEGIES[k], summ[k]
        ann, sh = s.get("ann_return", 0) * 100, s.get("sharpe", 0)
        kind = kind_of.get(spec["kind"]) or _KIND.get(spec["kind"], spec["kind"])
        c = st.columns(W, vertical_alignment="center")
        hold = _hold_label(k)
        chips = (f"<span style='font-size:10px;color:var(--primary);background:var(--primary-weak);"
                 f"border-radius:4px;padding:1px 6px'>{kind}</span> "
                 f"<span style='font-size:10px;color:#0e7490;background:#ecfeff;"
                 f"border-radius:4px;padding:1px 6px'>{hold}</span>")
        for tg in spec.get("tags", []):
            chips += (f" <span style='font-size:10px;color:#a16207;background:#fefce8;"
                      f"border-radius:4px;padding:1px 6px'>{tg}</span>")
        if s.get("unit_na"):
            chips += (" <span style='font-size:10px;color:#6b7280;background:#f3f4f6;"
                      "border-radius:4px;padding:1px 6px'>単元非対応</span>")
        c[0].markdown(
            f"<div style='line-height:1.25'><b>{spec['title']}</b> {chips}<br>"
            f"<span style='font-size:11px;color:var(--muted)'>{spec['thesis'][:56]}…</span></div>",
            unsafe_allow_html=True)
        if s.get("unit_na"):
            for i in (1, 2, 3, 4):
                c[i].markdown("<span style='font-size:14px;color:var(--muted)'>—</span>",
                              unsafe_allow_html=True)
        else:
            wr = s.get("win_rate", 0) * 100
            c[1].markdown(f"<span style='font-size:15px;font-weight:600;color:{_clr(ann)}'>{ann:.1f}</span>", unsafe_allow_html=True)
            c[2].markdown(f"<span style='font-size:15px;font-weight:700;color:{_clr(sh)}'>{sh:.2f}</span>", unsafe_allow_html=True)
            c[3].markdown(f"<span style='font-size:15px;color:{_clr(wr - 50)}'>{wr:.1f}</span>", unsafe_allow_html=True)
            c[4].markdown(f"<span style='font-size:15px;color:var(--neg)'>{s.get('max_drawdown',0)*100:.1f}</span>", unsafe_allow_html=True)
        c[5].button("詳細 ▶", key=f"open_{k}", on_click=_open, args=(k,), width="stretch")
        st.markdown("<hr>", unsafe_allow_html=True)

    for k in ranked:
        _row(k)
    if hidden_keys:
        with st.expander(f"🫥 フィルタで非表示の {len(hidden_keys)}件を開く"
                         "（Sharpe下限未満・保有区分タグ対象外）", expanded=False):
            for k in hidden_keys:
                _row(k)
    st.stop()

# =====================================================================
# SHOW — one strategy
# =====================================================================
strat = st.selectbox("戦略を切替", list(STRATEGIES), key="strat",
                     format_func=lambda k: STRATEGIES[k]["title"])
spec = STRATEGIES[strat]
con = spec.get("construction", "dollar_neutral") if spec["kind"] != "ensemble" else "capital_split"
is_ml = spec["kind"] == "ml"
if not mode.startswith("理想") and not _is_flat(strat):
    st.info("この戦略は場中フラットではない（オーバーナイト/翌日跨ぎ保有）ため、"
            "¥単元シミュレーション（一日信用・寄成建て/引成返済・片側銘柄数の前提）は非対応です。"
            "上部のモードを**理想バックテスト**に切り替えて評価してください。")
    st.stop()
daily, blot = _book_any(liq, strat, mode, p)
if daily.empty:
    st.warning("この制約（市場区分・時価総額・流動性）ではデータまたはML学習行数が不足し、"
               "バックテストを構築できません。制約を緩めてください。")
    st.stop()
daily["date"] = pd.to_datetime(daily["date"]); blot["date"] = pd.to_datetime(blot["date"])

st.markdown(f"### {spec['title']}　<span style='font-size:12px;color:gray'>{_KIND.get(spec['kind'])}/{con}／{mode}</span>",
            unsafe_allow_html=True)
st.info(f"**考え方**: {spec['thesis']}\n\n**具体的な取引**: {spec['rule']}")

sret = annualized_stats(daily, "net"); sg = annualized_stats(daily, "gross")
c = st.columns(6)
c[0].metric("年率リターン(ネット)", f"{sret['ann_return']*100:.1f}%", f"グロス {sg['ann_return']*100:.1f}%")
c[1].metric("Sharpe", f"{sret['sharpe']:.2f}")
c[2].metric("日次勝率", f"{sret['win_rate']*100:.1f}%")
c[3].metric("最大DD", f"{sret['max_drawdown']*100:.1f}%")
c[4].metric("年率ボラ", f"{sret['ann_vol']*100:.1f}%")
c[5].metric("累積リターン", f"{sret['total_return']*100:.0f}%")
if not mode.startswith("理想") and "margin_used_yen" in daily.columns:
    st.caption(f"信用取引の実態: 平均建玉 ¥{daily['deployed_yen'].mean()/1e6:.1f}百万"
               f"（実効倍率 {daily['deployed_yen'].mean()/p['capital']:.2f}x）／"
               f"平均拘束保証金 ¥{daily['margin_used_yen'].mean()/1e6:.1f}百万"
               f"（保証金の{daily['margin_used_yen'].mean()/p['capital']*100:.0f}%・ストップ高×30%基準）／"
               f"平均 {daily['total_lots'].mean():.0f}単元/日")

if is_ml:
    with st.expander("学習期間 と 取引期間（過去学習→翌年OOS）", expanded=False):
        st.dataframe(walk_forward_folds(_panel(liq, MKT, MINCAP, MAXCAP)), width="stretch", hide_index=True)
else:
    st.caption(f"ルールベース（学習なし）。取引: {daily['date'].min().date()} 〜 {daily['date'].max().date()}。")

st.subheader("エクイティカーブ")
eq = daily.set_index("date")
if mode.startswith("理想"):
    eq["ネット"] = (1 + eq["net"]).cumprod(); eq["グロス"] = (1 + eq["gross"]).cumprod()
    st.line_chart(eq[["ネット", "グロス"]], height=280)
else:
    eq["資産(¥ネット)"] = p["capital"] * (1 + eq["net"]).cumprod()
    st.line_chart(eq[["資産(¥ネット)"]], height=280)

st.subheader("日次トレード明細（最新の取引）")
dates = sorted(blot["date"].dt.date.unique())
pick = st.select_slider("取引日", options=dates, value=dates[-1]) if dates else None
if pick is not None:
    d = blot[blot["date"].dt.date.eq(pick)].copy()
    d["ギャップ%"] = (d["residual_gap"] * 100).round(2)
    d["日中%"] = (d["intraday_ret"] * 100).round(2)
    if mode.startswith("理想"):
        d["寄与bps"] = (d["pnl"] * 1e4).round(1)
        longs, shorts = d[d.side == "LONG"], d[d.side == "SHORT"]
        m = st.columns(3)
        m[0].metric("当日ネット寄与", f"{d['pnl'].sum()*100:.2f}%")
        m[1].metric("買い/売り", f"{len(longs)}/{len(shorts)}")
        m[2].metric("勝率(銘柄)", f"{(d['pnl']>0).mean()*100:.0f}%")
        show = ["symbol", "name", "sector", "ギャップ%", "日中%", "寄与bps"]
        lc, rc = st.columns(2)
        lc.caption(f"買い 上位/{len(longs)}"); lc.dataframe(longs.head(12)[show], width="stretch", hide_index=True)
        rc.caption(f"売り 上位/{len(shorts)}"); rc.dataframe(shorts.tail(12)[show].iloc[::-1], width="stretch", hide_index=True)
    else:
        d["株価"] = d["px"].round(0)
        d["単元数"] = d["units"].astype(int)
        d["株数"] = d["単元数"] * 100
        d["建玉¥"] = d["position_yen"].round(0); d["損益¥"] = d["pnl_yen"].round(0)
        longs, shorts = d[d.side_label == "LONG"], d[d.side_label == "SHORT"]
        m = st.columns(6)
        m[0].metric("当日損益", f"¥{d['pnl_yen'].sum():,.0f}")
        m[1].metric("買い/売り", f"{len(longs)}/{len(shorts)}銘柄")
        m[2].metric("合計単元数", f"{int(d['単元数'].sum()):,}単元")
        m[3].metric("建玉合計", f"¥{d['position_yen'].abs().sum()/1e6:.1f}百万")
        if "margin_yen" in d.columns:
            m[4].metric("拘束保証金", f"¥{d['margin_yen'].sum()/1e6:.1f}百万",
                        f"維持余力 {(1 - d['margin_yen'].sum()/p['capital'])*100:.0f}%")
        m[5].metric("勝率(銘柄)", f"{(d['pnl_yen']>0).mean()*100:.0f}%")
        show = ["symbol", "name", "sector", "株価", "単元数", "株数", "建玉¥", "ギャップ%", "日中%", "損益¥"]
        lc, rc = st.columns(2)
        lc.caption(f"買い {len(longs)}銘柄 / {int(longs['単元数'].sum()):,}単元")
        lc.dataframe(longs[show], width="stretch", hide_index=True)
        rc.caption(f"売り {len(shorts)}銘柄 / {int(shorts['単元数'].sum()):,}単元（各≤50単元）")
        rc.dataframe(shorts[show], width="stretch", hide_index=True)

st.subheader("年次パフォーマンス")
by = daily.copy(); by["year"] = by["date"].dt.year
rows = [{"年": y, "リターン%": round(annualized_stats(g, "net")["ann_return"]*100, 1),
         "Sharpe": round(annualized_stats(g, "net")["sharpe"], 2),
         "最大DD%": round(annualized_stats(g, "net")["max_drawdown"]*100, 1)} for y, g in by.groupby("year")]
st.dataframe(pd.DataFrame(rows).set_index("年"), width="stretch")
