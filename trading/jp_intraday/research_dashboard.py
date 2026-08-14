"""Standalone, persistent research ledger dashboard.

Run:
    PYTHONPATH=. streamlit run trading/jp_intraday/research_dashboard.py --server.port 8502
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trading.jp_intraday.research_registry import research_rows

st.set_page_config(page_title="JP戦略 研究台帳", layout="wide")
st.title("🧪 JP戦略 研究台帳")
st.caption("OOS・フォワード結果だけを表示。ブラウザの画面状態に依存せず、結果ファイルから毎回再構築します。")

research = research_rows()
families = sorted(research["ファミリー"].dropna().unique())
selected = st.multiselect("ファミリー", families, default=families)
show_invalid = st.checkbox("INVALIDATEDも監査記録として表示", value=True)
view = research[research["ファミリー"].isin(selected)].copy()
if not show_invalid:
    view = view[view["状態"] != "INVALIDATED"]

metrics = st.columns(5)
tested = view[view["OOS Sharpe"].notna()]
metrics[0].metric("記録戦略", len(view))
metrics[1].metric("有効OOS数値", len(tested))
metrics[2].metric("Sharpe≥1", int(tested["OOS Sharpe"].ge(1).sum()))
metrics[3].metric("Forward", int(view["状態"].astype(str).str.contains("FORWARD").sum()))
metrics[4].metric("Invalidated", int(view["状態"].eq("INVALIDATED").sum()))

st.dataframe(
    view.style.background_gradient(cmap="Blues", subset=["OOS Sharpe"]),
    width="stretch", hide_index=True,
    column_config={
        "OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "年率%": st.column_config.NumberColumn(format="%.1f%%"),
        "最大DD%": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.subheader("自社株買い研究")
buyback = research[research["ファミリー"].eq("BUYBACK_PRESSURE")]
st.dataframe(buyback, width="stretch", hide_index=True)
st.info("過去のSh 2.37とSh -0.53は特徴量分母の不備によりINVALIDATED。現在利用可能なSharpe主張はありません。")
from pathlib import Path
signal_path = Path(__file__).resolve().parents[2] / "data/jp_buybacks/forward/signals_v33_20260801.parquet"
if signal_path.exists():
    signals = pd.read_parquet(signal_path)
    st.subheader("自社株買い forward v3.3 候補")
    st.caption("ロングは実行適格性まで確認済み。ショート候補は貸借・規制・当日在庫確認前なので発注不可。")
    cols = ["symbol", "state", "submit_at", "period_end", "report_age_days",
            "calendar_days_to_end", "remaining_pressure", "estimated_adv_yen",
            "unit_lot_yen", "short_execution_status"]
    st.dataframe(signals[[c for c in cols if c in signals]], width="stretch", hide_index=True)
emergent_path = Path(__file__).resolve().parents[2] / "data/jp_buybacks/forward/emergent_v1_20260801.parquet"
if emergent_path.exists():
    emergent = pd.read_parquet(emergent_path)
    st.subheader("創発仮説 — 企業買付価格アンカー×実行加速")
    st.caption("主仕様はCORPORATE_PUT_LONGだけ。OBSERVEは成績を比較して後から採用しません。")
    ecols = ["symbol", "state", "submit_at", "anchor_gap", "execution_acceleration_ratio",
             "remaining_pressure", "estimated_adv_yen", "unit_lot_yen"]
    st.dataframe(emergent[[c for c in ecols if c in emergent]], width="stretch", hide_index=True)
st.caption("データ正本: research_registry.py → results/ と data/ の保存済みファイル。ページ再読込でも消えません。")
