"""Read-only registry of OOS/forward research outcomes for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def _json(rel):
    p=ROOT/rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _performance_row(name,family,path,status,tradable=True,note="",branch=None):
    x=_json(path)
    if x is None: return None
    node=x
    if branch: node=x[branch]
    e=node.get("evaluation",{})
    return {"戦略":name,"ファミリー":family,"状態":status,"実取引":tradable,
            "OOS Sharpe":e.get("sharpe"),"年率%":100*e.get("ann_return",float("nan")),
            "最大DD%":100*e.get("max_drawdown",float("nan")),"案件/日数":e.get("days"),
            "注記":note or str(node.get("decision",x.get("decision",""))),"結果":path}


def research_rows() -> pd.DataFrame:
    rows=[]
    specs=[
      ("動的close-to-close LASSO","CORPORATE/PRICE","data/jp_dynamic_cc_results/summary.json","NO-GO",True,"金利後Sh<1"),
      ("TOPIX500階層LASSO","CORPORATE/PRICE","data/jp_hierarchical_lasso_results/summary.json","NO-GO",True,"コスト後基準未達"),
      ("中期個別残差Ridge","CORPORATE/PRICE","data/jp_medium_residual_results/summary.json","NO-GO",True,"グロスα不足"),
      ("ATM IV−RVストラドル","OPTIONS","data/jp_option_iv_rv_straddle/summary.json","NO-GO",False,"清算値のみ・裸売り破綻"),
      ("OTMスキュー縦スプレッド","OPTIONS","data/jp_otm_skew_vertical/summary.json","NO-GO",False,"清算値段階で全年度マイナス"),
    ]
    for s in specs:
        r=_performance_row(*s)
        if r: rows.append(r)
    for name,branch,tradable,note in [
        ("先物ナイト×β","sensitivity",True,"先物単体・基準未達"),
        ("先物ナイト×ストレス","stress",True,"先物単体・基準未達"),
        ("先物ナイト×短期状態","state",True,"先物単体・グロス負"),
    ]:
        r=_performance_row(name,"FUTURES","data/jp_futures_night_interactions/summary.json",
                           "NO-GO",tradable,note,branch)
        if r: rows.append(r)
    for name,family,branch,tradable,note in [
        ("現物ロング","EQUITY_LONG","long_only",True,"ロング脚にαなし"),
        ("現物＋TOPIX先物","MIXED_TAX","mini_topix_hedged",False,"個人税務で損益通算不可"),
    ]:
        r=_performance_row(name,family,"data/jp_long_topix_hedged/summary.json",
                           "NO-GO",tradable,note,branch)
        if r: rows.append(r)
    runs=sorted((ROOT/"results/value_event_v1").glob("run_*.json"))
    tested=[]
    for p in runs:
        x=json.loads(p.read_text(encoding="utf-8"))
        if x.get("decision"): tested.append((p,x))
    if tested:
        p,x=tested[-1]
        rows.append({"戦略":"増配×低PBR Ridge","ファミリー":"CORPORATE_EVENTS",
          "状態":x["decision"],"実取引":True,"OOS Sharpe":None,"年率%":None,"最大DD%":None,
          "案件/日数":x.get("cases"),"注記":f"案件中央値{x.get('median',0)*100:.2f}%・利益集中{x.get('top_case_profit_share',0)*100:.1f}%",
          "結果":str(p.relative_to(ROOT))})
    v2runs=sorted((ROOT/"results/value_event_v2").glob("run_*.json"))
    if v2runs:
        p=v2runs[-1]; x=json.loads(p.read_text(encoding="utf-8"))
        labels={"dividend_resumption":"復配×低PBR",
                "treasury_cancellation_proxy":"自己株消却代理×低PBR"}
        for m in x.get("models",[]):
            note=(f"案件中央値{m.get('median',float('nan'))*100:.2f}%・"
                  f"勝率{m.get('win_rate',float('nan'))*100:.1f}%") if m.get("cases") else m.get("reason","")
            rows.append({"戦略":labels.get(m["model"],m["model"]),"ファミリー":"CORPORATE_EVENTS",
              "状態":m.get("decision",m.get("status")),"実取引":True,"OOS Sharpe":None,
              "年率%":None,"最大DD%":None,"案件/日数":m.get("cases",0),"注記":note,
              "結果":str(p.relative_to(ROOT))})
    v3runs=sorted((ROOT/"results/value_event_v3").glob("run_*.json"))
    if v3runs:
        p=v3runs[-1]; x=json.loads(p.read_text(encoding="utf-8"))
        note=(f"案件中央値{x.get('median',float('nan'))*100:.2f}%・"
              f"勝率{x.get('win_rate',float('nan'))*100:.1f}%・"
              f"利益集中{x.get('top_case_profit_share',float('nan'))*100:.1f}%"
              if x.get("cases") else x.get("reason",""))
        rows.append({"戦略":"固定株(政策保有)縮減×低PBR","ファミリー":"CORPORATE_EVENTS",
          "状態":x.get("decision",x.get("status")),"実取引":True,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":x.get("cases",0),"注記":note,
          "結果":str(p.relative_to(ROOT))})
    mom=_json("data/jp_momentum/summary.json")
    if mom:
        best=max(((v["daily_flat"]["1.0bps"].get("sharpe"),k) for k,v in mom["variants"].items()
                  if v.get("daily_flat",{}).get("1.0bps",{}).get("sharpe") is not None),
                 default=(None,None))
        gross=[c.get("sharpe") for v in mom["variants"].values()
               for c in v.get("gross_tranche",{}).values() if c.get("sharpe") is not None]
        rows.append({"戦略":"モメンタム(標準5定義×保有4)","ファミリー":"MOMENTUM",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":best[0],"年率%":None,"最大DD%":None,
          "案件/日数":len(mom["variants"])*4,
          "注記":f"日中実行の最良{best[1]}。グロス(コスト0)の最良{max(gross):.2f}/最悪{min(gross):.2f}＝α不在",
          "結果":"data/jp_momentum/summary.json"})
    crash=_json("data/jp_crash_dipbuy/summary.json")
    if crash:
        pr=crash["primary"]
        rows.append({"戦略":"暴落ディップバイ(ロングonly)","ファミリー":"CRASH_DIPBUY",
          "状態":crash.get("decision","NO_GO"),"実取引":True,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":pr.get("episodes"),
          "注記":(f"超過平均{pr['excess_mean']*100:+.2f}%(素{pr['raw_net_mean']*100:+.2f}% vs "
                 f"市場{pr['market_mean']*100:+.2f}%)＝全部β。勝率{pr['win_rate']*100:.0f}%"),
          "結果":"data/jp_crash_dipbuy/summary.json"})
    cit=_json("data/jp_crash_index_timing/summary.json")
    if cit:
        s,b=cit["strategy"],cit["buy_and_hold"]
        rows.append({"戦略":"暴落後の指数タイミング(1306現物)","ファミリー":"CRASH_DIPBUY",
          "状態":cit.get("decision","NO_GO"),"実取引":True,"OOS Sharpe":s.get("sharpe"),
          "年率%":100*s.get("ann_return",float("nan")),
          "最大DD%":100*s.get("max_drawdown",float("nan")),"案件/日数":s.get("events"),
          "注記":(f"バイ&ホールドSh{b['sharpe']:.2f}/年率{b['ann_return']*100:.1f}%に完敗。"
                 f"投資日数{s['time_in_market']*100:.1f}%。DDだけ浅いのは市場に居ないため"),
          "結果":"data/jp_crash_index_timing/summary.json"})
    qf=_json("data/jp_quotefree_v2_verify/summary.json")
    if qf:
        rows.append({"戦略":"気配不要ML v2（基準値の再現検証）","ファミリー":"QUOTEFREE_ML",
          "状態":"INVALIDATED","実取引":True,"OOS Sharpe":qf.get("best_unit_lot_sharpe"),
          "年率%":None,"最大DD%":None,"案件/日数":len(qf.get("unit_lot",{})),
          "注記":("記録1.43は24構成のどれでも再現不能（最良0.462）。"
                 "verify_baselineは4/4 PASSなので環境ではなく記録側の問題。1.43は根拠に使わない"),
          "結果":"data/jp_quotefree_v2_verify/summary.json"})
    gap=_json("data/jp_ideal_vs_unitlot/summary.json")
    if gap:
        L=gap["ladder"]
        rows.append({"戦略":"（診断）理想BTと単元BTの落差の分解","ファミリー":"QUOTEFREE_ML",
          "状態":"DIAGNOSTIC","実取引":False,"OOS Sharpe":None,"年率%":None,"最大DD%":None,
          "案件/日数":len(L),
          "注記":(f"理想{L['A_ideal_no_constraints']['sharpe']:.2f}→貸借フィルタで"
                 f"{L['B_ideal_plus_borrow_filter']['sharpe']:.2f}。単元粒度は無害"
                 f"(資本100倍でも{L['C_unitlot_noshortconstraints_capital2000M']['sharpe']:.2f}"
                 f"→{L['D_unitlot_noshortconstraints_capital20M']['sharpe']:.2f})＝死因は借株制約"),
          "結果":"data/jp_ideal_vs_unitlot/summary.json"})
    rows.extend([
      {"戦略":"TOPIX 2026改革","ファミリー":"TOPIX2026","状態":"INPUT BLOCKED","実取引":True,
       "OOS Sharpe":None,"年率%":None,"最大DD%":None,"案件/日数":None,"注記":"公式FFWなし・weight/株式数が古い。raw収集のみ可","結果":"data/topix_2026_forward/readiness_20260801.json"},
    ])
    buyback=_json("results/buyback_steady_oos_20260801.json")
    if buyback:
        b=buyback["results"]["steady"]; e=b["stats"]
        rows.append({"戦略":"自社株買い steady execution","ファミリー":"BUYBACK_PRESSURE",
          "状態":"INVALIDATED","実取引":False,"OOS Sharpe":None,"年率%":None,"最大DD%":None,
          "案件/日数":b["trades"],"注記":"同じ月内営業日分母の不備を使用。旧NO-GO判定も利用禁止",
          "結果":"results/buyback_steady_oos_20260801.json"})
    persistence=_json("results/buyback_persistence_oos_20260801.json")
    if persistence:
        b=persistence["results"]["20bps"]; e=b["stats"]
        rows.append({"戦略":"自社株買い実買付継続ロング","ファミリー":"BUYBACK_PRESSURE",
          "状態":"INVALIDATED","実取引":False,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,
          "案件/日数":b["trades"],"注記":"月内営業日分母の実装不備。成績は利用禁止・新規フォワード待ち",
          "結果":"results/buyback_persistence_oos_20260801.json"})
    rows.append({"戦略":"自社株買い実行圧力（フォワード観測）","ファミリー":"BUYBACK_PRESSURE",
       "状態":"FORWARD READY","実取引":True,"OOS Sharpe":None,"年率%":None,"最大DD%":None,
       "案件/日数":2,"注記":"v3.3実行適格ロング2件。鮮度・期限・品質・流動性・単元ゲート済み",
       "結果":"data/jp_buybacks/forward/signals_v33_20260801.parquet"})
    rows.append({"戦略":"企業買付価格アンカー×実行加速","ファミリー":"BUYBACK_CORPORATE_PUT",
       "状態":"FORWARD READY","実取引":True,"OOS Sharpe":None,"年率%":None,"最大DD%":None,
       "案件/日数":2,"注記":"主仕様2件（5020・9005）。支持価格単独/加速単独は観測のみ",
       "結果":"data/jp_buybacks/forward/emergent_v1_20260801.parquet"})
    return pd.DataFrame(rows).sort_values("OOS Sharpe",ascending=False,na_position="last").reset_index(drop=True)
