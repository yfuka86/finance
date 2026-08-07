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
    v4runs=sorted((ROOT/"results/value_event_v4").glob("run_*.json"))
    if v4runs:
        p=v4runs[-1]; x=json.loads(p.read_text(encoding="utf-8"))
        rows.append({"戦略":"増配×低PBR V4（フロア¥1億再定義）","ファミリー":"CORPORATE_EVENTS",
          "状態":x.get("decision","NO_GO"),"実取引":True,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":x.get("cases"),
          "注記":(f"中央値{x.get('median',0)*100:+.2f}%(40bps)・勝率{x.get('win_rate',0)*100:.1f}%・"
                 f"利益集中{x.get('top_case_profit_share',0)*100:.1f}%で基準20%に6pt届かず。"
                 f"増分50件単独でも中央値{(x.get('increment_median') or 0)*100:+.2f}%＝ドリフト実在の公算。"
                 "事前登録によりV5はなし・残る道はフォワードのみ"),
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
    fxs=_json("data/fx_alpha_sweep/summary.json")
    if fxs:
        best=max((v["selection"].get("sharpe") or -9, k) for k,v in fxs["results"].items())
        rows.append({"戦略":"FX 独立4因子スイープ","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":best[0],
          "年率%":None,"最大DD%":None,"案件/日数":len(fxs["results"]),
          "注記":(f"ドルタイミング−0.06/長期リバーサル0.11/金利モメンタム−0.08/週次リバーサル−0.36。"
                 f"全て選択窓で全滅・確認窓2020+は未消費のまま温存。最良={best[1]}"),
          "結果":"data/fx_alpha_sweep/summary.json"})
    fxs2=_json("data/fx_hourly_seasonality/summary.json")
    if fxs2:
        w=fxs2["walk_forward"]
        rows.append({"戦略":"FX 時間帯セズナリティWF(高頻度)","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":w.get("sharpe"),
          "年率%":100*w.get("ann_return",float("nan")),"最大DD%":100*w.get("max_drawdown",float("nan")),
          "案件/日数":w.get("trade_days"),
          "注記":"凍結規則の毎年再選択・全リターンOOSでSh0.11/後半−0.21。t≥2.5の時間帯効果はOOSで持続しない",
          "結果":"data/fx_hourly_seasonality/summary.json"})
    fxg=_json("data/fx_gotobi/summary.json")
    if fxg:
        sl=fxg["primary"]["selection"]; au=fxg.get("authenticity_selection",{})
        rows.append({"戦略":"FX ゴトー日・東京仲値(USDJPY)","ファミリー":"FX_BASKET",
          "状態":"SEALED→2027-08-06","実取引":True,"OOS Sharpe":sl.get("sharpe"),
          "年率%":100*sl.get("ann_return",float("nan")),"最大DD%":None,
          "案件/日数":sl.get("trades"),
          "注記":(f"カレンダーバグ修正後: midドリフト実在（対照差t={au.get('t_diff')}）だが"
                 "GMO 0.2銭でも+0.63bps/回=Sh0.28で未達。時間足は上げ→反落を相殺するため"
                 "9:00→9:55ティック形+1.93bps/回・Sh0.856でSharpeのみ未達。"
                 "封印再判定を事前登録: 未閲覧の2020-2026+フォワードを2027-08-06に1回開封"
                 "（約560取引・SE≈0.37）。落ちたら恒久クローズ"),
          "結果":"data/fx_gotobi/summary.json"})
    fxb=_json("data/fx_basket/summary.json")
    if fxb:
        a=fxb["A_carry"]["selection"]
        rows.append({"戦略":"FX キャリー/モメンタム（バスケット）","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":a.get("sharpe"),
          "年率%":100*a.get("ann_return",float("nan")),"最大DD%":100*a.get("max_drawdown",float("nan")),
          "案件/日数":a.get("days"),
          "注記":"キャリー0.14/0.07・モメンタム0.55→0.23(集中で崩壊)。スワップ年+2.3%を価格変動が食う",
          "結果":"data/fx_basket/summary.json"})
    fxd=_json("data/fx_carry_dip/summary.json")
    if fxd:
        sl=fxd["selection"]
        rows.append({"戦略":"FX 正キャリーのディップバイ","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":sl.get("sharpe"),
          "年率%":100*sl.get("ann_return",float("nan")),"最大DD%":100*sl.get("max_drawdown",float("nan")),
          "案件/日数":sl.get("trades"),
          "注記":"「暴落後は安い」が不成立（勝率50.9%・1取引平均−0.11%）。無条件キャリーより悪化",
          "結果":"data/fx_carry_dip/summary.json"})
    et=_json("data/jp_earnings_timing/summary.json")
    if et:
        a=et["A_pre_earnings"]; fc=a["confirmation_full_calendar"]
        rows.append({"戦略":"決算発表前ドリフト(ロングα+指数ヘッジ)","ファミリー":"EARNINGS_TIMING",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":fc.get("sharpe"),
          "年率%":100*fc.get("ann_return",float("nan")),
          "最大DD%":100*fc.get("max_drawdown",float("nan")),"案件/日数":fc.get("sessions"),
          "注記":("両窓を通過したが**利益集中で棄却**: 上位5日で総利益の92.2%(選択)/75.2%(確認)、"
                 "上位10日を除くとSh−0.68/−0.82。集中は銘柄でなく日＝残存共通要因の疑い。"
                 f"稼働日のみSh{a['confirmation']['sharpe']}/全暦日{fc['sharpe']}"),
          "結果":"data/jp_earnings_timing/summary.json"})
    cal=_json("data/jp_option_calendar/summary.json")
    if cal:
        c=cal["primary"]
        rows.append({"戦略":"日経225OP カレンダースプレッド","ファミリー":"OPTIONS",
          "状態":"NO-GO","実取引":False,"OOS Sharpe":c.get("sharpe"),
          "年率%":None,"最大DD%":None,"案件/日数":c.get("trades"),
          "注記":("敵対的検証で棄却＝同一バー・アーティファクト。シグナルと建玉が同じ清算値で、"
                 "ノイズ結合を収穫していた。実行可能なT+1建てはSh0.47(put)/0.96(パリティ平均)・"
                 "前後半−0.45/+1.29と不安定。1.97は実行不能な数字"),
          "結果":"data/jp_option_calendar/summary.json"})
    qs=_json("data/jp_quote_shortlist/summary.json")
    if qs:
        k50=qs["results"].get("K=50",{}); full=qs["results"].get("all",{})
        rows.append({"戦略":"気配候補プリスクリーン K=50（ensemble_core）","ファミリー":"QUOTE_SHORTLIST",
          "状態":qs.get("decision","NO_GO"),"実取引":False,"OOS Sharpe":k50.get("sharpe"),
          "年率%":100*k50.get("ann_return",float("nan")),
          "最大DD%":100*k50.get("max_drawdown",float("nan")),"案件/日数":k50.get("days"),
          "注記":(f"維持率{k50.get('retention',0)*100:.1f}%で基準50%に未達＝NO-GO。"
                 f"ただし絶対Sharpeは台帳最良（全銘柄{full.get('sharpe')}）。"
                 f"Kカーブが非単調(K=100で{qs['results'].get('K=100',{}).get('sharpe')})＝誤差大。"
                 "シミュレーションのみで、気配の予測力は未実測"),
          "結果":"data/jp_quote_shortlist/summary.json"})
    rev=_json("data/jp_reversal_leg_sweep/summary.json")
    if rev:
        c=rev["cells"]
        sf=max(v["short_free"] for v in c.values() if v["short_free"] is not None)
        sb=max(v["short_borrow"] for v in c.values() if v["short_borrow"] is not None)
        rows.append({"戦略":"リバーサル 脚別×借株可否 24セル","ファミリー":"REVERSAL",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":rev.get("best_long_excess"),
          "年率%":None,"最大DD%":None,"案件/日数":len(c),
          "注記":(f"執行可能な唯一の列(負け組ロングの市場超過)は最良{rev['best_long_excess']:+.3f}。"
                 f"ショート無制約{sf:+.3f}→貸借+規制で{sb:+.3f}＝借株の壁6例目"),
          "結果":"data/jp_reversal_leg_sweep/summary.json"})
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
