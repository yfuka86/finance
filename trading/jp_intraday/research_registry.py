"""Read-only registry of OOS/forward research outcomes for the dashboard."""
from __future__ import annotations

import json
import pathlib
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
    uss=_json("data/us_index_sweep/summary.json")
    if uss:
        spx=uss.get("SPX500_USD",{})
        h1=spx.get("H1_intraday_momentum",{}).get("selection",{})
        h4=spx.get("H4_drawdown_entry_longterm",{})
        rows.append({"戦略":"米指数CFD 4セル(日中モメ/TOM/夜間/押し目長期)","ファミリー":"US_INDEX",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":h1.get("sharpe"),
          "年率%":None,"最大DD%":None,"案件/日数":4,
          "注記":(f"全滅。日中モメンタムは符号反転Sh−1.33（公開済みアノマリーの死）、"
                 f"夜間はCFD金利で死亡、ATH−15%押し目長期はSh{h4.get('strategy',{}).get('sharpe')}vs "
                 f"B&H{h4.get('buy_and_hold',{}).get('sharpe')}＝タイミングはB&Hに勝てない(US/JP共通)"),
          "結果":"data/us_index_sweep/summary.json"})
    fxs2=_json("data/fx_hourly_seasonality/summary.json")
    if fxs2:
        w=fxs2["walk_forward"]
        rows.append({"戦略":"FX 時間帯セズナリティWF(高頻度)","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":w.get("sharpe"),
          "年率%":100*w.get("ann_return",float("nan")),"最大DD%":100*w.get("max_drawdown",float("nan")),
          "案件/日数":w.get("trade_days"),
          "注記":"凍結規則の毎年再選択・全リターンOOSでSh0.11/後半−0.21。t≥2.5の時間帯効果はOOSで持続しない",
          "結果":"data/fx_hourly_seasonality/summary.json"})
    caf=pathlib.Path("data/jp_close_auction_overnight/detail.json")
    if caf.exists():
        dd=_json("data/jp_close_auction_overnight/detail.json")
        rows.append({"戦略":"引けオークション反転 Long-only(採用・封印)","ファミリー":"QUOTEFREE",
          "状態":"SEALED→2028-08-14","実取引":True,"OOS Sharpe":dd.get("sel_sharpe"),
          "年率%":dd.get("sel_ann_pct"),"最大DD%":None,"案件/日数":dd.get("n_book_days"),
          "注記":("気配不要(前日15:24選択→引成→翌寄成・寄前気配不使用)。αはロング脚に宿り頑健"
                 f"(選択窓Sh{dd.get('sel_sharpe')}・上位10日除去{dd.get('sel_ir_ex_top10')})。"
                 "L/Sはショート裾依存で死。ユーザー承認で集中無視・採用しフォワード封印"),
          "結果":"data/jp_close_auction_overnight/detail.json"})
    cao=_json("data/jp_close_auction_overnight/summary.json")
    if cao:
        s3=cao["selection"].get("S3_ridge",{})
        rows.append({"戦略":"引けオークション反転×オーバーナイト往復(分足マイクロ)","ファミリー":"QUOTEFREE",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":s3.get("sharpe"),
          "年率%":s3.get("ann_pct"),"最大DD%":None,"案件/日数":s3.get("days"),
          "注記":("Optiver診断の残路。分足マイクロ特徴がオーバーナイト残差に実在の予測力"
                 "(分足ネイティブ Sh2.03/IC0.019/t3.0・リーク検証5点通過)=R7を更新(R7は日足特徴)。"
                 "同一バー結合auction_jumpを捕捉除外。NO-GO理由は集中top5=46-63%のみ。"
                 "唯一集中で落ちた実行可能気配不要シグナル。要ユーザー判断(封印)"),
          "結果":"data/jp_close_auction_overnight/summary.json"})
    clp=_json("data/jp_close_prediction/summary.json")
    if clp:
        pr=clp["pooled_ridge"]
        rows.append({"戦略":"引け予測 Optiver転用診断(後場→15:30残差)","ファミリー":"DIAGNOSTIC",
          "状態":"DIAGNOSTIC","実取引":False,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":pr.get("oos_days"),
          "注記":(f"2年1分足。後場→引け残差は予測可(OOS IC{pr['oos_ic_mean']}/t{pr['oos_ic_t']})・"
                 "効くのは反転系(r_last10 t−6)＝投稿の『特徴量は有効』を実証。"
                 f"だが10分位グロス{pr['decile_ls_gross_bps_per_day']}bps<往復7.2bps(15:00テイカー)で非取引。"
                 "板imbalance代理は無情報。翌寄り残差予測なら往復オークションで取れる可能性(要事前登録)"),
          "結果":"data/jp_close_prediction/summary.json"})
    x11l=pathlib.Path("data/jp_oversold_x11_forward/candidates.jsonl")
    if x11l.exists() or True:
        n=len([l for l in x11l.read_text(encoding="utf-8").splitlines() if l.strip()]) if x11l.exists() else 0
        rows.append({"戦略":"X11 売られすぎz20×低ボラ(セカンドチャンス封印)","ファミリー":"JP_FLOW_TIMING",
          "状態":"SEALED→2028-08-12","実取引":True,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":n,
          "注記":("ユーザー承認で集中基準を30%に緩和し昇格。選択窓IR1.15/7年全勝だが"
                 "開示済み窓外実測IR0.38=期待値はこちら寄り。24か月no-peek封印・"
                 "シグナル台帳のみ日次追記。落ちたら恒久クローズ"),
          "結果":"data/jp_oversold_x11_forward/candidates.jsonl"})
    osw=_json("data/jp_oversold_sweep2/cross.json")
    if osw:
        x11=next((c for c in osw if "X11" in str(c.get("name",""))), {})
        rows.append({"戦略":"売られすぎ族・第2掃引(111セル+交差+近傍検証)","ファミリー":"JP_FLOW_TIMING",
          "状態":"NO-GO(恒久)","実取引":True,"OOS Sharpe":x11.get("ir"),
          "年率%":x11.get("excess_ann_pct"),"最大DD%":None,
          "案件/日数":x11.get("active_days_per_year"),
          "注記":("z20×低ボラ×h5=IR1.15・7年全勝・前後半1.15/1.15・近傍安定で過去最良の1基準差。"
                 "落ちたのは上位5日集中(26%vs20%)のみ。小型は継続・出来高スパイクは幻影・"
                 "強売りフローは継続信号・TPはh10でΔ+0.15の初の正例。"
                 "再開はユーザー承認のフォワード封印セカンドチャンスのみ"),
          "結果":"data/jp_oversold_sweep2/"})
    ovi=_json("data/jp_oversold_interaction/summary.json")
    if ovi:
        ml=ovi["selection"].get("ML_ridge_h3",{})
        rows.append({"戦略":"売られすぎ交互作用×利確×ML/MLP","ファミリー":"JP_FLOW_TIMING",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":ml.get("ir"),
          "年率%":ml.get("excess_ann_pct"),"最大DD%":None,
          "案件/日数":ml.get("active_days_per_year"),
          "注記":("全17セル死。交互作用は逆符号(市場が弱いほど売られすぎは続落)。"
                 "TPは+0.04-0.07 IRの正寄与だが桁不足。ML Ridge IR0.59が最良もtop5日42%集中。"
                 "MLP=Ridge(deep増分なし)。売られすぎ反転はルール・ML両輪で決着"),
          "結果":"data/jp_oversold_interaction/summary.json"})
    sdl=_json("data/jp_sector_dip_long/summary.json")
    if sdl:
        s1=sdl["selection"]["S1"]
        rows.append({"戦略":"平時セクター・ディップのロング×フロー×RSI","ファミリー":"JP_FLOW_TIMING",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":None,
          "年率%":s1.get("excess_median_pct"),"最大DD%":s1.get("es5_pct"),
          "案件/日数":s1.get("episodes"),
          "注記":("素は中央値+0.49%/203件と惜しいが上位5%が利益の96%。正体=鉄鋼・海運の"
                 "2021-23スーパーサイクル1回(2018-20は3年連続負)。フローゲートは中央値を潰し"
                 "(イベント文脈と逆=汎用部品でない)、RSI<30は標本を1/9に枯らす。"
                 "「安いものを買う」系はどの粒度でもレジーム1回の塊(3例目)"),
          "結果":"data/jp_sector_dip_long/summary.json"})
    pvh=_json("data/jp_pead_value_hedged/summary.json")
    if pvh:
        e2=pvh["selection"]["E2_excess_flowgate"]
        rows.append({"戦略":"決算上方修正×低PBR×1306ヘッジ×フローゲート","ファミリー":"CORPORATE_EVENTS",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":None,
          "年率%":e2.get("median_pct"),"最大DD%":e2.get("es5_pct"),
          "案件/日数":e2.get("cases"),
          "注記":("上方修正PEAD素は中央値−1.1%で不成立(増配+4.6%とはイベント種で4pp差)。"
                 "フローゲートで+0.61%に反転=日中ML以外で初のフロー有効例(相関+0.09)。"
                 "決算後は信用の利確売りがt+1〜9に系統発生(谷t+3)。1306ヘッジは機構OKだが"
                 "載せるαが先。集中90%で全セル不合格"),
          "結果":"data/jp_pead_value_hedged/summary.json"})
    lof=_json("data/jp_long_only_frontier/summary.json")
    if lof:
        best=lof["selection"].get("V_h20",{})
        rows.append({"戦略":"ロングオンリー断面(バリュー/クオリティ×h5-60・現物)","ファミリー":"QUOTEFREE",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":best.get("net_sharpe"),
          "年率%":best.get("ann_return_pct"),"最大DD%":None,
          "案件/日数":best.get("days"),
          "注記":("全9セル死だがV_h20は断面系最良(超過+4.83%/年・IR0.52)。死因=超過の44%が"
                 "上位5日集中で、正体は2020-21 COVIDバリューローテーション1回の塊"
                 "(最大日はワクチン発表+511bps)。残りは+1.4bps/日の細流。"
                 "バリューは日次αでなく数年周期の因子回転と確定"),
          "結果":"data/jp_long_only_frontier/summary.json"})
    fhz=_json("data/jp_fund_horizon/summary.json")
    if fhz:
        best=max((v for v in fhz["selection"].values() if v.get("sharpe") is not None),
                 key=lambda v: v["sharpe"], default={})
        rows.append({"戦略":"保有期間フロンティア(ファンダ×フロー×h1-60)","ファミリー":"QUOTEFREE",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":best.get("sharpe"),
          "年率%":best.get("ann_return_pct"),"最大DD%":best.get("max_drawdown_pct"),
          "案件/日数":best.get("days"),
          "注記":("全12セル死。正しくマージした財務の断面αはどのhでも不在(最良グロス0.74bps/日)。"
                 "構造発見: 中長期L/Sの拘束コストはショート金利0.85bps/日の床で保有延長では消えない。"
                 "ROEはグロスから負・flowのcc脚も負(日中専用再確認)"),
          "結果":"data/jp_fund_horizon/summary.json"})
    ffs=_json("data/jp_flow_fund_sector_on/summary.json")
    if ffs:
        st=ffs["selection"]["A"]
        rows.append({"戦略":"フロー×ファンダ・セクター内L/S・夜間","ファミリー":"QUOTEFREE",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":st.get("sharpe"),
          "年率%":st.get("ann_return_pct"),"最大DD%":st.get("max_drawdown_pct"),
          "案件/日数":st.get("days"),
          "注記":("全4セル死。ロング脚は市場夜間ドリフトの通過・ショート脚は逆行=横断面夜間α不在。"
                 "借株制約を外すと悪化する初の逆転例。バリュー追加は夜間グロスを毀損。"
                 "3レンズ敵対的検証済(陽性対照=日中IC再現)。旧「夜間IC+0.005」はv11で−0.003に訂正"),
          "結果":"data/jp_flow_fund_sector_on/summary.json"})
    jlh=_json("data/jp_large_holdings/summary.json")
    if jlh and jlh.get("selection"):
        st=jlh["selection"]
        rows.append({"戦略":"大量保有報告書・新規5%イベント(機関あしあと銘柄レベル)","ファミリー":"CORPORATE_EVENTS",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":None,
          "年率%":st.get("excess_median_pct"),"最大DD%":st.get("es5_pct"),
          "案件/日数":st.get("cases"),
          "注記":("メタデータ63,695件を新規収集(縦覧5年ローリング・日次追記化済み)。"
                 "新規報告→翌寄り20営業日は超過中央値−1.64%・勝率42%=負のドリフト。"
                 "報告時点は買い圧力の出口。アクティビスト名簿凍結版は未検証(新窓要)"),
          "結果":"data/jp_large_holdings/summary.json"})
    jif=_json("data/jp_investor_flow/summary.json")
    if jif:
        st=jif["A1_foreign_4w_long_flat_selection"]
        rows.append({"戦略":"投資部門別フロー・タイミング(機関あしあと市場レベル)","ファミリー":"JP_FLOW_TIMING",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":st.get("sharpe"),
          "年率%":st.get("ann_return_pct"),"最大DD%":st.get("max_drawdown_pct"),
          "案件/日数":jif.get("weeks"),
          "注記":("18年971週を新規収集・初使用。海外4週ネット順張りSh0.15 vs B&H0.42=完敗。"
                 "全12部門の翌週IC|t|≤1.35=集計フローに前方予測力なし(同時相関のみ)。"
                 "銘柄粒度の大量保有報告書へ移行(収集中・縦覧5年ローリング消滅に注意)"),
          "結果":"data/jp_investor_flow/summary.json"})
    fxs3=_json("data/fx_session/summary.json")
    if fxs3:
        st=fxs3["S_primary_selection"]["portfolio"]
        rows.append({"戦略":"FX セッション効果(Ranaldo・7レッグ)","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":st.get("sharpe"),
          "年率%":st.get("ann_return_pct"),"最大DD%":st.get("max_drawdown_pct"),
          "案件/日数":st.get("days"),
          "注記":("JPYレッグはグロスから負=仲値後反落がフローを食う。EUR/USDのW形"
                 "(欧州下げ・NY上げ)は実在するがSh0.3止まり。確認窓未開封"),
          "結果":"data/fx_session/summary.json"})
    fxm3=_json("data/fx_micro3/summary.json")
    if fxm3:
        h1=fxm3["selection"]["H1_weekend_gap"]["stats"]
        rows.append({"戦略":"FX 残余3セル(週末ギャップ/水曜スワップ/TS週次反転)","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":h1.get("sharpe"),
          "年率%":h1.get("ann_return_pct"),"最大DD%":h1.get("max_drawdown_pct"),
          "案件/日数":h1.get("days"),
          "注記":("週末ギャップの初出+5.0bps/回は保有期間バグで訂正(正しくは−1.33bps/回)。"
                 "水曜ロールオーバーは完全調整の帰無成立。TS反転はVR<1実在も規模不足。"
                 "G8公開価格の探索は20仮説で枯れた"),
          "結果":"data/fx_micro3/summary.json"})
    fxwg=_json("data/fx_weekend_gap_v2/summary.json")
    if fxwg:
        c0=fxwg["selection_grid"]["C0_all13"]
        rows.append({"戦略":"週末ギャップV2(13銘柄×商品/初動条件付け)","ファミリー":"FX_BASKET",
          "状態":"NO-GO(恒久)","実取引":True,"OOS Sharpe":c0.get("sharpe"),
          "年率%":c0.get("ann_return_pct"),"最大DD%":c0.get("max_drawdown_pct"),
          "案件/日数":c0.get("trades"),
          "注記":("機構特定: 再開1時間の流動性プレミアム+5.2bps(t=24.7・9年全プラス)だが"
                 "往復スプレッド6.3bpsと同額=テイカーは収穫不能(流動性供給者の壁7例目)。"
                 "商品条件付けはCME再開18:00の時点でプレミアム消尽=原理的に不可能。"
                 "micro3のH1保有期間バグもこの検証で発見・訂正"),
          "結果":"data/fx_weekend_gap_v2/summary.json"})
    fxgrid=_json("data/fx_grid/summary.json")
    if fxgrid:
        st=fxgrid["S_primary_selection"]
        rows.append({"戦略":"FX 複数通貨L/Sグリッド(トラリピ型)","ファミリー":"FX_BASKET",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":st.get("sharpe"),
          "年率%":st.get("ann_return_pct"),"最大DD%":st.get("max_drawdown_pct"),
          "案件/日数":st.get("days"),
          "注記":("恒等式: グリッド期待値=平均回帰プレミアムのみ(マルチンゲールで0)。"
                 "VR中央値0.88-0.98=弱い回帰は実在するがグロス+0.3%/年規模。"
                 "上位5日111%集中・キャリー整合片側は悪化。勝率69%でも期待値0を計測器で実証"),
          "結果":"data/fx_grid/summary.json"})
    qcc=_json("data/jp_quotefree_cc1/summary.json")
    if qcc:
        st=qcc["selection"]["executable_delay1"]["stats"]
        rows.append({"戦略":"B案: 気配不要ML×cc1引け集中","ファミリー":"QUOTEFREE",
          "状態":"NO-GO","実取引":True,"OOS Sharpe":st.get("sharpe"),
          "年率%":100*st.get("ann_return",float("nan")),"最大DD%":100*st.get("max_drawdown",float("nan")),
          "案件/日数":st.get("days"),
          "注記":(f"AGENTS残る選択肢の最後の未決着枠。グロス{qcc['selection']['executable_delay1']['gross_bps_per_day']}bps vs "
                 f"全込み{qcc['selection']['executable_delay1']['all_in_cost_bps_per_day']}bps=2倍基準未達・上位5日137%集中。"
                 "遅延なし上界も負＝グロス不在が死因。気配不要の場中/cc1系はこれで全滅確定"),
          "結果":"data/jp_quotefree_cc1/summary.json"})
    v4f=pathlib.Path("data/value_event_v4_forward/events.jsonl")
    if v4f.exists():
        n=sum(1 for l in v4f.read_text(encoding="utf-8").splitlines() if l.strip())
        rows.append({"戦略":"V4増配×低PBR フォワード封印","ファミリー":"CORPORATE_EVENTS",
          "状態":"SEALED→2027-12-01","実取引":True,"OOS Sharpe":None,
          "年率%":None,"最大DD%":None,"案件/日数":n,
          "注記":(f"V4は中央値+4.61%頑健・集中25.96%の1点でNO-GO→仕様不変のまま新期間"
                 f"2026-05〜2027-08のイベントで1回だけ再判定。台帳{n}件（リターン非計算・"
                 "凍結Ridge予測のみ記録）。毎営業日19:30にbars→fins→台帳を自動収集。"
                 "基準: 採用≥30・40bps後中央値>0・最大案件シェア<20%"),
          "結果":"data/value_event_v4_forward/events.jsonl"})
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
