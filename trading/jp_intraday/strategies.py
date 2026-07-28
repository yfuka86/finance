"""Intraday, flat-overnight strategy library for JP equities.

Every strategy trades the same way — enter at the open, exit at the close, hold
NOTHING overnight — and differs only in the cross-sectional signal used at the
open. All signals are point-in-time (known at the open); returns are the tradable
open->close move. This module returns both the daily P&L and a per-position
"blotter" so the exact trades of any day can be inspected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .daily_model import (
    BASE_FEATURES, FUT_FEATURES, _ls_weights, walk_forward_predictions,
)

# 保有区分 (spec["holding"], 省略時 "intraday"):
#   intraday  = 場中フラット（寄→引・夜間なし） / overnight = 引→翌寄 / cc1 = 引→翌引
HOLDING_RET = {"intraday": "intraday_ret", "overnight": "ret_on_fwd", "cc1": "ret_cc_fwd"}
HOLDING_LABEL = {"intraday": "場中フラット", "overnight": "オーバーナイト", "cc1": "翌日跨ぎ"}

# name -> spec. kind "xs": cross-sectional rank of `score`. kind "ml": trained ridge.
STRATEGIES: dict[str, dict] = {
    "gap_reversal": {
        "kind": "xs", "score": lambda p: -p["residual_gap"], "need": ["residual_gap"],
        "title": "オーバーナイト・ギャップ反転",
        "thesis": "夜間に個別材料や地合いで大きく飛んだ寄り値は、日中に行き過ぎが是正されやすい。"
                  "市場平均を除いた“個別ギャップ”を逆張りする。",
        "rule": "寄付きで、残差ギャップ（当日始値/前日終値−市場平均）が最も低い銘柄を買い、"
                "最も高い銘柄を空売り。各サイド等金額でドルニュートラル。引けで全て手仕舞い。",
    },
    "vol_scaled_reversal": {
        "kind": "xs", "score": lambda p: -p["residual_gap"] / p["ivol"], "need": ["residual_gap", "ivol"],
        "title": "ボラ調整ギャップ反転",
        "thesis": "同じ%のギャップでも低ボラ銘柄の方が“異常”で反転しやすい。ギャップを20日ボラで基準化して逆張り。",
        "rule": "残差ギャップ÷20日ボラが最も低い銘柄を買い、高い銘柄を売り。ドルニュートラル、寄→引で手仕舞い。",
    },
    "sector_neutral_reversal": {
        "kind": "xs", "score": lambda p: -p["sector_resid_gap"], "need": ["sector_resid_gap"],
        "title": "セクター中立ギャップ反転",
        "thesis": "セクター全体が動いた分は継続しやすく、セクター内で突出して飛んだ分が反転しやすい。"
                  "同一33業種内の相対ギャップを逆張り。",
        "rule": "同業種内でギャップが突出して低い銘柄を買い、高い銘柄を売り。寄→引、ドルニュートラル。",
    },
    "prior_day_reversal": {
        "kind": "xs", "score": lambda p: -p["prev_intraday"], "need": ["prev_intraday"],
        "title": "前日日中リバーサル",
        "thesis": "オーバーナイトを使わない純日中シグナル。前日の寄→引で大きく動いた銘柄は翌日の日中に戻しやすい。",
        "rule": "前日の日中リターン（寄→引）が最も低い銘柄を買い、高い銘柄を売り。当日寄→引で手仕舞い。",
    },
    "ml_combined": {
        "kind": "ml", "features": BASE_FEATURES,
        "title": "MLコンバイン（ridge学習）",
        "thesis": "ギャップ・ギャップ規模・前日ダイナミクス・ボラ・流動性・出来高由来の"
                  "Amihud非流動性（前日まで・PIT）を線形結合し、日中リターンを予測。"
                  "出来高特徴量はIS選択→OOS確認を通過したamihud20のみ採用（+0.2〜0.3 Sh）。",
        "rule": "毎年、過去全年で学習したridgeで当日の日中リターン予測を作り、上位を買い・下位を売り。寄→引で手仕舞い。",
    },
    "ml_combined_futures": {
        "kind": "ml", "features": BASE_FEATURES + FUT_FEATURES,
        "title": "MLコンバイン＋先物(US夜間)",
        "thesis": "上記に先物ナイトセッション（NK225/Dow＝US夜間）を追加した版。市場レベル要因の寄与を検証。",
        "rule": "特徴量に先物夜間リターンを加えて同様に学習・運用。",
    },
    "self_normalized_gap_z": {
        "kind": "xs", "score": lambda p: np.where(p["gap_z"].abs() >= 1.5, -p["gap_z"], np.nan),
        "need": ["gap_z"], "construction": "dollar_neutral",
        "title": "自己ギャップZ極値反転",
        "thesis": "残差ギャップを『その銘柄自身の過去60日ギャップ標準偏差』で基準化(Z化)。普段よく飛ぶ銘柄の3%は正常、静かな銘柄の3%は真のサプライズ。|Z|≥1.5の極値だけを建て、1本あたりの反転期待を高める。",
        "rule": "寄付きで gap_z=残差ギャップ/自己ギャップ60日σ が −1.5以下の銘柄を買い、+1.5以上を空売り。極値のみ。引けで手仕舞い。",
    },
    "two_day_gap_reversal": {
        "kind": "xs", "score": lambda p: -(p["residual_gap"] + 0.5 * p["prev_resid_gap"]),
        "need": ["residual_gap", "prev_resid_gap"], "construction": "dollar_neutral",
        "title": "2日ギャップ反転",
        "thesis": "同方向の夜間ギャップが2日続いた銘柄は累積で行き過ぎ、より戻しやすい。当日＋前日(×0.5)の合成夜間ギャップを逆張り。前日“日中”リバーサル(効かない)とは別物。",
        "rule": "寄付きで (当日残差ギャップ+0.5×前日残差ギャップ) が低い銘柄を買い・高い銘柄を売り。引けで手仕舞い。",
    },
    "risk_parity_gap_reversal": {
        "kind": "xs", "score": lambda p: -p["residual_gap"],
        "need": ["residual_gap", "ivol"], "construction": "risk_parity",
        "title": "リスクパリティ・ギャップ反転",
        "thesis": "銘柄選択は素の反転のまま、建玉サイズを1/ボラに比例させ各銘柄のリスク寄与を均等化。高ボラ銘柄がP&Lを支配するのを防ぎ、分散を効かせてSharpeを底上げ。“選択”でなく“サイズ”を変える点がボラ調整と別。",
        "rule": "残差ギャップで上位/下位を選び、各銘柄を1/前日20日ボラに比例配分（低ボラほど厚く）。ドルニュートラル、寄→引。",
    },
    "beta_neutral_gap_reversal": {
        "kind": "xs", "score": lambda p: -p["residual_gap"],
        "need": ["residual_gap", "beta"], "construction": "beta_neutral",
        "title": "ベータ中立ギャップ反転",
        "thesis": "同じ反転シグナルに Σw·β=0 を重ね、意図せぬ市場方向リスクを明示ゼロ化。大きく飛ぶ寄り値は高β銘柄に偏るため等金額だと残るネットβを、TOPIX60日βで除去し純個別αを取り出す。",
        "rule": "残差ギャップで上位/下位を選び、ドル中立かつ Σ(ウェイト×TOPIXβ)=0 になるようβオーバーレイ調整。寄→引で手仕舞い。",
    },
    "short_squeeze_gap_reversal": {
        "kind": "xs",
        "score": lambda p: -p["residual_gap"] + 0.5 * np.maximum(-p["residual_gap"], 0.0)
        * p["sector_short_ratio_z"].clip(lower=0),
        "need": ["residual_gap", "sector_short_ratio_z"], "construction": "dollar_neutral",
        "title": "需給非対称反転(ショートスクイーズ)",
        "thesis": "業種の空売り比率が平常より高い(ショート積み上がり)ときの“下ギャップ”は投機的売り仕掛けを含み、日中の買い戻しで戻りやすい。下ギャップの買い戻し方向にのみ空売り比率zで増幅する非対称シグナル。",
        "rule": "残差ギャップ逆張りに、下ギャップ×(業種空売り比率z, 正のみ)を加点。全体ランクでドルニュートラル、寄→引。",
    },
    "sector_vol_double_neutral": {
        "kind": "xs", "score": lambda p: -p["sector_resid_gap"] / p["ivol"],
        "need": ["sector_resid_gap", "ivol"], "construction": "sector_neutral",
        "title": "セクター×ボラ二重中立反転",
        "thesis": "継続しやすいセクター全体の動きを除いた業種内相対ギャップを、さらに銘柄ボラで基準化。系統的要因と固有ボラ差の両方を除去し最も純粋な個別オーバーシュートを逆張り。構築も業種内でL/S均衡し残存セクター露出ゼロ。",
        "rule": "各33業種内で (−業種内残差ギャップ/ボラ) の上位/下位をL/S。業種ごとに均衡させドル・セクター中立。寄→引。",
    },
    "size_tiered_reversal": {
        "kind": "xs", "score": lambda p: -p["residual_gap"] * (2.0 - p["liq_rank"]),
        "need": ["residual_gap", "liq_rank"], "construction": "dollar_neutral",
        "title": "流動性増幅ギャップ反転",
        "thesis": "板が薄く裁定が効きにくい低流動性銘柄ほど夜間の行き過ぎと日中反転が大きい。残差ギャップに(2−流動性ランク)を掛け、低流動の中ギャップを高流動の大ギャップより優先。※netではコスト増に注意。",
        "rule": "残差ギャップ×(2−前日売買代金ランク) を逆張り。低流動を厚めに選ぶためコスト高めで要net検証。寄→引。",
    },
    "ml_mag_adaptive": {
        "kind": "ml", "features": BASE_FEATURES + ["idio_gap2", "sector_index_gap"],
        "construction": "magnitude_adaptive",
        "title": "ML予測強度ウェイト＋適応絞込",
        "thesis": "ridge予測を|予測|比例（上限3倍）で配分し、低分散日は上位2%に集中（+1.3〜2.3 Sh検証済み）。"
                  "R5で指数分解特徴量（idio_gap2＋sector_index_gap）を融合し、ギャップの『セクター成分/純個別成分』を"
                  "分けてモデルに提示 → OOS両コスト帯で更に+1.0 Sh・全年改善（敵対的検証で小数3桁一致）。",
        "rule": "毎年過去年で学習→翌年予測。上位/下位5%を|予測|比例ウェイトでL/S（低分散日は2%に集中）。寄→引で手仕舞い。",
    },
    "ml_rank_target": {
        "kind": "ml", "features": BASE_FEATURES, "target": "rank",
        "title": "MLランクターゲット学習",
        "thesis": "学習ターゲットをリターンでなく日次クロスセクションの順位（rank−0.5）に変更。ファットテールのノイズを殺し、"
                  "同じ特徴量・同じridgeでOOSネットSharpe +0.8〜0.9（全OOS年で改善、検証済み）。",
        "rule": "順位を予測するridgeで上位を買い・下位を売り（等金額）。寄→引で手仕舞い。",
    },
    "idio_gap_reversal": {
        "kind": "xs", "score": lambda p: -p["idio_gap2"] / p["ivol"],
        "need": ["idio_gap2", "ivol"], "construction": "sector_neutral",
        "title": "指数分解・純個別ギャップ反転",
        "thesis": "33業種TOPIX指数の寄付きギャップ（指数O/前日C−1）で銘柄ギャップを『セクター由来』と『純個別』に厳密分解し、"
                  "純個別分だけをフェード。銘柄平均プロキシ(svdn)より分解が正確で、同一条件比較でOOS優位（8.12 vs 7.28@3bps、検証済み）。データは2021-09以降。",
        "rule": "(銘柄ギャップ−所属業種指数ギャップ)÷ボラ を業種内で順位付けし業種中立L/S。寄→引で手仕舞い。",
    },
    "svdn_concentrated": {
        "kind": "xs",
        "score": lambda p: (lambda s: s.where(
            s.abs() >= s.abs().groupby(p["date"]).transform(lambda x: x.quantile(0.95))))(
            -p["sector_resid_gap"] / p["ivol"]),
        "need": ["sector_resid_gap", "ivol"], "construction": "sign_neutral",
        "title": "セクターボラ反転・集中版",
        "thesis": "svdnのシグナル|z|が当日上位5%の銘柄だけに絞った高コスト耐性版（約24銘柄/日）。"
                  "ISでP=0.95を選択しOOS確認：7bpsで+0.65・10bpsで+1.34 Sh（3bpsでは-0.25と広い版に劣る）。",
        "rule": "業種内相対ギャップ/ボラの絶対値が当日95パーセンタイル以上の銘柄のみを業種中立L/S。寄→引。",
    },
    "gap_short_hedged": {
        "kind": "xs", "score": lambda p: -p["residual_gap"],
        "need": ["residual_gap"], "construction": "pure_short_hedged",
        "title": "純ショート＋TOPIXヘッジ",
        "thesis": "ギャップ反転のアルファはショート脚（上ギャップの売り）に集中：ロング脚を捨ててショートだけを建て、"
                  "βマッチ（0.25グロス）のTOPIXロングで市場中立化。5年ネットSh 3.95@3bps/2.96@7bps・DD−8%・β≈0（検証済み）。7bps帯の最強スリーブ。",
        "rule": "残差ギャップ上位（上ギャップ）を空売り（グロス0.5）＋TOPIX(ETF/ミニ先物)を0.25ロング。両方寄→引で手仕舞い。",
    },
    "on_day_reversal": {
        "kind": "xs", "holding": "overnight",
        "score": lambda p: -(p["intraday_ret"] - p["intraday_ret"].groupby(p["date"]).transform("mean")),
        "need": ["intraday_ret"], "construction": "dollar_neutral", "tags": ["実験的"],
        "title": "🌙 日中反転の夜間持ち（実験枠）",
        "thesis": "実験枠（未検証）: 当日日中に大きく下げた銘柄を引けで買い翌朝寄りで売る（上げは逆）。"
                  "夜間リターンの平均回帰を検証するための土台。シグナルは当日引け時点で確定（PIT）。",
        "rule": "引けで当日日中リターン（市場平均比）下位を買い・上位を売り、翌日の寄付きで全て手仕舞い。夜間のみ保有。",
    },
    "cc1_st_reversal": {
        "kind": "xs", "holding": "cc1",
        "score": lambda p: -(p["ret"] - p["ret"].groupby(p["date"]).transform("mean")),
        "need": ["ret"], "construction": "dollar_neutral", "tags": ["実験的"],
        "title": "📅 短期リバーサル1日（実験枠）",
        "thesis": "実験枠（未検証）: 古典的な短期リバーサル。当日終値ベースの騰落（市場平均比）を"
                  "引けで逆張りし翌日引けまで1日保有。日跨ぎ戦略の検証土台。",
        "rule": "引けで当日騰落下位を買い・上位を売り、翌日の引けで手仕舞い（1営業日保有）。",
    },
    "ensemble_core": {
        "kind": "ensemble", "members": [("ml_mag_adaptive", 0.5), ("svdn_concentrated", 0.5)],
        "title": "🏆 コア・アンサンブル（ML強度×集中svdn）",
        "thesis": "新世代の2本柱: ML予測強度ウェイト版とセクターボラ集中版（クラスタ間相関0.31）を50/50資本分割。"
                  "56組合せ×2コスト帯の全探索でIS両帯上位の唯一のK=2で、OOS(2024-08〜) ネットSh 7.81@3bps / 6.24@7bps・DD−3〜4%。"
                  "最良単体(+0.6〜1.1)と旧アンサンブル(+0.7〜1.5)を全コスト帯で上回る（敵対的検証で完全一致再現）。"
                  "※日中±バリア執行は約定現実性検証で棄却済み（ブリーチ足終値約定は実装不能な好条件。タッチ/OCO現実では改善消失）。"
                  "執行は寄成→引成のまま。",
        "rule": "資本を半分ずつ両戦略に配分し、それぞれ寄付き建て・引け手仕舞い。等ウェイト固定（逆ボラ加重は検証の結果不採用）。",
    },
    "ensemble_core_lowcost": {
        "kind": "ensemble",
        "members": [("ml_mag_adaptive", 1 / 3), ("idio_gap_reversal", 1 / 3), ("svdn_concentrated", 1 / 3)],
        "title": "コア低コスト版（＋指数分解idio）",
        "thesis": "コアに指数分解idioを加えたK=3等分割。@3bpsのIS首位でOOS 8.64@3bps・DD−2.4%と全組合せ最強。"
                  "指数バックフィル(2018+)後の再審査でidioは9年連続プラス@3bpsを確認。ただしidioの7bps弱は長期でも残るため"
                  "**片道~3-5bps以下の執行が確実な場合限定**（7bps帯では標準コアが優位）。",
        "rule": "資本を1/3ずつ3戦略に配分。各々寄付き建て・引け手仕舞い。",
    },
    "ensemble_svdn_ml": {
        "kind": "ensemble", "members": [("sector_vol_double_neutral", 0.5), ("ml_combined", 0.5)],
        "title": "アンサンブル（セクターボラ＋ML）",
        "thesis": "相関0.28しかない2本柱（セクター×ボラ二重中立＝ルール系代表、MLコンバイン＝学習系）に資本を50/50分割。"
                  "全数探索(K=2〜4)でも本ペアが最良で、3bpsでSharpe 4.4→5.0、現実的な7bpsでも2.5と全コスト帯で最良単体を上回る（検証済み）。",
        "rule": "資本を半分ずつ両戦略に配分し、それぞれ通常どおり寄付き建て・引け手仕舞い。日次リターンは両スリーブの平均。",
    },
}

_SCORE_COLS = ["residual_gap", "vol20_floor", "sector_resid_gap", "prev_intraday", "intraday_ret"]
_BLOTTER_COLS = ["date", "symbol", "name", "sector", "residual_gap", "intraday_ret"]


def score_frame(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    """Compute a strategy's per-row signal (the expensive part; cache this)."""
    spec = STRATEGIES[name]
    if spec["kind"] == "ml":
        feats = [f for f in spec["features"] if f in panel.columns]
        frame = walk_forward_predictions(panel, feats, target=spec.get("target", "demeaned"))
        if frame.empty:  # 学習データ不足（強いユニバース制約など）
            return frame.assign(_s=pd.Series(dtype=float))
        cols = [c for c in ["date", "symbol", "name", "sector", "residual_gap",
                            "open", "raw_open", "shortable", "ivol", "prev_value",
                            "prev_close", "prev_low", "prev_close2", "topix_oc"] if c in panel.columns]
        frame = frame.merge(panel[cols], on=["date", "symbol"], how="left")
        frame = frame.assign(_s=frame["pred"])
    else:
        ret_col = HOLDING_RET[spec.get("holding", "intraday")]
        frame = panel.dropna(subset=spec["need"] + [ret_col]).copy()
        # スコアは必ず元の列で計算（エイリアス前！ 当日intraday_retをシグナルに使う戦略があるため）
        frame = frame.assign(_s=spec["score"](frame))
        if ret_col != "intraday_ret":
            # 下流（book/blotter/unit_lot）は intraday_ret 列名で「取りに行くリターン」を
            # 参照するため、保有区分のフォワードリターンをここでエイリアスする。
            frame["intraday_ret"] = frame[ret_col]
        frame = frame[np.isfinite(frame["_s"])]
    return frame


def _select_masks(frame, quantile):
    rank = frame["_s"].groupby(frame["date"]).rank(pct=True)
    long, short = rank.ge(1 - quantile), rank.le(quantile)
    nl = long.groupby(frame["date"]).transform("sum")
    ns = short.groupby(frame["date"]).transform("sum")
    both = nl.gt(0) & ns.gt(0)
    return (both & long), (both & short)


def _w_risk_parity(frame, quantile):
    """Inverse-vol sizing within each side (equalise per-name risk contribution)."""
    long, short = _select_masks(frame, quantile)
    inv = (1.0 / frame["ivol"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = pd.Series(0.0, index=frame.index)
    for mask, sign in ((long, 0.5), (short, -0.5)):
        s = inv.where(mask, 0.0)
        denom = s.groupby(frame["date"]).transform("sum").replace(0, np.nan)
        w = w + sign * (s / denom).fillna(0.0)
    return w


def _w_beta_neutral(frame, quantile):
    """Equal-weight L/S plus a β overlay so Σw=0 AND Σw·β=0."""
    long, short = _select_masks(frame, quantile)
    sel = long | short
    nl = long.groupby(frame["date"]).transform("sum")
    ns = short.groupby(frame["date"]).transform("sum")
    w0 = pd.Series(0.0, index=frame.index)
    w0.loc[long] = 0.5 / nl.loc[long]
    w0.loc[short] = -0.5 / ns.loc[short]
    beta = frame["beta"].fillna(1.0)
    b = beta.where(sel)
    bbar = b.groupby(frame["date"]).transform("mean")
    num = (w0 * beta).groupby(frame["date"]).transform("sum")
    den = ((b - bbar) ** 2).groupby(frame["date"]).transform("sum").replace(0, np.nan)
    lam = -(num / den)
    return w0 + (lam * (beta - bbar)).where(sel, 0.0).fillna(0.0)


def _w_sector_neutral(frame, quantile):
    """Rank & balance L/S within each 33-sector, then scale to ±0.5 per day."""
    key = [frame["date"], frame["sector"]]
    rank = frame["_s"].groupby(key).rank(pct=True)
    long, short = rank.ge(1 - quantile), rank.le(quantile)
    nl = long.groupby(key).transform("sum")
    ns = short.groupby(key).transform("sum")
    both = nl.gt(0) & ns.gt(0)
    w = pd.Series(0.0, index=frame.index)
    w.loc[both & long] = 1.0 / nl.loc[both & long]
    w.loc[both & short] = -1.0 / ns.loc[both & short]
    lpos, lneg = w.clip(lower=0.0), (-w).clip(lower=0.0)
    lsum = lpos.groupby(frame["date"]).transform("sum").replace(0, np.nan)
    ssum = lneg.groupby(frame["date"]).transform("sum").replace(0, np.nan)
    return (lpos / lsum * 0.5).fillna(0.0) - (lneg / ssum * 0.5).fillna(0.0)


def _w_magnitude(frame, quantile, cap: float = 3.0, adaptive: bool = False):
    """|score|-proportional weights within the L/S quantiles (cap × equal-weight).

    OOS-verified: +1.3〜2.3 net Sharpe on the ML book at 3-10bps vs equal-weight.
    ``adaptive``: shrink breadth to q=0.02 on days whose trailing-60d cross-sectional
    |score| dispersion is in its bottom tercile (PIT).
    """
    q_eff = pd.Series(quantile, index=frame.index)
    if adaptive:
        ddisp = frame.groupby("date")["_s"].apply(lambda s: s.abs().std()).sort_index()
        thresh = ddisp.rolling(60, min_periods=30).quantile(1 / 3).shift(1)
        low_days = ddisp.index[ddisp <= thresh]
        q_eff = pd.Series(np.where(frame["date"].isin(low_days), 0.02, quantile),
                          index=frame.index)
    rank = frame["_s"].groupby(frame["date"]).rank(pct=True)
    long = rank.ge(1 - q_eff)
    short = rank.le(q_eff)
    w = pd.Series(0.0, index=frame.index)
    for mask, sign in ((long, 0.5), (short, -0.5)):
        mag = frame["_s"].abs().where(mask, 0.0)
        n = mask.groupby(frame["date"]).transform("sum").replace(0, np.nan)
        cap_w = cap / n                                      # cap at 3x equal-weight
        raw = mag.groupby(frame["date"]).transform(lambda s: s / s.sum() if s.sum() else s)
        raw = np.minimum(raw, cap_w.fillna(np.inf))
        denom = raw.groupby(frame["date"]).transform("sum").replace(0, np.nan)
        w = w + sign * (raw / denom).fillna(0.0)
    return w


_CONSTRUCTIONS = {
    "dollar_neutral": lambda f, q: _ls_weights(f, f["_s"], q),
    "risk_parity": _w_risk_parity,
    "beta_neutral": _w_beta_neutral,
    "sector_neutral": _w_sector_neutral,
    "magnitude": lambda f, q: _w_magnitude(f, q, adaptive=False),
    "magnitude_adaptive": lambda f, q: _w_magnitude(f, q, adaptive=True),
    "pure_short_hedged": lambda f, q: _w_pure_short(f, q),
    "sign_neutral": lambda f, q: _w_sign_neutral(f),
}


def _w_sign_neutral(frame):
    """All surviving (already-masked) names traded: score>0 long / <0 short, ±0.5 each."""
    long = frame["_s"].gt(0)
    short = frame["_s"].lt(0)
    nl = long.groupby(frame["date"]).transform("sum")
    ns = short.groupby(frame["date"]).transform("sum")
    both = nl.gt(0) & ns.gt(0)
    w = pd.Series(0.0, index=frame.index)
    w.loc[both & long] = 0.5 / nl.loc[both & long]
    w.loc[both & short] = -0.5 / ns.loc[both & short]
    return w

HEDGE_WEIGHT = 0.25          # βマッチ (ショート銘柄の実現β~0.5 × 0.5グロス)
HEDGE_COST_BPS_SIDE = 1.0    # TOPIX先物/ETFの往復コスト想定


def _w_pure_short(frame, quantile):
    """Short-only book (bottom quantile, sums to −0.5). Hedge applied in book_from_scores."""
    rank = frame["_s"].groupby(frame["date"]).rank(pct=True)
    short = rank.le(quantile)
    ns = short.groupby(frame["date"]).transform("sum").replace(0, np.nan)
    w = pd.Series(0.0, index=frame.index)
    w.loc[short] = (-0.5 / ns).loc[short]
    return w.fillna(0.0)


def book_from_scores(frame: pd.DataFrame, quantile: float = 0.05,
                     gross_leverage: float = 1.0, cost_bps_side: float = 3.0,
                     construction: str = "dollar_neutral"):
    """Build the open->close book + blotter from a scored frame (flat overnight)."""
    build = _CONSTRUCTIONS.get(construction, _CONSTRUCTIONS["dollar_neutral"])
    w = build(frame, quantile) * gross_leverage
    frame = frame.assign(weight=w)
    gross = (frame["weight"] * frame["intraday_ret"]).groupby(frame["date"]).sum()
    expo = frame["weight"].abs().groupby(frame["date"]).sum()
    net = gross.sub(expo * 2 * cost_bps_side / 10_000)
    if construction == "pure_short_hedged" and "topix_oc" in frame.columns:
        # βマッチのTOPIXロング・オーバーレイ（日次往復、コスト込み）。
        oc = frame.groupby("date")["topix_oc"].first().reindex(gross.index).fillna(0.0)
        hedge = HEDGE_WEIGHT * gross_leverage * oc
        hedge_cost = HEDGE_WEIGHT * gross_leverage * 2 * HEDGE_COST_BPS_SIDE / 10_000
        gross = gross + hedge
        net = net + hedge - hedge_cost
    daily = pd.DataFrame({"date": gross.index, "gross": gross.values, "net": net.values})

    held = frame[frame["weight"].ne(0)].copy()
    held["pnl"] = held["weight"] * held["intraday_ret"]
    held["side"] = np.where(held["weight"] > 0, "LONG", "SHORT")
    cols = [c for c in _BLOTTER_COLS if c in held.columns] + ["side", "weight", "pnl"]
    return daily, held[cols].sort_values(["date", "weight"], ascending=[True, False])


def _combine_sleeves(results: list[tuple[float, pd.DataFrame, pd.DataFrame]]):
    """Capital-split combination: weighted daily columns + weight-scaled blotters.

    ``gross``/``net`` (and any yen/count columns) are summed after scaling by the
    sleeve weight, so returns stay relative to TOTAL capital.
    """
    daily = None
    blots = []
    for weight, d, b in results:
        cols = [c for c in d.columns if c != "date" and pd.api.types.is_numeric_dtype(d[c])]
        d = d.set_index("date")[cols] * weight
        daily = d if daily is None else daily.add(d, fill_value=0.0)
        b = b.copy()
        for col in ("weight", "pnl", "position_yen", "pnl_yen", "cost_yen"):
            if col in b.columns:
                b[col] = b[col] * weight
        blots.append(b)
    out = daily.reset_index()
    return out, pd.concat(blots, ignore_index=True).sort_values("date")


def run_strategy(panel: pd.DataFrame, name: str, quantile: float = 0.05,
                 gross_leverage: float = 1.0, cost_bps_side: float = 3.0):
    """Convenience: score + book in one call. Returns (daily P&L, blotter)."""
    spec = STRATEGIES[name]
    if spec["kind"] == "ensemble":
        return _combine_sleeves([
            (w, *run_strategy(panel, member, quantile, gross_leverage, cost_bps_side))
            for member, w in spec["members"]])
    con = spec.get("construction", "dollar_neutral")
    return book_from_scores(score_frame(panel, name), quantile, gross_leverage, cost_bps_side, con)


def run_unit_lot(panel: pd.DataFrame, name: str, capital_yen: float = 2e7,
                 names_per_side: int = 15, margin_ratio: float = 2.0,
                 cost_bps_side: float = 7.0):
    """Unit-lot (単元) backtest for any strategy incl. ensembles (capital split)."""
    spec = STRATEGIES[name]
    if spec["kind"] == "ensemble":
        sleeves = []
        for member, w in spec["members"]:
            d, b = run_unit_lot(panel, member, capital_yen * w, names_per_side,
                                margin_ratio, cost_bps_side)
            # net/gross are already relative to the sleeve capital; rescale to total.
            d = d.copy()
            d[["gross", "net"]] = d[["gross", "net"]] * w
            sleeves.append((1.0, d, b))
        daily, blot = _combine_sleeves(sleeves)
        return daily, blot
    con = spec.get("construction", "dollar_neutral")
    return unit_lot_backtest(score_frame(panel, name), capital_yen=capital_yen,
                             names_per_side=names_per_side, margin_ratio=margin_ratio,
                             cost_bps_side=cost_bps_side, construction=con)


# TSE 値幅制限 (daily price-limit band by 基準値段). (upper bound exclusive, width)
_PRICE_BANDS = (
    (100, 30), (200, 50), (500, 80), (700, 100), (1000, 150),
    (1500, 300), (2000, 400), (3000, 500), (5000, 700), (7000, 1000),
    (10000, 1500), (15000, 3000), (20000, 4000), (30000, 5000),
    (50000, 7000), (70000, 10000), (100000, 15000), (150000, 30000),
    (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000),
    (1000000, 150000), (1500000, 300000), (2000000, 400000), (3000000, 500000),
    (5000000, 700000), (7000000, 1000000), (10000000, 1500000), (float("inf"), 2000000),
)
MARGIN_REQ = 0.30          # 委託保証金率 30% -> 信用倍率上限 ~3.3x
MAX_SHORT_LOTS = 50        # 50単元以内の新規空売りは価格規制(アップティック)適用除外


def limit_up_price(base: pd.Series) -> pd.Series:
    """ストップ高価格 = 基準値段(前日終値) + 値幅制限."""
    bounds = np.array([b for b, _ in _PRICE_BANDS])
    widths = np.array([w for _, w in _PRICE_BANDS])
    idx = np.searchsorted(bounds, base.to_numpy(), side="right")
    return base + widths[np.minimum(idx, len(widths) - 1)]


def unit_lot_backtest(frame: pd.DataFrame, capital_yen: float = 2e7, names_per_side: int = 15,
                      margin_ratio: float = 2.0, cost_bps_side: float = 7.0,
                      lot_size: int = 100, construction: str = "dollar_neutral",
                      short_min_value_yen: float = 1e9,
                      gross_leverage: float | None = None):
    """信用取引に忠実な ¥ 単元バックテスト（一日信用・寄成建て/引成返済）.

    - ``capital_yen`` は委託保証金（現金）。``margin_ratio`` は信用倍率
      （建玉総額の目標 = 保証金 × 信用倍率、上限 1/0.30 ≈ 3.3x）。
    - 発注時の保証金拘束は実務どおり **ストップ高価格 × 30%**（成行の両建て共通）。
      拘束合計が保証金を超える日は全銘柄の単元数を比例縮小（両サイド均衡維持）。
    - ショートは制度貸借かつ前日売買代金 ≥ ``short_min_value_yen``、
      1銘柄 ``MAX_SHORT_LOTS`` 単元以内（空売り価格規制の適用除外を維持）。
    Returns (daily, blotter). blotter には units（単元数）と拘束保証金を含む。
    """
    if gross_leverage is not None:  # backward-compat alias
        margin_ratio = gross_leverage
    margin_ratio = min(margin_ratio, 1.0 / MARGIN_REQ)
    f = frame.copy()
    px = f["raw_open"].fillna(f["open"]) if "raw_open" in f.columns else f["open"]
    f["px"] = pd.to_numeric(px, errors="coerce")
    f["_s"] = pd.to_numeric(f["_s"], errors="coerce")
    f = f[(f["px"] > 0) & f["_s"].notna()]
    # ストップ高価格（保証金拘束の基準）: 前日終値ベース。無ければ当日始値で近似。
    base = pd.to_numeric(f["prev_close"], errors="coerce").fillna(f["px"]) \
        if "prev_close" in f.columns else f["px"]
    f["limit_up"] = limit_up_price(base)
    side_cap = capital_yen * margin_ratio / 2.0
    per_name_budget = side_cap / names_per_side

    f["side"] = 0
    rank_hi = f["_s"].groupby(f["date"]).rank(method="first", ascending=False)
    f.loc[rank_hi <= names_per_side, "side"] = 1                   # highest score -> long
    # Shorts: 制度信用貸借のみ + 流動性フロア（一日信用在庫切れ/プレミアム料リスク回避）.
    pool = f[f["shortable"] != False] if "shortable" in f.columns else f  # noqa: E712
    if "prev_value" in pool.columns and short_min_value_yen:
        pool = pool[pool["prev_value"] >= short_min_value_yen]
    rank_lo = pool["_s"].groupby(pool["date"]).rank(method="first", ascending=True)
    short_idx = pool.index[rank_lo <= names_per_side]
    short_idx = short_idx[f.loc[short_idx, "side"].eq(0)]
    f.loc[short_idx, "side"] = -1
    sel = f[f["side"].ne(0)].copy()

    if construction == "risk_parity" and "ivol" in sel.columns:
        inv = 1.0 / sel["ivol"]
        denom = inv.groupby([sel["date"], sel["side"]]).transform("sum")
        sel["target_yen"] = side_cap * (inv / denom)
    elif construction in ("magnitude", "magnitude_adaptive"):
        # |予測|比例（上限3×等金額）— 検証済みの本番ウェイト。adaptive の絞込は
        # 単元モードでは names_per_side 選択と競合するため magnitude のみ適用。
        mag = sel["_s"].abs()
        n = sel.groupby(["date", "side"])["_s"].transform("count")
        raw = mag / mag.groupby([sel["date"], sel["side"]]).transform("sum")
        raw = np.minimum(raw, 3.0 / n)
        denom = raw.groupby([sel["date"], sel["side"]]).transform("sum")
        sel["target_yen"] = side_cap * (raw / denom)
    else:                                                          # equal-yen per name
        sel["target_yen"] = per_name_budget
    sel["units"] = np.floor(sel["target_yen"] / (sel["px"] * lot_size)).astype("int64")
    # 空売り価格規制の50単元キャップ: トリガー銘柄（前日または当日寄りで基準値比-10%到達）
    # のみに条件適用。当戦略のショートは上ギャップ銘柄でトリガー該当は稀なため、
    # 一律キャップ（旧実装）はキャパシティを不必要に制限していた（R6で機構特定・検証済み）。
    if "prev_low" in sel.columns and "prev_close2" in sel.columns:
        trig_prev = sel["prev_low"] <= sel["prev_close2"] * 0.9
    else:
        trig_prev = pd.Series(False, index=sel.index)
    trig_today = (sel["px"] <= pd.to_numeric(sel.get("prev_close"), errors="coerce") * 0.9)
    triggered = (trig_prev | trig_today).fillna(True)              # 不明時は保守的にキャップ
    short_mask = sel["side"].lt(0)
    capped = short_mask & triggered
    sel.loc[capped, "units"] = sel.loc[capped, "units"].clip(upper=MAX_SHORT_LOTS)
    sel = sel[sel["units"] >= 1]                                   # must afford >=1 lot

    # 発注時の保証金拘束（ストップ高×30%）が保証金を超える日は比例縮小して再floor.
    sel["margin_yen"] = sel["units"] * lot_size * sel["limit_up"] * MARGIN_REQ
    day_margin = sel.groupby("date")["margin_yen"].transform("sum")
    scale = np.minimum(1.0, capital_yen / day_margin.replace(0, np.nan)).fillna(1.0)
    sel["units"] = np.floor(sel["units"] * scale).astype("int64")
    sel = sel[sel["units"] >= 1]
    sel["margin_yen"] = sel["units"] * lot_size * sel["limit_up"] * MARGIN_REQ

    sel["position_yen"] = sel["units"] * lot_size * sel["px"] * sel["side"]
    sel["pnl_yen"] = sel["position_yen"] * sel["intraday_ret"]
    sel["cost_yen"] = sel["position_yen"].abs() * 2 * cost_bps_side / 10_000

    # ベクトル化集計（lambda agg はPythonループになるため列を前計算してsumのみに）
    sel["_pos_long"] = sel["position_yen"].clip(lower=0.0)
    sel["_pos_short"] = (-sel["position_yen"]).clip(lower=0.0)
    sel["_is_long"] = (sel["side"] > 0).astype("int64")
    sel["_is_short"] = (sel["side"] < 0).astype("int64")
    daily = sel.groupby("date").agg(
        pnl_yen=("pnl_yen", "sum"), cost_yen=("cost_yen", "sum"),
        long_yen=("_pos_long", "sum"), short_yen=("_pos_short", "sum"),
        n_long=("_is_long", "sum"), n_short=("_is_short", "sum"),
        total_lots=("units", "sum"),
        margin_used_yen=("margin_yen", "sum"),
    ).reset_index()
    daily["net_yen"] = daily["pnl_yen"] - daily["cost_yen"]
    daily["deployed_yen"] = daily["long_yen"] + daily["short_yen"]
    daily["margin_util"] = daily["margin_used_yen"] / capital_yen
    daily["net"] = daily["net_yen"] / capital_yen
    daily["gross"] = daily["pnl_yen"] / capital_yen

    sel["side_label"] = np.where(sel["side"] > 0, "LONG", "SHORT")
    blot_cols = [c for c in ["date", "symbol", "name", "sector", "px", "units",
                             "position_yen", "margin_yen", "residual_gap", "intraday_ret",
                             "pnl_yen", "side_label"] if c in sel.columns]
    return daily, sel[blot_cols].sort_values(["date", "position_yen"], ascending=[True, False])


def walk_forward_folds(panel: pd.DataFrame, min_train_years: int = 2) -> pd.DataFrame:
    """Train/test structure of the ML walk-forward (train on all prior years)."""
    years = sorted(panel["date"].dt.year.unique())
    rows = []
    for i in range(min_train_years, len(years)):
        rows.append({"学習期間": f"{years[0]}–{years[i-1]}", "取引期間(OOS)": str(years[i]),
                     "学習年数": i})
    return pd.DataFrame(rows)
