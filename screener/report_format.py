"""
Stock Report Evaluation Format v2.0
====================================
バリュースコア: ファンダメンタルのみ (70点満点)

評価軸:
  1. 割安 (20点)   — PER/fPER/PBR/MIX
  2. 財務 (15点)   — 現金比率, 資産裏付け(PBR)
  3. 成長 (15点)   — fPER/PER比, 推定ROE
  4. 触媒 (10点)   — 決算近接度
  5. リスク (10点, 減点方式) — 流動性リスク, バリュートラップ兆候

判定 (FundaScore = 上記5軸合計, 70点満点):
  ≥45: Strong Buy  (★★★★★)
  ≥40: Buy         (★★★★)
  ≥35: Hold        (★★★)
  ≥30: Sell        (★★)
  <30: Strong Sell (★)

※ テクニカル(20点)/需給(10点)は evaluate_stock() 内で算出されるが
  ダッシュボードのスコア・判定には使用しない。
"""

FORMAT_VERSION = "v2.0"


def evaluate_stock(row: dict) -> dict:
    """定量データから統一スコアを算出。row は CSV の1行 (dict)."""
    import math

    scores = {}

    # ── 1. バリュエーション (20点) ──
    val = 0.0
    per = row.get("PER")
    fper = row.get("fPER")
    pbr = row.get("PBR")
    mix = row.get("MIX")

    if _valid(per):
        if per <= 0:
            val += 0
        elif per < 8:
            val += 5.0
        elif per < 12:
            val += 4.0
        elif per < 15:
            val += 3.0
        elif per < 20:
            val += 2.0
        elif per < 30:
            val += 1.0

    if _valid(fper):
        if 0 < fper < 10:
            val += 5.0
        elif fper < 15:
            val += 4.0
        elif fper < 20:
            val += 3.0
        elif fper < 30:
            val += 1.5

    if _valid(pbr):
        if 0 < pbr < 0.5:
            val += 5.0
        elif pbr < 0.8:
            val += 4.0
        elif pbr < 1.0:
            val += 3.5
        elif pbr < 1.5:
            val += 2.5
        elif pbr < 2.0:
            val += 1.5
        elif pbr < 3.0:
            val += 0.5

    if _valid(mix):
        if mix < 10:
            val += 5.0
        elif mix < 15:
            val += 4.0
        elif mix < 22.5:
            val += 3.0
        elif mix < 30:
            val += 1.5
    scores["valuation"] = min(val, 20.0)

    # ── 2. 財務健全性 (15点) ──
    fin = 0.0
    cash = row.get("CashRatio")
    if _valid(cash):
        if cash > 50:
            fin += 10.0
        elif cash > 30:
            fin += 8.0
        elif cash > 20:
            fin += 6.0
        elif cash > 10:
            fin += 4.0
        elif cash > 5:
            fin += 2.0

    if _valid(pbr) and pbr > 0:
        fin += min(5.0, 5.0 / pbr)  # PBR低いほど資産裏付け
    scores["financial"] = min(fin, 15.0)

    # ── 3. 成長性 (15点) ──
    growth = 0.0
    if _valid(per) and _valid(fper) and per > 0 and fper > 0:
        # fPER < PER → 利益成長見込み
        ratio = fper / per
        if ratio < 0.5:
            growth += 10.0
        elif ratio < 0.7:
            growth += 8.0
        elif ratio < 0.9:
            growth += 5.0
        elif ratio < 1.0:
            growth += 3.0
    elif _valid(fper) and 0 < fper < 15:
        growth += 5.0  # fPERのみでも低ければ加点

    # PBR高め × PER低め = 高ROE期待
    if _valid(per) and _valid(pbr) and per > 0:
        roe_est = pbr / per * 100
        if roe_est > 15:
            growth += 5.0
        elif roe_est > 10:
            growth += 3.0
        elif roe_est > 7:
            growth += 1.5
    scores["growth"] = min(growth, 15.0)

    # ── 4. テクニカル (20点) ──
    tech = row.get("TechScore", 0)
    scores["technical"] = round(_valid_or(tech, 0) * 20, 1)

    # ── 5. 需給 (10点) ──
    supply = 0.0
    vr = row.get("VolRatio")
    if _valid(vr):
        if vr > 2.0:
            supply += 7.0
        elif vr > 1.5:
            supply += 5.0
        elif vr > 1.2:
            supply += 3.0
        elif vr > 1.0:
            supply += 1.5

    va5 = row.get("Va_avg5")
    if _valid(va5):
        if va5 >= 50:
            supply += 3.0
        elif va5 >= 10:
            supply += 2.5
        elif va5 >= 3:
            supply += 2.0
        elif va5 >= 1:
            supply += 1.0
    scores["supply"] = min(supply, 10.0)

    # ── 6. カタリスト (10点) ──
    cat = 0.0
    earn_date = row.get("EarningsDate")
    if earn_date and str(earn_date) != "nan":
        import datetime as dt
        try:
            ed = dt.datetime.strptime(str(earn_date)[:10], "%Y-%m-%d").date()
            days = (ed - dt.date.today()).days
            if 0 <= days <= 14:
                cat += 8.0
            elif 0 <= days <= 30:
                cat += 5.0
            elif 0 <= days <= 60:
                cat += 3.0
            elif days > 60:
                cat += 1.0
        except Exception:
            pass
    scores["catalyst"] = min(cat, 10.0)

    # ── 7. リスク (10点, 減点方式 = 10から減点) ──
    risk = 10.0
    # バリュートラップ: PBR < 0.5 かつ CashRatio < 5 → 資産があるが換金不能?
    if _valid(pbr) and pbr < 0.5 and _valid(cash) and cash < 5:
        risk -= 3.0
    # 超高PER (赤字もしくは利益僅少)
    if _valid(per) and per > 100:
        risk -= 4.0
    elif _valid(per) and per > 50:
        risk -= 2.0
    # 低流動性
    if _valid(va5) and va5 < 1:
        risk -= 3.0
    elif _valid(va5) and va5 < 3:
        risk -= 1.0
    scores["risk"] = max(risk, 0.0)

    # ── 総合 ──
    total = sum(scores.values())
    total = round(min(total, 100.0), 1)

    # Verdict は相対ランクで決定 (evaluate_batch で上書き)
    # ここでは暫定判定
    if total >= 60:
        verdict = "Strong Buy"
        stars = 5
    elif total >= 50:
        verdict = "Buy"
        stars = 4
    elif total >= 42:
        verdict = "Hold"
        stars = 3
    elif total >= 35:
        verdict = "Sell"
        stars = 2
    else:
        verdict = "Strong Sell"
        stars = 1

    return {
        "total_score": total,
        "verdict": verdict,
        "stars": stars,
        "breakdown": scores,
        "format_version": FORMAT_VERSION,
    }


def _valid(v) -> bool:
    import math
    if v is None:
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _valid_or(v, default):
    return float(v) if _valid(v) else default


def format_verdict_html(verdict: str, stars: int) -> str:
    star_str = "★" * stars + "☆" * (5 - stars)
    cls_map = {
        "Strong Buy": "verdict-strong-buy",
        "Buy": "verdict-buy",
        "Hold": "verdict-hold",
        "Sell": "verdict-sell",
        "Strong Sell": "verdict-strong-sell",
    }
    cls = cls_map.get(verdict, "verdict-hold")
    return f'<span class="{cls}">{star_str} {verdict}</span>'
