from trading.jp_intraday.research_registry import research_rows


def test_registry_contains_only_explicit_oos_or_forward_status():
    x=research_rows()
    assert {"戦略","状態","OOS Sharpe","実取引","結果"}.issubset(x.columns)
    # IS（イン・サンプル）成績を台帳に出さないためのガード。単語として現れた場合だけを
    # 見る（"PENDING_FULL_RISK_TESTS" の R-IS-K を誤検出しないよう単語境界で判定する）。
    assert not x["状態"].astype(str).str.contains(r"\bIS\b", case=False, regex=True).any()
    assert (x["戦略"]=="増配×低PBR Ridge").any()
    assert (x["戦略"]=="TOPIX 2026改革").any()
