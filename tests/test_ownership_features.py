"""Regression tests for 有報 大株主 ownership features."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from trading.jp_intraday.ownership_features import (
    OWNERSHIP_FEATURES_DAILY, attach_ownership_features, filing_panel, is_custodian,
    load_filings, ownership_release_events,
)


def _holders(symbol, doc_id, sub_date, period_end, pairs):
    return pd.DataFrame([
        {"symbol": symbol, "doc_id": doc_id, "sub_date": pd.Timestamp(sub_date),
         "period_end": pd.Timestamp(period_end), "holder": name, "rank": i + 1,
         "shares": 1000, "ratio": ratio}
        for i, (name, ratio) in enumerate(pairs)
    ])


def test_pooled_nominee_accounts_are_float_not_fixed():
    assert is_custodian("日本マスタートラスト信託銀行株式会社（信託口）")
    assert is_custodian("株式会社日本カストディ銀行（信託口）")
    assert is_custodian("STATE STREET BANK AND TRUST COMPANY 505001")
    # Operating companies and named individuals are fixed ownership.
    assert not is_custodian("トヨタ自動車株式会社")
    assert not is_custodian("佐藤 肇")


def test_fixed_ratio_excludes_custodians():
    h = _holders("1234", "D1", "2024-06-20", "2024-03-31", [
        ("日本マスタートラスト信託銀行株式会社（信託口）", .15),
        ("親会社株式会社", .30),
        ("創業者 太郎", .05),
    ])
    p = filing_panel(h).iloc[0]
    assert p.fixed_ratio == 0.35
    assert p.custodian_ratio == 0.15
    assert p.top10_ratio == 0.50


def test_amended_filing_supersedes_original_for_same_period():
    original = _holders("1234", "D1", "2024-06-20", "2024-03-31", [("親会社株式会社", .40)])
    amended = _holders("1234", "D2", "2024-08-01", "2024-03-31", [("親会社株式会社", .20)])
    p = filing_panel(pd.concat([original, amended], ignore_index=True))
    assert len(p) == 1
    assert p.iloc[0].doc_id == "D2"
    assert p.iloc[0].fixed_ratio == 0.20


def test_year_skip_never_counts_as_a_one_year_unwind():
    """A missing fiscal year must not turn a multi-year drift into an event."""
    frames = [
        _holders("1234", "D1", "2022-06-20", "2022-03-31", [("親会社株式会社", .50)]),
        # 2023 report absent; the next period end is two years later.
        _holders("1234", "D3", "2024-06-20", "2024-03-31", [("親会社株式会社", .40)]),
    ]
    panel = filing_panel(pd.concat(frames, ignore_index=True))
    assert ownership_release_events(panel).empty

    consecutive = pd.concat(frames[:1] + [
        _holders("1234", "D2", "2023-06-20", "2023-03-31", [("親会社株式会社", .40)])
    ], ignore_index=True)
    events = ownership_release_events(filing_panel(consecutive))
    assert len(events) == 1
    assert events.iloc[0].delta_fixed == pytest.approx(-0.10)
    # PIT: the tradable timestamp is the filing date, not the fiscal period end.
    assert events.iloc[0].event_date == pd.Timestamp("2023-06-20")


def test_declines_below_threshold_are_not_events():
    frames = [
        _holders("1234", "D1", "2023-06-20", "2023-03-31", [("親会社株式会社", .50)]),
        _holders("1234", "D2", "2024-06-20", "2024-03-31", [("親会社株式会社", .49)]),
    ]
    panel = filing_panel(pd.concat(frames, ignore_index=True))
    assert ownership_release_events(panel).empty


def test_load_filings_skips_records_without_a_security_code(tmp_path):
    path = tmp_path / "filings.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"DocId": "D1", "Code": None, "EdinetCode": "E001", "DocTypeCode": "120",
         "SubDate": "2024-06-20", "PerEn": "2024-03-31",
         "Hldrs": [{"Rank": 1, "HldrName": "X", "ShsHeld": 1, "ShsRatio": .1}]},
        {"DocId": "D2", "Code": "12340", "EdinetCode": "E002", "DocTypeCode": "120",
         "SubDate": "2024-06-20", "PerEn": "2024-03-31",
         "Hldrs": [{"Rank": 1, "HldrName": "Y", "ShsHeld": 1, "ShsRatio": .2}]},
    ]), encoding="utf-8")
    df = load_filings(path)
    assert set(df.symbol) == {"1234"}


def test_non_listed_filers_are_dropped(tmp_path):
    """Code "00000" は非上場の有報提出者。残すと別会社が同一symbolに潰れ、
    前年比が**企業をまたいで**計算される（実際に踏んだ）。"""
    path = tmp_path / "filings.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"DocId": "D1", "Code": "00000", "DocTypeCode": "120", "SubDate": "2023-06-20",
         "PerEn": "2023-03-31",
         "Hldrs": [{"Rank": 1, "HldrName": "甲ゴルフ倶楽部", "ShsHeld": 1, "ShsRatio": .9}]},
        {"DocId": "D2", "Code": "00000", "DocTypeCode": "120", "SubDate": "2024-06-20",
         "PerEn": "2024-03-31",
         "Hldrs": [{"Rank": 1, "HldrName": "乙ゴルフ倶楽部", "ShsHeld": 1, "ShsRatio": .1}]},
        {"DocId": "D3", "Code": "72030", "DocTypeCode": "120", "SubDate": "2024-06-20",
         "PerEn": "2024-03-31",
         "Hldrs": [{"Rank": 1, "HldrName": "実在株式会社", "ShsHeld": 1, "ShsRatio": .2}]},
    ]), encoding="utf-8")
    df = load_filings(path)
    assert set(df.symbol) == {"7203"}
    # 除外しなければ 0.9 -> 0.1 の -80pt が偽イベントになっていた。
    assert ownership_release_events(filing_panel(df)).empty


def test_daily_features_join_across_code_widths(monkeypatch):
    """パネルは5桁コード、大株主は4桁。揃えないと全特徴量が無言で0になる。"""
    import trading.jp_intraday.ownership_features as of
    filings = pd.concat([
        _holders("7203", "D1", "2024-06-20", "2024-03-31",
                 [("親会社株式会社", .40), ("日本カストディ銀行（信託口）", .10)]),
        _holders("1301", "D2", "2024-06-20", "2024-03-31", [("創業家 太郎", .10)]),
    ], ignore_index=True)
    monkeypatch.setattr(of, "load_filings", lambda *a, **k: filings)
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-21"] * 3),
        "symbol": ["72030", "13010", "99840"],   # 5桁。大株主側は4桁。
    })
    out = of.attach_ownership_features(panel)
    # 5桁/4桁を揃えないとここが全て0になる（無言の不具合だった）。
    assert (out.loc[out.symbol.isin(["72030", "13010"]), "own_fixed_z"] != 0).all()
    assert out.loc[out.symbol.eq("99840"), "own_fixed_z"].eq(0).all()  # 未開示は0
    for c in OWNERSHIP_FEATURES_DAILY:
        assert out[c].notna().all()      # 欠損は0埋め済み（中核のdropnaを壊さない）
