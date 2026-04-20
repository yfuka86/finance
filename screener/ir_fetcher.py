"""
IR資料リンク生成 — IRBANK + EDINET
"""
from __future__ import annotations


def fetch_ir_links_batch(codes: list) -> dict:
    """全銘柄のIRリンクを一括生成 (外部リクエスト不要)。"""
    results = {}
    for code in codes:
        results[code] = {
            "ir_url": f"https://irbank.net/{code}/results",
            "presentation_url": f"https://irbank.net/{code}/presentation",
            "tdnet_url": f"https://disclosure2.edinet-fsa.go.jp/weee0010.aspx?search_text={code}",
            "shikiho_url": f"https://shikiho.toyokeizai.net/stocks/{code}",
        }
    return results
