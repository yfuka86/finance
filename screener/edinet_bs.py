"""
EDINET API v2 — BS (貸借対照表) 詳細取得モジュール.

有報・四半期報告書から BS 個別項目を取得し、
広義ネットキャッシュ（準現金性資産 − 有利子負債）を算出する。

準現金性資産:
  現金及び預金 + 受取手形及び売掛金 + 有価証券(流動)

有利子負債:
  短期借入金 + CP + 1年内償還社債 + 1年内返済長期借入金
  + 社債 + 長期借入金
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from data.collectors.config import EDINET_API_KEY

EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
BS_CACHE_PATH = os.path.join(CACHE_DIR, "edinet_bs.json")
DOC_INDEX_PATH = os.path.join(CACHE_DIR, "edinet_doc_index.json")

# ── XBRL element IDs for BS items ──────────────────────────────

# 準現金性資産
NEAR_CASH_ITEMS = {
    "CashAndDeposits": "現金及び預金",
    "NotesAndAccountsReceivableTrade": "受取手形及び売掛金",
    "AccountsReceivableTrade": "売掛金",
    "NotesReceivableTrade": "受取手形",
    "NotesAndAccountsReceivableTradeAndContractAssets": "受取手形、売掛金及び契約資産",
    "ShortTermInvestmentSecurities": "有価証券",
}

# 有利子負債
INTEREST_BEARING_DEBT = {
    "ShortTermLoansPayable": "短期借入金",
    "ShortTermBondsPayable": "短期社債",
    "CommercialPapersLiabilities": "CP",
    "CurrentPortionOfBonds": "1年内償還予定社債",
    "CurrentPortionOfLongTermLoansPayable": "1年内返済予定長期借入金",
    "Bonds": "社債",
    "BondsPayable": "社債",
    "LongTermLoansPayable": "長期借入金",
    "LongTermBondsPayable": "長期社債",
}

# 参考項目
REFERENCE_ITEMS = {
    "CurrentAssets": "流動資産",
    "NoncurrentAssets": "固定資産",
    "Assets": "総資産",
    "CurrentLiabilities": "流動負債",
    "NoncurrentLiabilities": "固定負債",
    "Liabilities": "負債合計",
    "NetAssets": "純資産",
    "InvestmentSecurities": "投資有価証券",
}

ALL_TARGET_ITEMS = {**NEAR_CASH_ITEMS, **INTEREST_BEARING_DEBT, **REFERENCE_ITEMS}


# ── Cache helpers ──────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ── EDINET document index ─────────────────────────────────────

def _fetch_doc_list(date_str: str) -> list[dict]:
    """1日分の EDINET 提出書類一覧を取得."""
    url = f"{EDINET_BASE}/documents.json"
    params = {"date": date_str, "type": 2, "Subscription-Key": EDINET_API_KEY}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
        except requests.RequestException:
            time.sleep(1)
    return []


def build_doc_index(days_back: int = 365, force: bool = False) -> dict:
    """過去 N 日分の EDINET 提出書類から、各銘柄の最新有報/四半期報 docID を特定.

    Returns: {secCode4: {"docID": ..., "date": ..., "type": ..., "name": ...}}
    """
    index = {} if force else _load_json(DOC_INDEX_PATH)

    # 既存 index の最新日を起点にする
    latest_in_cache = None
    if index and not force:
        dates = [v.get("date", "") for v in index.values()]
        if dates:
            latest_in_cache = max(dates)

    today = dt.date.today()
    start = today - dt.timedelta(days=days_back)

    # 対象期間
    dates_to_scan = []
    d = today
    while d >= start:
        if d.weekday() < 5:  # weekdays only
            ds = d.strftime("%Y-%m-%d")
            # Skip dates already fully covered
            if latest_in_cache and ds <= latest_in_cache and not force:
                break
            dates_to_scan.append(ds)
        d -= dt.timedelta(days=1)

    if not dates_to_scan:
        print(f"  EDINET doc index: {len(index)} codes (cache up to date)")
        return index

    print(f"  EDINET doc index: scanning {len(dates_to_scan)} days...")

    for i, ds in enumerate(dates_to_scan):
        docs = _fetch_doc_list(ds)
        for doc in docs:
            sc = doc.get("secCode")
            if not sc or len(str(sc)) != 5:
                continue
            doc_type = doc.get("docTypeCode", "")
            # 120=有報, 130=四半期報告書, 140=半期報告書
            if doc_type not in ("120", "130", "140"):
                continue
            if str(doc.get("csvFlag", "")) != "1":
                continue

            code4 = str(sc)[:4]
            existing = index.get(code4)
            # Keep the latest filing
            if existing and existing.get("date", "") >= ds:
                continue

            index[code4] = {
                "docID": doc["docID"],
                "date": ds,
                "type": doc_type,
                "edinetCode": doc.get("edinetCode", ""),
                "name": doc.get("filerName", ""),
            }

        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(dates_to_scan)} days ({len(index)} codes)")
            _save_json(DOC_INDEX_PATH, index)
        time.sleep(0.3)

    _save_json(DOC_INDEX_PATH, index)
    print(f"  → doc index: {len(index)} codes")
    return index


# ── BS CSV parser ──────────────────────────────────────────────

def _parse_bs_csv(zip_content: bytes) -> dict | None:
    """EDINET CSV ZIP からBS詳細を抽出.

    Returns: {"CashAndDeposits": 12345000, ...} (連結・当期末の値)
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_content))
    except zipfile.BadZipFile:
        return None

    # jpcrp で始まる CSV が本体
    csv_name = None
    for name in zf.namelist():
        if "jpcrp" in name and name.endswith(".csv"):
            csv_name = name
            break
    if not csv_name:
        zf.close()
        return None

    with zf.open(csv_name) as f:
        raw = f.read()
    zf.close()

    # UTF-16 (BOM 付き)
    try:
        text = raw.decode("utf-16")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None

    reader = csv.reader(text.splitlines(), delimiter="\t")
    rows = list(reader)
    if len(rows) < 2:
        return None

    # header: 要素ID, 項目名, コンテキストID, 相対年度, 連結・個別, 期間・時点, ユニットID, 単位, 値
    result = {}
    for row in rows[1:]:
        if len(row) < 9:
            continue
        elem_id = row[0]
        context = row[2]
        consol = row[4]
        value_str = row[8]

        # 連結・当期末のみ (CurrentYearInstant)
        if "CurrentYear" not in context:
            continue
        if consol != "連結":
            continue

        # XBRL ID の短縮名を取得
        short_id = elem_id.split(":")[-1] if ":" in elem_id else elem_id
        if short_id not in ALL_TARGET_ITEMS:
            continue

        # 重複回避 (最初のヒットを採用)
        if short_id in result:
            continue

        # 値のパース
        if not value_str or value_str in ("－", "-", "―", ""):
            result[short_id] = 0
        else:
            try:
                result[short_id] = int(value_str.replace(",", ""))
            except ValueError:
                try:
                    result[short_id] = float(value_str.replace(",", ""))
                except ValueError:
                    result[short_id] = 0

    # 連結なしの場合は個別を使う
    if not result:
        for row in rows[1:]:
            if len(row) < 9:
                continue
            elem_id = row[0]
            context = row[2]
            consol = row[4]
            value_str = row[8]
            if "CurrentYear" not in context:
                continue
            if consol != "個別":
                continue
            short_id = elem_id.split(":")[-1] if ":" in elem_id else elem_id
            if short_id not in ALL_TARGET_ITEMS:
                continue
            if short_id in result:
                continue
            if not value_str or value_str in ("－", "-", "―", ""):
                result[short_id] = 0
            else:
                try:
                    result[short_id] = int(value_str.replace(",", ""))
                except ValueError:
                    try:
                        result[short_id] = float(value_str.replace(",", ""))
                    except ValueError:
                        result[short_id] = 0

    return result if result else None


def _download_csv(doc_id: str) -> bytes | None:
    """EDINET から CSV ZIP をダウンロード."""
    url = f"{EDINET_BASE}/documents/{doc_id}"
    params = {"type": 5, "Subscription-Key": EDINET_API_KEY}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.content
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
        except requests.RequestException:
            time.sleep(1)
    return None


# ── Net cash calculation ───────────────────────────────────────

def calc_net_cash(bs: dict) -> dict:
    """BS 詳細から広義ネットキャッシュを算出.

    Returns:
        near_cash: 準現金性資産
        interest_debt: 有利子負債
        net_cash: 広義ネットキャッシュ (= near_cash - interest_debt)
        breakdown: 内訳 dict
    """
    # 準現金性資産
    cash = bs.get("CashAndDeposits", 0)

    # 売掛金: 複数の XBRL タグがあるため優先順位で取得
    receivables = (
        bs.get("NotesAndAccountsReceivableTradeAndContractAssets", 0)
        or bs.get("NotesAndAccountsReceivableTrade", 0)
        or (bs.get("AccountsReceivableTrade", 0) + bs.get("NotesReceivableTrade", 0))
    )

    securities = bs.get("ShortTermInvestmentSecurities", 0)

    near_cash = cash + receivables + securities

    # 有利子負債
    short_loans = bs.get("ShortTermLoansPayable", 0)
    cp = bs.get("CommercialPapersLiabilities", 0) + bs.get("ShortTermBondsPayable", 0)
    cur_bonds = bs.get("CurrentPortionOfBonds", 0)
    cur_ltloans = bs.get("CurrentPortionOfLongTermLoansPayable", 0)
    bonds = bs.get("Bonds", 0) or bs.get("BondsPayable", 0) or bs.get("LongTermBondsPayable", 0)
    lt_loans = bs.get("LongTermLoansPayable", 0)

    interest_debt = short_loans + cp + cur_bonds + cur_ltloans + bonds + lt_loans

    breakdown = {
        "現金及び預金": cash,
        "売掛金等": receivables,
        "有価証券(流動)": securities,
        "短期借入金": short_loans,
        "CP": cp,
        "1年内償還社債": cur_bonds,
        "1年内返済長期借入": cur_ltloans,
        "社債": bonds,
        "長期借入金": lt_loans,
        "投資有価証券": bs.get("InvestmentSecurities", 0),
    }

    return {
        "near_cash": near_cash,
        "interest_debt": interest_debt,
        "net_cash": near_cash - interest_debt,
        "breakdown": breakdown,
    }


# ── Batch fetch ────────────────────────────────────────────────

def fetch_bs_batch(codes: list[str], days_back: int = 365,
                   max_workers: int = 4) -> dict[str, dict]:
    """複数銘柄の BS 詳細を一括取得.

    Returns: {code4: {"bs": {...}, "net_cash_info": {...}, "doc_date": ...}}
    """
    # 1) doc index を構築/更新
    doc_index = build_doc_index(days_back=days_back)

    # 2) BS キャッシュ読み込み
    bs_cache = _load_json(BS_CACHE_PATH)

    # 3) 取得が必要な銘柄を特定
    to_fetch = []
    for code in codes:
        if code in bs_cache:
            continue
        if code in doc_index:
            to_fetch.append((code, doc_index[code]["docID"]))

    cached_count = len(codes) - len(to_fetch)
    no_doc = len(codes) - cached_count - len(to_fetch)
    print(f"  EDINET BS: {cached_count} cached, {len(to_fetch)} to fetch, {no_doc} no doc")

    if to_fetch:
        print(f"  EDINET BS ダウンロード中 ({len(to_fetch)} 銘柄)...")

        def _fetch_one(item):
            code, doc_id = item
            content = _download_csv(doc_id)
            if content is None:
                return code, None
            bs = _parse_bs_csv(content)
            return code, bs

        done = 0
        # Sequential with rate limiting (EDINET is strict)
        for code, doc_id in to_fetch:
            _, bs = _fetch_one((code, doc_id))
            if bs:
                nc = calc_net_cash(bs)
                bs_cache[code] = {
                    "bs": bs,
                    "net_cash_info": nc,
                    "doc_date": doc_index[code]["date"],
                    "doc_type": doc_index[code]["type"],
                }
            else:
                bs_cache[code] = {"bs": {}, "net_cash_info": None, "doc_date": None}

            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(to_fetch)}")
                _save_json(BS_CACHE_PATH, bs_cache)
            time.sleep(0.5)  # Rate limit

        _save_json(BS_CACHE_PATH, bs_cache)
        print(f"  → EDINET BS 完了 ({done} 銘柄)")

    # 4) 結果を返す
    result = {}
    for code in codes:
        if code in bs_cache and bs_cache[code].get("net_cash_info"):
            result[code] = bs_cache[code]
    return result


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EDINET BS詳細取得")
    ap.add_argument("--codes", nargs="*", help="銘柄コード (空=全銘柄)")
    ap.add_argument("--days", type=int, default=365, help="遡る日数")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()

    if args.rebuild_index:
        build_doc_index(days_back=args.days, force=True)
    elif args.codes:
        result = fetch_bs_batch(args.codes, days_back=args.days)
        for code, data in result.items():
            nc = data["net_cash_info"]
            print(f"\n{code} (doc: {data['doc_date']})")
            for k, v in nc["breakdown"].items():
                if v:
                    print(f"  {k}: {v/1e8:,.1f}億")
            print(f"  → 準現金: {nc['near_cash']/1e8:,.1f}億")
            print(f"  → 有利子負債: {nc['interest_debt']/1e8:,.1f}億")
            print(f"  → 広義ネットキャッシュ: {nc['net_cash']/1e8:,.1f}億")
    else:
        print("Use --codes CODE1 CODE2 or --rebuild-index")
