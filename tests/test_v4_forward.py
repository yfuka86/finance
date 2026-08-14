"""V4 forward-seal regression tests: guard, ledger key, eligibility gates."""
import datetime as dt
import json

import pandas as pd
import pytest

from scripts import collect_fins_incremental as cfi
from scripts import collect_v4_forward_events as cve
import scripts.verify_v4_forward as vf


def test_verdict_is_sealed_until_judgment_date():
    assert dt.date.today() < vf.JUDGMENT_DATE, "update this suite at judgment time"
    with pytest.raises(SystemExit, match="SEALED"):
        vf.main()


def test_fins_cursor_falls_back_to_snapshot_end_and_reads_max_discdate():
    assert cfi._cursor({}) == cfi.SNAPSHOT_END
    payload = {"1301": [{"DiscDate": "2026-05-01"}, {"DiscDate": "2026-06-15"}],
               "1302": [{"DiscDate": "2026-05-20"}]}
    assert cfi._cursor(payload) == dt.date(2026, 6, 15)


def test_ledger_keys_dedupe_on_symbol_fy_and_date(tmp_path, monkeypatch):
    ledger = tmp_path / "events.jsonl"
    row = {"symbol": "7128", "CurFYEn": "2026-12-31", "event_date": "2026-07-09"}
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(cve, "LEDGER", ledger)
    assert cve._existing_keys() == {("7128", "2026-12-31", "2026-07-09")}


def test_forward_eligible_applies_frozen_gates():
    events = pd.DataFrame({
        "symbol": ["0001", "0002", "0003"],
        "event_date": pd.to_datetime(["2026-06-02"] * 3),
        "div_raise": [.25, .25, .25],
        "BPS": [1000.0, 100.0, 1000.0],   # 0002: pbr=8 (fails), 0003 fine
        "EPS": [50.0, 50.0, 50.0], "OP": [1.0, 1.0, 1.0],
        "Sales": [10.0] * 3, "Eq": [5.0] * 3, "TA": [10.0] * 3,
        "CashEq": [1.0] * 3, "NP": [1.0] * 3,
    })
    daily = pd.DataFrame({
        "Date": pd.to_datetime(["2026-06-01"] * 3),
        "Code": ["00010", "00020", "00030"],
        "raw_close": [800.0, 800.0, 800.0],
        "Va": [2e8, 2e8, 5e7],            # 0003: below the 1e8 floor
    })
    out = cve.forward_eligible(events, daily)
    assert list(out["symbol"]) == ["0001"]
    assert out["pbr"].iloc[0] == pytest.approx(0.8)
