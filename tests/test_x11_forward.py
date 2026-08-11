"""X11 sealed-forward guards: date lock and signal-only ledger."""
import datetime as dt
import json

import pytest

import scripts.verify_x11_forward as vx
from scripts import collect_x11_forward as cx


def test_verdict_is_sealed_until_judgment_date():
    assert dt.date.today() < vx.JUDGMENT_DATE, "update this suite at judgment time"
    with pytest.raises(SystemExit, match="SEALED"):
        vx.main()


def test_ledger_rows_never_contain_returns(tmp_path, monkeypatch):
    ledger = tmp_path / "candidates.jsonl"
    ledger.write_text(json.dumps({"signal_date": "2026-08-12", "n": 2,
                                  "symbols": ["13010", "16050"],
                                  "recorded_at": "2026-08-12T10:00:00+00:00"}) + "\n",
                      encoding="utf-8")
    monkeypatch.setattr(cx, "LEDGER", ledger)
    assert cx.seen_dates() == {"2026-08-12"}
    row = json.loads(ledger.read_text().splitlines()[0])
    banned = {"ret", "return", "pnl", "excess", "ir", "sharpe"}
    assert not (set(row) & banned), "ledger must stay signal-only (no-peek)"
