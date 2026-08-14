"""Close-auction forward seal: date guard + signal-only ledger."""
import datetime as dt
import json

import pytest

import scripts.verify_close_auction_forward as vc
from scripts import collect_close_auction_forward as cc


def test_verdict_sealed_until_judgment():
    assert dt.date.today() < vc.JUDGMENT_DATE
    with pytest.raises(SystemExit, match="SEALED"):
        vc.main()


def test_ledger_is_signal_only(tmp_path, monkeypatch):
    led = tmp_path / "forward_candidates.jsonl"
    led.write_text(json.dumps({"signal_date": "2026-08-12", "n": 2,
                               "symbols": ["13010", "72030"],
                               "recorded_at": "2026-08-12T06:00:00+00:00"}) + "\n",
                   encoding="utf-8")
    monkeypatch.setattr(cc, "LEDGER", led)
    assert cc.seen() == {"2026-08-12"}
    row = json.loads(led.read_text().splitlines()[0])
    assert not (set(row) & {"ret", "pnl", "return", "excess", "sharpe"})
