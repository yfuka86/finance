import json
import pytest
from trading.jp_intraday.event_store import append_events, validate_event

def event(revision=0):
    return {"event_id":"TDNET:1234:x","security_code":"1234","event_family":"buyback",
            "source":"TDnet","source_published_at":"2026-08-03T15:00:00+09:00",
            "first_received_at":"2026-08-03T15:00:01+09:00","effective_at":"2026-08-03T15:00:00+09:00",
            "revision_no":revision,"document_hash":f"hash{revision}","state":"announced",
            "terms_json":{"max_yen":100},"first_tradable_at":"2026-08-04T09:00:00+09:00"}

def test_append_only_allows_revision_but_not_overwrite(tmp_path):
    p=tmp_path/"events.jsonl"; assert append_events(p,[event(0)])==1
    assert append_events(p,[event(1)])==1
    with pytest.raises(ValueError,match="append-only duplicate"): append_events(p,[event(1)])
    assert len(p.read_text().splitlines())==2

def test_pit_timestamp_order_is_enforced():
    x=event(); x["first_tradable_at"]="2026-08-03T14:59:00+09:00"
    with pytest.raises(ValueError,match="precedes receipt"): validate_event(x)
