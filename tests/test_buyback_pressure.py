import pandas as pd
import pytest
from trading.jp_intraday.buyback_pressure import normalize_tdnet_record, pressure_features
from trading.jp_intraday.event_store import validate_event

def test_pressure_uses_tighter_share_or_yen_capacity():
    p=pd.DataFrame([{"max_shares":1000,"max_yen":8000,"cumulative_shares":200,
        "cumulative_yen":2000,"prior_close":10,"remaining_sessions":10,"elapsed_sessions":5,
        "total_sessions":20,"adv20_shares":100}])
    x=pressure_features(p).iloc[0]
    assert x.remaining_shares==800 and x.remaining_capacity_shares==600
    assert x.remaining_pressure==pytest.approx(.6)
    assert x.pace_surprise==pytest.approx(-.2)

def test_tdnet_normalization_is_pit_valid():
    r={"DisclosedDate":"2026-08-03","DisclosedTime":"15:30:00","Code":"12340",
       "DisclosureType":"start","BoardMeetingDate":"2026-08-03","MaximumSharesToBeAcquired":100}
    e=normalize_tdnet_record(r,"2026-08-03T15:30:02+09:00","2026-08-04T09:00:00+09:00")
    assert validate_event(e)["state"]=="announced"
