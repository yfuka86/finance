import pandas as pd
from scripts.buyback_emergent_signals import candidates

def test_corporate_put_requires_anchor_and_acceleration():
    base={"program_id":"a","doc_id":"1","month_shares":1_000_000,"purchase_month_sessions":10,
      "adv20_shares":10_000_000,"prior_close":100,"cumulative_yen":1_000_000_000,
      "cumulative_shares":10_000_000,"max_shares":50_000_000,"remaining_pressure":.1,"period_end":"2026-12-31",
      "daily_detail_consistent":True}
    x=pd.DataFrame([base|{"submit_at":"2026-06-01"},base|{"doc_id":"2","month_shares":2_500_000,
      "cumulative_yen":4_950_000_000,"cumulative_shares":50_000_000,
      "max_shares":100_000_000,"submit_at":"2026-07-01"}])
    x.submit_at=pd.to_datetime(x.submit_at);x.period_end=pd.to_datetime(x.period_end)
    out=candidates(x,pd.Timestamp("2026-07-15"))
    assert out.iloc[0].state=="CORPORATE_PUT_LONG"
