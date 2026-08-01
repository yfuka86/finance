import pandas as pd
from scripts.backtest_buyback_persistence import persistence_candidates

def test_persistence_requires_current_and_prior_execution():
    base={"program_id":"a","doc_id":"1","month_shares":60,"purchase_month_sessions":10,
      "adv20_shares":100,"purchase_day_ratio":.6,"max_daily_concentration":.2,
      "remaining_pressure":.1,"remaining_sessions":15,"daily_detail_consistent":True}
    x=pd.DataFrame([base|{"submit_at":"2026-01-01"},base|{"doc_id":"2","submit_at":"2026-02-01"},
                    base|{"program_id":"b","doc_id":"3","submit_at":"2026-02-01"}])
    assert persistence_candidates(x).program_id.tolist()==["a"]
