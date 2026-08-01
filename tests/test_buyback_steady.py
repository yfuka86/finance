import pandas as pd
from scripts.backtest_buyback_steady import candidates

def test_steady_requires_cadence_and_low_daily_concentration():
    base={"remaining_pressure":.15,"pace_surprise":.1,"purchase_day_ratio":.6,
          "max_daily_concentration":.2,"remaining_sessions":15}
    x=pd.DataFrame([base,base|{"max_daily_concentration":.3},base|{"purchase_day_ratio":.4}])
    out=candidates(x)
    assert out.index.tolist()==[0]
    assert bool(out.iloc[0].steady)
