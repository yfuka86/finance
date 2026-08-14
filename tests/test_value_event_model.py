import pandas as pd
import pytest

from trading.jp_intraday.value_event_model import (
    attach_market_and_features, dividend_raise_events, dividend_resumption_events,
    fit_and_select_oos, treasury_cancellation_events,
)


def test_raise_must_be_same_fiscal_year_and_at_least_twenty_percent():
    f = pd.DataFrame([
        {"Code":"12340","DiscDate":"2022-01-01","DiscTime":"15:00","DiscNo":"1","CurFYEn":"2022-12-31","FDivAnn":10},
        {"Code":"12340","DiscDate":"2022-02-01","DiscTime":"15:00","DiscNo":"2","CurFYEn":"2022-12-31","FDivAnn":11},
        {"Code":"12340","DiscDate":"2022-03-01","DiscTime":"15:00","DiscNo":"3","CurFYEn":"2022-12-31","FDivAnn":14},
        {"Code":"12340","DiscDate":"2023-01-01","DiscTime":"15:00","DiscNo":"4","CurFYEn":"2023-12-31","FDivAnn":20},
    ])
    out = dividend_raise_events(f)
    assert out["DiscNo"].tolist() == ["3"]
    assert out.iloc[0].div_raise == pytest.approx(3/11)


def test_financial_state_is_past_only_forward_filled():
    f = pd.DataFrame([
        {"Code":"12340","DiscDate":"2022-01-01","DiscTime":"15:00","DiscNo":"1","CurFYEn":"2022-12-31","FDivAnn":10,"BPS":200},
        {"Code":"12340","DiscDate":"2022-02-01","DiscTime":"15:00","DiscNo":"2","CurFYEn":"2022-12-31","FDivAnn":12},
        {"Code":"99990","DiscDate":"2022-01-01","DiscTime":"15:00","DiscNo":"3","CurFYEn":"2022-12-31","FDivAnn":10},
        {"Code":"99990","DiscDate":"2022-02-01","DiscTime":"15:00","DiscNo":"4","CurFYEn":"2022-12-31","FDivAnn":12,"BPS":999},
    ])
    out = dividend_raise_events(f).set_index("symbol")
    assert out.loc["1234", "BPS"] == 200
    assert out.loc["9999", "BPS"] == 999  # same-row value, never borrowed across issuers


def test_market_join_uses_prior_day_and_next_session_not_event_close():
    event = pd.DataFrame([{"symbol":"1234", "event_date":pd.Timestamp("2022-01-03"),
        "div_raise":.2,"BPS":200,"EPS":10,"OP":20,"Sales":100,"Eq":100,"TA":200,
        "CashEq":20,"NP":10}])
    dates = pd.bdate_range("2021-12-31", periods=62)
    daily = pd.DataFrame({"Date":dates,"Code":"12340","AdjO":100.,"AdjC":110.,
                          "raw_close":110.,"Va":2e9})
    out = attach_market_and_features(event, daily).iloc[0]
    assert out.prior_close == 110
    assert out.entry_date == pd.Timestamp("2022-01-04")
    assert out.exit_date == dates[61]


def test_market_join_accepts_string_dividend_from_raw_financial_cache():
    event=pd.DataFrame([{"symbol":"1234","event_date":pd.Timestamp("2022-01-03"),
      "FDivAnn":"5.0","BPS":200,"EPS":10,"OP":20,"Sales":100,"Eq":100,"TA":200,
      "CashEq":20,"NP":10}])
    dates=pd.bdate_range("2021-12-31",periods=62)
    daily=pd.DataFrame({"Date":dates,"Code":"12340","AdjO":100.,"AdjC":110.,
                        "raw_close":110.,"Va":2e9})
    out=attach_market_and_features(event,daily).iloc[0]
    assert out.dividend_yield==pytest.approx(5/110)


def test_oos_capacity_never_exceeds_ten():
    rows=[]
    for i in range(24):
        event=pd.Timestamp("2022-01-01") if i < 12 else pd.Timestamp("2024-01-02")
        rows.append({"event_date":event,"entry_date":event+pd.Timedelta(days=i),
                     "exit_date":event+pd.Timedelta(days=i+90),"forward_return":.1,
                     **{c:1. for c in ["div_raise","book_to_price","earnings_yield","op_margin","equity_ratio","cash_assets","roe"]}})
    _, out=fit_and_select_oos(pd.DataFrame(rows))
    assert out.selected.sum() == 10


def test_resumption_requires_explicit_zero_same_fiscal_year():
    f=pd.DataFrame([
      {"Code":"12340","DiscDate":"2022-01-01","CurFYEn":"2022-12-31","FDivAnn":0},
      {"Code":"12340","DiscDate":"2022-02-01","CurFYEn":"2022-12-31","FDivAnn":5},
      {"Code":"99990","DiscDate":"2022-01-01","CurFYEn":"2022-12-31","FDivAnn":None},
      {"Code":"99990","DiscDate":"2022-02-01","CurFYEn":"2022-12-31","FDivAnn":5},
    ])
    out=dividend_resumption_events(f)
    assert out.symbol.tolist()==["1234"]


def test_cancellation_requires_issued_and_treasury_to_fall_together():
    base={"BPS":100,"EPS":10,"OP":10,"Sales":100,"Eq":50,"TA":100,"CashEq":10,"NP":5}
    f=pd.DataFrame([
      {"Code":"12340","DiscDate":"2022-01-01","ShOutFY":1000,"TrShFY":100,**base},
      {"Code":"12340","DiscDate":"2022-02-01","ShOutFY":990,"TrShFY":90,**base},
      {"Code":"99990","DiscDate":"2022-01-01","ShOutFY":1000,"TrShFY":100,**base},
      {"Code":"99990","DiscDate":"2022-02-01","ShOutFY":990,"TrShFY":80,**base},
    ])
    out=treasury_cancellation_events(f)
    assert out.symbol.tolist()==["1234"]
    assert out.iloc[0].cancel_fraction==pytest.approx(.01)
