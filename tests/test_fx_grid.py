"""FX grid simulator instrument tests (verify the instrument on known inputs).

The close-cross detection bug this suite guards against: detecting fills at
bar closes conditions entries on adverse continuation (-2.7 SE bias on a
martingale). Touch-based fills must earn exactly dx per oscillation, book
exact inventory losses in trends, and have zero expectation on a martingale.
"""
import numpy as np
import pandas as pd
import pytest

import scripts.experiment_fx_grid as g

ZERO_RATES = pd.DataFrame(
    {c: 0.0 for c in ["USD", "JPY", "EUR", "GBP", "CHF", "AUD", "CAD", "NZD"]},
    index=pd.date_range("2000-01-01", periods=2, freq="YS"))
UNIT = (1.0 / (7 * 10))


def bars_from_path(prices):
    p = np.asarray(prices, dtype=float)
    ts = pd.date_range("2024-01-02", periods=len(p), freq="1min", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": p, "high": p, "low": p, "mid": p,
                         "half_spread": 0.0})


def run(monkeypatch, df):
    monkeypatch.setattr(g, "load_minute",
                        lambda pair, year: df if year == 2024
                        else bars_from_path([1.0] * 200))
    monkeypatch.setattr(g, "grid_spacing", lambda pair, year: 0.01)
    return g.simulate_pair_year("EUR_USD", 2024, 0.5, False, 1.0, ZERO_RATES)


def test_oscillation_earns_exactly_dx_per_cycle(monkeypatch):
    path = [1.0]
    for _ in range(3):
        path += [0.985, 1.005]
    out = run(monkeypatch, bars_from_path(path + [1.005]))
    assert out["pnl_usd"].sum() == pytest.approx(3 * 0.01 * UNIT, abs=1e-12)


def test_trend_books_exact_inventory_loss(monkeypatch):
    out = run(monkeypatch, bars_from_path(np.linspace(1.0, 0.875, 200)))
    lines = [1.0 - k * 0.01 for k in range(1, 11)]        # cap 10 lines
    expect = sum(0.875 - l for l in lines) * UNIT
    assert out["pnl_usd"].sum() == pytest.approx(expect, abs=1e-12)


def test_intrabar_touch_fills_and_takes_profit(monkeypatch):
    ts = pd.date_range("2024-01-02", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame({"ts": ts, "open": [1.0] * 3, "high": [1.0, 1.0005, 1.0],
                       "low": [1.0, 0.9885, 1.0], "mid": [1.0] * 3,
                       "half_spread": 0.0})
    out = run(monkeypatch, df)
    assert out["pnl_usd"].sum() == pytest.approx(0.01 * UNIT, abs=1e-12)


def test_martingale_expectation_is_zero(monkeypatch):
    rng = np.random.default_rng(1)
    sub, nbar = 20, 500
    tots = []
    for _ in range(40):
        fine = 1.0 + np.cumsum(rng.standard_normal(nbar * sub) * 0.002 / sub ** .5)
        fine = np.concatenate(([1.0], fine))
        bars = fine[1:].reshape(nbar, sub)
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-02", periods=nbar, freq="1min", tz="UTC"),
            "open": np.concatenate(([fine[0]], bars[:-1, -1])),
            "high": bars.max(axis=1), "low": bars.min(axis=1),
            "mid": bars[:, -1], "half_spread": 0.0})
        tots.append(run(monkeypatch, df)["pnl_usd"].sum())
    tots = np.asarray(tots)
    se = tots.std() / len(tots) ** .5
    assert abs(tots.mean()) < 3 * se
