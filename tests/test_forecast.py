"""Forecast output shapes, leakage and anomaly-treatment checks."""

import numpy as np
import pandas as pd
import pytest

from wfm.config import INTERVALS_PER_DAY, QUEUE_CODES
from wfm.forecast import (
    MODELS,
    BacktestWindow,
    clean_outages,
    daily_actuals,
    forecast_daily,
    interval_forecast,
    learn_profiles,
    run_backtest,
    wape,
)
from wfm.simulate import simulate

TRAIN_END = pd.Timestamp("2025-12-27")
HORIZON = pd.date_range("2025-12-28", "2026-01-24", freq="D")


@pytest.fixture(scope="module")
def data():
    sim = simulate("2024-10-01", "2026-01-31", seed=11)
    clean = clean_outages(sim["intervals"], sim["events"])
    return sim, clean, daily_actuals(clean)


def test_outage_cleaning_reduces_only_flagged_intervals(data):
    sim, clean, _ = data
    assert clean["is_outage"].any()
    unflagged = clean[~clean["is_outage"]]
    assert (unflagged["offered_clean"] == unflagged["offered"]).all()
    flagged = clean[clean["is_outage"] & (clean["queue"] == "AR_TECH")]
    assert flagged["offered_clean"].sum() < 0.7 * flagged["offered"].sum()


def test_daily_forecast_shape(data):
    sim, _, daily = data
    fc = forecast_daily(daily, sim["calendar"], sim["events"], TRAIN_END, HORIZON)
    assert len(fc) == len(HORIZON) * len(QUEUE_CODES) * len(MODELS)
    assert set(fc["model"]) == set(MODELS)
    assert fc["forecast"].notna().all() and (fc["forecast"] > 0).all()


def test_no_leakage_future_actuals_do_not_change_forecast(data):
    sim, _, daily = data
    base = forecast_daily(daily, sim["calendar"], sim["events"], TRAIN_END, HORIZON)
    tampered = daily.copy()
    future = tampered["date"] > TRAIN_END
    tampered.loc[future, ["offered", "offered_clean"]] *= 10
    again = forecast_daily(tampered, sim["calendar"], sim["events"], TRAIN_END, HORIZON)
    pd.testing.assert_frame_equal(base, again)


def test_interval_forecast_sums_to_daily_and_respects_hours(data):
    sim, clean, daily = data
    fc = forecast_daily(daily, sim["calendar"], sim["events"], TRAIN_END, HORIZON)
    lgb = fc[fc["model"] == "LightGBM"][["date", "queue", "forecast"]]
    ivl = interval_forecast(lgb, sim["calendar"], learn_profiles(clean, sim["calendar"], TRAIN_END))
    assert len(ivl) == len(lgb) * INTERVALS_PER_DAY
    totals = ivl.groupby(["date", "queue"])["forecast_offered"].sum().reset_index()
    merged = totals.merge(lgb, on=["date", "queue"])
    np.testing.assert_allclose(merged["forecast_offered"], merged["forecast"], rtol=1e-9)
    sales_night = ivl[(ivl["queue"] == "SALES") & (ivl["interval"] < 18)]
    assert sales_night["forecast_offered"].sum() == 0
    assert sales_night["forecast_aht_s"].isna().all()


def test_backtest_beats_naive_on_holiday_heavy_window(data):
    sim, clean, daily = data
    window = BacktestWindow("test", TRAIN_END, HORIZON[0], HORIZON[-1])
    res = run_backtest(clean, daily, sim["calendar"], sim["events"], window)
    total = res.daily_metrics[res.daily_metrics["queue"] == "All queues"].set_index("model")["wape"]
    assert total["LightGBM"] < total["Seasonal naive"]
    assert total["LightGBM"] < 0.10


def test_wape_definition():
    assert wape(np.array([100, 100]), np.array([90, 120])) == pytest.approx(0.15)
