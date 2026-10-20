"""Simulator reproducibility and realism checks."""

import numpy as np
import pandas as pd
import pytest

from wfm.config import INTERVALS_PER_DAY, QUEUE_CODES, RTA_QUEUE, RTA_REPLAY_DATE
from wfm.egypt_calendar import build_calendar
from wfm.simulate import intraday_profile, simulate


@pytest.fixture(scope="module")
def small_sim():
    return simulate("2026-02-01", "2026-03-31", seed=7)


def test_reproducible_with_fixed_seed(small_sim):
    again = simulate("2026-02-01", "2026-03-31", seed=7)
    pd.testing.assert_frame_equal(small_sim["intervals"], again["intervals"])
    pd.testing.assert_frame_equal(small_sim["events"], again["events"])


def test_different_seed_changes_data(small_sim):
    other = simulate("2026-02-01", "2026-03-31", seed=8)
    assert not small_sim["intervals"]["offered"].equals(other["intervals"]["offered"])


def test_shape_and_accounting(small_sim):
    iv = small_sim["intervals"]
    days = 59
    assert len(iv) == days * INTERVALS_PER_DAY * len(QUEUE_CODES)
    assert (iv["handled"] + iv["abandoned"] == iv["offered"]).all()
    assert (iv["answered_in_sl"] <= iv["handled"]).all()


def test_sales_closed_overnight(small_sim):
    iv = small_sim["intervals"]
    sales = iv[iv["queue"] == "SALES"]
    closed = sales[(sales["interval"] < 18) | (sales["interval"] >= 46)]
    assert closed["offered"].sum() == 0
    assert closed["staffed_agents"].sum() == 0


def test_profiles_sum_to_one_and_ramadan_has_iftar_dip():
    for q in QUEUE_CODES:
        for day_type in ("weekday", "friday", "saturday"):
            for ramadan in (False, True):
                assert intraday_profile(q, day_type, ramadan).sum() == pytest.approx(1.0)
    normal = intraday_profile("AR_BILLING", "weekday", False)
    ramadan = intraday_profile("AR_BILLING", "weekday", True)
    iftar, late_night = slice(36, 38), slice(44, 48)  # 18:00-19:00 and 22:00-24:00
    assert ramadan[iftar].sum() < 0.5 * normal[iftar].sum()
    assert ramadan[late_night].sum() > 2 * normal[late_night].sum()


def test_calendar_egyptian_week_and_holidays():
    cal = build_calendar("2025-01-01", "2025-12-31")
    assert cal.loc[cal["date"] == "2025-01-07", "holiday_name"].item() == "Coptic Christmas"
    assert cal.loc[cal["date"] == "2025-03-30", "is_eid"].item()
    assert cal.loc[cal["date"] == "2025-04-02", "is_post_eid"].item()
    assert cal["is_ramadan"].sum() == 29
    assert (cal.loc[cal["dow_name"] == "Fri", "day_type"] == "friday").all()
    assert (cal["week_start"].dt.dayofweek == 6).all()  # weeks start on Sunday


def test_sunday_busier_than_friday():
    data = simulate("2025-10-01", "2025-11-30", seed=1)
    iv = data["intervals"]
    daily = iv.groupby(["date"]).offered.sum()
    dow = daily.groupby(daily.index.dayofweek).mean()
    assert dow[6] > 1.3 * dow[4]


def test_rta_outage_day_exists():
    data = simulate("2026-09-20", "2026-09-23", seed=3)
    ev = data["events"]
    outage = ev[(ev["event_type"] == "outage") & (ev["start"].dt.normalize() == RTA_REPLAY_DATE)]
    assert len(outage) == 1 and outage["queue"].item() == RTA_QUEUE
    iv = data["intervals"]
    day = iv[(iv["queue"] == RTA_QUEUE) & (iv["date"] == RTA_REPLAY_DATE)]
    peak = day.loc[day["interval"].between(28, 33), "offered"].mean()
    before = day.loc[day["interval"].between(22, 26), "offered"].mean()
    assert peak > 2 * before
    assert np.nanmean(day.loc[day["interval"].between(29, 33), "abandon_rate"]) > 0.15
