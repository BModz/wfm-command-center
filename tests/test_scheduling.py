"""Shift MILP coverage and labour-law compliance."""

import numpy as np
import pytest

from wfm.config import INTERVALS_PER_DAY
from wfm.scheduling import ShiftRules, optimise_shifts, required_shifts_lower_bound, schedule_requirement


def _day_curve(peak: float) -> np.ndarray:
    hours = np.arange(INTERVALS_PER_DAY) / 2
    return np.round(2 + peak * np.exp(-0.5 * ((hours - 14) / 4) ** 2))


def test_break_options_respect_five_hour_rule():
    rules = ShiftRules()
    offsets = rules.break_offsets()
    work = rules.length_intervals - rules.break_intervals
    assert rules.length_intervals == 18 and rules.break_intervals == 2
    assert all(o <= 10 and work - o <= 10 for o in offsets)
    assert offsets == [6, 7, 8, 9, 10]


def test_single_day_full_coverage_with_hard_penalty():
    req = _day_curve(30)
    res = optimise_shifts(req, ShiftRules(under_penalty=10.0), cyclic=True)
    cov = res.coverage
    assert (cov["scheduled_on_phone"] >= cov["requirement"]).all()
    assert res.stats["coverage_pct"] == pytest.approx(1.0)
    assert res.stats["shifts"] >= required_shifts_lower_bound(req)
    assert 0.8 < res.stats["efficiency_pct"] <= 1.0


def test_week_cyclic_overnight_wraps():
    req = np.tile(np.full(INTERVALS_PER_DAY, 4.0), 7)
    res = optimise_shifts(req, ShiftRules(under_penalty=10.0), cyclic=True, time_limit_s=5)
    assert (res.coverage["scheduled_on_phone"] >= 4).all()
    assert res.stats["coverage_pct"] == pytest.approx(1.0)


def test_open_hours_restrict_shifts():
    req = np.zeros(INTERVALS_PER_DAY)
    open_mask = np.zeros(INTERVALS_PER_DAY, dtype=bool)
    open_mask[18:46] = True  # 09:00-23:00
    req[open_mask] = 5
    res = optimise_shifts(req, ShiftRules(under_penalty=10.0), cyclic=False, open_intervals=open_mask)
    assert res.coverage.loc[~open_mask, "on_site"].sum() == 0
    assert res.coverage.loc[open_mask, "scheduled_on_phone"].min() >= 5


def test_soft_penalty_allows_small_gaps():
    req = np.zeros(INTERVALS_PER_DAY)
    req[20] = 1  # a single isolated half-hour is not worth a whole shift
    res = optimise_shifts(req, ShiftRules(under_penalty=0.3), cyclic=False)
    assert res.stats["shifts"] == 0 and res.stats["under_hours"] == 0.5


def test_schedule_requirement_grosses_up_for_in_day_shrinkage():
    assert schedule_requirement(np.array([10, 0]), 0.1).tolist() == [12, 0]
    assert schedule_requirement(np.array([9]), 0.1).tolist() == [10]


def test_contract_respects_egyptian_weekly_limits():
    from wfm.labour import EGYPT_LABOUR

    rules = ShiftRules()
    assert rules.paid_hours <= EGYPT_LABOUR.max_daily_hours
    assert rules.paid_hours * rules.shifts_per_fte <= EGYPT_LABOUR.max_weekly_hours
    rest_days = 7 - rules.shifts_per_fte
    assert rest_days * 24 >= EGYPT_LABOUR.min_weekly_rest_hours


def _sales_like_week() -> tuple[np.ndarray, np.ndarray]:
    """Two daily peaks inside 09:00-23:00, the shape fixed 9-hour shifts cover poorly."""
    hours = np.arange(INTERVALS_PER_DAY) / 2
    day = 2 + 10 * np.exp(-0.5 * ((hours - 12.5) / 1.5) ** 2) + 10 * np.exp(-0.5 * ((hours - 20) / 1.5) ** 2)
    open_mask = (hours >= 9) & (hours < 23)
    return np.tile(np.where(open_mask, np.round(day), 0), 7), np.tile(open_mask, 7)


def test_part_time_shifts_lift_efficiency_and_respect_rules():
    req, open_mask = _sales_like_week()
    full = optimise_shifts(req, ShiftRules(under_penalty=10.0), open_intervals=open_mask, time_limit_s=10)
    rules = ShiftRules(under_penalty=10.0, part_time=True)
    mixed = optimise_shifts(req, rules, open_intervals=open_mask, time_limit_s=10)
    assert mixed.stats["efficiency_pct"] > full.stats["efficiency_pct"]
    assert mixed.stats["paid_hours"] < full.stats["paid_hours"]
    assert mixed.stats["part_time_hours_share"] <= rules.max_part_time_share + 1e-9
    pt = mixed.shifts[mixed.shifts["shift_type"] != "Full-time"]
    assert len(pt) and (pt["paid_hours"] <= rules.max_continuous_hours).all()
    assert (pt["break_start"] == "").all()  # no break needed up to 5 hours


def test_roster_gives_five_shifts_and_two_consecutive_rest_days():
    from wfm.scheduling import DAY_NAMES, optimise_roster

    req = np.tile(_day_curve(12), 7)
    res = optimise_roster(req, ShiftRules(under_penalty=10.0), time_limit_s=20)
    roster = res.roster
    assert len(roster) == res.stats["agents"] > 0
    off = roster[DAY_NAMES].eq("OFF")
    assert (off.sum(axis=1) == 2).all()
    for row in off.to_numpy():
        days = np.flatnonzero(row)
        assert (days[1] - days[0]) % 7 in (1, 6)  # consecutive, allowing Sat-Sun wrap
    assert res.stats["coverage_pct"] == pytest.approx(1.0)
    shift_fte = optimise_shifts(req, ShiftRules(under_penalty=10.0), time_limit_s=20).stats["fte_equivalent"]
    assert res.stats["agents"] >= shift_fte - 1e-9  # roster constraints can only cost heads
