"""Shrinkage maths and capacity-plan behaviour."""

import numpy as np
import pandas as pd
import pytest

from wfm.capacity import CapacityAssumptions, HiringClass, Shrinkage, project_capacity, recommend_hiring

WEEKS = pd.date_range("2026-10-04", periods=26, freq="7D")


def test_shrinkage_compounds_below_additive():
    s = Shrinkage()
    assert s.total < s.additive_total
    assert s.total == pytest.approx(1 - np.prod([1 - v for v in s.components().values()]))
    assert 1 - s.total == pytest.approx((1 - s.planned) * (1 - s.unplanned))


def test_flat_requirement_attrition_only():
    req = pd.Series(100.0, index=WEEKS)
    plan = project_capacity(req, CapacityAssumptions(starting_hc=100))
    weekly = CapacityAssumptions(starting_hc=100).weekly_attrition
    assert (1 - weekly) ** (52 / 12) == pytest.approx(1 - 0.035)
    assert plan["available_fte"].iloc[-1] == pytest.approx(100 * (1 - weekly) ** 25)


def test_new_hire_class_ramp():
    a = CapacityAssumptions(starting_hc=0, classes=(HiringClass(2, 20),))
    plan = project_capacity(pd.Series(10.0, index=WEEKS), a)
    assert (plan["new_hire_fte"].iloc[:6] == 0).all()  # weeks 2-5 are training at 0% productivity
    assert plan["in_training"].iloc[2] == 20
    nesting = plan["new_hire_fte"].iloc[6]
    assert nesting == pytest.approx(20 * (1 - a.training_attrition) * a.nesting_productivity)
    assert plan["new_hire_fte"].iloc[8] == pytest.approx(20 * (1 - a.training_attrition) * (1 - a.weekly_attrition) ** 2)


def test_recommended_hiring_closes_gap_after_lead_time():
    req = pd.Series(np.linspace(100, 130, len(WEEKS)), index=WEEKS)
    a = CapacityAssumptions(starting_hc=100)
    classes, plan = recommend_hiring(req, a)
    assert classes and min(c.start_week for c in classes) >= a.lead_time_weeks
    assert all(a.min_class <= c.size <= a.max_class for c in classes)
    fixable = plan.iloc[a.lead_time_weeks + a.training_weeks + a.nesting_weeks + 2 :]
    assert (fixable["gap_pct"] > -0.05).all()


def test_no_hiring_when_overstaffed():
    classes, _ = recommend_hiring(pd.Series(50.0, index=WEEKS), CapacityAssumptions(starting_hc=80))
    assert classes == []
