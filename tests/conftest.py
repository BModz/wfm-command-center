"""Shared fixtures: a small, fully synthetic RTA replay day with an afternoon surge."""

import numpy as np
import pandas as pd
import pytest

from wfm.config import INTERVALS_PER_DAY
from wfm.erlang import required_agents
from wfm.realtime import RTADay, build_rta_day
from wfm.scheduling import ShiftRules, optimise_shifts, schedule_requirement

DAY = pd.Timestamp("2026-09-22")
QUEUE = "AR_TECH"
SURGE = slice(28, 34)  # 14:00-17:00


@pytest.fixture(scope="session")
def rta_day() -> RTADay:
    hours = np.arange(INTERVALS_PER_DAY) / 2
    forecast = 20 + 140 * np.exp(-0.5 * ((hours - 15) / 4.5) ** 2)
    offered = np.round(forecast).astype(int)
    offered[SURGE] *= 3
    common = {"queue": QUEUE, "date": DAY, "interval": np.arange(INTERVALS_PER_DAY)}
    history = pd.DataFrame({**common, "offered": offered, "aht_s": np.full(INTERVALS_PER_DAY, 340.0)})
    day_fc = pd.DataFrame({**common, "forecast_offered": forecast, "forecast_aht_s": np.full(INTERVALS_PER_DAY, 380.0)})
    req = schedule_requirement(required_agents(forecast, 380.0), 0.03)
    shifts = optimise_shifts(req, ShiftRules(under_penalty=5.0), cyclic=True, start=DAY).shifts
    return build_rta_day(history, day_fc, shifts, DAY, DAY, QUEUE, in_day_offline=0.03, seed=1)
