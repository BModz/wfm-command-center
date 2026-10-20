"""RTA replay: roster, adherence vs conformance, incident detection and actions."""

import numpy as np
import pandas as pd
import pytest
from conftest import DAY, QUEUE, SURGE

from wfm.config import INTERVALS_PER_DAY
from wfm.realtime import (
    BREAK,
    OFF,
    PHONE,
    SLOTS_PER_INTERVAL,
    RTADay,
    adherence_table,
    detect_incident,
    reforecast,
    snapshot,
)


def test_roster_matches_schedule(rta_day):
    agents, sched = rta_day.agents, rta_day.scheduled
    assert len(agents) == sched.shape[0] > 0
    on_site = (sched != OFF).sum(axis=1)
    assert ((sched == BREAK).sum(axis=1)[on_site == 18 * SLOTS_PER_INTERVAL] == 2 * SLOTS_PER_INTERVAL).all()
    assert agents["agent_id"].is_unique


def test_adherence_vs_conformance():
    sched = np.full((2, INTERVALS_PER_DAY * SLOTS_PER_INTERVAL), OFF, dtype=np.int8)
    sched[:, 0:96] = PHONE
    sched[:, 40:52] = BREAK
    actual = sched.copy()
    # Agent 2 takes the full break 30 minutes late: same worked time, different timing.
    actual[1, 40:52] = PHONE
    actual[1, 46:58] = BREAK
    agents = pd.DataFrame(
        {
            "agent_id": ["A1", "A2"],
            "team": "T",
            "shift_start": DAY,
            "shift_end": DAY + pd.Timedelta(hours=8),
            "break_start": DAY + pd.Timedelta(hours=40 * 5 / 60),
            "exception": "",
        }
    )
    day = RTADay(DAY, QUEUE, pd.DataFrame(), agents, sched, actual)
    table = adherence_table(day, 16).set_index("agent_id")
    assert table.loc["A1", "adherence"] == 1.0
    assert table.loc["A2", "adherence"] < 0.95
    assert table.loc["A2", "conformance"] == pytest.approx(1.0)


def test_incident_detection_and_decaying_reforecast(rta_day):
    iv = rta_day.intervals
    assert not detect_incident(iv, SURGE.start)
    assert detect_incident(iv, SURGE.start + 1)
    rf = reforecast(iv, SURGE.start + 2, incident=True)
    ratio = (rf["reforecast_offered"] / rf["forecast_offered"]).iloc[SURGE.start + 2 :].to_numpy()
    assert ratio[0] > 1.5 and ratio[-1] < 1.2
    calm = reforecast(iv, SURGE.stop + 2, incident=False)
    assert calm.attrs["volume_ratio"] == pytest.approx(1.0, abs=0.1)


def test_snapshot_during_incident_alerts_and_actions(rta_day):
    snap = snapshot(rta_day, SURGE.start + 2)
    names = {a["alert"] for a in snap.alerts}
    assert {"Possible incident", "Volume above forecast"} <= names
    actions = [a["action"] for a in snap.actions]
    assert actions[0] == "Enable IVR outage message and callback"
    assert snap.action_effect["sl_after"] > snap.action_effect["sl_before"]


def test_snapshot_quiet_morning_has_no_incident(rta_day):
    snap = snapshot(rta_day, 20)
    assert not snap.incident
    assert all(a["alert"] != "Possible incident" for a in snap.alerts)
    assert 0.8 < snap.kpis["adherence"] <= 1.0
