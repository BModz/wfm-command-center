"""Agent flagging escalation ladder, Staffing Outlook, midday reforecast and RCA."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from conftest import DAY, QUEUE, SURGE

from wfm.config import INTERVALS_PER_DAY
from wfm.realtime import AUX, BREAK, OFF, PHONE, SLOTS_PER_INTERVAL, interval_label
from wfm.rta_reports import FlagRules, flag_agents, midday_reforecast, rca_markdown, rca_report, staffing_outlook

NOW = 30  # 15:00
SLOT_NOW = NOW * SLOTS_PER_INTERVAL


def _controlled_day(rta_day):
    """Copy of the fixture day where four on-shift agents get known exceptions ending at NOW."""
    sched, actual = rta_day.scheduled.copy(), rta_day.scheduled.copy()  # everyone perfectly adherent
    on_phone = np.flatnonzero((sched[:, SLOT_NOW - 30 : SLOT_NOW] == PHONE).all(axis=1))
    aux, overrun, noshow, worker = on_phone[:4]
    actual[aux, SLOT_NOW - 4 : SLOT_NOW] = AUX  # 20 minutes of unscheduled aux -> team leader
    actual[overrun, SLOT_NOW - 1 : SLOT_NOW] = BREAK  # 5 minutes on break while scheduled phone -> nudge
    actual[noshow, :] = OFF  # never logged in -> operations manager
    sched[worker, SLOT_NOW - 2 : SLOT_NOW] = BREAK  # scheduled break but still on the phone -> monitor
    day = dataclasses.replace(rta_day, scheduled=sched, actual=actual)
    ids = day.agents["agent_id"]
    return day, ids[aux], ids[overrun], ids[noshow], ids[worker]


def test_escalation_ladder(rta_day):
    day, aux, overrun, noshow, worker = _controlled_day(rta_day)
    flags = flag_agents(day, NOW).set_index("agent_id")
    assert flags.loc[aux, "level"] == "Escalate to team leader"
    assert flags.loc[aux, "minutes_out"] == 20
    assert flags.loc[overrun, "level"] == "Nudge agent"
    assert flags.loc[noshow, "level"] == "Escalate to operations manager"
    assert flags.loc[noshow, "issue"].startswith("No-show")
    assert flags.loc[worker, "level"] == "Monitor"
    assert flags["priority"].is_monotonic_increasing
    assert aux in flags.loc[aux, "message"] or "TL" in flags.loc[aux, "message"]


def test_thresholds_are_configurable(rta_day):
    day, aux, *_ = _controlled_day(rta_day)
    strict = flag_agents(day, NOW, FlagRules(team_leader_after_min=30, manager_after_min=60)).set_index("agent_id")
    assert strict.loc[aux, "level"] == "Nudge agent"


def test_coaching_follow_up_for_low_adherence(rta_day):
    sched, actual = rta_day.scheduled.copy(), rta_day.scheduled.copy()
    worked_3h = (sched[:, SLOT_NOW - 36 : SLOT_NOW] != OFF).all(axis=1) & (sched[:, SLOT_NOW - 1] == PHONE)
    agent = int(np.flatnonzero(worked_3h)[0])
    earlier = np.arange(SLOT_NOW - 36, SLOT_NOW - 6)
    actual[agent, earlier[sched[agent, earlier] == PHONE]] = AUX  # long aux earlier, back on the phone now
    day = dataclasses.replace(rta_day, scheduled=sched, actual=actual)
    flags = flag_agents(day, NOW).set_index("agent_id")
    assert flags.loc[day.agents["agent_id"][agent], "level"] == "Coaching follow-up"


def test_staffing_outlook_two_days():
    intervals = np.arange(INTERVALS_PER_DAY)
    plan = pd.concat(
        [
            pd.DataFrame({"date": d, "interval": intervals, "forecast_offered": 100.0, "forecast_aht_s": 360.0, "scheduled_on_phone": staff})
            for d, staff in ((DAY, 30.0), (DAY + pd.Timedelta(days=1), 18.0))
        ]
    )
    table, summary = staffing_outlook(plan, QUEUE, expected_loss=0.0)
    assert len(table) == 2 * INTERVALS_PER_DAY and len(summary) == 2
    assert summary.loc[0, "net_over_under_hours"] > 0 > summary.loc[1, "net_over_under_hours"]
    assert summary.loc[1, "short_intervals"] == INTERVALS_PER_DAY
    assert summary.loc[0, "projected_sl"] > 0.8 > summary.loc[1, "projected_sl"]


def test_midday_reforecast_accounts_for_the_day_so_far(rta_day):
    summary, table = midday_reforecast(rta_day, 24)
    iv = rta_day.intervals
    assert summary["offered_so_far"] == iv["offered"].iloc[:24].sum()
    assert table["offered"].iloc[24:].isna().all() and table["reforecast_offered"].iloc[:24].isna().all()


def test_rca_finds_the_incident(rta_day):
    rca = rca_report(rta_day)
    times = [t for t, _ in rca["timeline"]]
    assert times == sorted(times)
    assert times[0] == interval_label(SURGE.start)
    assert "outage" in rca["root_cause"].lower()
    md = rca_markdown(rca)
    for section in ("## Timeline", "## Impact", "## Root cause", "## Preventive actions"):
        assert section in md


def test_rca_requires_an_incident(rta_day):
    calm = rta_day.intervals.copy()
    calm["offered"] = calm["forecast_offered"].round().astype(int)
    with pytest.raises(ValueError):
        rca_report(dataclasses.replace(rta_day, intervals=calm))
