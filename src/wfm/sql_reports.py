"""Standard WFM reports written in SQL, run on an in-memory SQLite copy of the data.

Each report is a plain ``.sql`` file in ``wfm/sql`` with named parameters
(``:queue``, ``:date``, ``:start``, ``:end``), so the queries can be read,
reused in another database and shown next to their results in the app.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wfm.realtime import SLOTS_PER_DAY, STATE_LABELS, RTADay

SQL_DIR = Path(__file__).resolve().parent / "sql"


@dataclass(frozen=True)
class SqlReport:
    """A named report backed by ``sql/<key>.sql``."""

    key: str
    title: str
    purpose: str
    params: tuple[str, ...]

    @property
    def sql(self) -> str:
        return (SQL_DIR / f"{self.key}.sql").read_text()


REPORTS: dict[str, SqlReport] = {
    r.key: r
    for r in (
        SqlReport("interval_report", "Interval report", "Plan vs actual every 30 minutes for one queue and day.", ("queue", "date")),
        SqlReport("daily_kpis", "Daily KPI summary", "Daily SL, abandons, ASA and AHT with a 7-day rolling SL and week-on-week volume.", ("queue", "start", "end")),
        SqlReport("plan_vs_actual", "Plan vs actual (weekly)", "The centre's historical plan: forecast bias and interval accuracy, and staffing delivered vs scheduled, by week and queue.", ("start", "end")),
        SqlReport("shrinkage_report", "Shrinkage report", "Weekly in-day shrinkage (scheduled phone time lost), ranked within each queue.", ("start", "end")),
        SqlReport("worst_intervals", "Worst intervals (RCA input)", "The 10 worst service-level intervals per queue, with the likely driver.", ("start", "end")),
        SqlReport("outage_impact", "Outage impact", "Volume, abandons and service level inside every network-outage window.", ("start", "end")),
        SqlReport("adherence_report", "Adherence report", "Agent adherence and conformance on the RTA replay day, against the team average.", ()),
    )
}


def _iso(frame: pd.DataFrame, columns: list[str], fmt: str) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col] = pd.to_datetime(out[col]).dt.strftime(fmt)
    return out


def build_database(intervals: pd.DataFrame, calendar: pd.DataFrame, events: pd.DataFrame, rta: RTADay | None = None) -> sqlite3.Connection:
    """Load the WFM tables into an in-memory SQLite database.

    Tables: intervals, calendar, events and, when ``rta`` is given, agents and
    agent_states (one row per agent per 5-minute slot of the replay day).
    Dates are stored as ISO text (YYYY-MM-DD and YYYY-MM-DD HH:MM:SS).
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    iv = _iso(intervals, ["date"], "%Y-%m-%d")
    iv = _iso(iv, ["interval_start"], "%Y-%m-%d %H:%M:%S")
    iv["queue"] = iv["queue"].astype(str)
    iv.to_sql("intervals", conn, index=False)
    cal = _iso(calendar, ["date", "week_start"], "%Y-%m-%d")
    cal.to_sql("calendar", conn, index=False)
    _iso(events, ["start", "end"], "%Y-%m-%d %H:%M:%S").to_sql("events", conn, index=False)
    conn.execute("CREATE INDEX ix_intervals ON intervals (queue, date)")
    conn.execute("CREATE INDEX ix_calendar ON calendar (date)")
    if rta is not None:
        rta.agents[["agent_id", "team"]].to_sql("agents", conn, index=False)
        n = len(rta.agents)
        states = pd.DataFrame(
            {
                "agent_id": np.repeat(rta.agents["agent_id"].to_numpy(), SLOTS_PER_DAY),
                "slot": np.tile(np.arange(SLOTS_PER_DAY), n),
                "scheduled_state": pd.Series(rta.scheduled.ravel()).map(STATE_LABELS).to_numpy(),
                "actual_state": pd.Series(rta.actual.ravel()).map(STATE_LABELS).to_numpy(),
            }
        )
        states.to_sql("agent_states", conn, index=False)
    return conn


def run_report(conn: sqlite3.Connection, key: str, **params: str) -> pd.DataFrame:
    """Run report ``key`` with its named parameters and return the result as a DataFrame."""
    report = REPORTS[key]
    missing = [p for p in report.params if p not in params]
    if missing:
        raise ValueError(f"Report {key} needs parameters: {', '.join(missing)}")
    return pd.read_sql_query(report.sql, conn, params={p: params[p] for p in report.params})
