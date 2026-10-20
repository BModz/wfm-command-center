"""Roster: named weekly working patterns with two consecutive rest days, from a tour-scheduling MILP."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, QUEUE_COLORS, chart, kpi, load_metadata, load_processed, pct, queue_label, setup_page, style

from wfm.config import QUEUES
from wfm.labour import EGYPT_LABOUR
from wfm.scheduling import DAY_NAMES, days_off_label

setup_page("Roster")
st.title("Weekly roster")
st.caption(
    "The shift schedule says how many agents should start at each time; a roster says which agent works when. Each "
    "agent gets one weekly pattern: the same start time on 5 days and 2 consecutive rest days. The roster is solved "
    "directly as a tour-scheduling MILP (days-off pair x start time x break position) against the same requirement."
)

meta = load_metadata()
agents = load_processed("roster_agents")
coverage = load_processed("roster_coverage")
tours = load_processed("roster_tours")
queue = st.selectbox("Queue", list(QUEUES), format_func=queue_label, key="roster_queue")
stats = meta["roster_stats"][queue]
shift_fte = meta["schedule_stats"][queue]["fte_equivalent"]
ag = agents[agents["queue"] == queue]
cov = coverage[coverage["queue"] == queue]

k = st.columns(5)
kpi(k[0], "Agents rostered", f"{stats['agents']:.0f}", f"{stats['paid_hours']:,.0f} paid hours a week")
kpi(k[1], "Cost of rostering rules", f"{stats['agents'] / shift_fte - 1:+.1%}", f"vs {shift_fte:.1f} FTE from free shift scheduling")
kpi(k[2], "Requirement covered", pct(stats["coverage_pct"]), f"{stats['under_hours']:.1f} agent-hours short", "ok" if stats["coverage_pct"] > 0.99 else "warn")
kpi(k[3], "Roster efficiency", pct(stats["efficiency_pct"]), f"{stats['distinct_start_times']:.0f} distinct start times")
kpi(k[4], "Friday-Saturday off", pct(stats["weekend_off_share"]), "share of agents on the Egyptian weekend")

fig = go.Figure()
fig.add_scatter(x=cov["interval_start"], y=cov["requirement"], name="Requirement (incl. in-day shrinkage)", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
fig.add_scatter(x=cov["interval_start"], y=cov["scheduled_on_phone"], name="Rostered on phones", line=dict(color=QUEUE_COLORS[queue], width=2, shape="hv"))
fig.update_xaxes(tickformat="%a %H:%M")
chart(style(fig, 340, "Agents"))
st.caption(
    "Fixed weekly start times and paired rest days are less flexible than choosing every shift independently, so the "
    "roster needs a few more people than the shift schedule's FTE equivalent. That gap is the honest cost of giving "
    "agents a stable, legal week; it is what capacity planning should budget for."
)

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.subheader("Rest days")
    off = ag["days_off"].value_counts().reindex([days_off_label(d) for d in range(7)], fill_value=0)
    fig_o = go.Figure(go.Bar(x=off.index, y=off.to_numpy(), marker_color=QUEUE_COLORS[queue], hovertemplate="%{x}: %{y} agents<extra></extra>"))
    chart(style(fig_o, 300, "Agents", legend=False))
    friday = off[[lbl for lbl in off.index if "Fri" in lbl]].sum() / max(off.sum(), 1)
    st.caption(
        f"Almost nobody rests on Sunday-Monday, because Sunday is the busiest day of the Egyptian week. {friday:.0%} of "
        "agents have Friday off, the quietest day; the rest are spread so every weekday keeps enough people."
    )
with right:
    st.subheader("Working-time rules checked")
    rules = pd.DataFrame(
        [
            {"Rule": "Paid hours per day", "Limit": f"{EGYPT_LABOUR.max_daily_hours:.0f}h", "Roster": f"{EGYPT_LABOUR.max_daily_hours:.0f}h"},
            {"Rule": "Paid hours per week", "Limit": f"{EGYPT_LABOUR.max_weekly_hours:.0f}h", "Roster": f"{5 * EGYPT_LABOUR.max_daily_hours:.0f}h"},
            {"Rule": "Continuous work before a break", "Limit": f"{EGYPT_LABOUR.max_continuous_work_hours:.0f}h", "Roster": "3-5h"},
            {"Rule": "Weekly rest", "Limit": f"{EGYPT_LABOUR.min_weekly_rest_hours:.0f}h", "Roster": "2 consecutive days (48h+)"},
            {"Rule": "Rest between shifts", "Limit": "site policy 11h", "Roster": "15h (same start daily)"},
        ]
    )
    st.dataframe(rules, hide_index=True)
    st.caption("Labour-law values are planning assumptions based on Egypt's Labour Law No. 14 of 2025, not legal advice.")

st.subheader("Roster")
f1, f2 = st.columns([1, 1])
teams = f1.multiselect("Teams", sorted(ag["team"].unique()), placeholder="All teams")
day_off = f2.multiselect("Rest days", [days_off_label(d) for d in range(7)], placeholder="Any")
view = ag
if teams:
    view = view[view["team"].isin(teams)]
if day_off:
    view = view[view["days_off"].isin(day_off)]
st.dataframe(
    view[["agent_id", "team", "days_off", *DAY_NAMES, "break_start"]].rename(
        columns={"agent_id": "Agent", "team": "Team", "days_off": "Rest days", "break_start": "Break"}
    ),
    hide_index=True,
    height=420,
)
with st.expander("Patterns used"):
    st.dataframe(
        tours[tours["queue"] == queue][["days_off", "shift", "break_start", "agents"]].rename(
            columns={"days_off": "Rest days", "shift": "Shift", "break_start": "Break", "agents": "Agents"}
        ),
        hide_index=True,
    )
