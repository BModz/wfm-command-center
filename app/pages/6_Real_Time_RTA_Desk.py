"""Real-Time (RTA desk): replay of an outage day with live KPIs, alerts, actions and adherence."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, HIGHLIGHT, MUTED, QUEUE_COLORS, REFERENCE, STATUS, chart, kpi, pct, setup_page, style

from wfm.config import INTERVALS_PER_DAY, PROCESSED_DIR, QUEUES, SL_TARGET
from wfm.realtime import Snapshot, interval_label, load_rta_day, snapshot

setup_page("Real-Time RTA Desk")


@st.cache_resource(show_spinner=False)
def rta_day():
    return load_rta_day(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def view(now: int) -> Snapshot:
    return snapshot(rta_day(), now)


day = rta_day()
color = QUEUE_COLORS[day.queue]
st.title("Real-time desk")
st.caption(
    f"Replay of {QUEUES[day.queue].name} on {day.date:%A %d %B %Y}. The interval forecast was made the evening before; "
    f"the schedule was optimised about two weeks earlier; {len(day.agents)} agents are rostered. Move the clock to see "
    "what an RTA would have seen and done at each point in the day."
)

now = st.select_slider(
    "Time now (end of the last completed interval)",
    options=list(range(1, INTERVALS_PER_DAY + 1)),
    value=31,
    format_func=interval_label,
)
snap = view(now)
kp = snap.kpis

# KPI tiles ------------------------------------------------------------------------------
st.subheader(f"Day to date at {interval_label(now)}")
c = st.columns(5)
kpi(c[0], "Service level (20s)", pct(kp["service_level"]), f"Target {SL_TARGET:.0%}", "ok" if kp["service_level"] >= SL_TARGET else ("warn" if kp["service_level"] >= 0.7 else "bad"))
kpi(c[1], "Abandon rate", pct(kp["abandon_rate"]), "Alert above 8%", "ok" if kp["abandon_rate"] < 0.05 else ("warn" if kp["abandon_rate"] < 0.08 else "bad"))
kpi(c[2], "ASA", f"{kp['asa_s']:.0f}s", "Average wait of answered calls")
kpi(c[3], "Volume vs forecast", f"{kp['volume_var']:+.1%}", f"{kp['offered']:,.0f} offered vs {kp['forecast']:,.0f}", "bad" if kp["volume_var"] > 0.2 else ("warn" if kp["volume_var"] > 0.1 else "ok"))
kpi(c[4], "AHT vs forecast", f"{kp['aht_var']:+.1%}", f"{kp['aht_s']:.0f}s actual")
c = st.columns(5)
kpi(c[0], "Occupancy", pct(kp["occupancy"]), "Handle time / staffed time")
kpi(c[1], "Staffed vs scheduled", pct(kp["staffing_realisation"]), "Agents on phones vs plan")
kpi(c[2], "Adherence", pct(kp.get("adherence", 1.0)), "Right state at the right time")
kpi(c[3], "Conformance", pct(kp.get("conformance", 1.0)), "Phone time worked vs scheduled")
out_now = kp.get("out_of_adherence_now", 0)
kpi(c[4], "Out of adherence now", f"{out_now:.0f}", f"of {kp.get('agents_on_shift_now', 0):.0f} agents on shift", "bad" if out_now > 5 else ("warn" if out_now > 2 else "ok"))

# Alerts and actions ---------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Alerts")
    if snap.alerts:
        rows = "".join(
            f'<div class="panel-row"><span class="badge {a["severity"].lower()}">{a["severity"]}</span>'
            f'<span class="title">{html.escape(a["alert"])}</span><br>{html.escape(a["detail"])}</div>'
            for a in snap.alerts
        )
    else:
        rows = '<div class="panel-row"><span class="badge info">Clear</span>No thresholds breached in the last interval.</div>'
    st.markdown(rows, unsafe_allow_html=True)
with right:
    st.subheader("Recommended actions")
    if snap.actions:
        rows = "".join(
            f'<div class="panel-row"><span class="badge action">{i}</span><span class="title">{html.escape(a["action"])}</span> '
            f'<span class="when">{html.escape(a["when"])}</span><br>{html.escape(a["detail"])}</div>'
            for i, a in enumerate(snap.actions, 1)
        )
        st.markdown(rows, unsafe_allow_html=True)
        eff = snap.action_effect
        st.markdown(
            f"Projected service level for the next 4 hours: **{pct(eff['sl_before'])}** as things stand, "
            f"**{pct(eff['sl_after'])}** if the actions are taken (Erlang A on the reforecast)."
        )
    else:
        st.markdown(
            '<div class="panel-row"><span class="badge info">None</span>Projected staffing meets 80/20 for the next 4 hours.</div>',
            unsafe_allow_html=True,
        )

# Charts -----------------------------------------------------------------------------------
tl = snap.timeline.copy()
tl["ts"] = day.date + pd.to_timedelta(tl["interval"] * 30, unit="min")
past, future = tl[tl["interval"] < now], tl[tl["interval"] >= now]
now_ts = day.date + pd.Timedelta(minutes=30 * now)


def mark_now(fig: go.Figure) -> go.Figure:
    fig.add_vline(x=now_ts, line=dict(color=MUTED, width=1))
    fig.update_xaxes(tickformat="%H:%M", range=[day.date, day.date + pd.Timedelta(days=1)])
    return fig


st.subheader("Volume: actual, forecast and intraday reforecast")
fig_v = go.Figure()
fig_v.add_scatter(x=tl["ts"], y=tl["forecast_offered"], name="Forecast (made day before)", line=dict(color=REFERENCE, width=2, shape="hv"))
fig_v.add_scatter(x=past["ts"], y=past["offered"], name="Actual offered", line=dict(color=ACTUAL_COLOR, width=2.5, shape="hv"))
fig_v.add_scatter(x=future["ts"], y=future["reforecast_offered"], name="Reforecast", line=dict(color=color, width=2, dash="dash", shape="hv"))
chart(mark_now(style(fig_v, 320, "Calls per 30 minutes")))
ratio = snap.timeline.attrs.get("volume_ratio", 1.0)
st.caption(
    f"The reforecast scales the remaining forecast by the recent actual/forecast ratio ({ratio:.2f}x, weighted towards the "
    "latest intervals). "
    + (
        "An incident is detected, so the surge is assumed to decay back to the normal run rate over the next couple of hours."
        if snap.incident
        else "Surge intervals are excluded from the run rate, so a past incident does not inflate the rest of the day."
    )
)

st.subheader("Agents: scheduled, staffed and required")
fig_a = go.Figure()
fig_a.add_scatter(x=tl["ts"], y=tl["required_fc"], name="Required (forecast)", line=dict(color=REFERENCE, width=2, shape="hv"))
fig_a.add_scatter(x=future["ts"], y=future["required_reforecast"], name="Required (reforecast)", line=dict(color=ACTUAL_COLOR, width=2, dash="dash", shape="hv"))
fig_a.add_scatter(x=past["ts"], y=past["required_actual"], name="Required (actual demand)", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
fig_a.add_scatter(x=tl["ts"], y=tl["scheduled_on_phone"], name="Scheduled on phones", line=dict(color=color, width=1.5, dash="dot", shape="hv"))
fig_a.add_scatter(x=past["ts"], y=past["staffed"], name="Staffed (actual)", line=dict(color=color, width=2.5, shape="hv"))
if "staffed_after_actions" in tl and snap.actions:
    fig_a.add_scatter(x=future["ts"], y=future["staffed_after_actions"], name="Staffed after actions", line=dict(color=HIGHLIGHT, width=2.5, dash="dash", shape="hv"))
chart(mark_now(style(fig_a, 340, "Agents on the phones")))
st.caption(
    "Scheduled is the published plan net of breaks and coaching; staffed is agents actually in the on-phone state. The "
    "difference is lost time from late logins, extended breaks, unscheduled aux and no-shows. Required is Erlang C at 80/20."
)

st.subheader("Service level by interval")
fig_s = go.Figure()
fig_s.add_bar(x=past["ts"], y=past["service_level"], name="Actual SL", marker_color=[STATUS["good"] if v >= SL_TARGET else (STATUS["warning"] if v >= 0.7 else STATUS["critical"]) for v in past["service_level"]])
if "projected_sl_before" in tl:
    fig_s.add_scatter(x=future["ts"], y=future["projected_sl_before"], name="Projected, no action", line=dict(color=STATUS["critical"], width=2, dash="dash"))
    fig_s.add_scatter(x=future["ts"], y=future["projected_sl_after"], name="Projected, with actions", line=dict(color=HIGHLIGHT, width=2.5))
fig_s.add_hline(y=SL_TARGET, line=dict(color=REFERENCE, width=1), annotation_text="80% target", annotation_position="bottom right")
fig_s.update_yaxes(tickformat=".0%", range=[0, 1.05])
chart(mark_now(style(fig_s, 300, "Answered within 20s")))
st.caption("Bars: realised service level per interval (green on target, amber 70-80%, red below 70%). Lines: next 4 hours projected with Erlang A, before and after the recommended actions.")

# Adherence -------------------------------------------------------------------------------------
st.subheader("Agent adherence")
adh = snap.adherence
f1, f2 = st.columns([1, 2])
only_out = f1.toggle("Only agents out of adherence now", value=False)
teams = f2.multiselect("Teams", sorted(adh["team"].unique()), placeholder="All teams")
shown = adh[adh["out_of_adherence"]] if only_out else adh
if teams:
    shown = shown[shown["team"].isin(teams)]
st.dataframe(
    shown,
    hide_index=True,
    height=380,
    column_config={
        "agent_id": "Agent",
        "team": "Team",
        "shift": "Shift",
        "scheduled_now": "Scheduled now",
        "actual_now": "Actual now",
        "on_shift_now": st.column_config.CheckboxColumn("On shift"),
        "out_of_adherence": st.column_config.CheckboxColumn("Out now"),
        "minutes_out_now": st.column_config.NumberColumn("Minutes out (current)", format="%d"),
        "adherence": st.column_config.ProgressColumn("Adherence", min_value=0.0, max_value=1.0, format="percent"),
        "conformance": st.column_config.NumberColumn("Conformance", format="percent"),
        "minutes_out_today": st.column_config.NumberColumn("Minutes out today", format="%d"),
        "exception": "Exceptions today",
    },
)
team = adh.groupby("team").agg(agents=("agent_id", "count"), adherence=("adherence", "mean"), conformance=("conformance", "mean"), out_now=("out_of_adherence", "sum")).reset_index()
with st.expander("Adherence by team"):
    st.dataframe(
        team.rename(columns={"team": "Team", "agents": "Agents", "adherence": "Adherence", "conformance": "Conformance", "out_now": "Out of adherence now"}).style.format(
            {"Adherence": "{:.1%}", "Conformance": "{:.1%}"}
        ),
        hide_index=True,
    )
st.caption(
    "Adherence = share of scheduled time to date in the scheduled state (phone, break, coaching), measured in 5-minute "
    "slots. Conformance = phone time actually worked divided by phone time scheduled, regardless of when. An agent who "
    "takes a full break 30 minutes late keeps 100% conformance but loses adherence, and leaves the queue short "
    "exactly when the plan counted on them."
)
