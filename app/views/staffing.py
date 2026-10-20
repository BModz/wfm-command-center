"""Staffing: interactive Erlang calculator, interval requirement curve and Erlang C vs Erlang A."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (
    ACTUAL_COLOR,
    MODEL_COLORS,
    QUEUE_COLORS,
    REFERENCE,
    chart,
    kpi,
    load_processed,
    pct,
    page_header,
    queue_label,
    style,
)

from wfm.config import QUEUES
from wfm.erlang import erlang_a_metrics, erlang_c_metrics, required_agents, required_agents_erlang_a, traffic_intensity

C_COLOR, A_COLOR = MODEL_COLORS["LightGBM"], MODEL_COLORS["Holt-Winters"]

textbook_c = int(required_agents(100, 180)[0])
textbook_a = int(required_agents_erlang_a(100, 180, 180)[0])
page_header(
    "staffing",
    "Turning forecast calls and handle time into the number of agents needed on the phones in each 30-minute interval.",
    f"100 calls in 30 minutes at 180s AHT needs <b>{textbook_c} agents</b> for 80/20 under Erlang C but only "
    f"<b>{textbook_a}</b> under Erlang A, because real callers hang up. Erlang C is the safe default; it overstaffs "
    "most on small queues with impatient callers.",
)

# Calculator ----------------------------------------------------------------------
st.subheader("Erlang calculator")
inputs, results = st.columns([1, 2.2], gap="large")
with inputs:
    calls = st.number_input("Calls offered per 30 minutes", 1, 3000, 100, step=10)
    aht = st.number_input("AHT (seconds)", 30, 1800, 180, step=10)
    target = st.slider("Service level target", 0.50, 0.95, 0.80, 0.05, format="%.2f")
    threshold = st.slider("Answer within (seconds)", 5, 120, 20, 5)
    max_occ = st.slider("Maximum occupancy", 0.70, 0.95, 0.88, 0.01, format="%.2f")
    patience = st.slider("Average caller patience for Erlang A (seconds)", 30, 600, 180, 10)


@st.cache_data(show_spinner=False)
def calculator(calls: int, aht: int, target: float, threshold: int, max_occ: float, patience: int) -> tuple[int, int, pd.DataFrame]:
    n_c = int(required_agents(calls, aht, target, threshold, max_occ)[0])
    n_a = int(required_agents_erlang_a(calls, aht, patience, target, threshold, max_occ)[0])
    load = float(traffic_intensity(calls, aht)[()])
    agents = np.arange(max(int(load) + 1 - 4, 1), max(n_c, n_a) + 6)
    c = erlang_c_metrics(calls, aht, agents, threshold)
    a = erlang_a_metrics(calls, aht, agents, patience, threshold)
    table = pd.DataFrame(
        {
            "Agents": agents,
            "Erlang C SL": c["service_level"],
            "Erlang C ASA (s)": c["asa_s"],
            "Erlang C P(wait)": c["p_wait"],
            "Erlang A SL": a["service_level"],
            "Erlang A ASA (s)": a["asa_s"],
            "Erlang A abandon": a["abandon_rate"],
            "Occupancy": np.minimum(load / agents, 1.0),
        }
    )
    return n_c, n_a, table


n_c, n_a, table = calculator(calls, aht, target, threshold, max_occ, patience)
load = calls * aht / 1800
row_c = table[table["Agents"] == n_c].iloc[0]
with results:
    k = st.columns(4)
    kpi(k[0], "Traffic intensity", f"{load:.1f} Erl", "Calls x AHT / 1,800s")
    kpi(k[1], "Erlang C agents", f"{n_c}", f"SL {pct(row_c['Erlang C SL'])}, ASA {row_c['Erlang C ASA (s)']:.0f}s")
    kpi(k[2], "Erlang A agents", f"{n_a}", f"Patience {patience}s")
    kpi(k[3], "Occupancy at Erlang C", pct(load / n_c), f"Cap {max_occ:.0%}")
    fig = go.Figure()
    visible = table[table["Agents"] > load]
    fig.add_scatter(x=visible["Agents"], y=visible["Erlang C SL"], name="Erlang C (no abandons)", line=dict(color=C_COLOR, width=2), mode="lines+markers", marker=dict(size=8))
    fig.add_scatter(x=visible["Agents"], y=visible["Erlang A SL"], name="Erlang A (with abandons)", line=dict(color=A_COLOR, width=2), mode="lines+markers", marker=dict(size=8))
    fig.add_hline(y=target, line=dict(color=REFERENCE, width=1), annotation_text=f"{target:.0%} target", annotation_position="bottom right")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    chart(style(fig, 300, f"Service level ({threshold}s)", "Agents on the phones"))
st.caption(
    f"With {calls} calls in 30 minutes at {aht}s AHT the offered load is {load:.1f} Erlangs: that many agents would be "
    "busy 100% of the time. Erlang C adds agents until the target is met, assuming every caller waits forever. "
    "Erlang A lets callers hang up, which removes work from the queue, so it reaches the same target with fewer agents. "
    "The textbook check: 100 calls at 180s gives 13 agents at 79.6% (just short) and 14 agents at 88.8%."
)
with st.expander("Full table"):
    st.dataframe(
        table.style.format(
            {
                "Erlang C SL": "{:.1%}",
                "Erlang C ASA (s)": "{:.1f}",
                "Erlang C P(wait)": "{:.1%}",
                "Erlang A SL": "{:.1%}",
                "Erlang A ASA (s)": "{:.1f}",
                "Erlang A abandon": "{:.1%}",
                "Occupancy": "{:.1%}",
            }
        ),
        hide_index=True,
    )

# Interval requirement curve -----------------------------------------------------
st.subheader("Interval requirement for a forecast day")
req = load_processed("forecast_intervals")
c1, c2 = st.columns(2)
queue = c1.selectbox("Queue", list(QUEUES), format_func=queue_label, key="req_queue")
dates = sorted(req["date"].unique())
day = c2.select_slider("Day", options=dates, value=dates[1], format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))
d = req[(req["queue"] == queue) & (req["date"] == day)].copy()


@st.cache_data(show_spinner=False)
def erlang_a_curve(calls: tuple, aht: tuple, patience: float) -> np.ndarray:
    calls_arr, aht_arr = np.array(calls), np.array(aht)
    out = np.zeros(len(calls_arr), dtype=int)
    open_ = ~np.isnan(aht_arr)
    out[open_] = required_agents_erlang_a(calls_arr[open_], aht_arr[open_], patience)
    return out


d["required_a"] = erlang_a_curve(tuple(d["forecast_offered"]), tuple(d["forecast_aht_s"]), QUEUES[queue].patience_s)
left, right = st.columns(2)
fig_v = go.Figure()
fig_v.add_bar(x=d["interval_start"], y=d["forecast_offered"], name="Forecast calls", marker_color=QUEUE_COLORS[queue])
fig_v.update_xaxes(tickformat="%H:%M")
with left:
    chart(style(fig_v, 300, "Calls per 30 minutes", legend=False))
fig_r = go.Figure()
fig_r.add_scatter(x=d["interval_start"], y=d["workload_erl"], name="Workload (Erlangs)", line=dict(color=ACTUAL_COLOR, width=1.5, dash="dot"))
fig_r.add_scatter(x=d["interval_start"], y=d["required_agents"], name="Erlang C requirement", line=dict(color=C_COLOR, width=2, shape="hv"))
fig_r.add_scatter(x=d["interval_start"], y=d["required_a"], name="Erlang A requirement", line=dict(color=A_COLOR, width=2, shape="hv"))
fig_r.update_xaxes(tickformat="%H:%M")
with right:
    chart(style(fig_r, 300, "Agents on the phones"))
st.caption(
    "Left: forecast calls per interval. Right: the pure workload (Erlangs) and the agents needed to hit 80/20 under each "
    "model. The gap between workload and requirement is the queueing buffer that keeps waits short; it is relatively "
    "larger at night, when small teams lose the pooling benefit of large ones."
)

# Erlang C vs A summary --------------------------------------------------------------
st.subheader("How much does Erlang C overstaff?")


@st.cache_data(show_spinner=False)
def weekly_comparison() -> pd.DataFrame:
    week = req[req["week_start"] == req["week_start"].min()]
    rows = []
    for q in QUEUES:
        w = week[(week["queue"] == q) & week["forecast_aht_s"].notna()]
        a_req = required_agents_erlang_a(w["forecast_offered"].to_numpy(), w["forecast_aht_s"].to_numpy(), QUEUES[q].patience_s)
        rows.append({"queue": q, "erlang_c_hours": w["required_agents"].sum() / 2, "erlang_a_hours": a_req.sum() / 2})
    out = pd.DataFrame(rows)
    out["difference"] = out["erlang_c_hours"] / out["erlang_a_hours"] - 1
    return out


cmp = weekly_comparison()
fig_c = go.Figure()
fig_c.add_bar(x=cmp["queue"], y=cmp["erlang_c_hours"], name="Erlang C", marker_color=C_COLOR)
fig_c.add_bar(x=cmp["queue"], y=cmp["erlang_a_hours"], name="Erlang A", marker_color=A_COLOR)
fig_c.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
left, right = st.columns([3, 2])
with left:
    chart(style(fig_c, 320, "Agent-hours in week 1"))
with right:
    total_c, total_a = cmp["erlang_c_hours"].sum(), cmp["erlang_a_hours"].sum()
    kpi(st, "Extra agent-hours with Erlang C", f"{total_c - total_a:,.0f}", f"{total_c / total_a - 1:+.1%} vs Erlang A, all queues, one week")
    st.write("")
    st.dataframe(
        cmp.rename(columns={"queue": "Queue", "erlang_c_hours": "Erlang C hours", "erlang_a_hours": "Erlang A hours", "difference": "C vs A"}).style.format(
            {"Erlang C hours": "{:,.0f}", "Erlang A hours": "{:,.0f}", "C vs A": "{:+.1%}"}
        ),
        hide_index=True,
    )
st.caption(
    "Erlang C is the industry default and is used for the capacity plan here because it is conservative. Ignoring "
    "abandonment overstates the requirement; the effect is largest on small queues and short-patience queues such as "
    "SALES. Planners often target Erlang C but validate with Erlang A, or calibrate patience from abandon curves."
)
