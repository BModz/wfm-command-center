"""Overview: recent performance, trends, intraday heatmap and the Ramadan profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (
    MUTED,
    QUEUE_COLORS,
    REFERENCE,
    SEQUENTIAL,
    chart,
    kpi,
    load_calendar,
    load_intervals,
    pct,
    queue_label,
    queue_picker,
    setup_page,
    style,
)

from wfm.config import QUEUES, SL_TARGET
from wfm.egypt_calendar import RAMADAN_PERIODS

setup_page("Overview")
st.title("Operational overview")
st.caption("Realised performance from the simulated history. All service metrics are volume-weighted across intervals.")

intervals = load_intervals()
calendar = load_calendar()
queue = queue_picker("overview_queue")
data = intervals if queue == "ALL" else intervals[intervals["queue"] == queue]


def summarise(frame: pd.DataFrame) -> dict[str, float]:
    """Volume-weighted KPIs for any slice of intervals."""
    handled = frame["handled"].sum()
    talk = (frame["aht_s"].fillna(0) * frame["handled"]).sum()
    return {
        "offered": frame["offered"].sum(),
        "sl": frame["answered_in_sl"].sum() / frame["offered"].sum(),
        "asa": (frame["asa_s"].fillna(0) * frame["handled"]).sum() / handled,
        "abandon": frame["abandoned"].sum() / frame["offered"].sum(),
        "occupancy": talk / (frame["staffed_agents"].sum() * 1800),
        "aht": talk / handled,
    }


end = data["date"].max()
last4 = data[data["date"] > end - pd.Timedelta(weeks=4)]
prev4 = data[(data["date"] <= end - pd.Timedelta(weeks=4)) & (data["date"] > end - pd.Timedelta(weeks=8))]
now, before = summarise(last4), summarise(prev4)

st.subheader(f"Last 4 weeks: {queue_label(queue)}")
st.caption(f"{(end - pd.Timedelta(days=27)).date()} to {end.date()}, compared with the previous 4 weeks.")
c = st.columns(6)
kpi(c[0], "Calls offered", f"{now['offered']:,.0f}", f"{now['offered'] / before['offered'] - 1:+.1%} vs prior 4 weeks")
kpi(
    c[1],
    "Service level (20s)",
    pct(now["sl"]),
    f"Target {SL_TARGET:.0%} | prior {pct(before['sl'])}",
    "ok" if now["sl"] >= SL_TARGET else ("warn" if now["sl"] >= 0.7 else "bad"),
)
kpi(c[2], "ASA", f"{now['asa']:.0f}s", f"Prior {before['asa']:.0f}s")
kpi(c[3], "Abandon rate", pct(now["abandon"]), f"Prior {pct(before['abandon'])}", "ok" if now["abandon"] < 0.05 else "warn")
kpi(c[4], "Occupancy", pct(now["occupancy"]), "Handle time / staffed time")
kpi(c[5], "AHT", f"{now['aht']:.0f}s", f"Prior {before['aht']:.0f}s")

# Weekly trends ---------------------------------------------------------------
st.subheader("Two-year weekly trend")
weekly = intervals.assign(week=intervals["date"] - pd.to_timedelta((intervals["date"].dt.dayofweek + 1) % 7, unit="D"))
weekly = weekly.groupby(["week", "queue"], observed=True).agg(
    offered=("offered", "sum"), in_sl=("answered_in_sl", "sum"), abandoned=("abandoned", "sum")
).reset_index()
weekly = weekly[weekly["week"] > weekly["week"].min()]  # drop the partial first week
weekly = weekly[weekly["week"] < weekly["week"].max()]  # and the partial last week
shown = list(QUEUES) if queue == "ALL" else [queue]

left, right = st.columns(2)
fig = go.Figure()
for q in shown:
    w = weekly[weekly["queue"] == q]
    fig.add_scatter(x=w["week"], y=w["offered"], name=q, line=dict(color=QUEUE_COLORS[q], width=2))
with left:
    chart(style(fig, 320, "Calls offered per week"))
fig_sl = go.Figure()
for q in shown:
    w = weekly[weekly["queue"] == q]
    fig_sl.add_scatter(x=w["week"], y=w["in_sl"] / w["offered"], name=q, line=dict(color=QUEUE_COLORS[q], width=2))
fig_sl.add_hline(y=SL_TARGET, line=dict(color=REFERENCE, width=1), annotation_text="80% target", annotation_position="top left")
fig_sl.update_yaxes(tickformat=".0%")
with right:
    chart(style(fig_sl, 320, "Weekly service level (20s)"))
st.caption(
    "Volume grows year on year with a dip in Ramadan and the Eid weeks; billing peaks in the first week of each "
    "month. Service level dips mark weeks with network outages, which drive sudden spikes on AR_TECH."
)

# Heatmap ---------------------------------------------------------------------
st.subheader("When do calls arrive?")
recent = data[(data["date"] > end - pd.Timedelta(weeks=8))]
recent = recent.merge(calendar[["date", "is_ramadan", "is_holiday", "dow_name"]], on="date")
recent = recent[~recent["is_holiday"]]
heat = recent.groupby(["dow_name", "interval"], observed=True)["offered"].sum().unstack() / 8
heat = heat.reindex(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
labels = [f"{i // 2:02d}:{30 * (i % 2):02d}" for i in range(48)]
fig_h = go.Figure(
    go.Heatmap(
        z=heat.to_numpy(),
        x=labels,
        y=heat.index,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        colorbar=dict(title="Calls", thickness=10),
        xgap=1,
        ygap=1,
        hovertemplate="%{y} %{x}<br>%{z:.0f} calls per interval<extra></extra>",
    )
)
fig_h.update_yaxes(autorange="reversed")
chart(style(fig_h, 320, legend=False))
st.caption(
    "Average calls per 30-minute interval over the last 8 weeks (public holidays excluded). Sunday, the first day of "
    "the Egyptian working week, is busiest; Friday is quietest, with a later start and a dip around Friday prayer."
)

# Ramadan vs normal -------------------------------------------------------------
st.subheader("Ramadan changes the shape of the day")
year = 2026
r_start, r_end = (pd.Timestamp(d) for d in RAMADAN_PERIODS[year])
profile_q = "AR_BILLING" if queue == "ALL" else queue
base = intervals[(intervals["queue"] == profile_q)].merge(calendar[["date", "day_type", "is_holiday"]], on="date")
base = base[(base["day_type"] == "weekday") & ~base["is_holiday"]]
ramadan = base[base["date"].between(r_start, r_end)]
normal = base[base["date"].between(r_start - pd.Timedelta(weeks=6), r_start - pd.Timedelta(days=1))]


def shares(frame: pd.DataFrame) -> np.ndarray:
    wide = frame.pivot_table(index="date", columns="interval", values="offered", observed=True)
    return wide.div(wide.sum(axis=1), axis=0).mean().to_numpy()


fig_r = go.Figure()
fig_r.add_scatter(x=labels, y=shares(normal), name="Normal (6 weeks before)", line=dict(color=MUTED, width=2))
fig_r.add_scatter(x=labels, y=shares(ramadan), name=f"Ramadan {year}", line=dict(color=QUEUE_COLORS[profile_q], width=2.5))
fig_r.add_vrect(x0="17:30", x1="19:30", fillcolor="#f3f2ef", line_width=0, layer="below", annotation_text="Iftar", annotation_position="top left")
fig_r.update_yaxes(tickformat=".1%")
chart(style(fig_r, 340, "Share of daily calls"))
st.caption(
    f"{profile_q}, Sunday-Thursday. In Ramadan the morning peak starts later, volume collapses around iftar "
    "(about 17:30-19:30) and a strong late-night peak runs from 21:00 past midnight. A normal profile would "
    "badly under-staff the night and over-staff the iftar dip, which is why the forecast uses a separate Ramadan profile."
)
