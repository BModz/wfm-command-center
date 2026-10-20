"""Schedule: optimised weekly shifts, coverage against requirement and efficiency, plus a live single-day solve."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, QUEUE_COLORS, SEQUENTIAL, STATUS, chart, kpi, load_metadata, load_processed, pct, queue_label, setup_page, style

from wfm.capacity import Shrinkage
from wfm.config import INTERVALS_PER_DAY, QUEUES, open_mask
from wfm.scheduling import DAY_NAMES, ShiftRules, optimise_shifts, schedule_requirement

setup_page("Schedule")
st.title("Shift schedule")
rules = ShiftRules()
st.caption(
    f"Mixed-integer programme solved with HiGHS: choose how many agents start each {rules.shift_hours:.0f}-hour shift "
    f"({rules.paid_hours:.0f} paid hours plus a {rules.break_hours:.0f}-hour unpaid break) at any 30-minute start, with "
    "the break placed so nobody works more than 5 hours without a rest (Egyptian labour law). The week runs Sunday to "
    "Saturday and is cyclic, so Saturday night shifts wrap into Sunday morning. Optionally, 4- and 5-hour part-time "
    f"shifts (no break needed, at most {rules.max_part_time_share:.0%} of paid hours) can be mixed in."
)

meta = load_metadata()
coverage = load_processed("schedule_coverage")
shifts = load_processed("schedule_shifts")
VARIANTS = {"full_time": "Full-time shifts only", "part_time": "Full-time + part-time shifts"}
c1, c2 = st.columns([1, 1])
queue = c1.selectbox("Queue", list(QUEUES), format_func=queue_label, key="sched_queue")
variant = c2.radio("Shift mix", list(VARIANTS), format_func=VARIANTS.get, horizontal=True)
stats = meta["schedule_stats" if variant == "full_time" else "schedule_stats_part_time"][queue]
cov = coverage[(coverage["queue"] == queue) & (coverage["variant"] == variant)].copy()
sh = shifts[(shifts["queue"] == queue) & (shifts["variant"] == variant)]

k = st.columns(5)
pt_note = f"{stats['part_time_shifts']:.0f} part-time" if stats["part_time_shifts"] else "all full-time"
kpi(k[0], "Shifts in the week", f"{stats['shifts']:,.0f}", f"{pt_note}, {stats['paid_hours']:,.0f} paid hours")
kpi(k[1], "FTE equivalent", f"{stats['fte_equivalent']:.1f}", "Paid hours / 40")
kpi(k[2], "Requirement covered", pct(stats["coverage_pct"]), f"{stats['under_hours']:.1f} agent-hours short", "ok" if stats["coverage_pct"] > 0.99 else "warn")
kpi(k[3], "Schedule efficiency", pct(stats["efficiency_pct"]), f"{stats['over_hours']:.0f} agent-hours over requirement")
gap = stats.get("mip_gap", 0.0)
kpi(k[4], "Solver result", "Optimal" if "Optimal" in stats["status"] else "Feasible", f"Within {gap:.1%} of the proven bound")

fig = go.Figure()
fig.add_scatter(x=cov["interval_start"], y=cov["requirement"], name="Requirement (incl. in-day shrinkage)", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
fig.add_scatter(x=cov["interval_start"], y=cov["net_required"], name="Net Erlang C requirement", line=dict(color=ACTUAL_COLOR, width=1, dash="dot", shape="hv"))
fig.add_scatter(x=cov["interval_start"], y=cov["scheduled_on_phone"], name="Scheduled on phones", line=dict(color=QUEUE_COLORS[queue], width=2, shape="hv"))
fig.update_xaxes(tickformat="%a %H:%M")
chart(style(fig, 360, "Agents"))
fig_o = go.Figure()
fig_o.add_bar(
    x=cov["interval_start"],
    y=cov["over_under"],
    marker_color=[STATUS["critical"] if v < 0 else "#9ec5f4" for v in cov["over_under"]],
    name="Over / under",
    hovertemplate="%{x|%a %H:%M}: %{y:+.0f} agents<extra></extra>",
)
fig_o.update_xaxes(tickformat="%a %H:%M")
chart(style(fig_o, 200, "Over (+) / under (-)", legend=False))
st.caption(
    "The requirement is the Erlang C agents grossed up for in-day shrinkage (coaching, meetings, aux overrun), so the "
    "agents left on the phones after those activities still meet 80/20. Efficiency = covered requirement / scheduled "
    "time; the remaining overstaffing comes from fixed shift lengths not matching the curve exactly. Under-coverage is "
    "allowed only where adding a whole shift costs more than the shortfall (soft penalty)."
)

# Part-time comparison -------------------------------------------------------------------
st.subheader("What part-time shifts buy")
rows = []
for q in QUEUES:
    ft, pt = meta["schedule_stats"][q], meta["schedule_stats_part_time"][q]
    rows.append(
        {
            "Queue": q,
            "Hours": QUEUES[q].hours_label,
            "FTE, FT only": ft["fte_equivalent"],
            "FTE, with PT": pt["fte_equivalent"],
            "FTE saved": ft["fte_equivalent"] - pt["fte_equivalent"],
            "Efficiency, FT only": ft["efficiency_pct"],
            "Efficiency, with PT": pt["efficiency_pct"],
            "PT share of hours": pt["part_time_hours_share"],
        }
    )
st.dataframe(
    pd.DataFrame(rows).style.format(
        {
            "FTE, FT only": "{:.1f}",
            "FTE, with PT": "{:.1f}",
            "FTE saved": "{:+.1f}",
            "Efficiency, FT only": "{:.1%}",
            "Efficiency, with PT": "{:.1%}",
            "PT share of hours": "{:.0%}",
        }
    ),
    hide_index=True,
)
st.caption(
    "SALES is open 14 hours with two peaks, so 9-hour shifts must overlap heavily and efficiency stalls around 83%. "
    "Short part-time shifts sit exactly on the peaks, which removes most of the overstaffing. The 24/7 queues already "
    "schedule at about 97%, so part-time adds little there and mainly adds shift-management complexity. Differences "
    "under about 1 FTE are within the solver's optimality gap (FT = full-time, PT = part-time)."
)

# Shift start heatmap --------------------------------------------------------------------
st.subheader("When shifts start")
sh_grid = sh.pivot_table(index="day", columns="shift_start", values="agents", aggfunc="sum", fill_value=0)
starts = [f"{i // 2:02d}:{30 * (i % 2):02d}" for i in range(INTERVALS_PER_DAY)]
sh_grid = sh_grid.reindex(index=DAY_NAMES, columns=starts, fill_value=0)
fig_s = go.Figure(
    go.Heatmap(
        z=sh_grid.to_numpy(),
        x=starts,
        y=DAY_NAMES,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        xgap=1,
        ygap=1,
        colorbar=dict(title="Agents", thickness=10),
        hovertemplate="%{y} %{x}: %{z} agents start<extra></extra>",
    )
)
fig_s.update_yaxes(autorange="reversed")
chart(style(fig_s, 300, legend=False))
st.caption(
    f"Agents starting a shift at each time of day. {queue} uses {sh['shift_start'].nunique()} distinct start times; "
    "real operations usually restrict starts to a smaller shift menu, which costs a little efficiency."
)
with st.expander("Shift list"):
    st.dataframe(
        sh[["day", "shift_type", "shift_start", "shift_end", "break_start", "agents"]].rename(
            columns={"day": "Day", "shift_type": "Type", "shift_start": "Start", "shift_end": "End", "break_start": "Break", "agents": "Agents"}
        ),
        hide_index=True,
    )

# Single-day solve ----------------------------------------------------------------------
st.subheader("Optimise a single day")
st.caption("Re-solves the MILP live for one day of the forecast, so you can see how the under-coverage penalty trades shifts against service.")
req = load_processed("forecast_intervals")
c1, c2, c3 = st.columns([1, 1, 0.7])
dates = sorted(req.loc[req["queue"] == queue, "date"].unique())
day = c1.selectbox("Day", dates[:14], format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))
penalty = c2.slider("Penalty per uncovered agent-interval (in shifts)", 0.05, 2.0, rules.under_penalty, 0.05)
day_pt = c3.toggle("Allow part-time", value=variant == "part_time")


@st.cache_data(show_spinner=False)
def solve_day(queue: str, day: pd.Timestamp, penalty: float, part_time: bool) -> tuple[pd.DataFrame, dict]:
    d = req[(req["queue"] == queue) & (req["date"] == day)].sort_values("interval")
    need = schedule_requirement(d["required_agents"].to_numpy(), Shrinkage().in_day)
    cyclic = QUEUES[queue].is_24x7
    day_rules = ShiftRules(under_penalty=penalty, part_time=part_time)
    res = optimise_shifts(need, day_rules, cyclic=cyclic, open_intervals=open_mask(queue), start=pd.Timestamp(day), time_limit_s=10)
    return res.coverage, res.stats


with st.spinner("Solving..."):
    day_cov, day_stats = solve_day(queue, day, penalty, day_pt)
k = st.columns(4)
kpi(k[0], "Shifts", f"{day_stats['shifts']:.0f}", f"{day_stats['part_time_shifts']:.0f} part-time, {day_stats['paid_hours']:.0f} paid hours")
kpi(k[1], "Coverage", pct(day_stats["coverage_pct"]))
kpi(k[2], "Efficiency", pct(day_stats["efficiency_pct"]))
kpi(k[3], "Intervals short", f"{day_stats['intervals_under']:.0f}")
fig_d = go.Figure()
fig_d.add_scatter(x=day_cov["interval_start"], y=day_cov["requirement"], name="Requirement", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
fig_d.add_scatter(x=day_cov["interval_start"], y=day_cov["scheduled_on_phone"], name="Scheduled on phones", line=dict(color=QUEUE_COLORS[queue], width=2, shape="hv"))
fig_d.add_scatter(x=day_cov["interval_start"], y=day_cov["on_site"], name="On site (incl. breaks)", line=dict(color=QUEUE_COLORS[queue], width=1, dash="dot", shape="hv"))
fig_d.update_xaxes(tickformat="%H:%M")
chart(style(fig_d, 320, "Agents"))
st.caption(
    "24/7 queues are solved as a cyclic day (tonight's late shifts cover this morning's early hours); SALES only "
    "schedules inside 09:00-23:00. A low penalty accepts short gaps at the edges of the day instead of paying for a "
    "full extra shift; a high penalty forces full coverage at the cost of efficiency."
)
