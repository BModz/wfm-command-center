"""Capacity plan: shrinkage editor, required vs available FTE and the recommended hiring plan."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, MUTED, QUEUE_COLORS, STATUS, chart, kpi, load_processed, mark_period, page_header, pct, queue_label, style

from wfm.capacity import (
    STARTING_HEADCOUNT,
    CapacityAssumptions,
    Shrinkage,
    hiring_table,
    planning_requirement,
    recommend_hiring,
    weekly_requirement,
)
from wfm.config import QUEUES
from wfm.egypt_calendar import RAMADAN_PERIODS

req = load_processed("forecast_intervals")
defaults = Shrinkage()


@st.cache_data(show_spinner=False)
def default_plan_headline() -> tuple[int, int, float, pd.Timestamp, pd.Timestamp]:
    """Hires, classes, peak total FTE, peak week and first class start under the default assumptions."""
    weekly = weekly_requirement(req, defaults)
    hires = classes = 0
    starts = []
    for q in QUEUES:
        r = weekly[weekly["queue"] == q].set_index("week_start")["required_fte"]
        cls, _ = recommend_hiring(r, CapacityAssumptions(STARTING_HEADCOUNT[q]))
        hires += sum(c.size for c in cls)
        classes += len(cls)
        starts += [r.index[c.start_week] for c in cls]
    total = weekly.groupby("week_start")["required_fte"].sum()
    return hires, classes, float(total.max()), total.idxmax(), min(starts)


hires_all, classes_all, peak_fte, peak_week, first_class = default_plan_headline()
page_header(
    "capacity",
    "Converts the interval requirement into weekly FTE, projects the current team forward with attrition, and "
    "recommends new-hire classes to close the gaps. Every assumption below is editable.",
    f"Demand peaks at <b>{peak_fte:.0f} FTE</b> in the week of {peak_week:%d %b %Y}, the start of Ramadan 2027. "
    f"With 3.5% monthly attrition and a 6-week training ramp, the centre needs <b>{hires_all} hires in "
    f"{classes_all} classes</b>, the first starting training on {first_class:%d %b %Y}.",
)

# Shrinkage editor -------------------------------------------------------------------
st.subheader("Shrinkage")
planned_col, unplanned_col, summary_col = st.columns([1, 1, 1.1], gap="large")
values = {}
with planned_col:
    st.markdown("**Planned**")
    values["annual_leave"] = st.slider("Annual leave", 0.0, 0.15, defaults.annual_leave, 0.005, format="%.3f", help="21 days a year under Egyptian labour law = 21 / 260 working days")
    values["training"] = st.slider("Training", 0.0, 0.10, defaults.training, 0.005, format="%.3f")
    values["coaching"] = st.slider("Coaching", 0.0, 0.10, defaults.coaching, 0.005, format="%.3f")
    values["meetings"] = st.slider("Meetings", 0.0, 0.10, defaults.meetings, 0.005, format="%.3f")
with unplanned_col:
    st.markdown("**Unplanned**")
    values["sickness"] = st.slider("Sickness", 0.0, 0.15, defaults.sickness, 0.005, format="%.3f")
    values["absence"] = st.slider("Absence (no-show, emergency)", 0.0, 0.15, defaults.absence, 0.005, format="%.3f")
    values["aux_overrun"] = st.slider("Aux overrun", 0.0, 0.10, defaults.aux_overrun, 0.005, format="%.3f")
shrink = Shrinkage(**values)
with summary_col:
    kpi(st, "Total shrinkage (compounded)", pct(shrink.total), f"Simple sum would be {pct(shrink.additive_total)}")
    st.write("")
    c = st.columns(2)
    kpi(c[0], "Planned", pct(shrink.planned))
    kpi(c[1], "Unplanned", pct(shrink.unplanned))
st.caption(
    "Shrinkage is paid time when agents are not available for calls. It is different from occupancy, which measures "
    "how busy agents are while they are available. Components compound: 1 - (1 - leave)(1 - training)... Required "
    "FTE = interval agent-hours / (1 - shrinkage) / 40 contracted hours."
)

# Supply assumptions ------------------------------------------------------------------
st.subheader("Supply assumptions")
queue = st.selectbox("Team", list(QUEUES), format_func=queue_label, key="cap_queue")
a1, a2, a3, a4 = st.columns(4)
start_hc = a1.number_input("Starting productive FTE", 0, 1000, STARTING_HEADCOUNT[queue], key=f"hc_{queue}")
attrition = a1.slider("Monthly attrition", 0.0, 0.10, 0.035, 0.005, format="%.3f")
training_weeks = a2.slider("Training weeks (0% productive)", 1, 8, 4)
nesting_weeks = a2.slider("Nesting weeks", 0, 6, 2)
nesting_prod = a3.slider("Nesting productivity", 0.2, 1.0, 0.6, 0.05)
train_attr = a3.slider("Training attrition", 0.0, 0.40, 0.15, 0.01)
lead = a4.slider("Recruitment lead time (weeks)", 1, 10, 4)
max_class = a4.slider("Max class size", 6, 40, 25)
assumptions = CapacityAssumptions(
    starting_hc=start_hc,
    monthly_attrition=attrition,
    training_weeks=training_weeks,
    nesting_weeks=nesting_weeks,
    nesting_productivity=nesting_prod,
    training_attrition=train_attr,
    lead_time_weeks=lead,
    max_class=max_class,
)


@st.cache_data(show_spinner=False)
def weekly_fte(shrink_values: tuple) -> pd.DataFrame:
    return weekly_requirement(req, Shrinkage(*shrink_values))


weekly = weekly_fte(tuple(values.values()))
required = weekly[weekly["queue"] == queue].set_index("week_start")["required_fte"]
classes, plan = recommend_hiring(required, assumptions)
baseline = planning_requirement(required)

# Chart ---------------------------------------------------------------------------------
k = st.columns(4)
hires = sum(c.size for c in classes)
kpi(k[0], "Recommended hires", f"{hires}", f"{len(classes)} class{'es' if len(classes) != 1 else ''}")
kpi(k[1], "Peak requirement", f"{required.max():.0f} FTE", f"Week of {required.idxmax():%d %b %Y}")
short = plan[plan["gap_pct"] < -0.05]
kpi(k[2], "Spike weeks >5% short", f"{len(short)}", "cover with overtime / flex, not hires", "warn" if len(short) else "ok")
kpi(k[3], "Attrition leavers (26 wks)", f"{start_hc - plan['tenured_fte'].iloc[-1]:.0f}", f"{attrition:.1%} a month = {assumptions.weekly_attrition:.2%} a week")

fig = go.Figure()
fig.add_scatter(x=plan["week_start"], y=plan["required_fte"], name="Required FTE", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
fig.add_scatter(x=plan["week_start"], y=baseline, name="Hiring baseline (5-week median)", line=dict(color=MUTED, width=1.5, dash="dot", shape="hv"))
fig.add_scatter(x=plan["week_start"], y=plan["available_fte"], name="Available FTE", line=dict(color=QUEUE_COLORS[queue], width=2.5, shape="hv"))
fig.add_scatter(x=plan["week_start"], y=plan["tenured_fte"], name="Available without hiring", line=dict(color=QUEUE_COLORS[queue], width=1.5, dash="dash", shape="hv"))
for start, end in RAMADAN_PERIODS.values():
    if pd.Timestamp(start) <= plan["week_start"].max() and pd.Timestamp(end) >= plan["week_start"].min():
        mark_period(fig, pd.Timestamp(start), pd.Timestamp(end), "Ramadan 2027")
chart(style(fig, 380, "FTE (productive, after shrinkage)"))

fig_g = go.Figure()
fig_g.add_bar(
    x=plan["week_start"],
    y=plan["gap_fte"],
    marker_color=[STATUS["good"] if g >= 0 else STATUS["critical"] for g in plan["gap_fte"]],
    name="Surplus / gap",
    hovertemplate="%{x|%d %b}: %{y:+.1f} FTE<extra></extra>",
)
chart(style(fig_g, 220, "Surplus (+) / gap (-) FTE", legend=False))
st.caption(
    "Top: weekly FTE needed to deliver 80/20 after shrinkage, against FTE available from the current team (dashed, "
    "shrinking with attrition) plus new-hire classes. New hires are 0% productive in training and partly productive in "
    "nesting. Hiring is planned against a 5-week rolling median so one-off spikes (billing-cycle weeks, campaign weeks) "
    "are covered with overtime rather than permanent headcount. Bottom: remaining surplus (green) or gap (red)."
)

# Hiring plan ------------------------------------------------------------------------------
st.subheader("Recommended hiring plan")
if classes:
    ht = hiring_table(classes, plan["week_start"], assumptions)
    for col in ("recruit_by", "training_start", "nesting_start", "proficient_from"):
        ht[col] = pd.to_datetime(ht[col]).dt.strftime("%d %b %Y")
    st.dataframe(
        ht.rename(
            columns={
                "class": "Class",
                "recruit_by": "Start recruiting",
                "training_start": "Training starts",
                "nesting_start": "Nesting starts",
                "proficient_from": "Fully proficient",
                "hires": "Hires",
                "expected_graduates": "Expected graduates",
            }
        ),
        hide_index=True,
    )
else:
    st.info("No hiring needed: the current team covers the requirement within tolerance for the whole horizon.")


@st.cache_data(show_spinner=False)
def all_queue_summary(shrink_values: tuple, assumption_values: tuple) -> pd.DataFrame:
    attr, tw, nw, npd, ta, lt, mc = assumption_values
    wk = weekly_fte(shrink_values)
    rows = []
    for q in QUEUES:
        r = wk[wk["queue"] == q].set_index("week_start")["required_fte"]
        a = CapacityAssumptions(STARTING_HEADCOUNT[q], attr, tw, nw, npd, ta, lt, max_class=mc)
        cls, p = recommend_hiring(r, a)
        rows.append(
            {
                "Team": q,
                "Starting FTE": STARTING_HEADCOUNT[q],
                "Avg required FTE": round(r.mean(), 1),
                "Peak required FTE": round(r.max(), 1),
                "Hires": sum(c.size for c in cls),
                "Classes": len(cls),
                "Min weekly gap %": f"{p['gap_pct'].min():+.1%}",
            }
        )
    return pd.DataFrame(rows)


st.subheader("All teams")
st.dataframe(
    all_queue_summary(tuple(values.values()), (attrition, training_weeks, nesting_weeks, nesting_prod, train_attr, lead, max_class)),
    hide_index=True,
)
st.caption(
    "Uses each team's own starting FTE with the shrinkage and supply assumptions above. Negative weekly gaps that remain "
    "fall either before the first class can graduate or in single spike weeks; those are closed with overtime, "
    "cross-skilled agents from other queues or leave planning."
)
