"""NileConnect WFM Command Centre: landing page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import kpi, load_metadata, load_processed, pct, setup_page

from wfm.config import QUEUES, SL_TARGET, SL_THRESHOLD_S

setup_page("Home")
meta = load_metadata()
metrics = load_processed("backtest_metrics")
ivl_metrics = load_processed("backtest_interval_metrics")

st.title("NileConnect WFM Command Centre")
st.markdown(
    '<div class="fictional">NileConnect is a <b>fictional</b> Cairo telecom contact centre created for this portfolio '
    "project. Interval-level contact-centre data is not publicly available, so two years of 30-minute history were "
    "simulated with realistic Egyptian patterns (Sunday-Thursday week, Ramadan, Eid, billing cycles, network outages).</div>",
    unsafe_allow_html=True,
)
st.write("")
st.markdown(
    "An end-to-end workforce management system covering the full planning cycle: **forecast** demand, convert it "
    "into **staffing** with Erlang C and Erlang A, plan **capacity** and hiring six months out, build an optimised "
    "**schedule**, then manage the day in **real time** as an RTA."
)


def total(window: str, model: str = "LightGBM") -> float:
    m = metrics[(metrics["window"] == window) & (metrics["model"] == model) & (metrics["queue"] == "All queues")]
    return float(m["wape"].iloc[0])


normal, ramadan = "Normal (Aug-Sep 2026)", "Ramadan 2026"
c = st.columns(4)
kpi(c[0], "Daily forecast WAPE", pct(total(normal)), f"LightGBM, 8-week backtest (seasonal naive {pct(total(normal, 'Seasonal naive'))})")
kpi(c[1], "Ramadan forecast WAPE", pct(total(ramadan)), f"LightGBM, Ramadan 2026 (Holt-Winters {pct(total(ramadan, 'Holt-Winters'))})")
iv = ivl_metrics[(ivl_metrics["window"] == normal) & (ivl_metrics["queue"] == "All queues")]
kpi(c[2], "Interval forecast WAPE", pct(float(iv["wape"].iloc[0])), "30-minute level, top-down profile")
eff = pd.Series({q: s["efficiency_pct"] for q, s in meta["schedule_stats"].items()})
kpi(c[3], "Schedule efficiency", pct(eff.drop("SALES").mean()), "24/7 queues, MILP (HiGHS) weekly schedule")

st.subheader("Queues")
queues = pd.DataFrame(
    [
        {
            "Queue": q.code,
            "Description": q.name,
            "Hours": q.hours_label,
            "Calls per day (2024)": f"{q.daily_calls:,.0f}",
            "AHT (s)": int(q.aht_s),
            "Service goal": f"{SL_TARGET:.0%}/{SL_THRESHOLD_S}s",
        }
        for q in QUEUES.values()
    ]
)
st.dataframe(queues, hide_index=True)

st.subheader("What each page answers")
st.markdown(
    """
| Page | Question it answers | Main WFM concepts |
|---|---|---|
| **Overview** | How has the centre performed, and what do demand patterns look like? | Service level, ASA, abandon rate, occupancy, intraday and weekly seasonality |
| **Forecast** | How many calls will arrive, and how accurate are we? | Seasonal naive vs Holt-Winters vs LightGBM, WAPE and bias, outage cleaning, Ramadan profiles |
| **Staffing** | How many agents do we need each interval? | Erlang C, Erlang A, traffic intensity, occupancy cap, overstaffing from ignoring abandons |
| **Capacity Plan** | How many people do we need to hire, and when? | Shrinkage (planned vs unplanned), FTE requirement, attrition, training and nesting curves |
| **Schedule** | Which shifts cover the requirement at the lowest cost? | Shift-scheduling MILP, break placement, coverage, over/under, schedule efficiency |
| **Real-Time (RTA Desk)** | What is happening right now, and what should we do? | Intraday reforecast, alerts, adherence vs conformance, IVR deflection, VTO and overtime |
"""
)
st.caption(
    f"Data built {meta['built_at']} from seed {meta['seed']}. History {meta['history'][0]} to {meta['history'][1]}; "
    f"forecast horizon {meta['forecast'][0]} to {meta['forecast'][1]}. Use the sidebar to move between pages."
)
