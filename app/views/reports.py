"""Reports: standard WFM reports written in SQL, and the Excel workbook with live formulas."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import load_calendar, load_events, load_intervals, page_header, queue_label

from wfm.config import PROCESSED_DIR, QUEUES
from wfm.excel_report import WORKBOOK_NAME
from wfm.realtime import load_rta_day
from wfm.sql_reports import REPORTS, build_database, run_report

page_header(
    "reports",
    "Every report below is a plain SQL file (common table expressions, window functions, joins, CASE logic) run on an "
    "in-memory SQLite copy of the data. The Excel workbook rebuilds the core WFM calculations with live formulas.",
    "The same numbers an analyst would pull from a WFM tool or database every morning: <b>7 SQL reports</b> and a "
    "<b>5-sheet Excel workbook</b> whose formulas were checked cell by cell against the Python engine.",
)


@st.cache_resource(show_spinner="Loading the data into SQLite...")
def database():
    return build_database(load_intervals(), load_calendar(), load_events(), load_rta_day(PROCESSED_DIR))


sql_tab, excel_tab = st.tabs(["SQL reports", "Excel workbook"])

with sql_tab:
    conn = database()
    dates = load_intervals()["date"]
    first, last = dates.min().date(), dates.max().date()
    c1, c2 = st.columns([1.2, 2])
    key = c1.selectbox("Report", list(REPORTS), format_func=lambda k: REPORTS[k].title)
    report = REPORTS[key]
    c2.markdown(f"<br>{report.purpose}", unsafe_allow_html=True)

    params: dict[str, str] = {}
    cols = st.columns(4)
    if "queue" in report.params:
        params["queue"] = cols[0].selectbox("Queue", list(QUEUES), index=1, format_func=queue_label, key="sql_queue")
    if "date" in report.params:
        params["date"] = str(cols[1].date_input("Date", value=pd.Timestamp("2026-09-22").date(), min_value=first, max_value=last, key="sql_date"))
    if "start" in report.params:
        params["start"] = str(cols[1].date_input("From", value=pd.Timestamp("2026-08-01").date(), min_value=first, max_value=last, key="sql_start"))
        params["end"] = str(cols[2].date_input("To", value=last, min_value=first, max_value=last, key="sql_end"))

    result = run_report(conn, key, **params)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown(f"**Result** · {len(result):,} rows")
        st.dataframe(result, hide_index=True, height=440)
        st.download_button(
            "Download CSV", result.to_csv(index=False).encode(), file_name=f"{key}.csv", mime="text/csv", icon=":material/download:"
        )
    with right:
        st.markdown(f"**Query** · `src/wfm/sql/{key}.sql`")
        st.code(report.sql, language="sql", wrap_lines=True)
    st.caption(
        "Tables: intervals (30-minute history: offered, handled, abandoned, SL, AHT, planned, required, scheduled and "
        "staffed agents), calendar, events (outages and campaigns), agents and agent_states (5-minute states for the "
        "RTA replay day). Parameters are bound by name, never pasted into the SQL."
    )

with excel_tab:
    st.markdown(
        """
The workbook rebuilds the core WFM calculations in Excel, so they can be checked, changed and reused without Python.
Inputs are yellow; every other calculated cell is a formula.

| Sheet | What it shows | Formulas used |
|---|---|---|
| **Erlang C** | Staffing calculator: P(wait), service level, ASA, occupancy; finds the agents needed | `POISSON.DIST`, `EXP`, `INDEX`/`MATCH`, conditional formatting |
| **Interval report** | Plan vs actual every 30 minutes for the outage day | variance, over/under, `SUMPRODUCT`-weighted AHT and SL, nested `IF` status |
| **Shrinkage** | Planned components with the compounded total; actual weekly in-day shrinkage | `1-EXP(SUMPRODUCT(LN(1-range)))`, colour scale |
| **Capacity plan** | 26-week required vs available FTE for AR_TECH, linked to the shrinkage total; hires; chart | cross-sheet references, gap formulas |
| **Adherence** | Agent adherence and conformance, coaching flags, team summary | `COUNTIF`, `AVERAGEIFS`, data bars |
"""
    )
    st.download_button(
        "Download the Excel workbook",
        (PROCESSED_DIR / WORKBOOK_NAME).read_bytes(),
        file_name=WORKBOOK_NAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        icon=":material/download:",
    )
    st.caption(
        "Verification: the workbook was recalculated from scratch in LibreOffice and every formula cell (over 1,000) matched "
        "the Python engine; for example, 100 calls at 180s AHT gives 14 agents at 88.8% for 80/20. A test repeats this check."
    )
