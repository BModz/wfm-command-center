"""Excel workbook of standard WFM tools and reports, built with live formulas.

Every calculated cell is an Excel formula (Erlang C with POISSON.DIST, weighted
service level with SUMPRODUCT, compounded shrinkage with EXP/SUMPRODUCT/LN,
lookups with INDEX/MATCH, AVERAGEIFS and conditional formatting), so the
workbook stays usable when someone changes an input. Each formula also carries
the value Python computed, so viewers that do not recalculate still show
correct numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from wfm.capacity import STARTING_HEADCOUNT, CapacityAssumptions, Shrinkage, recommend_hiring, weekly_requirement
from wfm.config import QUEUES, SL_TARGET, SL_THRESHOLD_S
from wfm.erlang import erlang_c_metrics, traffic_intensity
from wfm.labour import EGYPT_LABOUR
from wfm.realtime import AUX, OFF, PHONE, SLOT_MINUTES, RTADay, interval_label

NAVY, BLUE, GREY, GREEN_FILL, RED_FILL, AMBER_FILL = "#0F2742", "#2A78D6", "#F3F2EF", "#E3F4E3", "#FBE3E3", "#FDF0D2"
ERLANG_ROWS = 30
WORKBOOK_NAME = "NileConnect_WFM_workbook.xlsx"


class _Formats:
    """Shared cell formats for the workbook."""

    def __init__(self, wb: xlsxwriter.Workbook) -> None:
        base = {"font_name": "Calibri", "font_size": 11}
        self.title = wb.add_format({**base, "bold": True, "font_size": 16, "font_color": NAVY})
        self.note = wb.add_format({**base, "italic": True, "font_color": "#52514E"})
        self.header = wb.add_format({**base, "bold": True, "font_color": "#FFFFFF", "bg_color": NAVY, "border": 1, "text_wrap": True, "valign": "vcenter"})
        self.label = wb.add_format({**base, "bold": True})
        self.input = wb.add_format({**base, "bg_color": "#FFF7D6", "border": 1})
        self.input_pct = wb.add_format({**base, "bg_color": "#FFF7D6", "border": 1, "num_format": "0.0%"})
        self.int = wb.add_format({**base, "num_format": "#,##0", "border": 1})
        self.dec = wb.add_format({**base, "num_format": "#,##0.0", "border": 1})
        self.pct = wb.add_format({**base, "num_format": "0.0%", "border": 1})
        self.text = wb.add_format({**base, "border": 1})
        self.total = wb.add_format({**base, "bold": True, "num_format": "#,##0", "top": 2, "border": 1})
        self.total_pct = wb.add_format({**base, "bold": True, "num_format": "0.0%", "top": 2, "border": 1})
        self.result = wb.add_format({**base, "bold": True, "font_size": 13, "font_color": BLUE, "num_format": "#,##0.0"})
        self.result_int = wb.add_format({**base, "bold": True, "font_size": 13, "font_color": BLUE, "num_format": "0"})
        self.result_pct = wb.add_format({**base, "bold": True, "font_size": 13, "font_color": BLUE, "num_format": "0.0%"})
        self.green = wb.add_format({"bg_color": GREEN_FILL})
        self.red = wb.add_format({"bg_color": RED_FILL, "font_color": "#A42828"})
        self.amber = wb.add_format({"bg_color": AMBER_FILL})


def _header(ws, row: int, labels: list[str], fmt: _Formats, widths: list[int] | None = None) -> None:
    for c, label in enumerate(labels):
        ws.write(row, c, label, fmt.header)
        if widths:
            ws.set_column(c, c, widths[c])


def _readme(wb: xlsxwriter.Workbook, fmt: _Formats, sheets: list[tuple[str, str]]) -> None:
    ws = wb.add_worksheet("README")
    ws.set_column(0, 0, 24)
    ws.set_column(1, 1, 110)
    ws.write(0, 0, "NileConnect WFM workbook", fmt.title)
    ws.write(1, 0, "NileConnect is a fictional Cairo telecom contact centre; all data is synthetic.", fmt.note)
    ws.write(2, 0, "Yellow cells are inputs. Every other calculated cell is a live Excel formula.", fmt.note)
    _header(ws, 4, ["Sheet", "What it does"], fmt)
    for i, (name, purpose) in enumerate(sheets, start=5):
        ws.write_url(i, 0, f"internal:'{name}'!A1", string=name)
        ws.write(i, 1, purpose, fmt.text)


def _erlang_sheet(wb: xlsxwriter.Workbook, fmt: _Formats, calls: float = 100, aht: float = 180) -> None:
    ws = wb.add_worksheet("Erlang C")
    ws.write(0, 0, "Erlang C staffing calculator", fmt.title)
    ws.write(1, 0, "Change the yellow inputs; the table and the required agents update. P(wait) uses POISSON.DIST.", fmt.note)
    inputs = [
        ("Calls per interval", calls, fmt.input),
        ("AHT (seconds)", aht, fmt.input),
        ("Interval length (seconds)", 1800, fmt.input),
        ("Service level target", SL_TARGET, fmt.input_pct),
        ("Answer within (seconds)", SL_THRESHOLD_S, fmt.input),
        ("Maximum occupancy", 0.88, fmt.input_pct),
    ]
    for r, (label, value, f) in enumerate(inputs, start=3):
        ws.write(r, 0, label, fmt.label)
        ws.write(r, 1, value, f)
    load = float(traffic_intensity(calls, aht)[()])
    ws.write(10, 0, "Traffic intensity (Erlangs)", fmt.label)
    ws.write_formula(10, 1, "=B4*B5/B6", fmt.result, load)
    first_row = 14
    agents = np.arange(max(int(load) + 1, 1), max(int(load) + 1, 1) + ERLANG_ROWS)
    m = erlang_c_metrics(calls, aht, agents)
    meets = (m["service_level"] >= SL_TARGET) & (load / agents <= 0.88)
    req_idx = int(np.argmax(meets))
    last_excel = first_row + ERLANG_ROWS
    ws.write(11, 0, "Required agents", fmt.label)
    ws.write_formula(11, 1, f"=INDEX(A{first_row + 1}:A{last_excel},MATCH(TRUE,F{first_row + 1}:F{last_excel},0))", fmt.result_int, int(agents[req_idx]))
    ws.write(12, 0, "Service level at that staffing", fmt.label)
    ws.write_formula(12, 1, f"=INDEX(C{first_row + 1}:C{last_excel},MATCH(TRUE,F{first_row + 1}:F{last_excel},0))", fmt.result_pct, float(m["service_level"][req_idx]))
    _header(ws, first_row - 1, ["Agents", "P(wait)", "Service level", "ASA (s)", "Occupancy", "Meets target"], fmt, [30, 14, 14, 12, 12, 14])
    for k, n in enumerate(agents):
        r = first_row + k
        e = r + 1  # Excel row number
        if k == 0:
            ws.write_formula(r, 0, "=MAX(1,INT(B11)+1)", fmt.int, int(n))
        else:
            ws.write_formula(r, 0, f"=A{e - 1}+1", fmt.int, int(n))
        ws.write_formula(
            r, 1,
            f"=IF(A{e}<=$B$11,1,POISSON.DIST(A{e},$B$11,FALSE)/(POISSON.DIST(A{e},$B$11,FALSE)+(1-$B$11/A{e})*POISSON.DIST(A{e}-1,$B$11,TRUE)))",
            fmt.pct, float(m["p_wait"][k]),
        )
        ws.write_formula(r, 2, f"=IF(A{e}<=$B$11,0,1-B{e}*EXP(-(A{e}-$B$11)*$B$8/$B$5))", fmt.pct, float(m["service_level"][k]))
        ws.write_formula(r, 3, f'=IF(A{e}<=$B$11,"",B{e}*$B$5/(A{e}-$B$11))', fmt.dec, float(m["asa_s"][k]))
        ws.write_formula(r, 4, f"=MIN(1,$B$11/A{e})", fmt.pct, float(min(1.0, load / n)))
        ws.write_formula(r, 5, f"=AND(C{e}>=$B$7,E{e}<=$B$9)", fmt.text, bool(meets[k]))
    ws.conditional_format(first_row, 0, last_excel - 1, 5, {"type": "formula", "criteria": f"=$F{first_row + 1}=TRUE", "format": fmt.green})
    ws.freeze_panes(first_row, 0)


def _interval_sheet(wb: xlsxwriter.Workbook, fmt: _Formats, rta: RTADay) -> None:
    ws = wb.add_worksheet("Interval report")
    iv = rta.intervals
    ws.write(0, 0, f"Interval report - {QUEUES[rta.queue].name} - {rta.date:%a %d %b %Y}", fmt.title)
    ws.write(1, 0, "Plan vs actual every 30 minutes. Variance, over/under, totals and status are formulas.", fmt.note)
    cols = ["Interval", "Forecast", "Offered", "Volume var", "AHT forecast", "AHT actual", "AHT var", "Required", "Scheduled",
            "Staffed", "Over / under", "Answered in SL", "Abandoned", "Service level", "Abandon rate", "Status"]
    _header(ws, 3, cols, fmt, [10, 10, 10, 11, 12, 11, 10, 10, 11, 10, 12, 13, 11, 13, 12, 14])
    top = 4
    for k, rec in enumerate(iv.itertuples()):
        r, e = top + k, top + k + 1
        ws.write(r, 0, interval_label(int(rec.interval)), fmt.text)
        ws.write(r, 1, float(rec.forecast_offered), fmt.int)
        ws.write(r, 2, int(rec.offered), fmt.int)
        ws.write_formula(r, 3, f'=IF(B{e}=0,"",C{e}/B{e}-1)', fmt.pct, rec.offered / rec.forecast_offered - 1)
        ws.write(r, 4, float(rec.forecast_aht_s), fmt.int)
        ws.write(r, 5, float(rec.aht_s), fmt.int)
        ws.write_formula(r, 6, f'=IF(E{e}=0,"",F{e}/E{e}-1)', fmt.pct, rec.aht_s / rec.forecast_aht_s - 1)
        ws.write(r, 7, int(rec.required_fc), fmt.int)
        ws.write(r, 8, float(rec.scheduled_on_phone), fmt.dec)
        ws.write(r, 9, float(rec.staffed), fmt.dec)
        ws.write_formula(r, 10, f"=J{e}-H{e}", fmt.dec, float(rec.staffed - rec.required_fc))
        ws.write(r, 11, int(rec.answered_in_sl), fmt.int)
        ws.write(r, 12, int(rec.abandoned), fmt.int)
        sl = rec.answered_in_sl / rec.offered if rec.offered else 1.0
        ws.write_formula(r, 13, f"=IF(C{e}=0,1,L{e}/C{e})", fmt.pct, sl)
        ws.write_formula(r, 14, f"=IF(C{e}=0,0,M{e}/C{e})", fmt.pct, rec.abandoned / rec.offered if rec.offered else 0.0)
        status = "Below 70%" if sl < 0.7 else ("Below target" if sl < SL_TARGET else "On target")
        ws.write_formula(r, 15, f'=IF(N{e}<0.7,"Below 70%",IF(N{e}<{SL_TARGET},"Below target","On target"))', fmt.text, status)
    last = top + len(iv)
    t = last + 1  # Excel row of totals
    ws.write(last, 0, "Day", fmt.total)
    for col, letter, source in ((1, "B", "forecast_offered"), (2, "C", "offered"), (11, "L", "answered_in_sl"), (12, "M", "abandoned")):
        ws.write_formula(last, col, f"=SUM({letter}{top + 1}:{letter}{last})", fmt.total, float(iv[source].sum()))
    ws.write_formula(last, 3, f"=C{t}/B{t}-1", fmt.total_pct, iv["offered"].sum() / iv["forecast_offered"].sum() - 1)
    ws.write_formula(last, 5, f"=SUMPRODUCT(F{top + 1}:F{last},C{top + 1}:C{last})/C{t}", fmt.total, float((iv["aht_s"] * iv["offered"]).sum() / iv["offered"].sum()))
    ws.write_formula(last, 13, f"=L{t}/C{t}", fmt.total_pct, iv["answered_in_sl"].sum() / iv["offered"].sum())
    ws.write_formula(last, 14, f"=M{t}/C{t}", fmt.total_pct, iv["abandoned"].sum() / iv["offered"].sum())
    ws.conditional_format(top, 13, last - 1, 13, {"type": "cell", "criteria": "<", "value": 0.7, "format": fmt.red})
    ws.conditional_format(top, 13, last - 1, 13, {"type": "cell", "criteria": "between", "minimum": 0.7, "maximum": SL_TARGET - 1e-9, "format": fmt.amber})
    ws.conditional_format(top, 10, last - 1, 10, {"type": "cell", "criteria": "<", "value": 0, "format": fmt.red})
    ws.autofilter(3, 0, last - 1, len(cols) - 1)
    ws.freeze_panes(4, 1)


def _shrinkage_sheet(wb: xlsxwriter.Workbook, fmt: _Formats, intervals: pd.DataFrame, calendar: pd.DataFrame) -> str:
    """Planned shrinkage assumptions plus actual weekly in-day shrinkage; returns the total-shrinkage cell."""
    ws = wb.add_worksheet("Shrinkage")
    s = Shrinkage()
    ws.write(0, 0, "Shrinkage report", fmt.title)
    ws.write(1, 0, "Planned components are inputs. Total = 1 - product of (1 - each), via EXP(SUMPRODUCT(LN(1 - range))).", fmt.note)
    _header(ws, 3, ["Component", "Type", "Planned"], fmt, [28, 12, 12])
    comps = s.table()
    for k, rec in enumerate(comps.itertuples(), start=4):
        ws.write(k, 0, rec.component, fmt.text)
        ws.write(k, 1, rec.type, fmt.text)
        ws.write(k, 2, float(rec.value), fmt.input_pct)
    end = 4 + len(comps)
    ws.write(end, 0, "Total (compounded)", fmt.label)
    ws.write_formula(end, 2, f"=1-EXP(SUMPRODUCT(LN(1-C5:C{end})))", fmt.result_pct, s.total)
    ws.write(end + 1, 0, "Simple sum (overstates)", fmt.label)
    ws.write_formula(end + 1, 2, f"=SUM(C5:C{end})", fmt.pct, s.additive_total)
    ws.write(end + 2, 0, "Unplanned (compounded)", fmt.label)
    ws.write_formula(end + 2, 2, f'=1-EXP(SUMPRODUCT((B5:B{end}="Unplanned")*LN(1-C5:C{end})))', fmt.pct, s.unplanned)
    total_cell = f"Shrinkage!$C${end + 1}"

    weeks = intervals.merge(calendar[["date", "week_start"]], on="date")
    weeks = weeks[weeks["week_start"] >= weeks["week_start"].max() - pd.Timedelta(weeks=8)]
    weeks = weeks[weeks["week_start"] < weeks["week_start"].max()]
    table = weeks.groupby(["week_start", "queue"], observed=True).agg(scheduled=("scheduled_agents", "sum"), staffed=("staffed_agents", "sum")).reset_index()
    table[["scheduled", "staffed"]] /= 2  # agent-hours
    top = end + 5
    ws.write(top - 1, 0, "Actual in-day shrinkage by week (scheduled phone time lost to absence and aux)", fmt.label)
    _header(ws, top, ["Week starting", "Queue", "Scheduled hours", "Staffed hours", "Hours lost", "In-day shrinkage"], fmt, [28, 12, 16, 14, 12, 16])
    for k, rec in enumerate(table.itertuples(), start=top + 1):
        e = k + 1
        ws.write(k, 0, f"{rec.week_start:%d %b %Y}", fmt.text)
        ws.write(k, 1, str(rec.queue), fmt.text)
        ws.write(k, 2, float(rec.scheduled), fmt.int)
        ws.write(k, 3, float(rec.staffed), fmt.int)
        ws.write_formula(k, 4, f"=C{e}-D{e}", fmt.int, float(rec.scheduled - rec.staffed))
        ws.write_formula(k, 5, f"=1-D{e}/C{e}", fmt.pct, float(1 - rec.staffed / rec.scheduled))
    last = top + len(table)
    ws.conditional_format(top + 1, 5, last, 5, {"type": "3_color_scale", "min_color": "#E3F4E3", "mid_color": "#FDF0D2", "max_color": "#FBE3E3"})
    ws.autofilter(top, 0, last, 5)
    return total_cell


def _capacity_sheet(wb: xlsxwriter.Workbook, fmt: _Formats, forecast_intervals: pd.DataFrame, queue: str, shrink_cell: str) -> None:
    ws = wb.add_worksheet("Capacity plan")
    s = Shrinkage()
    weekly = weekly_requirement(forecast_intervals, s)
    req = weekly[weekly["queue"] == queue].set_index("week_start")
    classes, plan = recommend_hiring(req["required_fte"], CapacityAssumptions(STARTING_HEADCOUNT[queue]))
    starts = {c.start_week: c.size for c in classes}
    ws.write(0, 0, f"26-week capacity plan - {QUEUES[queue].name}", fmt.title)
    ws.write(1, 0, f"Required FTE = agent-hours / (1 - total shrinkage on the Shrinkage sheet) / {EGYPT_LABOUR.contract_weekly_hours:.0f}h. Change shrinkage there and this sheet updates.", fmt.note)
    ws.write(2, 0, "Contract hours per week", fmt.label)
    ws.write(2, 1, EGYPT_LABOUR.contract_weekly_hours, fmt.input)
    cols = ["Week starting", "Forecast calls", "Agent-hours needed", "Required FTE", "Available FTE", "Gap FTE", "Gap %", "New hires starting training"]
    _header(ws, 4, cols, fmt, [16, 14, 18, 14, 14, 11, 10, 18])
    for k, (week, rec) in enumerate(req.iterrows()):
        r, e = 5 + k, 6 + k
        need = rec["agent_hours"] / (1 - s.total) / EGYPT_LABOUR.contract_weekly_hours
        available = float(plan["available_fte"].iloc[k])
        ws.write(r, 0, f"{week:%d %b %Y}", fmt.text)
        ws.write(r, 1, float(rec["offered"]), fmt.int)
        ws.write(r, 2, float(rec["agent_hours"]), fmt.int)
        ws.write_formula(r, 3, f"=C{e}/(1-{shrink_cell})/$B$3", fmt.dec, need)
        ws.write(r, 4, available, fmt.dec)
        ws.write_formula(r, 5, f"=E{e}-D{e}", fmt.dec, available - need)
        ws.write_formula(r, 6, f"=F{e}/D{e}", fmt.pct, (available - need) / need)
        ws.write(r, 7, starts.get(k, 0), fmt.int)
    last = 5 + len(req)
    ws.write(last, 0, "Total hires", fmt.total)
    ws.write_formula(last, 7, f"=SUM(H6:H{last})", fmt.total, sum(starts.values()))
    ws.conditional_format(5, 5, last - 1, 6, {"type": "cell", "criteria": "<", "value": 0, "format": fmt.red})
    ws.conditional_format(5, 5, last - 1, 6, {"type": "cell", "criteria": ">=", "value": 0, "format": fmt.green})
    chart = wb.add_chart({"type": "line"})
    chart.add_series({"name": "Required FTE", "categories": ["Capacity plan", 5, 0, last - 1, 0], "values": ["Capacity plan", 5, 3, last - 1, 3], "line": {"color": "#2B2B29", "width": 2}})
    chart.add_series({"name": "Available FTE", "categories": ["Capacity plan", 5, 0, last - 1, 0], "values": ["Capacity plan", 5, 4, last - 1, 4], "line": {"color": BLUE, "width": 2.25}})
    chart.set_title({"name": "Required vs available FTE"})
    chart.set_legend({"position": "top"})
    chart.set_size({"width": 760, "height": 340})
    ws.insert_chart(4, 9, chart)


def _adherence_sheet(wb: xlsxwriter.Workbook, fmt: _Formats, adherence: pd.DataFrame, rta: RTADay) -> None:
    ws = wb.add_worksheet("Adherence")
    ws.write(0, 0, f"Schedule adherence - {QUEUES[rta.queue].name} - {rta.date:%a %d %b %Y}", fmt.title)
    ws.write(1, 0, "Adherence = in-adherence minutes / scheduled minutes. Conformance = phone minutes worked / phone minutes scheduled.", fmt.note)
    ws.write(2, 0, "Coaching threshold", fmt.label)
    ws.write(2, 1, 0.85, fmt.input_pct)
    cols = ["Agent", "Team", "Scheduled min", "Out of adherence min", "Adherence", "Scheduled phone min", "Phone min worked", "Conformance", "Unscheduled aux min", "Action"]
    _header(ws, 4, cols, fmt, [14, 11, 14, 18, 12, 18, 16, 13, 18, 14])
    for k, rec in enumerate(adherence.itertuples(), start=5):
        e = k + 1
        ws.write(k, 0, rec.agent_id, fmt.text)
        ws.write(k, 1, rec.team, fmt.text)
        ws.write(k, 2, int(rec.scheduled_min), fmt.int)
        ws.write(k, 3, int(rec.out_of_adherence_min), fmt.int)
        ws.write_formula(k, 4, f"=1-D{e}/C{e}", fmt.pct, 1 - rec.out_of_adherence_min / rec.scheduled_min)
        ws.write(k, 5, int(rec.scheduled_phone_min), fmt.int)
        ws.write(k, 6, int(rec.actual_phone_min), fmt.int)
        conf = rec.actual_phone_min / rec.scheduled_phone_min if rec.scheduled_phone_min else 0.0
        ws.write_formula(k, 7, f"=IF(F{e}=0,0,G{e}/F{e})", fmt.pct, conf)
        ws.write(k, 8, int(rec.aux_min), fmt.int)
        action = "Coach" if 1 - rec.out_of_adherence_min / rec.scheduled_min < 0.85 else ""
        ws.write_formula(k, 9, f'=IF(E{e}<$B$3,"Coach","")', fmt.text, action)
    last = 5 + len(adherence)
    ws.conditional_format(5, 4, last - 1, 4, {"type": "data_bar", "bar_color": "#9EC5F4", "min_type": "num", "min_value": 0, "max_type": "num", "max_value": 1})
    ws.conditional_format(5, 9, last - 1, 9, {"type": "text", "criteria": "containing", "value": "Coach", "format": fmt.amber})
    ws.autofilter(4, 0, last - 1, len(cols) - 1)
    ws.freeze_panes(5, 1)

    teams = sorted(adherence["team"].unique())
    col0 = len(cols) + 1
    for c, w in zip(range(col0, col0 + 4), (11, 9, 18, 20), strict=True):
        ws.set_column(c, c, w)
    for c, label in enumerate(["Team", "Agents", "Average adherence", "Average conformance"]):
        ws.write(4, col0 + c, label, fmt.header)
    for k, team in enumerate(teams, start=5):
        team_cell = xl_rowcol_to_cell(k, col0)
        g = adherence[adherence["team"] == team]
        ws.write(k, col0, team, fmt.text)
        ws.write_formula(k, col0 + 1, f"=COUNTIF($B$6:$B${last},{team_cell})", fmt.int, len(g))
        ws.write_formula(k, col0 + 2, f"=AVERAGEIFS($E$6:$E${last},$B$6:$B${last},{team_cell})", fmt.pct, float((1 - g["out_of_adherence_min"] / g["scheduled_min"]).mean()))
        conf = np.where(g["scheduled_phone_min"] > 0, g["actual_phone_min"] / g["scheduled_phone_min"].replace(0, 1), 0.0)
        ws.write_formula(k, col0 + 3, f"=AVERAGEIFS($H$6:$H${last},$B$6:$B${last},{team_cell})", fmt.pct, float(conf.mean()))


def build_workbook(
    path: Path,
    rta: RTADay,
    adherence: pd.DataFrame,
    intervals: pd.DataFrame,
    calendar: pd.DataFrame,
    forecast_intervals: pd.DataFrame,
    queue: str = "AR_TECH",
) -> Path:
    """Write the WFM workbook to ``path``.

    Args:
        path: Output .xlsx path.
        rta: Replay day (interval report and adherence source).
        adherence: Agent adherence rows with scheduled_min, out_of_adherence_min, scheduled_phone_min,
            actual_phone_min and aux_min (see :func:`agent_minutes`).
        intervals: Interval history (weekly shrinkage).
        calendar: Calendar with week_start.
        forecast_intervals: Interval forecast with Erlang C requirement (capacity plan).
        queue: Queue for the capacity plan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # POISSON.DIST is an Excel 2010+ function and must be stored with its _xlfn. prefix to evaluate.
    wb = xlsxwriter.Workbook(str(path), {"use_future_functions": True})
    wb.set_properties({"title": "NileConnect WFM workbook", "comments": "Fictional company; synthetic data."})
    fmt = _Formats(wb)
    _readme(
        wb,
        fmt,
        [
            ("Erlang C", "Staffing calculator: P(wait), service level, ASA and occupancy for any calls/AHT; finds the agents needed."),
            ("Interval report", "Plan vs actual every 30 minutes for the outage day, with variance, over/under, totals and status."),
            ("Shrinkage", "Planned shrinkage components with the compounded total, and actual weekly in-day shrinkage."),
            ("Capacity plan", "26-week required vs available FTE for AR_TECH, driven by the shrinkage total; recommended hires."),
            ("Adherence", "Agent adherence and conformance for the outage day, coaching flags and a team summary."),
        ],
    )
    _erlang_sheet(wb, fmt)
    _interval_sheet(wb, fmt, rta)
    shrink_cell = _shrinkage_sheet(wb, fmt, intervals, calendar)
    _capacity_sheet(wb, fmt, forecast_intervals, queue, shrink_cell)
    _adherence_sheet(wb, fmt, adherence, rta)
    wb.close()
    return path


def agent_minutes(rta: RTADay) -> pd.DataFrame:
    """Per-agent scheduled, out-of-adherence, phone and aux minutes for the whole replay day."""
    on = rta.scheduled != OFF
    out = pd.DataFrame(
        {
            "agent_id": rta.agents["agent_id"],
            "team": rta.agents["team"],
            "scheduled_min": on.sum(axis=1) * SLOT_MINUTES,
            "out_of_adherence_min": ((rta.scheduled != rta.actual) & on).sum(axis=1) * SLOT_MINUTES,
            "scheduled_phone_min": (rta.scheduled == PHONE).sum(axis=1) * SLOT_MINUTES,
            "actual_phone_min": (rta.actual == PHONE).sum(axis=1) * SLOT_MINUTES,
            "aux_min": (rta.actual == AUX).sum(axis=1) * SLOT_MINUTES,
        }
    )
    return out[out["scheduled_min"] > 0].sort_values("out_of_adherence_min", ascending=False).reset_index(drop=True)
