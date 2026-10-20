"""Shared UI: import path, page registry, page header, KPI cards, chart styling and cached data loaders."""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _refresh_wfm_package() -> None:
    """Drop cached ``wfm`` modules when their source changed on disk.

    Streamlit only reloads modules inside the app folder. After a redeploy pulls
    new code, the server would otherwise keep serving the old ``wfm`` package to
    the new pages and fail with import or attribute errors.
    """
    files = sorted((SRC / "wfm").glob("*.py"))
    fingerprint = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    cached = sys.modules.get("wfm")
    if cached is not None and getattr(cached, "__source_fingerprint__", fingerprint) != fingerprint:
        for name in [m for m in sys.modules if m == "wfm" or m.startswith("wfm.")]:
            del sys.modules[name]
    import wfm

    wfm.__source_fingerprint__ = fingerprint


_refresh_wfm_package()

from wfm.config import DATA_DIR, PROCESSED_DIR, QUEUES  # noqa: E402

# Reference categorical palette, fixed order: colour follows the queue, never its rank.
QUEUE_COLORS = {"AR_BILLING": "#2a78d6", "AR_TECH": "#eb6834", "EN_SUPPORT": "#1baf7a", "SALES": "#eda100"}
MODEL_COLORS = {"LightGBM": "#2a78d6", "Holt-Winters": "#eb6834", "Seasonal naive": "#1baf7a"}
ACTUAL_COLOR = "#2b2b29"
HIGHLIGHT = "#6250d6"  # violet: a single emphasised projection line, never a queue
MUTED = "#8a8984"
REFERENCE = "#b0afa9"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL = ["#f3f2ef", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRID = "#e8e7e3"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"
EVENT_FILL = "#efeee9"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


@dataclass(frozen=True)
class PageInfo:
    """One app page: navigation entry, Home card and header metadata."""

    key: str
    title: str
    file: str
    icon: str
    section: str
    step: int | None
    url_path: str | None
    question: str
    blurb: str
    tag: str = ""  # header label for pages outside the numbered planning steps


PLANNING_STEPS = 7
PAGES: list[PageInfo] = [
    PageInfo("home", "Home", "views/home.py", ":material/home:", "Start", None, None,
             "An end-to-end WFM system for a fictional Cairo contact centre.", ""),
    PageInfo("overview", "Overview", "views/overview.py", ":material/monitoring:", "Start", None, "Overview",
             "How has the centre performed, and what does demand look like?",
             "Service level, ASA, abandons and occupancy for the last four weeks, and the shape of demand.",
             "Performance review"),
    PageInfo("forecast", "Forecast", "views/forecast.py", ":material/query_stats:", "Plan", 1, "Forecast",
             "How many calls will arrive, and how accurate are we?",
             "Three models backtested on a normal period and on Ramadan, plus a 26-week outlook."),
    PageInfo("staffing", "Staffing", "views/staffing.py", ":material/groups:", "Plan", 2, "Staffing",
             "How many agents do we need in each half-hour?",
             "Interactive Erlang C and Erlang A calculator and the interval requirement curve."),
    PageInfo("multi_skill", "Multi-Skill", "views/multi_skill.py", ":material/hub:", "Plan", 3, "Multi_Skill",
             "How much does cross-skilling the Arabic queues save?",
             "Discrete-event simulation of dedicated and cross-skilled agents on two queues."),
    PageInfo("capacity", "Capacity Plan", "views/capacity_plan.py", ":material/trending_up:", "Plan", 4, "Capacity_Plan",
             "How many people do we need to hire, and when?",
             "Shrinkage, attrition, training curves and recommended hiring classes for 26 weeks."),
    PageInfo("schedule", "Schedule", "views/schedule.py", ":material/calendar_view_week:", "Schedule", 5, "Schedule",
             "Which shifts cover the requirement at the lowest cost?",
             "Shift optimisation (MILP) with labour-law breaks and an optional part-time mix."),
    PageInfo("roster", "Roster", "views/roster.py", ":material/badge:", "Schedule", 6, "Roster",
             "Who works when, legally, every day of the week?",
             "Named weekly patterns: five days on, two consecutive rest days, fixed start times."),
    PageInfo("rta", "RTA Desk", "views/rta_desk.py", ":material/support_agent:", "Operate", 7, "Real_Time_RTA_Desk",
             "What is happening right now, and what should we do?",
             "Replay an outage day: live alerts, agent flags, recommended actions and shift reports."),
    PageInfo("reports", "Reports", "views/reports.py", ":material/table_view:", "Report", None, "Reports",
             "Which standard WFM reports does the team run, and how?",
             "Interval, KPI, shrinkage, adherence and RCA reports in SQL, plus an Excel workbook with live formulas.",
             "Standard reporting"),
]
PAGE_BY_KEY = {p.key: p for p in PAGES}

CSS = """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1320px;}
[data-testid="stAppDeployButton"], .stAppDeployButton, #MainMenu, footer {display: none !important;}
h2 {font-size: 1.25rem !important; font-weight: 650 !important; letter-spacing: -0.01em; margin-top: 1.4rem !important;}
h3 {font-size: 1.02rem !important; font-weight: 600 !important;}

/* Page header */
.page-head {margin: 0 0 14px;}
.eyebrow {font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #256abf; margin-bottom: 6px;}
.page-head h1 {font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; margin: 0 0 6px; padding: 0; color: #0b0b0b;}
.page-head .question {font-size: 1.05rem; color: #3d3c39; margin: 0;}
.page-head .intro {font-size: 0.9rem; color: #6b6a66; margin: 8px 0 0; max-width: 1000px; line-height: 1.55;}
.takeaway {display: flex; gap: 14px; align-items: baseline; background: #f2f7fd; border: 1px solid #d5e5fa;
           border-left: 4px solid #2a78d6; border-radius: 8px; padding: 12px 16px; margin: 4px 0 20px;
           font-size: 0.95rem; line-height: 1.5; color: #0b0b0b;}
.takeaway .tk-label {flex: none; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #184f95;}

/* KPI cards */
.kpi {background: #ffffff; border: 1px solid #e8e7e3; border-radius: 8px; padding: 14px 16px 12px; min-height: 118px; margin-bottom: 8px;}
.kpi .label {font-size: 0.74rem; font-weight: 500; color: #52514e; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;}
.kpi .value {font-size: 1.6rem; font-weight: 650; color: #0b0b0b; line-height: 1.2; font-variant-numeric: tabular-nums;}
.kpi .sub {font-size: 0.8rem; color: #52514e; margin-top: 4px;}
.kpi .delta {font-size: 0.8rem; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums;}
.delta.good {color: #0a7d0a;} .delta.bad {color: #b42f2f;} .delta.neutral {color: #52514e;}
.kpi.bad {border-left: 4px solid #d03b3b;} .kpi.warn {border-left: 4px solid #fab219;} .kpi.ok {border-left: 4px solid #0ca30c;}

/* Alert and action panels */
.badge {display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 10px; margin-right: 8px;
        text-transform: uppercase; letter-spacing: 0.04em;}
.badge.critical {background: #fbe3e3; color: #a42828;} .badge.warning {background: #fdf0d2; color: #7a5300;}
.badge.action {background: #e2edfb; color: #184f95;} .badge.info {background: #ecebe7; color: #52514e;}
.panel-row {padding: 10px 0; border-bottom: 1px solid #ecebe7; font-size: 0.9rem; line-height: 1.45;}
.panel-row:last-child {border-bottom: none;}
.panel-row .title {font-weight: 600;} .panel-row .when {color: #52514e; font-variant-numeric: tabular-nums;}

/* Home hero and page cards */
.hero {background: #0f2742; color: #ffffff; border-radius: 12px; padding: 30px 34px; margin-bottom: 18px;}
.hero .brand {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #9ec5f4;}
.hero h1 {font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em; margin: 6px 0 8px; padding: 0; color: #ffffff; line-height: 1.15;}
.hero .tagline {font-size: 1.05rem; color: #dbe7f6; max-width: 820px; line-height: 1.55; margin: 0;}
.hero .chips {margin-top: 16px;}
.hero .chip {display: inline-block; font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; margin: 0 6px 6px 0;
             border: 1px solid #3c5a7d; color: #dbe7f6;}
.hero .chip.note {background: #1d3a5c; border-color: #1d3a5c;}
[class*="st-key-card_"] {background: #ffffff; transition: border-color 0.15s ease, box-shadow 0.15s ease; min-height: 150px;}
[class*="st-key-card_"]:hover {border-color: #9ec5f4 !important; box-shadow: 0 2px 10px rgba(15, 39, 66, 0.08);}
[class*="st-key-card_"] [data-testid="stPageLink"] p {font-size: 1.02rem; font-weight: 650;}
.card-stage {font-size: 0.68rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #6b6a66;}
.card-blurb {font-size: 0.86rem; color: #52514e; line-height: 1.5;}

/* Real-time wallboard */
.wallboard {background: #121821; border-radius: 12px; padding: 14px; display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 6px 0 16px;}
.wb-tile {background: #1b2330; border-radius: 8px; padding: 10px 12px 12px; border-top: 3px solid #3b4757;}
.wb-tile.ok {border-top-color: #0ca30c;} .wb-tile.warn {border-top-color: #fab219;} .wb-tile.bad {border-top-color: #d03b3b;}
.wb-label {font-size: 0.68rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #a9b4c2;}
.wb-value {font-size: 1.75rem; font-weight: 650; color: #ffffff; font-variant-numeric: tabular-nums; line-height: 1.25;}
.wb-sub {font-size: 0.74rem; color: #a9b4c2;}
.clock {display: flex; align-items: center; gap: 10px; font-size: 0.95rem; color: #3d3c39;}
.clock .time {font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; color: #0b0b0b;}
.live {display: inline-flex; align-items: center; gap: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em;
       padding: 3px 9px; border-radius: 10px; background: #fbe3e3; color: #a42828;}
.live.paused {background: #ecebe7; color: #52514e;}
.live .dot {width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: pulse 1.4s ease-in-out infinite;}
.live.paused .dot {animation: none;}
@keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.25;}}
@media (prefers-reduced-motion: reduce) {.live .dot {animation: none;}}

.sidebar-note {font-size: 0.78rem; color: #6b6a66; line-height: 1.5;}

/* Phones */
@media (max-width: 640px) {
  .block-container {padding: 3.6rem 1rem 2.5rem;}  /* clear Streamlit's top bar */
  .page-head h1 {font-size: 1.5rem;}
  .page-head .question {font-size: 0.95rem;}
  .takeaway {flex-direction: column; gap: 4px;}
  .kpi {min-height: 0;}
  .kpi .value {font-size: 1.35rem;}
  .hero {padding: 20px;}
  .hero h1 {font-size: 1.6rem;}
  .wallboard {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .wb-value {font-size: 1.4rem;}
}
</style>
"""


def install_chrome() -> None:
    """Shared CSS and the sidebar note, rendered once per run by the entry point."""
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-note"><b>NileConnect</b> is a fictional Cairo telecom contact centre. All data is '
            "synthetic, generated to mirror real interval-level patterns.</div>",
            unsafe_allow_html=True,
        )


def release() -> None:
    """Forget this module so the next run re-imports it and repeats the wfm freshness check."""
    sys.modules.pop(__name__, None)


def page_header(key: str, intro: str = "", takeaway: str = "") -> None:
    """Stage label, title, the question the page answers, an optional intro and a key-takeaway box.

    ``takeaway`` may contain simple inline HTML such as <b>; everything else is escaped.
    """
    page = PAGE_BY_KEY[key]
    stage = f"{page.section} · Step {page.step} of {PLANNING_STEPS}" if page.step else f"{page.section} · {page.tag}"
    intro_html = f'<p class="intro">{html.escape(intro)}</p>' if intro else ""
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{stage}</div><h1>{html.escape(page.title)}</h1>'
        f'<p class="question">{html.escape(page.question)}</p>{intro_html}</div>',
        unsafe_allow_html=True,
    )
    if takeaway:
        st.markdown(f'<div class="takeaway"><span class="tk-label">Key takeaway</span><span>{takeaway}</span></div>', unsafe_allow_html=True)


def kpi(
    col,
    label: str,
    value: str,
    sub: str = "",
    state: str = "",
    delta: float | None = None,
    delta_text: str = "",
    higher_is_better: bool | None = True,
) -> None:
    """Render a KPI card.

    ``state`` ('', 'ok', 'warn', 'bad') adds a coloured edge; the sub text always says why. ``delta`` adds an
    arrow and signed change (arrow, sign and words carry the meaning, not colour alone).
    """
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        tone = "neutral" if higher_is_better is None or delta == 0 else ("good" if (delta > 0) == higher_is_better else "bad")
        delta_html = f'<div class="delta {tone}">{arrow} {delta_text}</div>'
    col.markdown(
        f'<div class="kpi {state}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'{delta_html}<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig: go.Figure, height: int = 360, y_title: str = "", x_title: str = "", legend: bool = True) -> go.Figure:
    """Apply the shared chart style: white surface, hairline grid, legend above the plot."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=40 if legend else 14, b=12),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12.5, color=TEXT),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font_size=12),
        showlegend=legend,
    )
    fig.update_xaxes(
        showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, automargin=True,
        title_text=x_title, title_font_color=TEXT_SECONDARY, tickfont_color=TEXT_SECONDARY,
    )
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True,
        title_text=y_title, title_font_color=TEXT_SECONDARY, tickfont_color=TEXT_SECONDARY,
    )
    for trace in fig.data:
        if trace.type == "scatter" and trace.mode is None:
            trace.mode = "lines"
    return fig


def mark_period(fig: go.Figure, start, end, label: str) -> go.Figure:
    """Shade an event window (Ramadan, an outage, a campaign) behind the data with a small label."""
    fig.add_vrect(
        x0=start, x1=end, fillcolor=EVENT_FILL, line_width=0, layer="below",
        annotation_text=label, annotation_position="top left", annotation_font=dict(size=11, color=TEXT_SECONDARY),
    )
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)


def queue_label(code: str) -> str:
    return "All queues" if code == "ALL" else f"{QUEUES[code].name}"


def queue_picker(key: str, include_all: bool = True, default: str | None = None) -> str:
    options = (["ALL"] if include_all else []) + list(QUEUES)
    index = options.index(default) if default in options else 0
    return st.selectbox("Queue", options, index=index, format_func=queue_label, key=key)


@st.cache_data(show_spinner=False)
def load_intervals() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "intervals.parquet")


@st.cache_data(show_spinner=False)
def load_calendar() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "calendar.parquet")


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "events.parquet")


@st.cache_data(show_spinner=False)
def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((PROCESSED_DIR / "metadata.json").read_text())


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
