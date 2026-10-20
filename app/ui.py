"""Shared UI helpers: import path, cached data loaders, chart styling and KPI cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1320px;}
h1 {font-size: 1.85rem !important; font-weight: 650 !important; letter-spacing: -0.01em;}
h2 {font-size: 1.3rem !important; font-weight: 620 !important; margin-top: 1.2rem !important;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important;}
.kpi {background: #ffffff; border: 1px solid #e8e7e3; border-radius: 8px; padding: 14px 16px 12px; min-height: 124px; margin-bottom: 8px;}
.kpi .label {font-size: 0.78rem; color: #52514e; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;}
.kpi .value {font-size: 1.6rem; font-weight: 650; color: #0b0b0b; line-height: 1.2; font-variant-numeric: tabular-nums;}
.kpi .sub {font-size: 0.8rem; color: #52514e; margin-top: 4px;}
.kpi.bad {border-left: 4px solid #d03b3b;} .kpi.warn {border-left: 4px solid #fab219;} .kpi.ok {border-left: 4px solid #0ca30c;}
.badge {display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 10px; margin-right: 8px;
        text-transform: uppercase; letter-spacing: 0.04em;}
.badge.critical {background: #fbe3e3; color: #a42828;} .badge.warning {background: #fdf0d2; color: #7a5300;}
.badge.action {background: #e2edfb; color: #184f95;} .badge.info {background: #ecebe7; color: #52514e;}
.panel-row {padding: 9px 0; border-bottom: 1px solid #ecebe7; font-size: 0.9rem; line-height: 1.45;}
.panel-row:last-child {border-bottom: none;}
.panel-row .title {font-weight: 600;} .panel-row .when {color: #52514e; font-variant-numeric: tabular-nums;}
.fictional {background: #f3f2ef; border: 1px solid #e8e7e3; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #52514e;}
</style>
"""


def setup_page(title: str) -> None:
    """Page config, shared CSS and the sidebar note shown on every page."""
    st.set_page_config(page_title=f"{title} | NileConnect WFM", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**NileConnect WFM Command Centre**")
        st.caption(
            "NileConnect is a fictional Cairo telecom contact centre. All data is synthetic, generated to "
            "mirror real interval-level patterns."
        )


def kpi(col, label: str, value: str, sub: str = "", state: str = "") -> None:
    """Render a KPI card; ``state`` is '', 'ok', 'warn' or 'bad' (border colour plus the sub text carry meaning)."""
    col.markdown(
        f'<div class="kpi {state}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
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
