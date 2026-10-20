"""Forecast: model comparison, backtests (normal and Ramadan) and the 26-week outlook."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (
    ACTUAL_COLOR,
    MODEL_COLORS,
    MUTED,
    QUEUE_COLORS,
    chart,
    kpi,
    load_events,
    load_metadata,
    load_processed,
    pct,
    queue_label,
    setup_page,
    style,
)

from wfm.config import QUEUES
from wfm.forecast import MODELS

setup_page("Forecast")
st.title("Demand forecast")
st.caption(
    "Daily volume per queue is forecast with three models and spread into 30-minute intervals using day-type "
    "profiles. Accuracy is measured with WAPE (weighted absolute percentage error) and bias on held-out periods."
)

meta = load_metadata()
metrics = load_processed("backtest_metrics")
ivl_metrics = load_processed("backtest_interval_metrics")
bt_daily = load_processed("backtest_daily")
bt_ivl = load_processed("backtest_intervals")
fc_daily = load_processed("forecast_daily")
daily = load_processed("daily_actuals")
events = load_events()
windows = list(metrics["window"].unique())

# Model comparison --------------------------------------------------------------
st.subheader("Model comparison")
summary = metrics[metrics["queue"] == "All queues"].pivot(index="model", columns="window", values=["wape", "bias"])
summary = summary.reindex(list(MODELS))
table = pd.DataFrame(index=summary.index)
for w in windows:
    table[f"WAPE: {w}"] = summary[("wape", w)].map(pct)
    table[f"Bias: {w}"] = summary[("bias", w)].map(lambda v: f"{v:+.1%}")
table.index.name = "Model"
st.dataframe(table)
st.caption(
    f"Champion model: **{meta['champion_model']}** (lowest average WAPE across both backtests). Seasonal naive repeats "
    "last week, so it cannot anticipate holidays, billing-cycle days or Ramadan. Holt-Winters learns weekly seasonality "
    "and trend but not calendar events. The single global LightGBM model uses calendar features (weekday, holidays, "
    "Ramadan day, post-Eid, billing-cycle day, campaigns) on a detrended target, so it can extrapolate growth."
)

per_queue = metrics[metrics["queue"] != "All queues"].pivot_table(index=["window", "queue"], columns="model", values="wape")
per_queue = per_queue[list(MODELS)].map(pct).reset_index().rename(columns={"window": "Backtest", "queue": "Queue"})
with st.expander("WAPE by queue"):
    st.dataframe(per_queue, hide_index=True)

# Backtest charts ---------------------------------------------------------------
st.subheader("Backtests")
c1, c2 = st.columns([2, 1])
window = c1.radio("Backtest window", windows, horizontal=True)
queue = c2.selectbox("Queue", list(QUEUES), format_func=queue_label, key="bt_queue")
bd = bt_daily[(bt_daily["window"] == window) & (bt_daily["queue"] == queue)]
m = metrics[(metrics["window"] == window) & (metrics["queue"] == queue)].set_index("model")
im = ivl_metrics[(ivl_metrics["window"] == window) & (ivl_metrics["queue"] == queue)].iloc[0]

k = st.columns(4)
kpi(k[0], "LightGBM daily WAPE", pct(m.loc["LightGBM", "wape"]), f"Bias {m.loc['LightGBM', 'bias']:+.1%}")
kpi(k[1], "Holt-Winters daily WAPE", pct(m.loc["Holt-Winters", "wape"]), f"Bias {m.loc['Holt-Winters', 'bias']:+.1%}")
kpi(k[2], "Seasonal naive daily WAPE", pct(m.loc["Seasonal naive", "wape"]), f"Bias {m.loc['Seasonal naive', 'bias']:+.1%}")
kpi(k[3], "Interval WAPE (volume / AHT)", f"{pct(im['wape'])} / {pct(im['aht_wape'])}", "LightGBM, 30-minute level")

fig = go.Figure()
actual = bd[bd["model"] == "LightGBM"]
fig.add_scatter(x=actual["date"], y=actual["actual_raw"], name="Actual (raw)", mode="markers", marker=dict(color=MUTED, size=8, symbol="circle-open"))
fig.add_scatter(x=actual["date"], y=actual["actual"], name="Actual (outages cleaned)", line=dict(color=ACTUAL_COLOR, width=2.5))
for model in MODELS:
    f = bd[bd["model"] == model]
    fig.add_scatter(x=f["date"], y=f["forecast"], name=model, line=dict(color=MODEL_COLORS[model], width=2))
chart(style(fig, 380, "Calls offered per day"))
st.caption(
    "Each model was trained only on data up to the day before the window. Hollow markers show raw volume; where "
    "they sit above the black line, a network outage inflated the day. Outages are unforecastable, so they are "
    "removed from training and accuracy is scored against cleaned actuals; the RTA desk handles them on the day."
)

if window.startswith("Ramadan"):
    st.info(
        "The Ramadan window starts ten days before Ramadan 2026 and runs past Eid al-Fitr. The only Ramadan in the "
        "training data is Ramadan 2025, so the model learns the Ramadan effect and the Ramadan intraday profile from a "
        "single previous year, as a real planner would."
    )

days = sorted(bt_ivl.loc[(bt_ivl["window"] == window), "date"].unique())
pick = st.select_slider("Day for the interval view", options=days, value=days[min(17, len(days) - 1)], format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))
iv = bt_ivl[(bt_ivl["window"] == window) & (bt_ivl["queue"] == queue) & (bt_ivl["date"] == pick)]
left, right = st.columns(2)
fig_i = go.Figure()
fig_i.add_scatter(x=iv["interval_start"], y=iv["offered"], name="Actual", line=dict(color=ACTUAL_COLOR, width=2.5))
fig_i.add_scatter(x=iv["interval_start"], y=iv["forecast_offered"], name="Forecast", line=dict(color=QUEUE_COLORS[queue], width=2))
fig_i.update_xaxes(tickformat="%H:%M")
with left:
    chart(style(fig_i, 300, "Calls per 30 minutes"))
fig_a = go.Figure()
fig_a.add_scatter(x=iv["interval_start"], y=iv["aht_s"], name="Actual AHT", line=dict(color=ACTUAL_COLOR, width=2))
fig_a.add_scatter(x=iv["interval_start"], y=iv["forecast_aht_s"], name="Forecast AHT", line=dict(color=QUEUE_COLORS[queue], width=2))
fig_a.update_xaxes(tickformat="%H:%M")
with right:
    chart(style(fig_a, 300, "AHT (seconds)"))
st.caption(
    "Top-down interval forecast: the daily forecast multiplied by the intraday profile for the day type "
    "(Sunday-Thursday, Friday or Saturday, normal or Ramadan). Interval-level error is higher than daily error "
    "because 30-minute arrivals are Poisson-noisy; with 100 calls per interval, noise alone is about 8%."
)

# Outlook ------------------------------------------------------------------------
st.subheader("26-week outlook")
champion = meta["champion_model"]
hist = daily[daily["date"] > daily["date"].max() - pd.Timedelta(weeks=26)]
fig_o = go.Figure()
for q in QUEUES:
    h = hist[hist["queue"] == q]
    f = fc_daily[(fc_daily["queue"] == q) & (fc_daily["model"] == champion)]
    fig_o.add_scatter(x=h["date"], y=h["offered_clean"], name=f"{q} actual", line=dict(color=QUEUE_COLORS[q], width=1.2), opacity=0.55, legendgroup=q)
    fig_o.add_scatter(x=f["date"], y=f["forecast"], name=f"{q} forecast", line=dict(color=QUEUE_COLORS[q], width=2), legendgroup=q)
fig_o.add_vline(x=pd.Timestamp(meta["forecast"][0]), line=dict(color=MUTED, width=1))
chart(style(fig_o, 400, "Calls offered per day"))
st.caption(
    f"{champion} forecast from {meta['forecast'][0]} to {meta['forecast'][1]}, after the last 26 weeks of cleaned "
    "actuals (faded). The horizon includes the November and December billing peaks, planned sales campaign weeks "
    "and Ramadan 2027 (about 8 February to 9 March, dates approximate), followed by the Eid al-Fitr trough."
)

weekly = fc_daily[fc_daily["model"] == champion].assign(week=lambda d: d["date"] - pd.to_timedelta((d["date"].dt.dayofweek + 1) % 7, unit="D"))
weekly = weekly.pivot_table(index="week", columns="queue", values="forecast", aggfunc="sum").round(0)
weekly["Total"] = weekly.sum(axis=1)
weekly.index = weekly.index.strftime("%d %b %Y")
weekly.index.name = "Week starting (Sun)"
with st.expander("Weekly forecast table"):
    st.dataframe(weekly.astype(int))

outages = events[events["event_type"] == "outage"]
st.caption(f"Anomaly treatment: {len(outages)} network outages in the history were replaced by a baseline before training.")
