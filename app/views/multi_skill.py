"""Multi-skill: how much cross-skilling the two Arabic queues saves, from a discrete-event simulation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, HIGHLIGHT, MUTED, QUEUE_COLORS, chart, kpi, load_metadata, load_processed, page_header, pct, style

from wfm.config import QUEUES
from wfm.multiskill import FLEX_SHARES, QueueLoad, compare_shares, pooling_gain_bound

meta = load_metadata()
detail = load_processed("cross_skill_day")
summary = pd.DataFrame(meta["cross_skill"]["summary"])
day = pd.Timestamp(meta["cross_skill"]["day"])
base_hours = summary.loc[summary["flex_share"] == 0, "agent_hours"].iloc[0]
best = summary.loc[summary["saving"].idxmax()]
near_best = summary[summary["saving"] >= 0.9 * best["saving"]].iloc[0]  # smallest share with 90% of the gain
page_header(
    "multi_skill",
    "Erlang formulas cover one skill group. If some agents can take both Arabic billing and Arabic technical calls, the "
    "two queues share spare capacity and need fewer people in total. There is no closed formula for partial "
    "cross-skilling, so this page uses a discrete-event simulation of calls, agents and abandoning callers.",
    f"Cross-skilling <b>{near_best['flex_share']:.0%}</b> of AR_BILLING and AR_TECH agents saves "
    f"<b>{near_best['saving']:.1%}</b> of agent-hours, about <b>{near_best['fte_saved']:.1f} FTE a week</b>. "
    "Training everyone saves less, because cross-skilled agents are slower on their second skill.",
)

k = st.columns(4)
kpi(k[0], "Agent-hours, dedicated teams", f"{base_hours:,.0f}", f"Both queues, {day:%a %d %b %Y}, 80/20 on each")
kpi(k[1], "Best saving", pct(best["saving"]), f"at {best['flex_share']:.0%} cross-skilled")
kpi(k[2], "Recommended share", f"{near_best['flex_share']:.0%}", f"captures {near_best['saving'] / best['saving']:.0%} of the best saving")
kpi(k[3], "FTE saved per week", f"{near_best['fte_saved']:.1f}", "if every day looks like this one")

left, right = st.columns([1, 1.6], gap="large")
with left:
    fig = go.Figure(
        go.Bar(
            x=[f"{f:.0%}" for f in summary["flex_share"]],
            y=summary["saving"],
            marker_color=[HIGHLIGHT if f == near_best["flex_share"] else QUEUE_COLORS["AR_BILLING"] for f in summary["flex_share"]],
            hovertemplate="%{x} cross-skilled: %{y:.1%} fewer agent-hours<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(type="category")
    chart(style(fig, 320, "Agent-hours saved vs dedicated", "Share of agents cross-skilled", legend=False))
with right:
    wide = detail.pivot_table(index="interval", columns="flex_share", values="total")
    times = [f"{i // 2:02d}:{30 * (i % 2):02d}" for i in wide.index]
    fig_d = go.Figure()
    fig_d.add_scatter(x=times, y=wide[0.0], name="Dedicated teams", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
    fig_d.add_scatter(x=times, y=wide[near_best["flex_share"]], name=f"{near_best['flex_share']:.0%} cross-skilled", line=dict(color=HIGHLIGHT, width=2, shape="hv"))
    fig_d.add_scatter(x=times, y=wide[1.0], name="Everyone cross-skilled", line=dict(color=MUTED, width=1.5, dash="dot", shape="hv"))
    chart(style(fig_d, 320, "Agents needed (both queues)"))
st.caption(
    "Most of the pooling benefit arrives with a minority of cross-skilled agents: they absorb the random peaks of "
    "whichever queue is busier, while dedicated agents keep the bulk of the work. Training everyone is no better, "
    "because cross-skilled agents handle technical calls about 5% slower and the queues are already large enough to "
    "be efficient on their own. Simulated every hour of the day; results move by about one agent between runs."
)

# Interactive ---------------------------------------------------------------------------
st.subheader("Try an interval")
c = st.columns(5)
calls_a = c[0].number_input("Billing calls / 30 min", 1, 600, 150, 10)
aht_a = c[1].number_input("Billing AHT (s)", 60, 900, 270, 10)
calls_b = c[2].number_input("Tech calls / 30 min", 1, 600, 140, 10)
aht_b = c[3].number_input("Tech AHT (s)", 60, 900, 410, 10)
penalty = c[4].slider("Cross-skilled AHT penalty on tech", 0.0, 0.3, 0.05, 0.01, format="%.2f")


@st.cache_data(show_spinner=False)
def run(calls_a: int, aht_a: int, calls_b: int, aht_b: int, penalty: float) -> pd.DataFrame:
    a = QueueLoad(calls_a, aht_a, QUEUES["AR_BILLING"].patience_s)
    b = QueueLoad(calls_b, aht_b, QUEUES["AR_TECH"].patience_s)
    return compare_shares(a, b, FLEX_SHARES, penalty, hours=12)


with st.spinner("Simulating 12 hours of calls for each scenario..."):
    table = run(calls_a, aht_a, calls_b, aht_b, penalty)
bound = pooling_gain_bound(QueueLoad(calls_a, aht_a, 1), QueueLoad(calls_b, aht_b, 1))
k = st.columns(3)
dedicated = int(table.loc[table["flex_share"] == 0, "total"].iloc[0])
best_row = table.loc[table["total"].idxmin()]
kpi(k[0], "Dedicated teams", f"{dedicated} agents", f"Erlang C (no abandons) says {int(table['erlang_c_separate'].iloc[0])}")
kpi(k[1], "With cross-skilling", f"{int(best_row['total'])} agents", f"at {best_row['flex_share']:.0%} cross-skilled ({best_row['saving_vs_dedicated']:.1%} fewer)")
kpi(k[2], "Full-pooling bound (Erlang C)", pct(bound), "saving if both queues were one perfect pool")
st.dataframe(
    table[["flex_share", "total", "dedicated_a", "dedicated_b", "flexible", "sl_a", "sl_b", "abandon_a", "abandon_b", "saving_vs_dedicated"]]
    .rename(
        columns={
            "flex_share": "Cross-skilled share",
            "total": "Agents",
            "dedicated_a": "Billing only",
            "dedicated_b": "Tech only",
            "flexible": "Both skills",
            "sl_a": "Billing SL",
            "sl_b": "Tech SL",
            "abandon_a": "Billing abandon",
            "abandon_b": "Tech abandon",
            "saving_vs_dedicated": "Saving",
        }
    )
    .style.format(
        {
            "Cross-skilled share": "{:.0%}",
            "Billing SL": "{:.1%}",
            "Tech SL": "{:.1%}",
            "Billing abandon": "{:.1%}",
            "Tech abandon": "{:.1%}",
            "Saving": "{:.1%}",
        }
    ),
    hide_index=True,
)
st.caption(
    "Each row is the smallest team that keeps both queues at 80/20, found by simulation on the same stream of calls. "
    "Routing: a call goes to an idle agent of its own skill first, then to a cross-skilled agent; a free cross-skilled "
    "agent takes whichever waiting call has waited longest. Try small volumes (night-time levels): pooling matters most "
    "when teams are small. The simulator reproduces Erlang A to within about one point of service level when nobody is cross-skilled."
)
