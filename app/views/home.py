"""Home: what the project is, headline results and a card for every page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import PAGES, PLANNING_STEPS, kpi, load_metadata, load_processed, pct

from wfm.config import PROCESSED_DIR, QUEUES, SL_TARGET, SL_THRESHOLD_S
from wfm.realtime import interval_label, load_rta_day, snapshot

meta = load_metadata()
metrics = load_processed("backtest_metrics")

st.markdown(
    """
<div class="hero">
  <div class="brand">NileConnect · Workforce Management</div>
  <h1>WFM Command Centre</h1>
  <p class="tagline">The full planning cycle for a Cairo telecom contact centre: forecast demand, staff it with
  Erlang C and A, plan hiring six months out, build legal schedules and rosters, then run the day as a real-time analyst.</p>
  <div class="chips">
    <span class="chip note">Fictional company · synthetic data</span>
    <span class="chip">4 queues · ~13,000 calls a day</span>
    <span class="chip">Egyptian calendar, Ramadan and labour law</span>
    <span class="chip">Python · SQL · Excel · LightGBM · HiGHS MILP · Streamlit</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def wape(window: str, model: str) -> float:
    m = metrics[(metrics["window"] == window) & (metrics["model"] == model) & (metrics["queue"] == "All queues")]
    return float(m["wape"].iloc[0])


@st.cache_data(show_spinner=False)
def outage_response() -> tuple[str, float, float]:
    """Projected SL without and with the recommended actions, one hour into the replayed outage."""
    day = load_rta_day(PROCESSED_DIR)
    now = 31
    snap = snapshot(day, now)
    return interval_label(now), snap.action_effect["sl_before"], snap.action_effect["sl_after"]


ramadan = "Ramadan 2026"
sales_ft = meta["schedule_stats"]["SALES"]["fte_equivalent"]
sales_pt = meta["schedule_stats_part_time"]["SALES"]["fte_equivalent"]
cross = pd.DataFrame(meta["cross_skill"]["summary"]).set_index("flex_share")
at, sl_before, sl_after = outage_response()

st.markdown("##### Headline results")
c = st.columns(4)
kpi(c[0], "Ramadan forecast error", pct(wape(ramadan, "LightGBM")), f"LightGBM daily WAPE, vs {pct(wape(ramadan, 'Seasonal naive'))} for seasonal naive")
kpi(c[1], "Outage response", f"{sl_before:.0%} → {sl_after:.0%}", f"Projected service level at {at} on the outage day, before → after the RTA actions")
kpi(c[2], "Part-time shifts on SALES", f"−{sales_ft - sales_pt:.1f} FTE", f"{(sales_ft - sales_pt) / sales_ft:.0%} fewer paid hours for the same 80/20 coverage")
kpi(c[3], "Cross-skilling 30% of agents", f"{cross.loc[0.3, 'fte_saved']:.1f} FTE", f"saved per week on the Arabic queues ({cross.loc[0.3, 'saving']:.1%} of agent-hours)")

for section in ("Start", "Plan", "Schedule", "Operate", "Report"):
    pages = [p for p in PAGES if p.section == section and p.key != "home"]
    if not pages:
        continue
    st.markdown(f"##### {section}")
    cols = st.columns(4 if section == "Plan" else 3)
    for col, page in zip(cols, pages, strict=False):
        with col.container(border=True, key=f"card_{page.key}"):
            stage = f"Step {page.step} of {PLANNING_STEPS}" if page.step else page.tag
            st.markdown(f'<div class="card-stage">{stage}</div>', unsafe_allow_html=True)
            st.page_link(page.file, label=page.title, icon=page.icon)
            st.markdown(f'<div class="card-blurb">{page.blurb}</div>', unsafe_allow_html=True)

with st.expander("The four queues"):
    st.dataframe(
        pd.DataFrame(
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
        ),
        hide_index=True,
    )
st.caption(
    "NileConnect is fictional. Interval-level contact-centre data is not publicly available, so two years of 30-minute "
    f"history were simulated with realistic Egyptian patterns. Data built {meta['built_at']} from seed {meta['seed']}; "
    f"history {meta['history'][0]} to {meta['history'][1]}, forecast {meta['forecast'][0]} to {meta['forecast'][1]}."
)
