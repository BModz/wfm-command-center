"""RTA desk: replay of an outage day as a live wallboard with alerts, actions and adherence."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACTUAL_COLOR, HIGHLIGHT, MUTED, QUEUE_COLORS, REFERENCE, STATUS, chart, kpi, load_processed, page_header, pct, style

from wfm.config import INTERVALS_PER_DAY, PROCESSED_DIR, QUEUES, SL_TARGET
from wfm.realtime import Snapshot, detect_incident, interval_label, load_rta_day, snapshot
from wfm.rta_reports import (
    FlagRules,
    flag_agents,
    midday_message,
    midday_reforecast,
    outlook_message,
    rca_markdown,
    rca_report,
    staffing_outlook,
)

START_AT = 27  # 13:30, an hour before the outage is detected
SPEEDS = {"Slow": 2.5, "Normal": 1.5, "Fast": 0.8}  # seconds per 30-minute interval


@st.cache_resource(show_spinner=False)
def rta_day():
    return load_rta_day(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def view(now: int) -> Snapshot:
    return snapshot(rta_day(), now)


day = rta_day()
color = QUEUE_COLORS[day.queue]
detected = next(n for n in range(1, INTERVALS_PER_DAY + 1) if detect_incident(day.intervals, n))
effect = view(detected + 2).action_effect
page_header(
    "rta",
    f"Replay of {QUEUES[day.queue].name} on {day.date:%A %d %B %Y}. The interval forecast was made the evening before; "
    f"the schedule was optimised about two weeks earlier; {len(day.agents)} agents are rostered. Press Play to watch "
    "the day unfold, or drag the clock to any half-hour.",
    f"A network outage nearly triples tech volume from 14:00. The desk flags a possible incident at <b>{interval_label(detected)}</b>, "
    f"30 minutes in, and by {interval_label(detected + 2)} its recommended actions lift projected service level from "
    f"<b>{effect['sl_before']:.0%}</b> to <b>{effect['sl_after']:.0%}</b> for the next four hours.",
)

state = st.session_state
if "rta_now" not in state:
    # A shareable deep link such as ?t=15:30 opens the desk at that time.
    labels = {interval_label(i): i for i in range(1, INTERVALS_PER_DAY + 1)}
    state.rta_now = labels.get(st.query_params.get("t", ""), START_AT)
state.setdefault("rta_playing", False)


def toggle_play() -> None:
    if not state.rta_playing and state.rta_now >= INTERVALS_PER_DAY:
        state.rta_now = 1
    state.rta_playing = not state.rta_playing


def restart() -> None:
    state.rta_now, state.rta_playing = START_AT, False


def pause() -> None:
    state.rta_playing = False


controls = st.columns([1, 1, 2, 3], vertical_alignment="bottom")
controls[0].button("Pause" if state.rta_playing else "Play", icon=":material/pause:" if state.rta_playing else ":material/play_arrow:",
                   on_click=toggle_play, type="primary", width="stretch")
controls[1].button("Restart", icon=":material/replay:", on_click=restart, width="stretch")
speed = controls[2].segmented_control("Speed", list(SPEEDS), default="Normal", key="rta_speed") or "Normal"


def status(value: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    """'ok', 'warn' or 'bad' for a wallboard tile."""
    if higher_is_better:
        return "ok" if value >= good else ("warn" if value >= warn else "bad")
    return "ok" if value <= good else ("warn" if value <= warn else "bad")


def tile(label: str, value: str, sub: str, state_: str = "") -> str:
    return (
        f'<div class="wb-tile {state_}"><div class="wb-label">{label}</div>'
        f'<div class="wb-value">{value}</div><div class="wb-sub">{sub}</div></div>'
    )


@st.fragment(run_every=SPEEDS[speed] if state.rta_playing else None)
def desk() -> None:
    """Everything that moves with the clock; reruns on its own while the replay is playing."""
    if state.rta_playing:
        if state.rta_now >= INTERVALS_PER_DAY:
            state.rta_playing = False
            st.rerun()
        state.rta_now += 1
    now = state.rta_now
    snap = view(now)
    kp = snap.kpis
    last = day.intervals.iloc[now - 1]

    live = '<span class="live"><span class="dot"></span>LIVE REPLAY</span>' if state.rta_playing else '<span class="live paused"><span class="dot"></span>PAUSED</span>'
    st.markdown(
        f'<div class="clock">{live}<span class="time">{interval_label(now)}</span>'
        f"<span>{day.date:%a %d %b %Y} · {QUEUES[day.queue].name}</span></div>",
        unsafe_allow_html=True,
    )
    st.select_slider(
        "Time now (end of the last completed interval)",
        options=list(range(1, INTERVALS_PER_DAY + 1)),
        format_func=interval_label,
        key="rta_now",
        on_change=pause,
    )

    staffed, required = last["staffed"], snap.timeline["required_actual"].iloc[now - 1]
    out_now = kp.get("out_of_adherence_now", 0)
    st.markdown(
        '<div class="wallboard">'
        + tile("Service level", pct(kp["service_level"], 0), f"day to date · target {SL_TARGET:.0%}", status(kp["service_level"], SL_TARGET, 0.7))
        + tile("Abandoned", pct(kp["abandon_rate"]), "day to date · alert above 8%", status(kp["abandon_rate"], 0.05, 0.08, higher_is_better=False))
        + tile("ASA", f"{kp['asa_s']:.0f}s", "average wait, answered calls", status(kp["asa_s"], 20, 45, higher_is_better=False))
        + tile("Volume vs forecast", f"{kp['volume_var']:+.0%}", f"{kp['offered']:,.0f} offered today", status(kp["volume_var"], 0.1, 0.2, higher_is_better=False))
        + tile("Agents on phones", f"{staffed:.0f} / {required:.0f}", "staffed / required, last 30 min", status(staffed / max(required, 1), 1.0, 0.9))
        + tile("Out of adherence", f"{out_now:.0f}", f"of {kp.get('agents_on_shift_now', 0):.0f} agents on shift", status(out_now, 2, 5, higher_is_better=False))
        + "</div>",
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    kpi(c[0], "Occupancy", pct(kp["occupancy"]), "Handle time / staffed time")
    kpi(c[1], "Staffed vs scheduled", pct(kp["staffing_realisation"]), "Agents on phones vs plan")
    kpi(c[2], "Adherence", pct(kp.get("adherence", 1.0)), "Right state at the right time")
    kpi(c[3], "Conformance", pct(kp.get("conformance", 1.0)), "Phone time worked vs scheduled")
    kpi(c[4], "AHT vs forecast", f"{kp['aht_var']:+.1%}", f"{kp['aht_s']:.0f}s actual today")

    live_tab, flags_tab, reports_tab = st.tabs(["Live view", "Agent flags", "Shift reports"])

    with live_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Alerts")
            if snap.alerts:
                rows = "".join(
                    f'<div class="panel-row"><span class="badge {a["severity"].lower()}">{a["severity"]}</span>'
                    f'<span class="title">{html.escape(a["alert"])}</span><br>{html.escape(a["detail"])}</div>'
                    for a in snap.alerts
                )
            else:
                rows = '<div class="panel-row"><span class="badge info">Clear</span>No thresholds breached in the last interval.</div>'
            st.markdown(rows, unsafe_allow_html=True)
        with right:
            st.subheader("Recommended actions")
            if snap.actions:
                rows = "".join(
                    f'<div class="panel-row"><span class="badge action">{i}</span><span class="title">{html.escape(a["action"])}</span> '
                    f'<span class="when">{html.escape(a["when"])}</span><br>{html.escape(a["detail"])}</div>'
                    for i, a in enumerate(snap.actions, 1)
                )
                st.markdown(rows, unsafe_allow_html=True)
                eff = snap.action_effect
                st.markdown(
                    f"Projected service level for the next 4 hours: **{pct(eff['sl_before'])}** as things stand, "
                    f"**{pct(eff['sl_after'])}** if the actions are taken (Erlang A on the reforecast)."
                )
            else:
                st.markdown(
                    '<div class="panel-row"><span class="badge info">None</span>Projected staffing meets 80/20 for the next 4 hours.</div>',
                    unsafe_allow_html=True,
                )

        tl = snap.timeline.copy()
        tl["ts"] = day.date + pd.to_timedelta(tl["interval"] * 30, unit="min")
        past, future = tl[tl["interval"] < now], tl[tl["interval"] >= now]
        now_ts = day.date + pd.Timedelta(minutes=30 * now)

        def mark_now(fig: go.Figure) -> go.Figure:
            fig.add_vline(x=now_ts, line=dict(color=MUTED, width=1))
            fig.update_xaxes(tickformat="%H:%M", range=[day.date, day.date + pd.Timedelta(days=1)])
            return fig

        st.subheader("Volume: actual, forecast and intraday reforecast")
        fig_v = go.Figure()
        fig_v.add_scatter(x=tl["ts"], y=tl["forecast_offered"], name="Forecast (made day before)", line=dict(color=REFERENCE, width=2, shape="hv"))
        fig_v.add_scatter(x=past["ts"], y=past["offered"], name="Actual offered", line=dict(color=ACTUAL_COLOR, width=2.5, shape="hv"))
        fig_v.add_scatter(x=future["ts"], y=future["reforecast_offered"], name="Reforecast", line=dict(color=color, width=2, dash="dash", shape="hv"))
        chart(mark_now(style(fig_v, 320, "Calls per 30 minutes")))
        ratio = snap.timeline.attrs.get("volume_ratio", 1.0)
        st.caption(
            f"The reforecast scales the remaining forecast by the recent actual/forecast ratio ({ratio:.2f}x, weighted towards the "
            "latest intervals). "
            + (
                "An incident is detected, so the surge is assumed to decay back to the normal run rate over the next couple of hours."
                if snap.incident
                else "Surge intervals are excluded from the run rate, so a past incident does not inflate the rest of the day."
            )
        )

        st.subheader("Agents: scheduled, staffed and required")
        fig_a = go.Figure()
        fig_a.add_scatter(x=tl["ts"], y=tl["required_fc"], name="Required (forecast)", line=dict(color=REFERENCE, width=2, shape="hv"))
        fig_a.add_scatter(x=future["ts"], y=future["required_reforecast"], name="Required (reforecast)", line=dict(color=ACTUAL_COLOR, width=2, dash="dash", shape="hv"))
        fig_a.add_scatter(x=past["ts"], y=past["required_actual"], name="Required (actual demand)", line=dict(color=ACTUAL_COLOR, width=2, shape="hv"))
        fig_a.add_scatter(x=tl["ts"], y=tl["scheduled_on_phone"], name="Scheduled on phones", line=dict(color=color, width=1.5, dash="dot", shape="hv"))
        fig_a.add_scatter(x=past["ts"], y=past["staffed"], name="Staffed (actual)", line=dict(color=color, width=2.5, shape="hv"))
        if "staffed_after_actions" in tl and snap.actions:
            fig_a.add_scatter(x=future["ts"], y=future["staffed_after_actions"], name="Staffed after actions", line=dict(color=HIGHLIGHT, width=2.5, dash="dash", shape="hv"))
        chart(mark_now(style(fig_a, 340, "Agents on the phones")))
        st.caption(
            "Scheduled is the published plan net of breaks and coaching; staffed is agents actually in the on-phone state. The "
            "difference is lost time from late logins, extended breaks, unscheduled aux and no-shows. Required is Erlang C at 80/20."
        )

        st.subheader("Service level by interval")
        fig_s = go.Figure()
        fig_s.add_bar(x=past["ts"], y=past["service_level"], name="Actual SL", marker_color=[STATUS["good"] if v >= SL_TARGET else (STATUS["warning"] if v >= 0.7 else STATUS["critical"]) for v in past["service_level"]])
        if "projected_sl_before" in tl:
            fig_s.add_scatter(x=future["ts"], y=future["projected_sl_before"], name="Projected, no action", line=dict(color=STATUS["critical"], width=2, dash="dash"))
            fig_s.add_scatter(x=future["ts"], y=future["projected_sl_after"], name="Projected, with actions", line=dict(color=HIGHLIGHT, width=2.5))
        fig_s.add_hline(y=SL_TARGET, line=dict(color=REFERENCE, width=1), annotation_text="80% target", annotation_position="bottom right")
        fig_s.update_yaxes(tickformat=".0%", range=[0, 1.05])
        chart(mark_now(style(fig_s, 300, "Answered within 20s")))
        st.caption("Bars: realised service level per interval (green on target, amber 70-80%, red below 70%). Lines: next 4 hours projected with Erlang A, before and after the recommended actions.")

    with flags_tab:
        render_flags(now)

        st.subheader("Agent adherence")
        adh = snap.adherence
        f1, f2 = st.columns([1, 2])
        only_out = f1.toggle("Only agents out of adherence now", value=False, key="rta_only_out")
        teams = f2.multiselect("Teams", sorted(adh["team"].unique()), placeholder="All teams", key="rta_teams")
        shown = adh[adh["out_of_adherence"]] if only_out else adh
        if teams:
            shown = shown[shown["team"].isin(teams)]
        st.dataframe(
            shown,
            hide_index=True,
            height=380,
            column_config={
                "agent_id": "Agent",
                "team": "Team",
                "shift": "Shift",
                "scheduled_now": "Scheduled now",
                "actual_now": "Actual now",
                "on_shift_now": st.column_config.CheckboxColumn("On shift"),
                "out_of_adherence": st.column_config.CheckboxColumn("Out now"),
                "minutes_out_now": st.column_config.NumberColumn("Minutes out (current)", format="%d"),
                "adherence": st.column_config.ProgressColumn("Adherence", min_value=0.0, max_value=1.0, format="percent"),
                "conformance": st.column_config.NumberColumn("Conformance", format="percent"),
                "minutes_out_today": st.column_config.NumberColumn("Minutes out today", format="%d"),
                "exception": "Exceptions today",
            },
        )
        team = adh.groupby("team").agg(agents=("agent_id", "count"), adherence=("adherence", "mean"), conformance=("conformance", "mean"), out_now=("out_of_adherence", "sum")).reset_index()
        with st.expander("Adherence by team"):
            st.dataframe(
                team.rename(columns={"team": "Team", "agents": "Agents", "adherence": "Adherence", "conformance": "Conformance", "out_now": "Out of adherence now"}).style.format(
                    {"Adherence": "{:.1%}", "Conformance": "{:.1%}"}
                ),
                hide_index=True,
            )
        st.caption(
            "Adherence = share of scheduled time to date in the scheduled state (phone, break, coaching), measured in 5-minute "
            "slots. Conformance = phone time actually worked divided by phone time scheduled, regardless of when. An agent who "
            "takes a full break 30 minutes late keeps 100% conformance but loses adherence, and leaves the queue short "
            "exactly when the plan counted on them."
        )

    with reports_tab:
        render_reports(now)


LEVEL_KEYS = ["Escalate to operations manager", "Escalate to team leader", "Nudge agent", "Coaching follow-up"]


def render_flags(now: int) -> None:
    """Escalation ladder for agents out of adherence, ready-to-send messages and the flag log."""
    st.subheader("Agent flags")
    st.caption(
        "Who needs attention right now, how long they have been out of adherence, and the next step on the escalation "
        "ladder: chat nudge, then the team leader, then the operations manager. Agents back in adherence but low for the "
        "day get a coaching follow-up. Mismatches that add capacity (working through a break) are only monitored."
    )
    c = st.columns(3)
    nudge = c[0].number_input("Nudge agent after (min)", 5, 60, 5, 5, key="flag_nudge")
    team_leader = c[1].number_input("Escalate to team leader after (min)", 10, 120, 15, 5, key="flag_tl")
    manager = c[2].number_input("Escalate to operations manager after (min)", 15, 240, 30, 5, key="flag_mgr")
    rules = FlagRules(nudge, max(team_leader, nudge + 5), max(manager, team_leader + 5))
    flags = flag_agents(day, now, rules)
    counts = flags["level"].value_counts() if len(flags) else pd.Series(dtype=int)
    k = st.columns(4)
    states = ["bad", "warn", "", ""]
    for col, level, tone in zip(k, LEVEL_KEYS, states, strict=True):
        n = int(counts.get(level, 0))
        kpi(col, level, f"{n}", "agents", tone if n else "ok")
    if flags.empty:
        st.success("Nobody needs a flag right now.")
        return
    st.dataframe(
        flags[["level", "agent_id", "team", "issue", "since", "minutes_out", "channel", "message"]],
        hide_index=True,
        height=min(60 + 35 * len(flags), 420),
        column_config={
            "level": "Next step",
            "agent_id": "Agent",
            "team": "Team",
            "issue": "Issue",
            "since": "Since",
            "minutes_out": st.column_config.NumberColumn("Minutes", format="%d"),
            "channel": "Channel",
            "message": st.column_config.TextColumn("Message to send", width="large"),
        },
    )
    with st.expander("Copy messages"):
        st.code("\n\n".join(f"[{r.channel}] {r.message}" for r in flags[flags["level"] != "Monitor"].itertuples()), language=None)
    log = state.setdefault("flag_log", [])
    if st.button("Log these flags as sent", icon=":material/send:", key="log_flags"):
        already = {(entry["agent_id"], entry["level"]) for entry in log}
        for rec in flags[flags["level"] != "Monitor"].itertuples():
            if (rec.agent_id, rec.level) not in already:
                log.append({"time": interval_label(now), "agent_id": rec.agent_id, "team": rec.team, "level": rec.level, "channel": rec.channel, "message": rec.message})
    if log:
        st.markdown(f"**Flag log** · {len(log)} sent today")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, hide_index=True, height=min(60 + 35 * len(log_df), 300))
        st.download_button("Download flag log (CSV)", log_df.to_csv(index=False).encode(), file_name="flag_log.csv", mime="text/csv", icon=":material/download:", key="dl_flags")


@st.cache_data(show_spinner=False)
def outlook() -> tuple[pd.DataFrame, pd.DataFrame]:
    return staffing_outlook(load_processed("rta_plan"), day.queue)


@st.cache_data(show_spinner=False)
def rca() -> dict:
    return rca_report(day)


def render_reports(now: int) -> None:
    """Start-of-shift Staffing Outlook, midday reforecast and the incident RCA."""
    st.subheader("Start-of-shift Staffing Outlook: today and tomorrow")
    table, summary = outlook()
    shown = summary.assign(date=pd.to_datetime(summary["date"]).dt.strftime("%a %d %b"))
    st.dataframe(
        shown.rename(
            columns={
                "date": "Day",
                "forecast_calls": "Forecast calls",
                "required_hours": "Required hours",
                "projected_hours": "Projected hours",
                "net_over_under_hours": "Net over/under (h)",
                "short_intervals": "Short intervals",
                "worst_interval": "Worst interval",
                "projected_sl": "Projected SL",
            }
        ).style.format({"Forecast calls": "{:,.0f}", "Required hours": "{:,.0f}", "Projected hours": "{:,.0f}", "Net over/under (h)": "{:+,.0f}", "Projected SL": "{:.1%}"}),
        hide_index=True,
    )
    fig = go.Figure()
    for i, (date, g) in enumerate(table.groupby("date")):
        fig.add_bar(x=g["time"], y=g["over_under"], name=f"{pd.Timestamp(date):%a %d %b}", marker_color=[QUEUE_COLORS[day.queue], MUTED][i])
    fig.update_layout(barmode="group", bargap=0.2)
    chart(style(fig, 260, "Over (+) / under (-) agents"))
    with st.expander("Outlook email"):
        message = outlook_message(summary, table, day.queue)
        st.code(message, language=None)
        d1, d2 = st.columns(2)
        d1.download_button("Download email (.txt)", message.encode(), file_name="staffing_outlook.txt", icon=":material/download:", key="dl_outlook_txt")
        d2.download_button("Download interval table (CSV)", table.to_csv(index=False).encode(), file_name="staffing_outlook.csv", icon=":material/download:", key="dl_outlook_csv")

    st.subheader(f"Intraday reforecast as of {interval_label(now)}")
    st.caption("The routine midday update goes out at 12:00; the desk can rerun it at any time from the actual-to-date and the run rate.")
    summ, mtable = midday_reforecast(day, now)
    k = st.columns(4)
    kpi(k[0], "Volume vs forecast so far", f"{summ['volume_variance']:+.1%}", f"{summ['offered_so_far']:,.0f} calls offered")
    kpi(k[1], "Service level so far", pct(summ["sl_so_far"]), f"abandon {pct(summ['abandon_so_far'])}", "ok" if summ["sl_so_far"] >= SL_TARGET else "bad")
    kpi(k[2], "Rest of day", f"{summ['reforecast_rest_of_day']:,.0f}", f"reforecast vs {summ['original_rest_of_day']:,.0f} planned")
    kpi(k[3], "Revised over/under", f"{summ['revised_net_hours']:+.0f} h", f"{summ['short_intervals']} short intervals", "bad" if summ["short_intervals"] else "ok")
    with st.expander("Reforecast email"):
        message = midday_message(summ, day.queue, view(now).actions)
        st.code(message, language=None)
        d1, d2 = st.columns(2)
        d1.download_button("Download email (.txt)", message.encode(), file_name="midday_reforecast.txt", icon=":material/download:", key="dl_mid_txt")
        d2.download_button("Download interval table (CSV)", mtable.to_csv(index=False).encode(), file_name="midday_reforecast.csv", icon=":material/download:", key="dl_mid_csv")

    st.subheader("Incident RCA")
    report = rca()
    if now < report["ends_at_interval"]:
        st.info(f"The RCA is written once the incident is over. Move the clock past {interval_label(report['ends_at_interval'])} to generate it.")
        return
    document = rca_markdown(report)
    with st.container(border=True):
        # Shift heading levels down for display inside the page; the download keeps the document's own levels.
        st.markdown(document.replace("\n## ", "\n#### ").replace("# ", "### ", 1))
    st.download_button("Download RCA (Markdown)", document.encode(), file_name="rca_outage.md", icon=":material/download:", key="dl_rca")


desk()
