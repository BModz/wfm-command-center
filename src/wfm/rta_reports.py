"""RTA deliverables: agent flagging, Staffing Outlook, midday reforecast and incident RCA.

These are the documents a real-time analyst produces every day:

- **Agent flags**: who is out of adherence, how long, and what to do about it,
  following an escalation ladder (nudge the agent, then the team leader, then
  the operations manager; persistent offenders go to coaching).
- **Staffing Outlook** (start of shift): today's and tomorrow's projected
  over/under by interval from the forecast and the published schedule.
- **Midday reforecast**: plan vs actual so far, a revised forecast for the rest
  of the day from the run rate, and the revised over/under.
- **RCA**: timeline, impact, root cause, contributing factors and preventive
  actions for an incident day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfm.config import INTERVAL_MINUTES, QUEUES, SL_TARGET
from wfm.erlang import erlang_a_metrics, required_agents
from wfm.realtime import (
    AUX,
    BREAK,
    INCIDENT_RATIO,
    OFF,
    OFFLINE,
    PHONE,
    SLOT_MINUTES,
    SLOTS_PER_INTERVAL,
    STATE_LABELS,
    RTADay,
    detect_incident,
    interval_label,
    snapshot,
)

# ---------------------------------------------------------------------------
# Agent flagging
# ---------------------------------------------------------------------------

LEVELS = {
    "manager": ("Escalate to operations manager", "Radio / floor walk", 1),
    "team_leader": ("Escalate to team leader", "Extension call", 2),
    "agent": ("Nudge agent", "Chat", 3),
    "coaching": ("Coaching follow-up", "End-of-shift note to team leader", 4),
    "monitor": ("Monitor", "None yet", 5),
}


@dataclass(frozen=True)
class FlagRules:
    """Escalation thresholds, in minutes out of adherence (site policy, editable on the desk)."""

    nudge_after_min: int = 5
    team_leader_after_min: int = 15
    manager_after_min: int = 30
    coaching_below_adherence: float = 0.85
    coaching_min_scheduled_min: int = 120  # judge adherence only after two scheduled hours


def _issue(scheduled: int, actual: int, logged_in_today: bool) -> tuple[str, bool]:
    """Describe a scheduled-vs-actual mismatch and whether it costs the queue capacity."""
    if scheduled == PHONE:
        if actual == OFF:
            return ("No-show / not logged in" if not logged_in_today else "Logged out during shift"), True
        if actual == BREAK:
            return "Break overrun or taken early", True
        if actual == AUX:
            return "Unscheduled aux", True
        if actual == OFFLINE:
            return "Unplanned offline activity", True
    if scheduled == BREAK and actual == PHONE:
        return "Working through scheduled break", False
    if scheduled == OFFLINE and actual == PHONE:
        return "On phones instead of scheduled coaching", False
    return f"{STATE_LABELS[actual]} instead of {STATE_LABELS[scheduled]}", True


def flag_agents(day: RTADay, now: int, rules: FlagRules | None = None) -> pd.DataFrame:
    """Flags for every agent who needs attention at the end of interval ``now - 1``.

    Agents currently out of adherence are escalated by how long they have been
    out; mismatches that add capacity (working through a break) are only
    monitored. Agents back in adherence but with low adherence today get a
    coaching follow-up. The message column is ready to paste into chat.
    """
    rules = rules or FlagRules()
    slot = now * SLOTS_PER_INTERVAL
    if slot == 0:
        return pd.DataFrame()
    sched, act = day.scheduled[:, :slot], day.actual[:, :slot]
    iv = day.intervals.iloc[now - 1]
    queue_note = (
        f" The queue is at {iv['service_level']:.0%} service level with {iv['abandon_rate']:.0%} abandoning."
        if iv["service_level"] < SL_TARGET
        else ""
    )
    now_time = day.date + pd.Timedelta(minutes=now * INTERVAL_MINUTES)
    rows = []
    for i, agent in day.agents.iterrows():
        on_shift = sched[i] != OFF
        if not on_shift.any():
            continue
        s_now, a_now = int(sched[i, -1]), int(act[i, -1])
        mismatch = (sched[i] != act[i]) & on_shift
        adherence = 1 - mismatch.sum() / on_shift.sum()
        team_leader = f"TL {agent['team']}"
        if s_now != OFF and s_now != a_now:
            run = int(np.argmax(~mismatch[::-1])) if (~mismatch).any() else mismatch.size
            minutes = run * SLOT_MINUTES
            since = now_time - pd.Timedelta(minutes=minutes)
            issue, costs_capacity = _issue(s_now, a_now, bool((act[i] != OFF).any()))
            if not costs_capacity:
                level = "monitor"
            elif issue.startswith("No-show") or minutes >= rules.manager_after_min:
                level = "manager"
            elif minutes >= rules.team_leader_after_min:
                level = "team_leader"
            elif minutes >= rules.nudge_after_min:
                level = "agent"
            else:
                continue  # too short to act on yet
            nudged = since + pd.Timedelta(minutes=rules.nudge_after_min)
            message = {
                "agent": f"Hi {agent['agent_id']}, WFM here. You've been in {STATE_LABELS[a_now].lower()} since {since:%H:%M} "
                f"but you're scheduled {STATE_LABELS[s_now].lower()}.{queue_note} Please go Available now. Thanks!",
                "team_leader": f"{team_leader}: {agent['agent_id']} has been out of adherence for {minutes} min "
                f"({STATE_LABELS[a_now].lower()} vs scheduled {STATE_LABELS[s_now].lower()}), nudged by chat at {nudged:%H:%M}. "
                "Please check in with them.",
                "manager": f"Escalation: {agent['agent_id']} ({agent['team']}) - {issue.lower()} for {minutes} min. "
                f"{team_leader} informed. Logged for the attendance and RCA reports.",
                "monitor": f"{agent['agent_id']} is {STATE_LABELS[a_now].lower()} instead of {STATE_LABELS[s_now].lower()}; "
                "this adds capacity, so no action. Make sure the break is taken before the 5-hour limit.",
            }[level]
            rows.append((agent, level, issue, s_now, a_now, since, minutes, adherence, message))
        elif on_shift.sum() * SLOT_MINUTES >= rules.coaching_min_scheduled_min and adherence < rules.coaching_below_adherence:
            out_min = int(mismatch.sum()) * SLOT_MINUTES
            message = (
                f"{agent['agent_id']} is at {adherence:.0%} adherence today ({out_min} min out of adherence: "
                f"{agent['exception'] or 'several short exceptions'}). Suggest an adherence conversation at end of shift."
            )
            rows.append((agent, "coaching", "Low adherence today", s_now, a_now, None, out_min, adherence, message))
    if not rows:
        return pd.DataFrame(columns=["priority", "agent_id", "team", "level", "channel", "issue", "since", "minutes_out", "message"])
    out = pd.DataFrame(
        [
            {
                "priority": LEVELS[level][2],
                "agent_id": agent["agent_id"],
                "team": agent["team"],
                "level": LEVELS[level][0],
                "channel": LEVELS[level][1],
                "issue": issue,
                "scheduled_now": STATE_LABELS[s],
                "actual_now": STATE_LABELS[a],
                "since": since.strftime("%H:%M") if since is not None else "",
                "minutes_out": minutes,
                "adherence_today": adh,
                "message": message,
            }
            for agent, level, issue, s, a, since, minutes, adh, message in rows
        ]
    )
    return out.sort_values(["priority", "minutes_out"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Staffing Outlook (start of shift, today and tomorrow)
# ---------------------------------------------------------------------------

SHORT_AGENTS = 1.0  # an interval counts as short when at least one full agent is missing


def staffing_outlook(plan: pd.DataFrame, queue: str, expected_loss: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Projected over/under per interval and per day from the forecast and the published schedule.

    Args:
        plan: One row per interval with date, interval, forecast_offered, forecast_aht_s, scheduled_on_phone.
        queue: Queue code (sets caller patience for the projected service level).
        expected_loss: Share of scheduled phone time expected to be lost in-day (late logins, aux, absence).

    Returns:
        Interval table and a one-row-per-day summary.
    """
    out = plan.copy()
    out["required"] = required_agents(out["forecast_offered"].to_numpy(), out["forecast_aht_s"].to_numpy())
    out["projected_staffed"] = out["scheduled_on_phone"] * (1 - expected_loss)
    out["over_under"] = out["projected_staffed"] - out["required"]
    service = erlang_a_metrics(
        out["forecast_offered"].to_numpy(), out["forecast_aht_s"].to_numpy(), np.round(out["projected_staffed"]), QUEUES[queue].patience_s
    )
    out["projected_sl"] = service["service_level"]
    out["time"] = [interval_label(i) for i in out["interval"]]
    hours = INTERVAL_MINUTES / 60
    rows = []
    for date, g in out.groupby("date"):
        worst = g.loc[g["over_under"].idxmin()]
        rows.append(
            {
                "date": date,
                "forecast_calls": g["forecast_offered"].sum(),
                "required_hours": g["required"].sum() * hours,
                "projected_hours": g["projected_staffed"].sum() * hours,
                "net_over_under_hours": g["over_under"].sum() * hours,
                "short_intervals": int((g["over_under"] < -SHORT_AGENTS).sum()),
                "worst_interval": f"{worst['time']} ({worst['over_under']:+.1f} agents)",
                "projected_sl": float((g["projected_sl"] * g["forecast_offered"]).sum() / g["forecast_offered"].sum()),
            }
        )
    return out, pd.DataFrame(rows)


def outlook_message(summary: pd.DataFrame, intervals: pd.DataFrame, queue: str) -> str:
    """Start-of-shift Staffing Outlook email body."""
    lines = [f"Subject: Staffing Outlook - {QUEUES[queue].name} - {pd.Timestamp(summary['date'].iloc[0]):%a %d %b}", "", "Hi all,", ""]
    lines.append("Projected staffing against the forecast and the published schedule (after expected in-day losses):")
    lines.append("")
    for rec in summary.itertuples():
        g = intervals[intervals["date"] == rec.date]
        short = g[g["over_under"] < -SHORT_AGENTS]
        lines.append(
            f"- {pd.Timestamp(rec.date):%a %d %b}: {rec.forecast_calls:,.0f} calls forecast, projected SL {rec.projected_sl:.0%}, "
            f"net {rec.net_over_under_hours:+.0f} agent-hours. "
            + (f"{len(short)} short intervals, worst {rec.worst_interval}." if len(short) else "No short intervals.")
        )
    first = intervals[intervals["date"] == summary["date"].iloc[0]]
    gaps = first[first["over_under"] < -SHORT_AGENTS]
    lines += ["", "Watch-outs today:" if len(gaps) else "No staffing risks flagged for today."]
    for rec in gaps.nsmallest(4, "over_under").sort_values("interval").itertuples():
        lines.append(f"- {rec.time}: {rec.over_under:+.1f} agents vs requirement; hold offline activities and consider OT.")
    lines += ["", "I'll share the midday reforecast at 12:00.", "", "Thanks,", "RTA / WFM"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Midday reforecast
# ---------------------------------------------------------------------------


def midday_reforecast(day: RTADay, now: int) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Plan vs actual so far and the revised rest-of-day view at ``now``.

    Returns:
        Summary figures and an interval table (actual so far, reforecast and revised over/under after).
    """
    snap = snapshot(day, now)
    tl = snap.timeline
    done, rest = tl[tl["interval"] < now], tl[tl["interval"] >= now]
    revised = rest["projected_staffed"] - rest["required_reforecast"]
    worst = rest.loc[revised.idxmin()] if len(rest) else None
    summary: dict[str, float | str] = {
        "now": interval_label(now),
        "offered_so_far": float(done["offered"].sum()),
        "forecast_so_far": float(done["forecast_offered"].sum()),
        "volume_variance": float(done["offered"].sum() / max(done["forecast_offered"].sum(), 1) - 1),
        "sl_so_far": snap.kpis["service_level"],
        "abandon_so_far": snap.kpis["abandon_rate"],
        "aht_variance": snap.kpis["aht_var"],
        "original_rest_of_day": float(rest["forecast_offered"].sum()),
        "reforecast_rest_of_day": float(rest["reforecast_offered"].sum()),
        "revised_net_hours": float(revised.sum() * INTERVAL_MINUTES / 60),
        "short_intervals": int((revised < -SHORT_AGENTS).sum()),
        "worst_interval": f"{interval_label(int(worst['interval']))} ({revised.min():+.1f} agents)" if worst is not None else "",
        "projected_sl_next_4h": snap.action_effect.get("sl_before", float("nan")),
    }
    table = tl[["interval", "forecast_offered", "offered", "reforecast_offered", "required_reforecast", "projected_staffed"]].copy()
    table["time"] = [interval_label(i) for i in table["interval"]]
    table["revised_over_under"] = np.where(table["interval"] >= now, table["projected_staffed"] - table["required_reforecast"], np.nan)
    table["offered"] = np.where(table["interval"] < now, table["offered"], np.nan)
    table["reforecast_offered"] = np.where(table["interval"] >= now, table["reforecast_offered"], np.nan)
    return summary, table


def midday_message(summary: dict[str, float | str], queue: str, actions: list[dict[str, str]]) -> str:
    """Midday reforecast email body."""
    lines = [
        f"Subject: Midday reforecast - {QUEUES[queue].name} - as of {summary['now']}",
        "",
        "Hi all,",
        "",
        f"Plan vs actual to {summary['now']}: {summary['offered_so_far']:,.0f} calls offered vs {summary['forecast_so_far']:,.0f} "
        f"forecast ({summary['volume_variance']:+.1%}); SL {summary['sl_so_far']:.1%}, abandon {summary['abandon_so_far']:.1%}, "
        f"AHT {summary['aht_variance']:+.1%} vs forecast.",
        f"Rest of day: reforecast {summary['reforecast_rest_of_day']:,.0f} calls (original {summary['original_rest_of_day']:,.0f}). "
        f"Revised over/under {summary['revised_net_hours']:+.0f} agent-hours with {summary['short_intervals']} short intervals"
        + (f", worst {summary['worst_interval']}." if summary["short_intervals"] else "."),
    ]
    if actions:
        lines += ["", "Actions in progress:"] + [f"- {a['action']} ({a['when']})" for a in actions]
    lines += ["", "Next update at end of day.", "", "Thanks,", "RTA / WFM"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Incident RCA
# ---------------------------------------------------------------------------


def _incident_window(iv: pd.DataFrame) -> tuple[int, int]:
    """First and last interval of the longest run at or above the incident ratio."""
    surge = (iv["offered"] >= INCIDENT_RATIO * iv["forecast_offered"].clip(lower=1)).to_numpy()
    best, start = (0, -1), None
    for i, flag in enumerate(np.r_[surge, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    if best[1] < 0:
        raise ValueError("No incident found on this day")
    return best


def rca_report(day: RTADay) -> dict[str, object]:
    """Root-cause analysis of the incident on the replay day, computed from the data."""
    iv = day.intervals
    first, last = _incident_window(iv)
    window = iv.iloc[first : last + 1]
    outside = iv.drop(window.index)
    detected = next(n for n in range(1, len(iv) + 1) if detect_incident(iv, n))
    peak = window.loc[(window["offered"] / window["forecast_offered"]).idxmax()]
    recovered = next((int(i) for i in iv.index[last + 1 :] if iv.loc[i, "service_level"] >= SL_TARGET), None)
    sl_day = iv["answered_in_sl"].sum() / iv["offered"].sum()
    sl_outside = outside["answered_in_sl"].sum() / outside["offered"].sum()
    excess = window["offered"].sum() - window["forecast_offered"].sum()
    base_abandon = outside["abandoned"].sum() / outside["offered"].sum()
    extra_abandons = window["abandoned"].sum() - base_abandon * window["offered"].sum()
    aht_drop = (window["aht_s"] * window["handled"]).sum() / (window["forecast_aht_s"] * window["handled"]).sum() - 1
    slots = slice(first * SLOTS_PER_INTERVAL, (last + 1) * SLOTS_PER_INTERVAL)
    sched_w, act_w = day.scheduled[:, slots], day.actual[:, slots]
    realisation = (act_w == PHONE).sum() / max((sched_w == PHONE).sum(), 1)
    coaching_agents = int((sched_w == OFFLINE).any(axis=1).sum())
    break_hours = (sched_w == BREAK).sum() * SLOT_MINUTES / 60
    aux_hours = ((act_w == AUX) & (sched_w == PHONE)).sum() * SLOT_MINUTES / 60
    actions_at = detected + 2
    actions = snapshot(day, actions_at)
    start_t, end_t = interval_label(first), interval_label(last + 1)
    return {
        "ends_at_interval": last + 1,
        "title": f"RCA - {QUEUES[day.queue].name} service failure, {day.date:%a %d %b %Y}",
        "summary": (
            f"A network outage drove AR_TECH volume to {peak['offered'] / peak['forecast_offered']:.1f}x forecast between "
            f"{start_t} and {end_t}. Service level fell to {window['answered_in_sl'].sum() / window['offered'].sum():.0%} in that "
            f"window and the day closed at {sl_day:.1%} against an {SL_TARGET:.0%} target ({sl_outside:.1%} outside the incident)."
        ),
        "timeline": sorted([
            (start_t, f"Volume jumps to {iv.loc[first, 'offered']:.0f} calls vs {iv.loc[first, 'forecast_offered']:.0f} forecast."),
            (interval_label(detected), "RTA desk flags a possible incident (volume and short AHT pattern); NOC contacted."),
            (interval_label(int(peak["interval"])), f"Peak volume: {peak['offered']:.0f} calls offered, {peak['abandon_rate']:.0%} abandoned."),
            (end_t, "Volume back below 1.5x forecast."),
            (interval_label(recovered) if recovered is not None else "24:00", "Service level back above target." if recovered is not None else "SL did not recover before end of day."),
        ], key=lambda event: event[0]),
        "impact": [
            ("Excess calls vs forecast", f"{excess:,.0f}"),
            ("Calls abandoned in the window", f"{window['abandoned'].sum():,.0f} ({extra_abandons:,.0f} above the normal rate)"),
            ("Service level in the window", f"{window['answered_in_sl'].sum() / window['offered'].sum():.1%}"),
            ("Day service level", f"{sl_day:.1%} (vs {sl_outside:.1%} outside the window)"),
            ("Peak ASA", f"{window['asa_s'].max():.0f}s"),
        ],
        "root_cause": (
            f"External: a network outage generated short 'is there an outage?' calls (AHT {aht_drop:+.0%} vs forecast). "
            "The interval forecast could not anticipate it, and no outage IVR message was active when it started, so "
            "every affected customer reached the queue."
        ),
        "contributing": [
            f"Staffing realisation in the window was {realisation:.0%} of scheduled phone time (late logins, aux, extended breaks); "
            f"{aux_hours:.1f} agent-hours of unscheduled aux.",
            f"{coaching_agents} agents had coaching or offline sessions scheduled inside the window.",
            f"{break_hours:.0f} agent-hours of breaks were scheduled inside the window.",
        ],
        "actions_at": interval_label(actions_at),
        "actions_taken": [f"{a['action']} ({a['when']})" for a in actions.actions],
        "preventive": [
            "Agree an outage trigger with the NOC so WFM is alerted before volume arrives; pre-approve the IVR outage message.",
            "Keep a standing overtime list for incident days and an incident playbook with break and coaching rules.",
            f"Keep coaching out of the {start_t}-{end_t} high-risk window on incident-prone days; move it to the morning trough.",
            "Watch AR_BILLING for spillover during AR_TECH incidents and cross-skill a share of billing agents.",
        ],
    }


def rca_markdown(rca: dict[str, object]) -> str:
    """Render an RCA as a Markdown document."""
    lines = [f"# {rca['title']}", "", "## Summary", "", str(rca["summary"]), "", "## Timeline", "", "| Time | Event |", "|---|---|"]
    lines += [f"| {t} | {e} |" for t, e in rca["timeline"]]
    lines += ["", "## Impact", "", "| Measure | Value |", "|---|---|"]
    lines += [f"| {m} | {v} |" for m, v in rca["impact"]]
    lines += ["", "## Root cause", "", str(rca["root_cause"]), "", "## Contributing factors", ""]
    lines += [f"- {c}" for c in rca["contributing"]]
    lines += ["", f"## Actions recommended by the RTA desk at {rca['actions_at']}", ""] + [f"- {a}" for a in rca["actions_taken"]]
    lines += ["", "## Preventive actions", ""] + [f"- {p}" for p in rca["preventive"]]
    return "\n".join(lines) + "\n"
