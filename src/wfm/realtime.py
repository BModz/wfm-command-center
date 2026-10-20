"""Real-time analyst (RTA) desk: replay of one day with agent states, alerts and actions.

The replay combines three planning artefacts the way a real operation would:
- the interval forecast produced the evening before,
- the published schedule from the shift optimiser (built two weeks earlier),
- agent-level states at 5-minute granularity, with late logins, extended breaks,
  unscheduled aux, no-shows and planned coaching or offline blocks.

At any "now" the desk shows performance to date, an intraday reforecast for the
rest of the day, alerts, recommended actions with numbers and times, and an
adherence table.

Adherence asks whether an agent was in the scheduled state at the scheduled time.
Conformance compares the total time actually worked on the phones with the time
scheduled, regardless of when. An agent can be 100% conformant and still poorly
adherent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wfm.config import INTERVAL_MINUTES, INTERVAL_SECONDS, INTERVALS_PER_DAY, QUEUES, SL_TARGET
from wfm.erlang import erlang_a_metrics, required_agents
from wfm.labour import EGYPT_LABOUR
from wfm.scheduling import ShiftRules

SLOT_MINUTES = 5
SLOTS_PER_INTERVAL = INTERVAL_MINUTES // SLOT_MINUTES
SLOTS_PER_DAY = INTERVALS_PER_DAY * SLOTS_PER_INTERVAL

OFF, PHONE, BREAK, OFFLINE, AUX = range(5)
STATE_LABELS = {OFF: "Off / logged out", PHONE: "On phone", BREAK: "Break", OFFLINE: "Coaching / offline", AUX: "Unscheduled aux"}
OFFLINE_TYPES = ("Coaching 1:1", "Team huddle", "E-learning", "Quality calibration")

ALERT_SL = 0.70
ALERT_ABANDON = 0.08
ALERT_AHT_DRIFT = 0.10
ALERT_VOLUME = 0.20
ALERT_MIN_CALLS = 30  # drift alerts on smaller intervals are mostly noise
INCIDENT_RATIO = 1.5
IVR_DEFLECTION = 0.30  # share of incident-driven excess calls handled by an IVR outage message plus callback offer
ACTION_WINDOW = 8  # intervals (4 hours) the action engine looks ahead
TEAM_SIZE = 14


@dataclass
class RTADay:
    """Everything needed to replay one day on one queue."""

    date: pd.Timestamp
    queue: str
    intervals: pd.DataFrame
    agents: pd.DataFrame
    scheduled: np.ndarray  # agents x 5-minute slots, scheduled state codes
    actual: np.ndarray  # agents x 5-minute slots, actual state codes


def interval_label(interval: int) -> str:
    """HH:MM label for the start of a 30-minute interval (48 means end of day)."""
    minutes = interval * INTERVAL_MINUTES
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}" if interval < INTERVALS_PER_DAY else "24:00"


# ---------------------------------------------------------------------------
# Building the replay day
# ---------------------------------------------------------------------------


def build_roster(
    week_shifts: pd.DataFrame,
    week_start: pd.Timestamp,
    day: pd.Timestamp,
    in_day_offline: float,
    rng: np.random.Generator,
    rules: ShiftRules | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn the optimised weekly shift counts into named agents with a scheduled state per 5-minute slot.

    Shifts that started the previous evening and run past midnight are included.
    Coaching and offline blocks (30 or 60 minutes, mostly in the afternoon) are
    placed inside phone time until they absorb the planned in-day offline share.
    """
    rules = rules or ShiftRules()
    week_len = 7 * INTERVALS_PER_DAY
    day_offset = (day - week_start).days * INTERVALS_PER_DAY
    rows = []
    for rec in week_shifts.itertuples():
        rel = (rec.start_interval - day_offset + week_len // 2) % week_len - week_len // 2
        if rel < INTERVALS_PER_DAY and rel + rules.length_intervals > 0:
            rows += [(rel, rec.break_offset)] * int(rec.agents)
    rows.sort()
    n = len(rows)
    scheduled = np.full((n, SLOTS_PER_DAY), OFF, dtype=np.int8)
    agents = []
    for i, (rel, brk) in enumerate(rows):
        first = rel * SLOTS_PER_INTERVAL
        span = np.arange(first, first + rules.length_intervals * SLOTS_PER_INTERVAL)
        brk_slots = np.arange((rel + brk) * SLOTS_PER_INTERVAL, (rel + brk + rules.break_intervals) * SLOTS_PER_INTERVAL)
        keep = (span >= 0) & (span < SLOTS_PER_DAY)
        scheduled[i, span[keep]] = PHONE
        keep_b = (brk_slots >= 0) & (brk_slots < SLOTS_PER_DAY)
        scheduled[i, brk_slots[keep_b]] = BREAK
        agents.append(
            {
                "agent_id": f"NC-TS-{i + 1:04d}",
                "team": f"Team {i // TEAM_SIZE + 1:02d}",
                "shift_start": day + pd.Timedelta(minutes=rel * INTERVAL_MINUTES),
                "shift_end": day + pd.Timedelta(minutes=(rel + rules.length_intervals) * INTERVAL_MINUTES),
                "break_start": day + pd.Timedelta(minutes=(rel + brk) * INTERVAL_MINUTES),
                "offline_type": "",
            }
        )
    agents = pd.DataFrame(agents)

    # Offline blocks on 30-minute boundaries inside phone time, weighted towards 13:00-17:30.
    target = in_day_offline * (scheduled == PHONE).sum()
    placed = 0
    interval_weight = np.where((np.arange(INTERVALS_PER_DAY) >= 26) & (np.arange(INTERVALS_PER_DAY) < 35), 4.0, 1.0)
    for i in rng.permutation(n):
        if placed >= target:
            break
        length = int(rng.choice([1, 2]))
        phone_intervals = [
            k
            for k in range(INTERVALS_PER_DAY - length + 1)
            if (scheduled[i, k * SLOTS_PER_INTERVAL : (k + length) * SLOTS_PER_INTERVAL] == PHONE).all()
        ]
        if not phone_intervals:
            continue
        w = interval_weight[phone_intervals]
        k = int(rng.choice(phone_intervals, p=w / w.sum()))
        scheduled[i, k * SLOTS_PER_INTERVAL : (k + length) * SLOTS_PER_INTERVAL] = OFFLINE
        agents.loc[i, "offline_type"] = str(rng.choice(OFFLINE_TYPES))
        placed += length * SLOTS_PER_INTERVAL
    return agents, scheduled


def simulate_agent_states(
    scheduled: np.ndarray, rng: np.random.Generator, incident_slots: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Generate actual agent states from the schedule with realistic adherence exceptions.

    Returns:
        Actual state matrix and a short exception note per agent.
    """
    actual = scheduled.copy()
    notes: list[str] = []
    for i in range(scheduled.shape[0]):
        on_shift = np.flatnonzero(scheduled[i] != OFF)
        note: list[str] = []
        if on_shift.size == 0:
            notes.append("")
            continue
        if rng.random() < 0.04:
            actual[i, on_shift] = OFF
            notes.append("No-show")
            continue
        start, end = on_shift[0], on_shift[-1] + 1
        if start > 0 and rng.random() < 0.15:
            late = int(rng.integers(1, 5))
            actual[i, start : start + late] = OFF
            note.append(f"Late login {late * SLOT_MINUTES} min")
        if end < SLOTS_PER_DAY and rng.random() < 0.05:
            early = int(rng.integers(1, 4))
            actual[i, end - early : end] = OFF
            note.append(f"Early logout {early * SLOT_MINUTES} min")
        breaks = np.flatnonzero(scheduled[i] == BREAK)
        if breaks.size and rng.random() < 0.25:
            # Break taken at the wrong time but full length: hurts adherence, not conformance.
            shift = int(rng.choice([-3, -2, -1, 1, 2, 3]))
            moved = np.clip(breaks + shift, start, end - 1)
            actual[i, breaks] = PHONE
            actual[i, moved] = BREAK
            note.append(f"Break {'early' if shift < 0 else 'late'} {abs(shift) * SLOT_MINUTES} min")
            breaks = np.flatnonzero(actual[i] == BREAK)
        if breaks.size and rng.random() < 0.22:
            extra = int(rng.integers(1, 4))
            after = np.arange(breaks[-1] + 1, min(breaks[-1] + 1 + extra, SLOTS_PER_DAY))
            actual[i, after[actual[i, after] == PHONE]] = BREAK
            note.append(f"Extended break {extra * SLOT_MINUTES} min")
        phone = np.flatnonzero(actual[i] == PHONE)
        if phone.size and rng.random() < 0.35:
            s = int(rng.choice(phone))
            length = int(rng.integers(1, 5))
            block = np.arange(s, min(s + length, SLOTS_PER_DAY))
            actual[i, block[actual[i, block] == PHONE]] = AUX
            note.append(f"Unscheduled aux {length * SLOT_MINUTES} min")
        if incident_slots is not None and rng.random() < 0.10:
            phone_in_incident = np.intersect1d(np.flatnonzero(actual[i] == PHONE), incident_slots)
            if phone_in_incident.size:
                s = int(rng.choice(phone_in_incident))
                actual[i, s : s + 2][actual[i, s : s + 2] == PHONE] = AUX
                note.append("System aux during incident")
        notes.append("; ".join(note))
    return actual, notes


def _per_interval(states: np.ndarray, code: int) -> np.ndarray:
    """Average number of agents in ``code`` per 30-minute interval."""
    counts = (states == code).sum(axis=0)
    return counts.reshape(INTERVALS_PER_DAY, SLOTS_PER_INTERVAL).mean(axis=1)


def build_rta_day(
    history: pd.DataFrame,
    day_forecast: pd.DataFrame,
    week_shifts: pd.DataFrame,
    week_start: pd.Timestamp,
    day: pd.Timestamp,
    queue: str,
    in_day_offline: float,
    seed: int = 7,
) -> RTADay:
    """Assemble the replay day.

    Args:
        history: Interval history (actual offered and AHT for the day are taken from it).
        day_forecast: Interval forecast for the day made the evening before.
        week_shifts: Optimised shift patterns for the week containing ``day``.
        week_start: Sunday the schedule week starts on.
        day: Replay date.
        queue: Queue to replay.
        in_day_offline: Share of phone time planned as coaching or offline.
        seed: Random seed for agent behaviour and service outcomes.
    """
    rng = np.random.default_rng(seed)
    agents, scheduled = build_roster(week_shifts, week_start, day, in_day_offline, rng)
    hist = history[(history["queue"] == queue) & (history["date"] == day)].sort_values("interval")
    fc = day_forecast[(day_forecast["queue"] == queue) & (day_forecast["date"] == day)].sort_values("interval")
    surge = (hist["offered"].to_numpy() >= INCIDENT_RATIO * np.maximum(fc["forecast_offered"].to_numpy(), 1))
    incident_slots = np.flatnonzero(np.repeat(surge, SLOTS_PER_INTERVAL))
    actual, notes = simulate_agent_states(scheduled, rng, incident_slots)
    agents["exception"] = notes

    offered = hist["offered"].to_numpy()
    aht = hist["aht_s"].fillna(fc["forecast_aht_s"].to_numpy().mean()).to_numpy()
    staffed = _per_interval(actual, PHONE)
    service = erlang_a_metrics(offered, aht, np.round(staffed), QUEUES[queue].patience_s)
    abandoned = rng.binomial(offered, np.clip(service["abandon_rate"], 0, 1))
    handled = offered - abandoned
    in_sl = np.minimum(rng.binomial(offered, np.clip(service["service_level"], 0, 1)), handled)
    with np.errstate(divide="ignore", invalid="ignore"):
        intervals = pd.DataFrame(
            {
                "interval": np.arange(INTERVALS_PER_DAY),
                "time": [interval_label(i) for i in range(INTERVALS_PER_DAY)],
                "forecast_offered": fc["forecast_offered"].to_numpy(),
                "forecast_aht_s": fc["forecast_aht_s"].to_numpy(),
                "required_fc": required_agents(fc["forecast_offered"].to_numpy(), fc["forecast_aht_s"].to_numpy()),
                "scheduled_on_phone": _per_interval(scheduled, PHONE),
                "staffed": staffed,
                "offered": offered,
                "handled": handled,
                "abandoned": abandoned,
                "answered_in_sl": in_sl,
                "aht_s": aht,
                "asa_s": np.where(handled > 0, service["asa_s"], 0.0),
                "service_level": np.where(offered > 0, in_sl / offered, np.nan),
                "abandon_rate": np.where(offered > 0, abandoned / offered, np.nan),
            }
        )
    return RTADay(day, queue, intervals, agents, scheduled, actual)


# ---------------------------------------------------------------------------
# Live snapshot at a chosen "now"
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """RTA desk view at the end of interval ``now - 1``."""

    now: int
    kpis: dict[str, float]
    timeline: pd.DataFrame
    alerts: list[dict[str, str]]
    actions: list[dict[str, str]]
    adherence: pd.DataFrame
    incident: bool
    action_effect: dict[str, float] = field(default_factory=dict)


def detect_incident(iv: pd.DataFrame, now: int) -> bool:
    """Possible incident: the last interval ran at least 2x forecast, or the last two at least 1.5x."""
    if now < 1:
        return False
    ratio = iv["offered"].iloc[:now] / iv["forecast_offered"].iloc[:now].clip(lower=1)
    return bool(ratio.iloc[-1] >= 2 * INCIDENT_RATIO / 1.5 or (now >= 2 and (ratio.iloc[-2:] >= INCIDENT_RATIO).all()))


def reforecast(iv: pd.DataFrame, now: int, incident: bool, half_life: float = 2.0, incident_decay: float = 4.0) -> pd.DataFrame:
    """Intraday reforecast of volume and AHT for intervals from ``now`` onwards.

    The actual/forecast ratio to date is weighted towards recent intervals
    (half-life ``half_life`` intervals). Normally the ratio persists for the rest of the
    day. During an incident the surge ratio decays back to the underlying
    (non-incident) ratio with time constant ``incident_decay`` intervals.
    """
    out = iv.copy()
    out["reforecast_offered"] = out["forecast_offered"]
    out["reforecast_aht_s"] = out["forecast_aht_s"]
    if now == 0:
        return out
    done = out.iloc[:now]
    age = (now - 1 - done["interval"]).to_numpy()
    w = 0.5 ** (age / half_life)
    vol_ratio = (w * done["offered"]).sum() / max((w * done["forecast_offered"]).sum(), 1e-9)
    aht_ratio = (w * done["aht_s"] * done["handled"]).sum() / max((w * done["forecast_aht_s"] * done["handled"]).sum(), 1e-9)
    steps = np.arange(1, INTERVALS_PER_DAY - now + 1)
    # Surge intervals are anomalies: they drive the incident path but never the underlying run rate.
    normal = (done["offered"] < INCIDENT_RATIO * done["forecast_offered"].clip(lower=1)).to_numpy()
    wn = w[normal]
    base = (wn * done["offered"].to_numpy()[normal]).sum() / max((wn * done["forecast_offered"].to_numpy()[normal]).sum(), 1e-9) if normal.any() else 1.0
    if incident:
        factor = base + (vol_ratio - base) * np.exp(-steps / incident_decay)
    else:
        factor = np.full(steps.size, base)
    out.loc[now:, "reforecast_offered"] = out["forecast_offered"].iloc[now:].to_numpy() * factor
    out.loc[now:, "reforecast_aht_s"] = out["forecast_aht_s"].iloc[now:].to_numpy() * aht_ratio
    out.attrs["volume_ratio"] = float(vol_ratio if incident else base)
    out.attrs["aht_ratio"] = float(aht_ratio)
    return out


def _kpis(iv: pd.DataFrame, now: int) -> dict[str, float]:
    """Day-to-date KPIs over completed intervals."""
    done = iv.iloc[:now]
    offered, handled = done["offered"].sum(), done["handled"].sum()
    staffed_hours = done["staffed"].sum() * INTERVAL_SECONDS
    return {
        "offered": float(offered),
        "forecast": float(done["forecast_offered"].sum()),
        "volume_var": float(offered / done["forecast_offered"].sum() - 1) if now else 0.0,
        "service_level": float(done["answered_in_sl"].sum() / offered) if offered else 1.0,
        "abandon_rate": float(done["abandoned"].sum() / offered) if offered else 0.0,
        "asa_s": float((done["asa_s"] * done["handled"]).sum() / handled) if handled else 0.0,
        "aht_s": float((done["aht_s"] * done["handled"]).sum() / handled) if handled else 0.0,
        "aht_var": float((done["aht_s"] * done["handled"]).sum() / max((done["forecast_aht_s"] * done["handled"]).sum(), 1e-9) - 1) if handled else 0.0,
        "occupancy": float(min((done["aht_s"] * done["handled"]).sum() / staffed_hours, 1.0)) if staffed_hours else 0.0,
        "staffing_realisation": float(done["staffed"].sum() / max(done["scheduled_on_phone"].sum(), 1e-9)),
    }


def _alerts(iv: pd.DataFrame, now: int, incident: bool, kpis: dict[str, float]) -> list[dict[str, str]]:
    """Threshold alerts on the last completed interval and the day so far."""
    if now == 0:
        return []
    last = iv.iloc[now - 1]
    t = f"{interval_label(now - 1)}-{interval_label(now)}"
    alerts: list[dict[str, str]] = []
    if incident:
        recent = iv.iloc[max(now - 2, 0) : now]
        ratio = recent["offered"].sum() / recent["forecast_offered"].sum()
        aht_gap = (recent["aht_s"] * recent["handled"]).sum() / (recent["forecast_aht_s"] * recent["handled"]).sum() - 1
        alerts.append(
            {
                "severity": "Critical",
                "alert": "Possible incident",
                "detail": f"Volume {ratio:.1f}x forecast for the last {len(recent) * INTERVAL_MINUTES} min with AHT {aht_gap:+.0%} vs forecast "
                "(short 'is there an outage?' calls). Confirm with the NOC and warn AR_BILLING of spillover.",
            }
        )
    if last["service_level"] < ALERT_SL:
        alerts.append(
            {
                "severity": "Critical" if last["service_level"] < 0.5 else "Warning",
                "alert": "Service level below 70%",
                "detail": f"{t}: SL {last['service_level']:.0%} (day to date {kpis['service_level']:.0%}, target 80/20).",
            }
        )
    if last["abandon_rate"] > ALERT_ABANDON:
        alerts.append(
            {
                "severity": "Critical" if last["abandon_rate"] > 2 * ALERT_ABANDON else "Warning",
                "alert": "Abandon rate above 8%",
                "detail": f"{t}: {last['abandon_rate']:.1%} abandoned ({int(last['abandoned'])} calls).",
            }
        )
    aht_drift = last["aht_s"] / last["forecast_aht_s"] - 1
    if abs(aht_drift) > ALERT_AHT_DRIFT and last["handled"] >= ALERT_MIN_CALLS:
        alerts.append(
            {
                "severity": "Warning",
                "alert": "AHT drift above 10%",
                "detail": f"{t}: AHT {last['aht_s']:.0f}s vs forecast {last['forecast_aht_s']:.0f}s ({aht_drift:+.0%}).",
            }
        )
    vol_var = last["offered"] / max(last["forecast_offered"], 1e-9) - 1
    if vol_var > ALERT_VOLUME and last["offered"] >= ALERT_MIN_CALLS:
        alerts.append(
            {
                "severity": "Critical" if vol_var > 0.5 else "Warning",
                "alert": "Volume above forecast",
                "detail": f"{t}: {int(last['offered'])} offered vs {last['forecast_offered']:.0f} forecast ({vol_var:+.0%}).",
            }
        )
    return alerts


def _future_service(offered: np.ndarray, aht: np.ndarray, staff: np.ndarray, patience: float) -> np.ndarray:
    return erlang_a_metrics(offered, aht, np.maximum(np.round(staff), 0), patience)["service_level"]


def _span_label(intervals: np.ndarray) -> str:
    return f"{interval_label(int(intervals.min()))}-{interval_label(int(intervals.max()) + 1)}"


def _actions(day: RTADay, tl: pd.DataFrame, now: int, incident: bool) -> tuple[list[dict[str, str]], pd.DataFrame, dict[str, float]]:
    """Rule-based recommendations for the next four hours, applied in order of cost.

    Order: deflect demand (IVR/callback) -> recover offline time (cancel coaching)
    -> reshape supply (move breaks) -> buy time (overtime). If the projection shows
    surplus instead, offer voluntary time off (VTO).
    """
    rules = ShiftRules()
    end = min(now + ACTION_WINDOW, INTERVALS_PER_DAY)
    window = np.arange(now, end)
    actions: list[dict[str, str]] = []
    if window.size == 0:
        return actions, tl, {}
    patience = QUEUES[day.queue].patience_s
    offered = tl["reforecast_offered"].to_numpy().copy()
    aht = tl["reforecast_aht_s"].to_numpy()
    staff = tl["projected_staffed"].to_numpy().copy()
    req_before = tl["required_reforecast"].to_numpy()

    if incident:
        excess = np.clip(offered[window] - tl["forecast_offered"].to_numpy()[window], 0, None)
        deflected = IVR_DEFLECTION * excess
        if deflected.sum() >= 1:
            offered[window] -= deflected
            req_after = required_agents(offered[window], aht[window])
            saving = req_before[window] - req_after
            actions.append(
                {
                    "action": "Enable IVR outage message and callback",
                    "when": "Now",
                    "detail": f"Est. {deflected.sum():.0f} calls deflected or converted to callbacks over {_span_label(window)}; "
                    f"peak requirement falls by {saving.max():.0f} agents.",
                }
            )
    required = np.zeros(INTERVALS_PER_DAY, dtype=int)
    required[window] = required_agents(offered[window], aht[window])
    present = (day.actual != OFF).any(axis=1)  # excludes no-shows
    sl_target = SL_TARGET

    def gap() -> np.ndarray:
        """Erlang C shortfall, counted only where Erlang A projects SL below target."""
        raw = required[window] - staff[window]
        at_risk = _future_service(offered[window], aht[window], staff[window], patience) < sl_target
        return np.where(at_risk, raw, np.minimum(raw, 0))

    # Cancel coaching / offline in short intervals.
    short = window[gap() > 0.5]
    if short.size:
        slots = np.concatenate([np.arange(k * SLOTS_PER_INTERVAL, (k + 1) * SLOTS_PER_INTERVAL) for k in short])
        offline_agents = np.flatnonzero((day.scheduled[:, slots] == OFFLINE).any(axis=1) & present)
        if offline_agents.size:
            for k in short:
                in_k = (day.scheduled[offline_agents, k * SLOTS_PER_INTERVAL : (k + 1) * SLOTS_PER_INTERVAL] == OFFLINE).mean(axis=1)
                staff[k] += in_k.sum()
            ids = day.agents.loc[offline_agents, "agent_id"].tolist()
            actions.append(
                {
                    "action": f"Cancel coaching/offline for {len(ids)} agents",
                    "when": _span_label(short),
                    "detail": f"Return to phones: {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}. "
                    "Reschedule sessions to a low-volume interval later in the week.",
                }
            )

    # Delay breaks out of short intervals, within the 5-hour continuous-work rule.
    short = window[gap() > 0.5]
    if short.size:
        moved, moves = 0, []
        latest_offset = round(rules.max_continuous_hours * 60 / INTERVAL_MINUTES)
        for i, ag in day.agents.iterrows():
            b = int((ag["break_start"] - day.date) / pd.Timedelta(minutes=INTERVAL_MINUTES))
            s = int((ag["shift_start"] - day.date) / pd.Timedelta(minutes=INTERVAL_MINUTES))
            if b not in short or not present[i]:
                continue
            candidates = [k for k in range(b + 1, s + latest_offset + 1) if k + rules.break_intervals <= INTERVALS_PER_DAY]
            target = next((k for k in candidates if all(k + j not in short for j in range(rules.break_intervals))), None)
            if target is None:
                continue
            for j in range(rules.break_intervals):
                if b + j < INTERVALS_PER_DAY:
                    staff[b + j] += 1
                staff[target + j] -= 1
            moved += 1
            moves.append((b, target))
        if moved:
            frm = np.array([m[0] for m in moves])
            to = np.array([m[1] for m in moves])
            actions.append(
                {
                    "action": f"Delay breaks for {moved} agents",
                    "when": _span_label(frm),
                    "detail": f"Move breaks from {_span_label(frm)} to {_span_label(to)}; every break still starts within "
                    f"{rules.max_continuous_hours:.0f}h of shift start (labour-law limit on continuous work).",
                }
            )

    # Overtime for the residual gap (max 2 hours per agent per day).
    residual = gap()
    short = window[residual > 0.5]
    if short.size:
        n_ot = int(math.ceil(residual.max()))
        hours = short.size * INTERVAL_MINUTES / 60
        ot_hours = min(hours, EGYPT_LABOUR.max_daily_overtime_hours)
        end_iv = int(short.max()) + 1
        ending = day.agents[
            (day.agents["shift_end"] > day.date + pd.Timedelta(minutes=int(short.min()) * INTERVAL_MINUTES - 120))
            & (day.agents["shift_end"] <= day.date + pd.Timedelta(minutes=end_iv * INTERVAL_MINUTES))
        ]
        night = any(
            (k * INTERVAL_MINUTES // 60) >= EGYPT_LABOUR.night_start_hour or (k * INTERVAL_MINUTES // 60) < EGYPT_LABOUR.night_end_hour
            for k in short
        )
        premium = EGYPT_LABOUR.overtime_premium_night if night else EGYPT_LABOUR.overtime_premium_day
        staff[short] += n_ot * min(1.0, ot_hours / hours)
        actions.append(
            {
                "action": f"Offer overtime to {n_ot} agents",
                "when": _span_label(short),
                "detail": f"Up to {ot_hours:.1f}h each (legal cap {EGYPT_LABOUR.max_daily_overtime_hours:.0f}h/day), "
                f"{len(ending)} agents finish their shift in or just before this window and can extend. "
                f"Premium +{premium:.0%}; about {n_ot * ot_hours:.0f} OT hours."
                + (" Gap is longer than 2h: split across two OT groups." if hours > ot_hours else ""),
            }
        )

    # Voluntary time off when the projection is comfortably overstaffed for 2h+.
    surplus = staff[window] - required[window]
    threshold = np.maximum(2, 0.12 * required[window])
    over = window[surplus >= threshold]
    if not actions and over.size >= 4:
        runs = np.split(over, np.flatnonzero(np.diff(over) > 1) + 1)
        best = max(runs, key=len)
        if best.size >= 4:
            n_vto = int(np.floor((surplus[np.isin(window, best)] - 1).min()))
            if n_vto >= 2:
                actions.append(
                    {
                        "action": f"Offer VTO to {n_vto} agents",
                        "when": _span_label(best),
                        "detail": f"Projected surplus of {n_vto}+ agents over required for {best.size * INTERVAL_MINUTES // 60}h "
                        "while keeping 80/20; release in order of VTO list, prioritise agents already over contract hours.",
                    }
                )
                staff[best] -= n_vto

    tl = tl.copy()
    tl["staffed_after_actions"] = np.where(tl["interval"] >= now, staff, np.nan)
    tl["offered_after_actions"] = np.where(tl["interval"] >= now, offered, np.nan)
    sl_before = _future_service(tl["reforecast_offered"].to_numpy()[window], aht[window], tl["projected_staffed"].to_numpy()[window], patience)
    sl_after = _future_service(offered[window], aht[window], staff[window], patience)
    tl["projected_sl_before"] = np.nan
    tl["projected_sl_after"] = np.nan
    tl.loc[window, "projected_sl_before"] = sl_before
    tl.loc[window, "projected_sl_after"] = sl_after
    weights = tl["reforecast_offered"].to_numpy()[window]
    effect = {
        "sl_before": float((sl_before * weights).sum() / weights.sum()) if weights.sum() else 1.0,
        "sl_after": float((sl_after * offered[window]).sum() / offered[window].sum()) if offered[window].sum() else 1.0,
    }
    return actions, tl, effect


def adherence_table(day: RTADay, now: int) -> pd.DataFrame:
    """Per-agent adherence and conformance to date, plus current state and exception duration."""
    slot = now * SLOTS_PER_INTERVAL
    if slot == 0:
        return pd.DataFrame()
    sched, act = day.scheduled[:, :slot], day.actual[:, :slot]
    on_shift = sched != OFF
    started = on_shift.any(axis=1)
    match = (sched == act) & on_shift
    adherence = match.sum(axis=1) / np.maximum(on_shift.sum(axis=1), 1)
    conformance = (act == PHONE).sum(axis=1) / np.maximum((sched == PHONE).sum(axis=1), 1)
    now_s, now_a = day.scheduled[:, slot - 1], day.actual[:, slot - 1]
    out_now = (now_s != now_a) & (now_s != OFF)
    current = np.zeros(len(day.agents), dtype=int)
    for i in np.flatnonzero(out_now):
        mism = (sched[i] != act[i])[::-1]
        current[i] = int(np.argmax(~mism)) if (~mism).any() else mism.size
    table = pd.DataFrame(
        {
            "agent_id": day.agents["agent_id"],
            "team": day.agents["team"],
            "shift": day.agents["shift_start"].dt.strftime("%H:%M")
            + "-"
            + day.agents["shift_end"].dt.strftime("%H:%M")
            + np.where(day.agents["shift_start"] < day.date, " (prev day)", ""),
            "scheduled_now": [STATE_LABELS[s] for s in now_s],
            "actual_now": [STATE_LABELS[a] for a in now_a],
            "on_shift_now": now_s != OFF,
            "out_of_adherence": out_now,
            "minutes_out_now": current * SLOT_MINUTES,
            "adherence": adherence,
            "conformance": conformance,
            "minutes_out_today": ((sched != act) & on_shift).sum(axis=1) * SLOT_MINUTES,
            "exception": day.agents["exception"],
        }
    )
    table = table[started].sort_values(
        ["out_of_adherence", "minutes_out_now", "on_shift_now", "adherence"], ascending=[False, False, False, True]
    )
    return table.reset_index(drop=True)


def snapshot(day: RTADay, now: int) -> Snapshot:
    """Compute the full RTA view after ``now`` completed intervals (0-48)."""
    iv = day.intervals
    incident = detect_incident(iv, now)
    tl = reforecast(iv, now, incident)
    recent = iv.iloc[max(now - 4, 0) : now]
    realisation = float(np.clip(recent["staffed"].sum() / max(recent["scheduled_on_phone"].sum(), 1e-9), 0.7, 1.05)) if now else 0.93
    tl["projected_staffed"] = np.where(tl["interval"] >= now, tl["scheduled_on_phone"] * realisation, tl["staffed"])
    tl["required_reforecast"] = required_agents(tl["reforecast_offered"].to_numpy(), tl["reforecast_aht_s"].to_numpy())
    tl["required_actual"] = np.where(
        tl["interval"] < now, required_agents(tl["offered"].to_numpy(), tl["aht_s"].to_numpy()), np.nan
    )
    kpis = _kpis(iv, now)
    adherence = adherence_table(day, now)
    if len(adherence):
        kpis["adherence"] = float(adherence["adherence"].mean())
        kpis["conformance"] = float(adherence["conformance"].mean())
        kpis["out_of_adherence_now"] = float(adherence["out_of_adherence"].sum())
        kpis["agents_on_shift_now"] = float((day.scheduled[:, now * SLOTS_PER_INTERVAL - 1] != OFF).sum())
    kpis["realisation_used"] = realisation
    actions, tl, effect = _actions(day, tl, now, incident)
    return Snapshot(now, kpis, tl, _alerts(iv, now, incident, kpis), actions, adherence, incident, effect)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_rta_day(day: RTADay, directory) -> None:
    """Write the replay day to parquet files in ``directory``."""
    day.intervals.assign(date=day.date, queue=day.queue).to_parquet(directory / "rta_intervals.parquet", index=False)
    day.agents.to_parquet(directory / "rta_agents.parquet", index=False)
    cols = [f"s{k:03d}" for k in range(SLOTS_PER_DAY)]
    pd.DataFrame(day.scheduled, columns=cols).to_parquet(directory / "rta_scheduled.parquet", index=False)
    pd.DataFrame(day.actual, columns=cols).to_parquet(directory / "rta_actual.parquet", index=False)


def load_rta_day(directory) -> RTADay:
    """Load a replay day written by :func:`save_rta_day`."""
    iv = pd.read_parquet(directory / "rta_intervals.parquet")
    return RTADay(
        date=pd.Timestamp(iv["date"].iloc[0]),
        queue=str(iv["queue"].iloc[0]),
        intervals=iv.drop(columns=["date", "queue"]),
        agents=pd.read_parquet(directory / "rta_agents.parquet"),
        scheduled=pd.read_parquet(directory / "rta_scheduled.parquet").to_numpy(dtype=np.int8),
        actual=pd.read_parquet(directory / "rta_actual.parquet").to_numpy(dtype=np.int8),
    )
