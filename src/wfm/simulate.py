"""Synthetic interval-level history for the fictional NileConnect contact centre.

Real interval-level contact-centre data is not publicly available, so this module
generates a realistic two-year history. Volume is built top-down:

    daily volume = base x growth x weekday x seasonality x billing cycle
                   x campaign x holiday / Ramadan effect x day-level noise
    interval volume ~ Poisson(daily volume x intraday profile x outage multiplier)

The centre's own historical planning is simulated too (forecast with error,
Erlang C requirement, schedule fit, absenteeism), and realised service is
computed with Erlang A so overloaded intervals produce realistic abandons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wfm.config import (
    DATA_DIR,
    DEFAULT_SEED,
    FORECAST_END,
    HISTORY_END,
    HISTORY_START,
    INTERVAL_MINUTES,
    INTERVAL_SECONDS,
    INTERVALS_PER_DAY,
    QUEUE_CODES,
    QUEUES,
    RTA_OUTAGE_HOURS,
    RTA_OUTAGE_PEAK,
    RTA_OUTAGE_START,
    RTA_QUEUE,
    RTA_REPLAY_DATE,
    open_mask,
)
from wfm.egypt_calendar import build_calendar, profile_day_type
from wfm.erlang import erlang_a_metrics, required_agents

HOURS = np.arange(INTERVALS_PER_DAY) * INTERVAL_MINUTES / 60 + INTERVAL_MINUTES / 120  # interval mid-points

# Weekday factors indexed by pandas dayofweek (Mon=0 ... Sun=6); Sunday is the first working day.
DOW_FACTORS = {
    "default": np.array([1.06, 1.03, 1.01, 0.96, 0.70, 0.86, 1.13]),
    "SALES": np.array([1.05, 1.04, 1.02, 1.00, 0.62, 0.90, 1.10]),
}
BILL_CYCLE = {  # uplift on days 1-5 of the month
    "AR_BILLING": [1.38, 1.30, 1.20, 1.12, 1.06],
    "AR_TECH": [1.04, 1.03, 1.02, 1.01, 1.00],
    "EN_SUPPORT": [1.14, 1.10, 1.07, 1.04, 1.02],
    "SALES": [1.0] * 5,
}
HOLIDAY_FACTOR = {"AR_BILLING": 0.58, "AR_TECH": 0.74, "EN_SUPPORT": 0.68, "SALES": 0.45}
EID_FACTOR = {"AR_BILLING": 0.45, "AR_TECH": 0.62, "EN_SUPPORT": 0.55, "SALES": 0.30}
POST_EID_FACTOR = {"AR_BILLING": 1.24, "AR_TECH": 1.14, "EN_SUPPORT": 1.12, "SALES": 1.10}
RAMADAN_FACTOR = {"AR_BILLING": 0.93, "AR_TECH": 1.03, "EN_SUPPORT": 0.96, "SALES": 0.86}
CAMPAIGN_UPLIFT = (1.25, 1.45)
PLANNED_SALES_CAMPAIGNS_PER_YEAR = 7

# Intraday shapes as sums of circular Gaussian bumps: (centre hour, width hours, height).
NORMAL_BUMPS: dict[str, list[tuple[float, float, float]]] = {
    "AR_BILLING": [(11.5, 2.4, 1.0), (17.0, 2.6, 0.85), (21.0, 1.6, 0.35)],
    "AR_TECH": [(12.0, 2.8, 0.75), (17.0, 2.0, 0.55), (20.8, 2.2, 0.95)],
    "EN_SUPPORT": [(10.5, 2.2, 1.0), (15.0, 2.5, 0.8), (20.0, 2.0, 0.35)],
    "SALES": [(12.8, 2.4, 1.0), (19.5, 2.0, 0.95)],
}
RAMADAN_BUMPS: dict[str, list[tuple[float, float, float]]] = {
    "AR_BILLING": [(13.5, 2.2, 0.9), (15.8, 1.4, 0.5), (22.5, 1.7, 1.2)],
    "AR_TECH": [(13.8, 2.2, 0.7), (15.8, 1.4, 0.45), (22.7, 1.8, 1.35)],
    "EN_SUPPORT": [(12.5, 2.2, 0.9), (15.5, 1.6, 0.6), (22.0, 1.8, 0.8)],
    "SALES": [(13.8, 2.0, 0.9), (21.8, 1.3, 1.25)],
}
NIGHT_FLOOR = {"AR_BILLING": 0.04, "AR_TECH": 0.08, "EN_SUPPORT": 0.07, "SALES": 0.0}

ABSENCE_BASE = 0.045
# In-day absence allowance the historical planners added on top of the Erlang requirement.
PLANNED_ABSENCE_ALLOWANCE = 0.04


def _bump(hours: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Gaussian bump on a 24-hour circle so late-night peaks wrap past midnight."""
    d = np.abs(hours - centre)
    d = np.minimum(d, 24 - d)
    return np.exp(-0.5 * (d / width) ** 2)


def intraday_profile(queue: str, day_type: str, ramadan: bool) -> np.ndarray:
    """Share of a day's calls landing in each 30-minute interval (sums to 1).

    Fridays shift the morning later and dip around Friday prayer (~12:00-13:30).
    Saturdays start later. Ramadan has a later morning peak, a sharp dip around
    iftar (~17:30-19:30) and a strong late-night peak from 21:00 to 01:00.
    """
    bumps = RAMADAN_BUMPS[queue] if ramadan else NORMAL_BUMPS[queue]
    shift = {"weekday": 0.0, "friday": 1.5, "saturday": 0.7}[day_type]
    shape = np.full(INTERVALS_PER_DAY, NIGHT_FLOOR[queue])
    for centre, width, height in bumps:
        moved = centre + shift if centre < 16 else centre
        shape += height * _bump(HOURS, moved, width)
    if day_type == "friday":
        shape *= 1 - 0.35 * _bump(HOURS, 12.7, 0.6)
    if ramadan:
        shape *= 1 - 0.62 * _bump(HOURS, 18.4, 0.7)  # iftar
        shape *= 1 - 0.30 * _bump(HOURS, 3.5, 1.2)  # suhoor and sleep
    shape = shape * open_mask(queue)
    return shape / shape.sum()


def aht_time_of_day_factor(queue: str, ramadan: bool) -> np.ndarray:
    """Relative AHT by interval: nights run longer (fewer, more complex calls), mid-day slightly shorter."""
    factor = 1 + 0.12 * _bump(HOURS, 3.0, 2.5) - 0.03 * _bump(HOURS, 14.0, 4.0)
    if ramadan:
        factor = factor * (1.05 + 0.04 * _bump(HOURS, 23.0, 2.0))
    return factor


def simulate_events(calendar: pd.DataFrame, rng: np.random.Generator, history_end: pd.Timestamp) -> pd.DataFrame:
    """Generate network outages on AR_TECH and sales campaign weeks.

    Outages only occur in the history (they are unplanned). Campaign weeks are
    marketing decisions known in advance, so they extend into the forecast horizon.
    """
    rows: list[dict] = []
    hist = calendar[calendar["date"] <= history_end]
    years = len(hist) / 365.25
    n_outages = rng.poisson(14 * years)
    candidates = hist.loc[hist["date"] != RTA_REPLAY_DATE, "date"].to_numpy()
    for date in np.sort(rng.choice(candidates, size=n_outages, replace=False)):
        start_hour = rng.choice(np.arange(7.0, 22.5, 0.5))
        rows.append(
            {
                "event_type": "outage",
                "queue": RTA_QUEUE,
                "start": pd.Timestamp(date) + pd.Timedelta(hours=float(start_hour)),
                "duration_h": float(rng.choice(np.arange(2.0, 6.5, 0.5))),
                "peak_multiplier": round(float(rng.uniform(2.0, 3.5)), 2),
                "aht_factor": round(float(rng.uniform(0.82, 0.92)), 3),
                "spillover_queue": "AR_BILLING",
                "spillover_share": round(float(rng.uniform(0.08, 0.14)), 3),
            }
        )
    rows.append(
        {
            "event_type": "outage",
            "queue": RTA_QUEUE,
            "start": RTA_REPLAY_DATE + pd.Timedelta(RTA_OUTAGE_START + ":00"),
            "duration_h": RTA_OUTAGE_HOURS,
            "peak_multiplier": RTA_OUTAGE_PEAK,
            "aht_factor": 0.86,
            "spillover_queue": "AR_BILLING",
            "spillover_share": 0.10,
        }
    )
    sundays = calendar.loc[calendar["dow"] == 6, "date"].to_numpy()
    n_campaigns = int(round(PLANNED_SALES_CAMPAIGNS_PER_YEAR * len(calendar) / 365.25))
    chosen = np.sort(rng.choice(sundays[:-1], size=n_campaigns, replace=False))
    for week_start in chosen:
        rows.append(
            {
                "event_type": "campaign",
                "queue": "SALES",
                "start": pd.Timestamp(week_start),
                "duration_h": 7 * 24.0,
                "peak_multiplier": round(float(rng.uniform(*CAMPAIGN_UPLIFT)), 2),
                "aht_factor": 1.0,
                "spillover_queue": "",
                "spillover_share": 0.0,
            }
        )
    events = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    events["end"] = events["start"] + pd.to_timedelta(events["duration_h"], unit="h")
    events.insert(0, "event_id", [f"EV{i:03d}" for i in range(1, len(events) + 1)])
    return events


def campaign_flags(calendar: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Campaign uplift multiplier per date for the SALES queue (1.0 when no campaign)."""
    uplift = pd.Series(1.0, index=calendar["date"])
    for ev in events[events["event_type"] == "campaign"].itertuples():
        days = pd.date_range(ev.start.normalize(), ev.end - pd.Timedelta(seconds=1), freq="D")
        uplift.loc[uplift.index.isin(days)] = ev.peak_multiplier
    return uplift


def outage_multipliers(
    dates: pd.DatetimeIndex, events: pd.DataFrame
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Interval volume multipliers per queue and the AHT factor caused by outages.

    Volume jumps to the peak within the first hour then decays as the network is
    restored; billing receives a share of the excess as spillover.
    """
    n = len(dates) * INTERVALS_PER_DAY
    mult = {q: np.ones(n) for q in QUEUE_CODES}
    aht = np.ones(n)
    day_index = {d: i for i, d in enumerate(dates)}
    for ev in events[events["event_type"] == "outage"].itertuples():
        day = ev.start.normalize()
        if day not in day_index:
            continue
        first = day_index[day] * INTERVALS_PER_DAY + int((ev.start - day) / pd.Timedelta(minutes=INTERVAL_MINUTES))
        length = int(round(ev.duration_h * 60 / INTERVAL_MINUTES))
        ramp = np.r_[0.65, np.linspace(1.0, 0.3, length - 1)]
        sl = slice(first, min(first + length, n))
        excess = (ev.peak_multiplier - 1) * ramp[: sl.stop - sl.start]
        mult[ev.queue][sl] *= 1 + excess
        aht[sl] = ev.aht_factor
        if ev.spillover_queue:
            mult[ev.spillover_queue][sl] *= 1 + ev.spillover_share * excess
    return mult, aht


def expected_daily_volume(calendar: pd.DataFrame, events: pd.DataFrame, queue: str) -> np.ndarray:
    """Deterministic expected daily volume per date (before noise and outages)."""
    spec = QUEUES[queue]
    years = (calendar["date"] - HISTORY_START).dt.days.to_numpy() / 365.25
    growth = (1 + spec.yoy_growth) ** years
    dow = DOW_FACTORS.get(queue, DOW_FACTORS["default"])
    weekday = dow[calendar["dow"].to_numpy()] / dow.mean()
    doy = calendar["day_of_year"].to_numpy()
    season = 1 + 0.03 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    if queue == "AR_TECH":
        season = season * (1 + 0.14 * np.exp(-0.5 * ((doy - 205) / 32) ** 2))
    if queue == "EN_SUPPORT":
        season = season * (1 + 0.06 * np.exp(-0.5 * ((doy - 200) / 40) ** 2))
    bill_day = calendar["bill_cycle_day"].to_numpy()
    bill = np.where(bill_day > 0, np.array([1.0] + BILL_CYCLE[queue])[bill_day], 1.0)
    campaign = campaign_flags(calendar, events).to_numpy() if queue == "SALES" else np.ones(len(calendar))
    special = np.ones(len(calendar))
    ramadan = calendar["is_ramadan"].to_numpy()
    rday = calendar["ramadan_day"].to_numpy()
    special = np.where(ramadan, RAMADAN_FACTOR[queue] * np.where(rday >= 27, 0.86, 1.0), special)
    holiday = calendar["is_holiday"].to_numpy()
    eid = calendar["is_eid"].to_numpy()
    weekend_holiday = np.isin(calendar["dow"].to_numpy(), [4, 5])
    weekday = np.where(holiday, np.where(weekend_holiday, 0.92, 1.0), weekday)
    special = np.where(holiday, np.where(eid, EID_FACTOR[queue], HOLIDAY_FACTOR[queue]), special)
    special = np.where(calendar["is_post_eid"].to_numpy(), POST_EID_FACTOR[queue], special)
    return spec.daily_calls * growth * weekday * season * bill * campaign * special


def _ar1_noise(rng: np.random.Generator, n: int, sigma: float, phi: float) -> np.ndarray:
    """Stationary AR(1) Gaussian noise with marginal standard deviation sigma."""
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def simulate(
    start: str | pd.Timestamp = HISTORY_START,
    end: str | pd.Timestamp = HISTORY_END,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Simulate interval history, events and calendar.

    Args:
        start: First history date.
        end: Last history date.
        seed: Random seed; the same seed always returns identical data.

    Returns:
        Dict with ``intervals`` (one row per queue x 30-minute interval),
        ``events`` (outages and campaigns) and ``calendar``.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    rng_events, rng_volume, rng_plan, rng_service = streams

    full_calendar = build_calendar(start, max(end, FORECAST_END))
    events = simulate_events(full_calendar, rng_events, end)
    calendar = full_calendar[full_calendar["date"] <= end].reset_index(drop=True)
    dates = pd.DatetimeIndex(calendar["date"])
    n_days = len(dates)
    outage_mult, outage_aht = outage_multipliers(dates, events)
    common_noise = _ar1_noise(rng_volume, n_days, 0.02, 0.5)
    absence_day = ABSENCE_BASE + rng_plan.gamma(4, 0.004, n_days)
    absence_day += np.where(calendar["dow"] == 6, 0.015, 0.0)
    absence_day += np.where(calendar["is_post_eid"], 0.06, 0.0)
    absence_day += np.where(calendar["is_ramadan"], 0.02, 0.0)
    years = (dates - HISTORY_START).days.to_numpy() / 365.25

    frames = []
    for queue in QUEUE_CODES:
        spec = QUEUES[queue]
        daily = expected_daily_volume(calendar, events, queue)
        daily = daily * np.exp(common_noise + _ar1_noise(rng_volume, n_days, 0.035, 0.4))
        profiles = np.vstack(
            [
                intraday_profile(queue, profile_day_type(dt, hol), ram)
                for dt, hol, ram in zip(calendar["day_type"], calendar["is_holiday"], calendar["is_ramadan"])
            ]
        )
        expected = (daily[:, None] * profiles).ravel()
        offered = rng_volume.poisson(expected * outage_mult[queue])

        tod = np.vstack([aht_time_of_day_factor(queue, ram) for ram in calendar["is_ramadan"]])
        aht_day = spec.aht_s * np.exp(-0.03 * years) * np.exp(rng_volume.normal(0, 0.02, n_days))
        expected_aht = (aht_day[:, None] * tod).ravel()
        is_open = np.tile(open_mask(queue), n_days)

        # Historical planning: the centre's own forecast, Erlang C requirement and schedule fit.
        plan_err = np.repeat(np.exp(rng_plan.normal(0, 0.06, n_days)), INTERVALS_PER_DAY)
        planned_offered = expected * plan_err * np.exp(rng_plan.normal(0, 0.05, expected.size))
        planned_aht = expected_aht * np.exp(rng_plan.normal(0, 0.03, expected.size))
        required = required_agents(planned_offered, planned_aht)
        fit = np.exp(0.01 + _ar1_noise(rng_plan, expected.size, 0.07, 0.8))
        scheduled = np.where(is_open, np.maximum(np.round(required * fit / (1 - PLANNED_ABSENCE_ALLOWANCE)), 1), 0).astype(int)
        absence = np.repeat(absence_day, INTERVALS_PER_DAY) + rng_plan.gamma(2, 0.006, expected.size)
        staffed = rng_plan.binomial(scheduled, 1 - np.clip(absence, 0, 0.6))

        # Realised service via Erlang A.
        actual_aht_mean = expected_aht * outage_aht if queue == RTA_QUEUE else expected_aht
        service = erlang_a_metrics(offered, actual_aht_mean, staffed, spec.patience_s)
        abandoned = rng_service.binomial(offered, np.clip(service["abandon_rate"], 0, 1))
        handled = offered - abandoned
        in_sl = np.minimum(rng_service.binomial(offered, np.clip(service["service_level"], 0, 1)), handled)
        aht_noise = np.exp(rng_service.normal(0, 0.8 / np.sqrt(np.maximum(handled, 1))))
        aht = np.where(handled > 0, actual_aht_mean * np.clip(aht_noise, 0.4, 2.5), np.nan)
        asa = np.where(handled > 0, service["asa_s"] * np.exp(rng_service.normal(0, 0.15, offered.size)), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            occupancy = np.where(staffed > 0, np.minimum(handled * np.nan_to_num(aht) / (staffed * INTERVAL_SECONDS), 0.99), np.nan)
            sl = np.where(offered > 0, in_sl / offered, np.nan)
            abandon = np.where(offered > 0, abandoned / offered, np.nan)

        frames.append(
            pd.DataFrame(
                {
                    "interval_start": np.repeat(dates.to_numpy(), INTERVALS_PER_DAY)
                    + np.tile(np.arange(INTERVALS_PER_DAY) * np.timedelta64(INTERVAL_MINUTES, "m"), n_days),
                    "queue": queue,
                    "offered": offered,
                    "handled": handled,
                    "abandoned": abandoned,
                    "answered_in_sl": in_sl,
                    "aht_s": aht,
                    "asa_s": asa,
                    "service_level": sl,
                    "abandon_rate": abandon,
                    "occupancy": occupancy,
                    "planned_offered": np.round(np.where(is_open, planned_offered, 0), 1),
                    "planned_aht_s": np.round(planned_aht, 1),
                    "required_agents": np.where(is_open, required, 0),
                    "scheduled_agents": scheduled,
                    "staffed_agents": staffed,
                }
            )
        )
    intervals = pd.concat(frames, ignore_index=True)
    intervals.insert(1, "date", intervals["interval_start"].dt.normalize())
    intervals.insert(2, "interval", intervals["interval_start"].dt.hour * 2 + intervals["interval_start"].dt.minute // 30)
    intervals["queue"] = intervals["queue"].astype("category")
    for col in ("aht_s", "asa_s", "service_level", "abandon_rate", "occupancy"):
        intervals[col] = intervals[col].round(4)
    return {"intervals": intervals, "events": events, "calendar": full_calendar}


def save(data: dict[str, pd.DataFrame]) -> None:
    """Write the simulated tables to data/ as parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_parquet(DATA_DIR / f"{name}.parquet", index=False)


if __name__ == "__main__":
    save(simulate())
