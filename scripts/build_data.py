"""Build every dataset the Streamlit app reads.

Runs the simulator, the forecast backtests, the 26-week forecast, interval
requirements, the weekly MILP schedules (full-time only and with part-time),
the weekly rosters, the cross-skilling simulation and the RTA replay day, and
writes the results to data/ and data/processed/. Usage:

    python scripts/build_data.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfm.capacity import Shrinkage, interval_requirement  # noqa: E402
from wfm.config import (  # noqa: E402
    DEFAULT_SEED,
    FORECAST_END,
    FORECAST_START,
    HISTORY_END,
    INTERVALS_PER_DAY,
    PROCESSED_DIR,
    QUEUE_CODES,
    QUEUES,
    RTA_QUEUE,
    RTA_REPLAY_DATE,
    open_mask,
)
from wfm.egypt_calendar import week_start  # noqa: E402
from wfm.forecast import (  # noqa: E402
    BACKTESTS,
    clean_outages,
    daily_actuals,
    forecast_forward,
    run_backtest,
    select_champion,
)
from wfm.excel_report import WORKBOOK_NAME, agent_minutes, build_workbook  # noqa: E402
from wfm.multiskill import day_comparison, summarise_day  # noqa: E402
from wfm.realtime import build_rta_day, save_rta_day  # noqa: E402
from wfm.scheduling import ShiftRules, optimise_roster, optimise_shifts, schedule_requirement  # noqa: E402
from wfm.simulate import save, simulate  # noqa: E402

SCHEDULE_LEAD_DAYS = 15  # schedules are published about two weeks ahead
SCHEDULE_VARIANTS = {"full_time": ShiftRules(), "part_time": ShiftRules(part_time=True)}
CROSS_SKILL_PAIR = ("AR_BILLING", "AR_TECH")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build_schedules(requirement: pd.DataFrame, week_start: pd.Timestamp, shrinkage: Shrinkage) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Optimise one Sunday-Saturday week per queue, full-time only and with part-time shifts."""
    shifts, coverage, stats = [], [], {v: {} for v in SCHEDULE_VARIANTS}
    for queue in QUEUE_CODES:
        req = requirement[(requirement["queue"] == queue) & (requirement["week_start"] == week_start)].sort_values("interval_start")
        need = schedule_requirement(req["required_agents"].to_numpy(), shrinkage.in_day)
        for variant, rules in SCHEDULE_VARIANTS.items():
            result = optimise_shifts(need, rules, open_intervals=np.tile(open_mask(queue), 7), start=week_start, time_limit_s=20)
            coverage.append(result.coverage.assign(queue=queue, variant=variant, net_required=req["required_agents"].to_numpy()))
            shifts.append(result.shifts.assign(queue=queue, variant=variant))
            stats[variant][queue] = {**result.stats, "status": result.status}
    return pd.concat(shifts, ignore_index=True), pd.concat(coverage, ignore_index=True), stats


def build_rosters(requirement: pd.DataFrame, week_start: pd.Timestamp, shrinkage: Shrinkage) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Weekly rosters (5 shifts, 2 consecutive rest days, fixed start) per queue."""
    agents, tours, coverage, stats = [], [], [], {}
    for queue in QUEUE_CODES:
        req = requirement[(requirement["queue"] == queue) & (requirement["week_start"] == week_start)].sort_values("interval_start")
        need = schedule_requirement(req["required_agents"].to_numpy(), shrinkage.in_day)
        result = optimise_roster(need, open_intervals=np.tile(open_mask(queue), 7), start=week_start, agent_prefix=queue.split("_")[-1][:3], time_limit_s=45)
        agents.append(result.roster.assign(queue=queue))
        tours.append(result.tours.assign(queue=queue))
        coverage.append(result.coverage.assign(queue=queue))
        stats[queue] = {**result.stats, "status": result.status}
    return pd.concat(agents, ignore_index=True), pd.concat(tours, ignore_index=True), pd.concat(coverage, ignore_index=True), stats


def build_cross_skill(requirement: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Simulated agents needed through one day for each cross-skilled share of the Arabic queues."""
    qa, qb = CROSS_SKILL_PAIR
    a = requirement[(requirement["queue"] == qa) & (requirement["date"] == day)].set_index("interval")
    b = requirement[(requirement["queue"] == qb) & (requirement["date"] == day)].set_index("interval")
    frame = pd.DataFrame(
        {
            "interval": a.index,
            "calls_a": a["forecast_offered"].to_numpy(),
            "aht_a": a["forecast_aht_s"].to_numpy(),
            "calls_b": b["forecast_offered"].to_numpy(),
            "aht_b": b["forecast_aht_s"].to_numpy(),
        }
    )
    detail = day_comparison(frame, QUEUES[qa].patience_s, QUEUES[qb].patience_s, hours=8)
    return detail.merge(frame, on="interval")


def main() -> None:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = simulate(seed=DEFAULT_SEED)
    save(data)
    intervals, events, calendar = data["intervals"], data["events"], data["calendar"]
    _log(f"simulated {len(intervals):,} interval rows", t0)

    clean = clean_outages(intervals, events)
    daily = daily_actuals(clean)
    daily.to_parquet(PROCESSED_DIR / "daily_actuals.parquet", index=False)

    results = [run_backtest(clean, daily, calendar, events, w) for w in BACKTESTS]
    champion = select_champion(results)
    pd.concat([r.daily.assign(window=r.window.name) for r in results]).to_parquet(PROCESSED_DIR / "backtest_daily.parquet", index=False)
    pd.concat([r.daily_metrics.assign(window=r.window.name) for r in results]).to_parquet(PROCESSED_DIR / "backtest_metrics.parquet", index=False)
    pd.concat([r.intervals.assign(window=r.window.name) for r in results]).to_parquet(PROCESSED_DIR / "backtest_intervals.parquet", index=False)
    pd.concat([r.interval_metrics.assign(window=r.window.name) for r in results]).to_parquet(
        PROCESSED_DIR / "backtest_interval_metrics.parquet", index=False
    )
    _log(f"backtests done, champion = {champion}", t0)

    fc_daily, fc_intervals = forecast_forward(clean, daily, calendar, events, HISTORY_END, FORECAST_START, FORECAST_END, champion)
    requirement = interval_requirement(fc_intervals)
    requirement["week_start"] = week_start(requirement["date"])
    fc_daily.to_parquet(PROCESSED_DIR / "forecast_daily.parquet", index=False)
    requirement.to_parquet(PROCESSED_DIR / "forecast_intervals.parquet", index=False)
    _log("26-week forecast and interval requirement done", t0)

    shrinkage = Shrinkage()
    shifts, coverage, stats = build_schedules(requirement, FORECAST_START, shrinkage)
    shifts.to_parquet(PROCESSED_DIR / "schedule_shifts.parquet", index=False)
    coverage.to_parquet(PROCESSED_DIR / "schedule_coverage.parquet", index=False)
    _log("weekly schedules optimised (full-time and part-time variants)", t0)

    roster_agents, roster_tours, roster_coverage, roster_stats = build_rosters(requirement, FORECAST_START, shrinkage)
    roster_agents.to_parquet(PROCESSED_DIR / "roster_agents.parquet", index=False)
    roster_tours.to_parquet(PROCESSED_DIR / "roster_tours.parquet", index=False)
    roster_coverage.to_parquet(PROCESSED_DIR / "roster_coverage.parquet", index=False)
    _log(f"weekly rosters built for {len(roster_agents)} agents", t0)

    cross_skill = build_cross_skill(requirement, FORECAST_START)
    cross_skill.to_parquet(PROCESSED_DIR / "cross_skill_day.parquet", index=False)
    cross_summary = summarise_day(cross_skill)
    _log("cross-skilling simulation done", t0)

    # RTA replay: schedule from a forecast made two weeks before, intraday plan from the day before.
    week = week_start(pd.Series([RTA_REPLAY_DATE])).iloc[0]
    _, week_fc = forecast_forward(
        clean, daily, calendar, events, week - pd.Timedelta(days=SCHEDULE_LEAD_DAYS), week, week + pd.Timedelta(days=6), champion
    )
    week_req = interval_requirement(week_fc[week_fc["queue"] == RTA_QUEUE]).sort_values("interval_start")
    rta_schedule = optimise_shifts(schedule_requirement(week_req["required_agents"].to_numpy(), shrinkage.in_day), start=week)
    tomorrow = RTA_REPLAY_DATE + pd.Timedelta(days=1)
    _, day_fc = forecast_forward(
        clean, daily, calendar, events, RTA_REPLAY_DATE - pd.Timedelta(days=1), RTA_REPLAY_DATE, tomorrow, champion
    )
    rta_day = build_rta_day(intervals, day_fc, rta_schedule.shifts, week, RTA_REPLAY_DATE, RTA_QUEUE, shrinkage.coaching + shrinkage.meetings)
    save_rta_day(rta_day, PROCESSED_DIR)

    # Start-of-shift Staffing Outlook inputs: today's and tomorrow's forecast against the published schedule,
    # with planned in-day offline time (coaching, meetings) taken off the scheduled phone time.
    plan = day_fc[day_fc["queue"] == RTA_QUEUE][["date", "interval", "interval_start", "forecast_offered", "forecast_aht_s"]]
    sched_cov = rta_schedule.coverage[["interval_start", "scheduled_on_phone"]]
    plan = plan.merge(sched_cov, on="interval_start")
    plan["scheduled_on_phone"] *= 1 - (shrinkage.coaching + shrinkage.meetings)
    plan.drop(columns="interval_start").to_parquet(PROCESSED_DIR / "rta_plan.parquet", index=False)

    build_workbook(PROCESSED_DIR / WORKBOOK_NAME, rta_day, agent_minutes(rta_day), intervals, calendar, requirement)
    _log("Excel workbook written", t0)
    _log(f"RTA replay day built with {len(rta_day.agents)} agents", t0)

    meta = {
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seed": DEFAULT_SEED,
        "champion_model": champion,
        "history": [str(intervals["date"].min().date()), str(intervals["date"].max().date())],
        "forecast": [str(FORECAST_START.date()), str(FORECAST_END.date())],
        "schedule_week": str(FORECAST_START.date()),
        "schedule_stats": stats["full_time"],
        "schedule_stats_part_time": stats["part_time"],
        "roster_stats": roster_stats,
        "cross_skill": {"pair": list(CROSS_SKILL_PAIR), "day": str(FORECAST_START.date()), "summary": cross_summary.to_dict("records")},
        "rta": {"date": str(RTA_REPLAY_DATE.date()), "queue": RTA_QUEUE, "schedule_stats": rta_schedule.stats},
        "intervals_per_day": INTERVALS_PER_DAY,
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    _log("done", t0)


if __name__ == "__main__":
    main()
