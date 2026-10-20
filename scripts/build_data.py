"""Build every dataset the Streamlit app reads.

Runs the simulator, the forecast backtests, the 26-week forecast, interval
requirements, the weekly MILP schedules and the RTA replay day, and writes the
results to data/ and data/processed/. Usage:

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
from wfm.realtime import build_rta_day, save_rta_day  # noqa: E402
from wfm.scheduling import optimise_shifts, schedule_requirement  # noqa: E402
from wfm.simulate import save, simulate  # noqa: E402

SCHEDULE_LEAD_DAYS = 15  # schedules are published about two weeks ahead


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build_schedules(requirement: pd.DataFrame, week_start: pd.Timestamp, shrinkage: Shrinkage) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Optimise one Sunday-Saturday week per queue."""
    shifts, coverage, stats = [], [], {}
    for queue in QUEUE_CODES:
        req = requirement[(requirement["queue"] == queue) & (requirement["week_start"] == week_start)].sort_values("interval_start")
        need = schedule_requirement(req["required_agents"].to_numpy(), shrinkage.in_day)
        result = optimise_shifts(need, open_intervals=np.tile(open_mask(queue), 7), start=week_start)
        cov = result.coverage.assign(queue=queue, net_required=req["required_agents"].to_numpy())
        shifts.append(result.shifts.assign(queue=queue))
        coverage.append(cov)
        stats[queue] = {**result.stats, "status": result.status}
    return pd.concat(shifts, ignore_index=True), pd.concat(coverage, ignore_index=True), stats


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
    _log("weekly schedules optimised", t0)

    # RTA replay: schedule from a forecast made two weeks before, intraday plan from the day before.
    week = week_start(pd.Series([RTA_REPLAY_DATE])).iloc[0]
    _, week_fc = forecast_forward(
        clean, daily, calendar, events, week - pd.Timedelta(days=SCHEDULE_LEAD_DAYS), week, week + pd.Timedelta(days=6), champion
    )
    week_req = interval_requirement(week_fc[week_fc["queue"] == RTA_QUEUE]).sort_values("interval_start")
    rta_schedule = optimise_shifts(schedule_requirement(week_req["required_agents"].to_numpy(), shrinkage.in_day), start=week)
    _, day_fc = forecast_forward(
        clean, daily, calendar, events, RTA_REPLAY_DATE - pd.Timedelta(days=1), RTA_REPLAY_DATE, RTA_REPLAY_DATE, champion
    )
    rta_day = build_rta_day(intervals, day_fc, rta_schedule.shifts, week, RTA_REPLAY_DATE, RTA_QUEUE, shrinkage.coaching + shrinkage.meetings)
    save_rta_day(rta_day, PROCESSED_DIR)
    _log(f"RTA replay day built with {len(rta_day.agents)} agents", t0)

    meta = {
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seed": DEFAULT_SEED,
        "champion_model": champion,
        "history": [str(intervals["date"].min().date()), str(intervals["date"].max().date())],
        "forecast": [str(FORECAST_START.date()), str(FORECAST_END.date())],
        "schedule_week": str(FORECAST_START.date()),
        "schedule_stats": stats,
        "rta": {"date": str(RTA_REPLAY_DATE.date()), "queue": RTA_QUEUE, "schedule_stats": rta_schedule.stats},
        "intervals_per_day": INTERVALS_PER_DAY,
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    _log("done", t0)


if __name__ == "__main__":
    main()
