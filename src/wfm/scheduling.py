"""Shift optimisation as a mixed-integer programme solved with HiGHS (scipy.optimize.milp).

Decision variables: how many agents start each shift pattern, where a pattern is
a start interval plus a break position. Each shift is 9 hours on site with a
1-hour unpaid break (8 paid hours). Egyptian labour law caps continuous work at
5 hours, so the break must start between 3 and 5 hours into the shift.

    minimise   sum_k cost_k * x_k  +  penalty * sum_t under_t
    subject to sum_k cover[t, k] * x_k + under_t >= requirement_t   for every interval t
               x_k integer >= 0, under_t >= 0

A week is solved cyclically (Sunday to Saturday), so a Saturday-night shift wraps
into Sunday morning. The same function solves a single day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, identity

from wfm.config import INTERVAL_MINUTES, INTERVALS_PER_DAY
from wfm.labour import EGYPT_LABOUR

DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


@dataclass(frozen=True)
class ShiftRules:
    """Shift construction rules derived from labour law and site policy."""

    shift_hours: float = EGYPT_LABOUR.shift_length_hours
    break_hours: float = EGYPT_LABOUR.min_daily_break_hours
    max_continuous_hours: float = EGYPT_LABOUR.max_continuous_work_hours
    paid_hours: float = EGYPT_LABOUR.max_daily_hours
    under_penalty: float = 0.3  # cost of one uncovered agent-interval, relative to one shift
    shifts_per_fte: int = 5

    def break_offsets(self) -> list[int]:
        """Interval offsets (from shift start) at which the break may begin."""
        per_hour = 60 // INTERVAL_MINUTES
        length, brk = self.length_intervals, round(self.break_hours * per_hour)
        work = length - brk
        max_block = round(self.max_continuous_hours * per_hour)
        return [pre for pre in range(1, work) if pre <= max_block and work - pre <= max_block]

    @property
    def length_intervals(self) -> int:
        return round(self.shift_hours * 60 / INTERVAL_MINUTES)

    @property
    def break_intervals(self) -> int:
        return round(self.break_hours * 60 / INTERVAL_MINUTES)


@dataclass
class ScheduleResult:
    """Optimised schedule with coverage and efficiency statistics."""

    shifts: pd.DataFrame  # one row per used pattern: day, start, end, break_start, agents
    coverage: pd.DataFrame  # one row per interval: requirement, coverage, over/under
    stats: dict[str, float]
    status: str


def schedule_requirement(net_required: np.ndarray, in_day_shrinkage: float) -> np.ndarray:
    """Agents to schedule per interval so that, after in-day shrinkage, net requirement is met."""
    return np.ceil(np.asarray(net_required, dtype=float) / (1.0 - in_day_shrinkage) - 1e-9)


def _patterns(n_intervals: int, rules: ShiftRules, cyclic: bool, allowed: np.ndarray) -> tuple[list[tuple[int, int]], csc_matrix]:
    """Enumerate (start, break offset) patterns and their interval coverage matrix."""
    length, brk = rules.length_intervals, rules.break_intervals
    patterns, rows, cols = [], [], []
    last_start = n_intervals if cyclic else n_intervals - length + 1
    for start in range(last_start):
        span = (start + np.arange(length)) % n_intervals
        if not cyclic and span[-1] < start:
            continue
        if not allowed[span].all():
            continue
        for offset in rules.break_offsets():
            working = np.r_[span[:offset], span[offset + brk :]]
            k = len(patterns)
            patterns.append((start, offset))
            rows.extend(working.tolist())
            cols.extend([k] * len(working))
    matrix = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_intervals, len(patterns)))
    return patterns, matrix


def optimise_shifts(
    requirement: np.ndarray,
    rules: ShiftRules | None = None,
    cyclic: bool = True,
    open_intervals: np.ndarray | None = None,
    start: pd.Timestamp | None = None,
    time_limit_s: float = 30.0,
) -> ScheduleResult:
    """Solve the shift-scheduling MILP for a requirement curve.

    Args:
        requirement: Agents to schedule per 30-minute interval (already grossed up for in-day shrinkage).
            Length 336 for a Sunday-Saturday week or 48 for a single day.
        rules: Shift rules; defaults to 9h shifts with a 1h break placed per labour law.
        cyclic: Treat the horizon as repeating so overnight shifts wrap.
        open_intervals: Boolean mask; shifts may only cover open intervals (e.g. SALES 09:00-23:00).
        start: Timestamp of the first interval, used to label the output.
        time_limit_s: HiGHS time limit.
    """
    rules = rules or ShiftRules()
    req = np.asarray(requirement, dtype=float)
    n = req.size
    allowed = np.ones(n, dtype=bool) if open_intervals is None else np.asarray(open_intervals, dtype=bool)
    patterns, cover = _patterns(n, rules, cyclic, allowed)
    n_x = len(patterns)

    # A tiny tie-break on break position removes symmetric optima and speeds up branch-and-bound.
    tie_break = 1e-4 * np.array([offset for _, offset in patterns])
    cost = np.r_[1.0 + tie_break, np.full(n, rules.under_penalty)]
    a = hstack([cover, identity(n, format="csc")], format="csc")
    constraints = LinearConstraint(a, lb=req, ub=np.inf)
    integrality = np.r_[np.ones(n_x), np.zeros(n)]
    res = milp(
        cost,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(0, np.inf),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.01, "disp": False},
    )
    if res.x is None:
        raise RuntimeError(f"Shift optimisation failed: {res.message}")
    x = np.round(res.x[:n_x]).astype(int)
    coverage_agents = cover @ x
    on_shift = _on_shift(patterns, x, n, rules, cyclic)
    result = _summarise(req, coverage_agents, on_shift, patterns, x, rules, start, res.message)
    result.stats["mip_gap"] = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return result


def _on_shift(patterns: list[tuple[int, int]], x: np.ndarray, n: int, rules: ShiftRules, cyclic: bool) -> np.ndarray:
    """Agents on site per interval (including those on break)."""
    on_site = np.zeros(n)
    for (start, _), count in zip(patterns, x):
        if count:
            span = (start + np.arange(rules.length_intervals)) % n
            on_site[span] += count
    return on_site


def _summarise(
    req: np.ndarray,
    cov: np.ndarray,
    on_site: np.ndarray,
    patterns: list[tuple[int, int]],
    x: np.ndarray,
    rules: ShiftRules,
    start: pd.Timestamp | None,
    status: str,
) -> ScheduleResult:
    """Build output tables and schedule-quality statistics."""
    n = req.size
    origin = start if start is not None else pd.Timestamp("2026-01-04")  # a Sunday
    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    times = origin + step * np.arange(n)
    coverage = pd.DataFrame(
        {
            "interval_start": times,
            "requirement": req,
            "scheduled_on_phone": cov,
            "on_site": on_site,
            "on_break": on_site - cov,
            "over_under": cov - req,
        }
    )
    used = [(p, c) for p, c in zip(patterns, x) if c > 0]
    shifts = pd.DataFrame(
        [
            {
                "day": DAY_NAMES[(origin.dayofweek + 1 + s // INTERVALS_PER_DAY) % 7],
                "shift_start": (origin + step * s).strftime("%H:%M"),
                "shift_end": (origin + step * (s + rules.length_intervals)).strftime("%H:%M"),
                "break_start": (origin + step * (s + off)).strftime("%H:%M"),
                "start_interval": s,
                "agents": int(c),
            }
            for (s, off), c in used
        ]
    )
    hours = INTERVAL_MINUTES / 60
    covered = np.minimum(cov, req).sum()
    total_shifts = int(x.sum())
    paid_hours = total_shifts * rules.paid_hours
    stats = {
        "shifts": float(total_shifts),
        "paid_hours": paid_hours,
        "fte_equivalent": paid_hours / (rules.paid_hours * rules.shifts_per_fte),
        "requirement_hours": req.sum() * hours,
        "scheduled_hours": cov.sum() * hours,
        "over_hours": np.clip(cov - req, 0, None).sum() * hours,
        "under_hours": np.clip(req - cov, 0, None).sum() * hours,
        "coverage_pct": covered / req.sum() if req.sum() else 1.0,
        "efficiency_pct": covered / cov.sum() if cov.sum() else 1.0,
        "intervals_under": (cov < req).sum(),
    }
    return ScheduleResult(shifts, coverage, {k: float(v) for k, v in stats.items()}, status)


def required_shifts_lower_bound(requirement: np.ndarray, rules: ShiftRules | None = None) -> int:
    """Simple lower bound: requirement hours / productive hours per shift."""
    rules = rules or ShiftRules()
    productive = (rules.length_intervals - rules.break_intervals) * INTERVAL_MINUTES / 60
    return math.ceil(np.asarray(requirement).sum() * INTERVAL_MINUTES / 60 / productive)
