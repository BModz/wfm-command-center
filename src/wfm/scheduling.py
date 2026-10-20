"""Shift and roster optimisation as mixed-integer programmes solved with HiGHS (scipy.optimize.milp).

Shift scheduling decides how many agents start each shift pattern, where a
pattern is a shift type, a start interval and a break position:

- Full-time: 9 hours on site with a 1-hour unpaid break (8 paid hours).
  Egyptian labour law caps continuous work at 5 hours, so the break starts
  between 3 and 5 hours into the shift.
- Part-time (optional): 4 or 5 paid hours with no break, since neither exceeds
  the 5-hour continuous-work limit. Their share of paid hours is capped by policy.

    minimise   sum_k (paid_hours_k / 8) * x_k  +  penalty * sum_t under_t
    subject to sum_k cover[t, k] * x_k + under_t >= requirement_t   for every interval t
               sum_{k part-time} h_k x_k <= cap * sum_k h_k x_k
               x_k integer >= 0, under_t >= 0

Roster (tour) scheduling then turns demand into named weekly working patterns:
each agent works the same start time on 5 days with 2 consecutive rest days,
which satisfies the 24-hour weekly-rest rule. A week is solved cyclically
(Sunday to Saturday), so a Saturday-night shift wraps into Sunday morning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, identity, vstack

from wfm.config import INTERVAL_MINUTES, INTERVALS_PER_DAY
from wfm.labour import EGYPT_LABOUR

DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKEND_OFF = 5  # days-off pattern index for Friday + Saturday, the Egyptian weekend
ROSTER_TEAM_SIZE = 14


@dataclass(frozen=True)
class ShiftType:
    """A shift template: length on site, break length and allowed break offsets (in intervals)."""

    name: str
    length: int
    break_len: int
    break_offsets: tuple[int, ...]
    paid_hours: float

    @property
    def part_time(self) -> bool:
        return self.name != "Full-time"


@dataclass(frozen=True)
class ShiftRules:
    """Shift construction rules derived from labour law and site policy."""

    shift_hours: float = EGYPT_LABOUR.shift_length_hours
    break_hours: float = EGYPT_LABOUR.min_daily_break_hours
    max_continuous_hours: float = EGYPT_LABOUR.max_continuous_work_hours
    paid_hours: float = EGYPT_LABOUR.max_daily_hours
    under_penalty: float = 0.3  # cost of one uncovered agent-interval, relative to one full-time shift
    shifts_per_fte: int = 5
    part_time: bool = False
    part_time_hours: tuple[float, ...] = (4.0, 5.0)
    max_part_time_share: float = 0.30  # cap on part-time share of paid hours

    def break_offsets(self) -> list[int]:
        """Interval offsets (from shift start) at which the full-time break may begin."""
        per_hour = 60 // INTERVAL_MINUTES
        work = self.length_intervals - self.break_intervals
        max_block = round(self.max_continuous_hours * per_hour)
        return [pre for pre in range(1, work) if pre <= max_block and work - pre <= max_block]

    @property
    def length_intervals(self) -> int:
        return round(self.shift_hours * 60 / INTERVAL_MINUTES)

    @property
    def break_intervals(self) -> int:
        return round(self.break_hours * 60 / INTERVAL_MINUTES)

    def shift_types(self) -> list[ShiftType]:
        """Full-time type, plus no-break part-time types when part-time is enabled."""
        types = [ShiftType("Full-time", self.length_intervals, self.break_intervals, tuple(self.break_offsets()), self.paid_hours)]
        if self.part_time:
            for hours in self.part_time_hours:
                if hours > self.max_continuous_hours:
                    raise ValueError("Part-time shifts without a break cannot exceed the continuous-work limit")
                types.append(ShiftType(f"Part-time {hours:g}h", round(hours * 60 / INTERVAL_MINUTES), 0, (), hours))
        return types


@dataclass
class ScheduleResult:
    """Optimised schedule with coverage and efficiency statistics."""

    shifts: pd.DataFrame  # one row per used pattern: day, type, start, end, break_start, agents
    coverage: pd.DataFrame  # one row per interval: requirement, coverage, over/under
    stats: dict[str, float]
    status: str


@dataclass
class RosterResult:
    """Weekly roster: tours (pattern counts), named agents and coverage."""

    tours: pd.DataFrame
    roster: pd.DataFrame
    coverage: pd.DataFrame
    stats: dict[str, float]
    status: str


Pattern = tuple[int, int, int]  # (start interval, shift-type index, break offset or -1)


def schedule_requirement(net_required: np.ndarray, in_day_shrinkage: float) -> np.ndarray:
    """Agents to schedule per interval so that, after in-day shrinkage, net requirement is met."""
    return np.ceil(np.asarray(net_required, dtype=float) / (1.0 - in_day_shrinkage) - 1e-9)


def _working_intervals(start: int, stype: ShiftType, offset: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """On-site span and on-phone intervals of one shift (wrapping modulo n)."""
    span = (start + np.arange(stype.length)) % n
    if offset < 0:
        return span, span
    return span, np.r_[span[:offset], span[offset + stype.break_len :]]


def _patterns(n_intervals: int, types: list[ShiftType], cyclic: bool, allowed: np.ndarray) -> tuple[list[Pattern], csc_matrix]:
    """Enumerate shift patterns and their interval coverage matrix."""
    patterns: list[Pattern] = []
    rows: list[int] = []
    cols: list[int] = []
    for t, stype in enumerate(types):
        last_start = n_intervals if cyclic else n_intervals - stype.length + 1
        for start in range(last_start):
            span = (start + np.arange(stype.length)) % n_intervals
            if not allowed[span].all():
                continue
            for offset in stype.break_offsets or (-1,):
                _, working = _working_intervals(start, stype, offset, n_intervals)
                k = len(patterns)
                patterns.append((start, t, offset))
                rows.extend(working.tolist())
                cols.extend([k] * len(working))
    matrix = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_intervals, len(patterns)))
    return patterns, matrix


def _solve(cost: np.ndarray, a: csc_matrix, lb: np.ndarray, ub: np.ndarray, n_int: int, time_limit_s: float):
    """Run HiGHS with the first ``n_int`` variables integer."""
    res = milp(
        cost,
        constraints=LinearConstraint(a, lb=lb, ub=ub),
        integrality=np.r_[np.ones(n_int), np.zeros(cost.size - n_int)],
        bounds=Bounds(0, np.inf),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.01, "disp": False},
    )
    if res.x is None:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    return res


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
        rules: Shift rules; defaults to full-time 9h shifts with a 1h break placed per labour law.
        cyclic: Treat the horizon as repeating so overnight shifts wrap.
        open_intervals: Boolean mask; shifts may only cover open intervals (e.g. SALES 09:00-23:00).
        start: Timestamp of the first interval, used to label the output.
        time_limit_s: HiGHS time limit.
    """
    rules = rules or ShiftRules()
    types = rules.shift_types()
    req = np.asarray(requirement, dtype=float)
    n = req.size
    allowed = np.ones(n, dtype=bool) if open_intervals is None else np.asarray(open_intervals, dtype=bool)
    patterns, cover = _patterns(n, types, cyclic, allowed)
    n_x = len(patterns)
    hours = np.array([types[t].paid_hours for _, t, _ in patterns])

    # Cost in full-time-shift units; a tiny tie-break on break position removes symmetric optima.
    tie_break = 1e-4 * np.array([max(offset, 0) for _, _, offset in patterns])
    cost = np.r_[hours / rules.paid_hours + tie_break, np.full(n, rules.under_penalty)]
    a = hstack([cover, identity(n, format="csc")], format="csc")
    lb, ub = req, np.full(n, np.inf)
    if rules.part_time:
        is_pt = np.array([types[t].part_time for _, t, _ in patterns])
        cap_row = csc_matrix(np.r_[hours * (is_pt - rules.max_part_time_share), np.zeros(n)][None, :])
        a = vstack([a, cap_row], format="csc")
        lb, ub = np.r_[lb, -np.inf], np.r_[ub, 0.0]
    res = _solve(cost, a, lb, ub, n_x, time_limit_s)
    x = np.round(res.x[:n_x]).astype(int)
    coverage_agents = cover @ x
    on_site = np.zeros(n)
    for (s, t, offset), count in zip(patterns, x, strict=True):
        if count:
            span, _ = _working_intervals(s, types[t], offset, n)
            on_site[span] += count
    result = _summarise(req, coverage_agents, on_site, patterns, x, types, rules, start, res.message)
    result.stats["mip_gap"] = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return result


def _coverage_frame(req: np.ndarray, cov: np.ndarray, on_site: np.ndarray, origin: pd.Timestamp) -> pd.DataFrame:
    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    return pd.DataFrame(
        {
            "interval_start": origin + step * np.arange(req.size),
            "requirement": req,
            "scheduled_on_phone": cov,
            "on_site": on_site,
            "on_break": on_site - cov,
            "over_under": cov - req,
        }
    )


def _coverage_stats(req: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    hours = INTERVAL_MINUTES / 60
    covered = np.minimum(cov, req).sum()
    return {
        "requirement_hours": req.sum() * hours,
        "scheduled_hours": cov.sum() * hours,
        "over_hours": np.clip(cov - req, 0, None).sum() * hours,
        "under_hours": np.clip(req - cov, 0, None).sum() * hours,
        "coverage_pct": covered / req.sum() if req.sum() else 1.0,
        "efficiency_pct": covered / cov.sum() if cov.sum() else 1.0,
        "intervals_under": (cov < req).sum(),
    }


def _summarise(
    req: np.ndarray,
    cov: np.ndarray,
    on_site: np.ndarray,
    patterns: list[Pattern],
    x: np.ndarray,
    types: list[ShiftType],
    rules: ShiftRules,
    start: pd.Timestamp | None,
    status: str,
) -> ScheduleResult:
    """Build output tables and schedule-quality statistics."""
    origin = start if start is not None else pd.Timestamp("2026-01-04")  # a Sunday
    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    rows = []
    for (s, t, off), c in zip(patterns, x, strict=True):
        if c <= 0:
            continue
        stype = types[t]
        rows.append(
            {
                "day": DAY_NAMES[(origin.dayofweek + 1 + s // INTERVALS_PER_DAY) % 7],
                "shift_type": stype.name,
                "shift_start": (origin + step * s).strftime("%H:%M"),
                "shift_end": (origin + step * (s + stype.length)).strftime("%H:%M"),
                "break_start": (origin + step * (s + off)).strftime("%H:%M") if off >= 0 else "",
                "start_interval": s,
                "break_offset": off,
                "paid_hours": stype.paid_hours,
                "agents": int(c),
            }
        )
    shifts = pd.DataFrame(rows)
    pt = shifts["shift_type"] != "Full-time" if len(shifts) else pd.Series(dtype=bool)
    paid = float((shifts["paid_hours"] * shifts["agents"]).sum()) if len(shifts) else 0.0
    pt_paid = float((shifts.loc[pt, "paid_hours"] * shifts.loc[pt, "agents"]).sum()) if len(shifts) else 0.0
    stats = {
        "shifts": float(x.sum()),
        "part_time_shifts": float(shifts.loc[pt, "agents"].sum()) if len(shifts) else 0.0,
        "paid_hours": paid,
        "part_time_hours_share": pt_paid / paid if paid else 0.0,
        "fte_equivalent": paid / (rules.paid_hours * rules.shifts_per_fte),
        **_coverage_stats(req, cov),
    }
    return ScheduleResult(shifts, _coverage_frame(req, cov, on_site, origin), {k: float(v) for k, v in stats.items()}, status)


def required_shifts_lower_bound(requirement: np.ndarray, rules: ShiftRules | None = None) -> int:
    """Simple lower bound on full-time shifts: requirement hours / productive hours per shift."""
    rules = rules or ShiftRules()
    productive = (rules.length_intervals - rules.break_intervals) * INTERVAL_MINUTES / 60
    return math.ceil(np.asarray(requirement).sum() * INTERVAL_MINUTES / 60 / productive)


# ---------------------------------------------------------------------------
# Roster (tour) scheduling
# ---------------------------------------------------------------------------


def days_off_label(pattern: int) -> str:
    """Label for days-off pattern ``pattern``: rest on that day and the next (Sunday = 0)."""
    return f"{DAY_NAMES[pattern]}-{DAY_NAMES[(pattern + 1) % 7]}"


def optimise_roster(
    requirement: np.ndarray,
    rules: ShiftRules | None = None,
    open_intervals: np.ndarray | None = None,
    start: pd.Timestamp | None = None,
    agent_prefix: str = "AG",
    time_limit_s: float = 60.0,
) -> RosterResult:
    """Tour-scheduling MILP: weekly patterns of 5 shifts at a fixed start time with 2 consecutive rest days.

    Each variable is a tour (days-off pair, start interval, break offset) and
    counts agents; one tour is one full-time agent. The week is cyclic. Ties
    are broken towards Friday-Saturday off, the Egyptian weekend.

    Args:
        requirement: Agents to schedule per interval for a Sunday-Saturday week (length 336).
        rules: Shift rules (full-time shifts only).
        open_intervals: Boolean mask of intervals shifts may cover.
        start: Sunday the week starts on, used for labels.
        agent_prefix: Prefix for generated agent IDs.
        time_limit_s: HiGHS time limit.
    """
    rules = rules or ShiftRules()
    ft = replace(rules, part_time=False).shift_types()[0]
    req = np.asarray(requirement, dtype=float)
    n = req.size
    if n != 7 * INTERVALS_PER_DAY:
        raise ValueError("A roster needs a full Sunday-Saturday week of intervals")
    allowed = np.ones(n, dtype=bool) if open_intervals is None else np.asarray(open_intervals, dtype=bool)
    tours: list[tuple[int, int, int]] = []
    rows: list[int] = []
    cols: list[int] = []
    for days_off in range(7):
        work_days = [d for d in range(7) if d not in (days_off, (days_off + 1) % 7)]
        for s in range(INTERVALS_PER_DAY):
            spans = [(d * INTERVALS_PER_DAY + s + np.arange(ft.length)) % n for d in work_days]
            if not all(allowed[sp].all() for sp in spans):
                continue
            for offset in ft.break_offsets:
                k = len(tours)
                tours.append((days_off, s, offset))
                for d in work_days:
                    _, working = _working_intervals(d * INTERVALS_PER_DAY + s, ft, offset, n)
                    rows.extend(working.tolist())
                    cols.extend([k] * len(working))
    cover = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(tours)))
    n_x = len(tours)
    tie = np.array([1e-4 * off - (1e-3 if d == WEEKEND_OFF else 0.0) for d, _, off in tours])
    # One tour costs one FTE (5 shifts), so the per-interval penalty is scaled to match shift scheduling.
    cost = np.r_[1.0 + tie, np.full(n, rules.under_penalty / rules.shifts_per_fte)]
    a = hstack([cover, identity(n, format="csc")], format="csc")
    res = _solve(cost, a, req, np.full(n, np.inf), n_x, time_limit_s)
    x = np.round(res.x[:n_x]).astype(int)
    cov = cover @ x
    origin = start if start is not None else pd.Timestamp("2026-01-04")
    on_site = np.zeros(n)
    for (days_off, s, _), count in zip(tours, x, strict=True):
        if count:
            for d in range(7):
                if d not in (days_off, (days_off + 1) % 7):
                    on_site[(d * INTERVALS_PER_DAY + s + np.arange(ft.length)) % n] += count
    tour_df, roster = _roster_tables(tours, x, ft, origin, agent_prefix)
    heads = int(x.sum())
    stats = {
        "agents": heads,
        "paid_hours": heads * rules.shifts_per_fte * rules.paid_hours,
        "weekend_off_share": float(x[[i for i, t in enumerate(tours) if t[0] == WEEKEND_OFF]].sum() / heads) if heads else 0.0,
        "distinct_start_times": float(tour_df["start"].nunique()) if len(tour_df) else 0.0,
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        **_coverage_stats(req, cov),
    }
    return RosterResult(tour_df, roster, _coverage_frame(req, cov, on_site, origin), {k: float(v) for k, v in stats.items()}, res.message)


def _roster_tables(
    tours: list[tuple[int, int, int]], x: np.ndarray, ft: ShiftType, origin: pd.Timestamp, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tour counts and one row per named agent with each day's shift or OFF."""
    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    tour_rows, agents = [], []
    used = sorted(((t, c) for t, c in zip(tours, x, strict=True) if c > 0), key=lambda tc: (tc[0][1], tc[0][0], tc[0][2]))
    for (days_off, s, off), count in used:
        shift = f"{(origin + step * s):%H:%M}-{(origin + step * (s + ft.length)):%H:%M}"
        brk = f"{(origin + step * (s + off)):%H:%M}"
        tour_rows.append({"days_off": days_off_label(days_off), "start": f"{(origin + step * s):%H:%M}", "shift": shift, "break_start": brk, "agents": int(count)})
        for _ in range(count):
            i = len(agents)
            row = {"agent_id": f"{prefix}-{i + 1:04d}", "team": f"Team {i // ROSTER_TEAM_SIZE + 1:02d}", "days_off": days_off_label(days_off), "break_start": brk}
            row.update({day: ("OFF" if d in (days_off, (days_off + 1) % 7) else shift) for d, day in enumerate(DAY_NAMES)})
            agents.append(row)
    return pd.DataFrame(tour_rows), pd.DataFrame(agents)
