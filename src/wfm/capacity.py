"""Long-term capacity planning: shrinkage, FTE requirement, attrition and hiring.

Required FTE = interval agent-hours / (1 - shrinkage) / contracted weekly hours.
Available FTE projects the current headcount forward with weekly attrition and
adds new-hire classes as they move through training (0% productive) and
nesting (partly productive) to full proficiency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from wfm.config import INTERVAL_MINUTES, MAX_OCCUPANCY, SL_TARGET, SL_THRESHOLD_S
from wfm.egypt_calendar import week_start
from wfm.erlang import required_agents, traffic_intensity
from wfm.labour import EGYPT_LABOUR

# Working days per year a full-timer is scheduled: 52 weeks x 5 days.
WORKING_DAYS_PER_YEAR = 52 * 5
# Productive agents (FTE, all full-time) per team at the start of the plan, in line with September 2026 staffing.
STARTING_HEADCOUNT = {"AR_BILLING": 116, "AR_TECH": 152, "EN_SUPPORT": 40, "SALES": 63}


@dataclass(frozen=True)
class Shrinkage:
    """Shrinkage components as fractions of paid time.

    Planned items are known in advance (leave, training, coaching, meetings);
    unplanned items are not (sickness, absence, aux overrun). Components are
    compounded, total = 1 - prod(1 - s_i), because each loss applies to the time
    left after the others; simply adding them overstates shrinkage slightly.
    """

    annual_leave: float = round(EGYPT_LABOUR.annual_leave_days / WORKING_DAYS_PER_YEAR, 3)
    training: float = 0.025
    coaching: float = 0.02
    meetings: float = 0.01
    sickness: float = 0.04
    absence: float = 0.03
    aux_overrun: float = 0.015

    PLANNED = ("annual_leave", "training", "coaching", "meetings")
    UNPLANNED = ("sickness", "absence", "aux_overrun")
    # Losses that happen inside a scheduled shift, so the scheduler must cover them.
    IN_DAY = ("coaching", "meetings", "aux_overrun")

    @staticmethod
    def _compound(values: list[float]) -> float:
        return 1.0 - float(np.prod([1.0 - v for v in values]))

    def components(self) -> dict[str, float]:
        return asdict(self)

    @property
    def planned(self) -> float:
        return self._compound([getattr(self, k) for k in self.PLANNED])

    @property
    def unplanned(self) -> float:
        return self._compound([getattr(self, k) for k in self.UNPLANNED])

    @property
    def total(self) -> float:
        return self._compound(list(self.components().values()))

    @property
    def additive_total(self) -> float:
        return float(sum(self.components().values()))

    @property
    def in_day(self) -> float:
        return self._compound([getattr(self, k) for k in self.IN_DAY])

    def table(self) -> pd.DataFrame:
        """Component table for display."""
        rows = [
            {"component": k.replace("_", " ").capitalize(), "type": "Planned" if k in self.PLANNED else "Unplanned", "value": v}
            for k, v in self.components().items()
        ]
        return pd.DataFrame(rows)


def interval_requirement(
    interval_fc: pd.DataFrame,
    sl_target: float = SL_TARGET,
    threshold_s: float = SL_THRESHOLD_S,
    max_occupancy: float = MAX_OCCUPANCY,
) -> pd.DataFrame:
    """Add workload (Erlangs) and Erlang C required agents to an interval forecast."""
    out = interval_fc.copy()
    open_ = out["forecast_aht_s"].notna()
    out["workload_erl"] = np.where(open_, traffic_intensity(out["forecast_offered"], out["forecast_aht_s"].fillna(0)), 0.0)
    out["required_agents"] = 0
    out.loc[open_, "required_agents"] = required_agents(
        out.loc[open_, "forecast_offered"].to_numpy(),
        out.loc[open_, "forecast_aht_s"].to_numpy(),
        sl_target,
        threshold_s,
        max_occupancy,
    )
    return out


def weekly_requirement(requirement: pd.DataFrame, shrinkage: Shrinkage, weekly_hours: float = EGYPT_LABOUR.contract_weekly_hours) -> pd.DataFrame:
    """Weekly required FTE per queue from interval requirements.

    Args:
        requirement: Output of :func:`interval_requirement` (needs date, queue, required_agents, workload_erl).
        shrinkage: Shrinkage assumptions.
        weekly_hours: Contracted paid hours per FTE.

    Returns:
        week_start, queue, offered, workload_hours, agent_hours, required_fte.
    """
    hours = INTERVAL_MINUTES / 60
    req = requirement.assign(
        week_start=week_start(requirement["date"]),
        agent_hours=requirement["required_agents"] * hours,
        workload_hours=requirement["workload_erl"] * hours,
    )
    weekly = req.groupby(["week_start", "queue"], observed=True).agg(
        offered=("forecast_offered", "sum"), workload_hours=("workload_hours", "sum"), agent_hours=("agent_hours", "sum")
    )
    weekly["required_fte"] = weekly["agent_hours"] / (1 - shrinkage.total) / weekly_hours
    return weekly.reset_index()


@dataclass(frozen=True)
class HiringClass:
    """A new-hire class starting training in week index ``start_week`` (0 = first plan week)."""

    start_week: int
    size: int


@dataclass(frozen=True)
class CapacityAssumptions:
    """Supply-side assumptions for the capacity plan."""

    starting_hc: float
    monthly_attrition: float = 0.035
    training_weeks: int = 4
    nesting_weeks: int = 2
    nesting_productivity: float = 0.60
    training_attrition: float = 0.15
    lead_time_weeks: int = 4
    min_class: int = 6
    max_class: int = 25
    classes: tuple[HiringClass, ...] = field(default_factory=tuple)

    @property
    def weekly_attrition(self) -> float:
        """Monthly attrition converted to an equivalent weekly rate."""
        return 1.0 - (1.0 - self.monthly_attrition) ** (12 / 52)


def _class_curves(a: CapacityAssumptions, weeks: int, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per hire in a class: heads remaining, productive FTE and stage label index for each plan week."""
    heads = np.zeros(weeks)
    productive = np.zeros(weeks)
    stage = np.full(weeks, -1)  # -1 not started, 0 training, 1 nesting, 2 proficient
    train_weekly_loss = 1.0 - (1.0 - a.training_attrition) ** (1 / a.training_weeks)
    remaining = 1.0
    for w in range(start, weeks):
        age = w - start
        if age < a.training_weeks:
            stage[w], prod = 0, 0.0
        elif age < a.training_weeks + a.nesting_weeks:
            stage[w], prod = 1, a.nesting_productivity
        else:
            stage[w], prod = 2, 1.0
        heads[w] = remaining
        productive[w] = remaining * prod
        remaining *= 1.0 - (train_weekly_loss if age < a.training_weeks else a.weekly_attrition)
    return heads, productive, stage


def project_capacity(required_fte: pd.Series, assumptions: CapacityAssumptions) -> pd.DataFrame:
    """Project available FTE week by week against the requirement.

    Args:
        required_fte: Weekly required FTE indexed by week start.
        assumptions: Supply assumptions including any hiring classes.

    Returns:
        One row per week with tenured FTE, new-hire heads by stage, available FTE and the gap.
    """
    weeks = len(required_fte)
    tenured = assumptions.starting_hc * (1 - assumptions.weekly_attrition) ** np.arange(weeks)
    in_training = np.zeros(weeks)
    in_nesting = np.zeros(weeks)
    new_productive = np.zeros(weeks)
    new_heads = np.zeros(weeks)
    for cls in assumptions.classes:
        heads, productive, stage = _class_curves(assumptions, weeks, cls.start_week)
        in_training += cls.size * heads * (stage == 0)
        in_nesting += cls.size * heads * (stage == 1)
        new_heads += cls.size * heads
        new_productive += cls.size * productive
    available = tenured + new_productive
    plan = pd.DataFrame(
        {
            "week_start": required_fte.index,
            "required_fte": required_fte.to_numpy(),
            "tenured_fte": tenured,
            "new_hire_fte": new_productive,
            "in_training": in_training,
            "in_nesting": in_nesting,
            "total_headcount": tenured + new_heads,
            "available_fte": available,
        }
    )
    plan["gap_fte"] = plan["available_fte"] - plan["required_fte"]
    plan["gap_pct"] = plan["gap_fte"] / plan["required_fte"]
    return plan


def planning_requirement(required_fte: pd.Series, window: int = 5) -> pd.Series:
    """Centred rolling median of weekly required FTE.

    Permanent hires are planned against this smoothed baseline; single-week
    spikes (billing-cycle weeks, campaign weeks) are covered with overtime and
    flex rather than headcount that would sit idle afterwards. Edge weeks reuse
    the nearest full-window value so a spike at the end of the horizon is not
    mistaken for a new baseline.
    """
    return required_fte.rolling(window, center=True).median().ffill().bfill()


def recommend_hiring(
    required_fte: pd.Series, assumptions: CapacityAssumptions, tolerance: float = 0.02, lookahead_weeks: int = 8
) -> tuple[list[HiringClass], pd.DataFrame]:
    """Greedy hiring plan that closes baseline shortfalls larger than ``tolerance``.

    Works forward in time against :func:`planning_requirement`: for the first week
    with an uncovered shortfall it books a class early enough to be proficient
    that week (never earlier than the recruitment lead time) and sizes it to cover
    the trigger week's shortfall or the average shortfall over the next ``lookahead_weeks``, whichever is larger, allowing for training
    and ongoing attrition. Classes are capped at ``max_class``; a full class rolls
    the remainder into the following week. Shortfalls before the first class can
    graduate cannot be fixed by hiring and stay visible for overtime or cross-skilling.

    Returns:
        The hiring classes and the resulting capacity plan (against the raw requirement).
    """
    baseline = planning_requirement(required_fte)
    ramp = assumptions.training_weeks + assumptions.nesting_weeks
    first_fixable = assumptions.lead_time_weeks + assumptions.training_weeks
    classes: dict[int, int] = {c.start_week: c.size for c in assumptions.classes}
    weeks = len(required_fte)
    unfixable: set[int] = set()
    for _ in range(4 * weeks):
        plan = project_capacity(baseline, replace(assumptions, classes=_as_classes(classes)))
        short = (plan["gap_fte"] < -tolerance * plan["required_fte"]) & (plan.index >= first_fixable)
        short &= ~plan.index.isin(list(unfixable))
        if not short.any():
            break
        week = int(np.flatnonzero(short.to_numpy())[0])
        start = max(assumptions.lead_time_weeks, week - ramp)
        while classes.get(start, 0) >= assumptions.max_class and start < week - assumptions.training_weeks:
            start += 1
        if classes.get(start, 0) >= assumptions.max_class:
            unfixable.add(week)
            continue
        _, productive, _ = _class_curves(assumptions, weeks, start)
        window = slice(week, min(week + lookahead_weeks, weeks))
        need = (-plan["gap_fte"].to_numpy()[window]) / np.maximum(productive[window], 1e-9)
        size = max(math.ceil(max(need[0], need.mean())), assumptions.min_class)
        classes[start] = min(classes.get(start, 0) + size, assumptions.max_class)
    final = _as_classes(classes)
    return list(final), project_capacity(required_fte, replace(assumptions, classes=final))


def _as_classes(classes: dict[int, int]) -> tuple[HiringClass, ...]:
    return tuple(HiringClass(start, size) for start, size in sorted(classes.items()))


def hiring_table(classes: list[HiringClass], week_starts: pd.Series, assumptions: CapacityAssumptions) -> pd.DataFrame:
    """Human-readable hiring plan: class start, nesting and go-live dates and expected graduates."""
    rows = []
    for i, cls in enumerate(classes, 1):
        start = pd.Timestamp(week_starts.iloc[cls.start_week])
        rows.append(
            {
                "class": f"Class {i}",
                "recruit_by": start - pd.Timedelta(weeks=assumptions.lead_time_weeks),
                "training_start": start,
                "nesting_start": start + pd.Timedelta(weeks=assumptions.training_weeks),
                "proficient_from": start + pd.Timedelta(weeks=assumptions.training_weeks + assumptions.nesting_weeks),
                "hires": cls.size,
                "expected_graduates": round(cls.size * (1 - assumptions.training_attrition), 1),
            }
        )
    return pd.DataFrame(rows)
