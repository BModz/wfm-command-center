"""Cross-skilling two queues: discrete-event simulation of dedicated plus flexible agents.

Erlang formulas assume one skill group. When some agents can take both Arabic
billing and Arabic technical calls, the queues share capacity and need fewer
agents in total (pooling). There is no closed form for partial cross-skilling,
so this module simulates it.

Model:
- Two Poisson arrival streams (A and B), exponential handle times and
  exponential caller patience (callers abandon, as in Erlang A).
- Dedicated A agents take only A calls, dedicated B agents only B calls, and
  flexible agents take both. Flexible agents are slower on queue B, their
  secondary skill (``flex_aht_penalty``).
- Routing: a new call goes to an idle dedicated agent of its skill first, then
  to an idle flexible agent. A freed dedicated agent takes the head of its own
  queue; a freed flexible agent takes whichever queue's head call has waited longest.
- Service level is answered within T seconds / offered, measured after a warm-up.
  Common random numbers (the same call stream for every staffing level) keep
  comparisons between staffing levels stable.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfm.config import INTERVAL_SECONDS, SL_TARGET, SL_THRESHOLD_S
from wfm.erlang import required_agents, required_agents_erlang_a, traffic_intensity

FLEX_SHARES = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


@dataclass(frozen=True)
class QueueLoad:
    """Demand for one queue in a 30-minute interval."""

    calls: float
    aht_s: float
    patience_s: float

    @property
    def erlangs(self) -> float:
        return float(traffic_intensity(self.calls, self.aht_s)[()])


@dataclass(frozen=True)
class CallStream:
    """Pre-drawn calls for both queues (common random numbers across staffing levels)."""

    arrival: np.ndarray
    skill: np.ndarray  # 0 = queue A, 1 = queue B
    work: np.ndarray  # handle time before any flexible-agent penalty
    patience: np.ndarray
    measure_from: float


def draw_calls(a: QueueLoad, b: QueueLoad, hours: float = 6.0, warmup_h: float = 0.5, seed: int = 0) -> CallStream:
    """Draw a stationary call stream at the interval's rates for ``hours`` plus warm-up."""
    rng = np.random.default_rng(seed)
    horizon = (hours + warmup_h) * 3600
    parts = []
    for skill, q in enumerate((a, b)):
        rate = q.calls / INTERVAL_SECONDS
        n = rng.poisson(rate * horizon)
        parts.append(
            pd.DataFrame(
                {
                    "arrival": np.sort(rng.uniform(0, horizon, n)),
                    "skill": skill,
                    "work": rng.exponential(q.aht_s, n),
                    "patience": rng.exponential(q.patience_s, n),
                }
            )
        )
    calls = pd.concat(parts).sort_values("arrival", kind="stable")
    return CallStream(
        calls["arrival"].to_numpy(),
        calls["skill"].to_numpy(),
        calls["work"].to_numpy(),
        calls["patience"].to_numpy(),
        warmup_h * 3600,
    )


def simulate(
    stream: CallStream,
    dedicated_a: int,
    dedicated_b: int,
    flexible: int,
    flex_aht_penalty: float = 0.05,
    threshold_s: float = SL_THRESHOLD_S,
) -> dict[str, float]:
    """Run the two-skill simulation and return service level and abandon rate per queue."""
    idle = [dedicated_a, dedicated_b]
    idle_flex = flexible
    queues: list[deque[int]] = [deque(), deque()]
    busy: list[tuple[float, int, int]] = []  # (finish time, agent kind: 0/1 dedicated skill or 2 flex, call)
    offered = np.zeros(2)
    answered_sl = np.zeros(2)
    abandoned = np.zeros(2)
    arr, skill, work, pat = stream.arrival, stream.skill, stream.work, stream.patience
    start_at = stream.measure_from

    def start(call: int, kind: int, now: float) -> None:
        s = skill[call]
        wait = now - arr[call]
        if arr[call] >= start_at and wait <= threshold_s:
            answered_sl[s] += 1
        duration = work[call] * (1 + flex_aht_penalty if kind == 2 and s == 1 else 1)
        heapq.heappush(busy, (now + duration, kind, call))

    def head(q: deque[int], now: float) -> int | None:
        """Oldest call still waiting, dropping callers whose patience ran out."""
        while q:
            c = q[0]
            if arr[c] + pat[c] < now:
                q.popleft()
                if arr[c] >= start_at:
                    abandoned[skill[c]] += 1
                continue
            return c
        return None

    for i in range(arr.size):
        now = arr[i]
        while busy and busy[0][0] <= now:
            t, kind, _ = heapq.heappop(busy)
            if kind == 2:
                nxt = _flex_pick(queues, head, t)
                if nxt is None:
                    idle_flex += 1
                else:
                    start(nxt, 2, t)
            else:
                nxt = head(queues[kind], t)
                if nxt is None:
                    idle[kind] += 1
                else:
                    queues[kind].popleft()
                    start(nxt, kind, t)
        s = skill[i]
        if arr[i] >= start_at:
            offered[s] += 1
        if idle[s] > 0:
            idle[s] -= 1
            start(i, s, now)
        elif idle_flex > 0:
            idle_flex -= 1
            start(i, 2, now)
        else:
            queues[s].append(i)
    end = arr[-1] if arr.size else 0.0
    for q in queues:
        for c in q:
            if arr[c] >= start_at and arr[c] + pat[c] < end:
                abandoned[skill[c]] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = np.where(offered > 0, answered_sl / offered, 1.0)
        ab = np.where(offered > 0, abandoned / offered, 0.0)
    return {"sl_a": float(sl[0]), "sl_b": float(sl[1]), "abandon_a": float(ab[0]), "abandon_b": float(ab[1])}


def _flex_pick(queues: list[deque[int]], head, t: float) -> int | None:
    """A free flexible agent takes whichever queue's head call has waited longest."""
    heads = [head(queues[0], t), head(queues[1], t)]
    candidates = [(c, s) for s, c in enumerate(heads) if c is not None]
    if not candidates:
        return None
    call, s = min(candidates, key=lambda cs: cs[0])  # lower index = earlier arrival
    queues[s].popleft()
    return call


def _meets(res: dict[str, float], target: float) -> bool:
    return res["sl_a"] >= target and res["sl_b"] >= target


def _split(total: int, flex: int, a: QueueLoad, b: QueueLoad) -> tuple[int, int]:
    """Split dedicated agents between the queues in proportion to workload."""
    dedicated = total - flex
    share = a.erlangs / (a.erlangs + b.erlangs)
    da = int(round(dedicated * share))
    return da, dedicated - da


def min_staffing(
    a: QueueLoad,
    b: QueueLoad,
    flex_share: float,
    stream: CallStream,
    flex_aht_penalty: float = 0.05,
    target: float = SL_TARGET,
) -> dict[str, float]:
    """Smallest total team meeting the SL target on both queues for a given cross-skilled share.

    With ``flex_share = 0`` each queue is staffed on its own. Otherwise the total
    is searched upwards from the fully pooled Erlang C estimate minus a margin;
    flexible agents are ``round(flex_share x total)`` and the rest are split by workload.
    """
    if flex_share <= 0:
        out = {}
        for idx, (name, q) in enumerate((("a", a), ("b", b))):
            n = max(int(required_agents_erlang_a(q.calls, q.aht_s, q.patience_s)[0]) - 2, 1)
            for _ in range(60):
                res = simulate(stream, n if idx == 0 else 10**6, 10**6 if idx == 0 else n, 0, flex_aht_penalty)
                if res[f"sl_{name}"] >= target:
                    break
                n += 1
            out[name] = n
        res = simulate(stream, out["a"], out["b"], 0, flex_aht_penalty)
        return {"total": out["a"] + out["b"], "dedicated_a": out["a"], "dedicated_b": out["b"], "flexible": 0, **res}

    pooled_calls = a.calls + b.calls
    pooled_aht = (a.calls * a.aht_s + b.calls * b.aht_s) / pooled_calls
    total = max(int(required_agents(pooled_calls, pooled_aht)[0]) - 6, 2)
    for _ in range(80):
        flex = min(total, max(1, int(round(flex_share * total))))
        da, db = _split(total, flex, a, b)
        res = simulate(stream, da, db, flex, flex_aht_penalty)
        if _meets(res, target):
            return {"total": total, "dedicated_a": da, "dedicated_b": db, "flexible": flex, **res}
        total += 1
    raise RuntimeError("No staffing level met the target")


def compare_shares(
    a: QueueLoad,
    b: QueueLoad,
    shares: tuple[float, ...] = FLEX_SHARES,
    flex_aht_penalty: float = 0.05,
    hours: float = 6.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimum agents for each cross-skilled share in one interval (same call stream for all)."""
    stream = draw_calls(a, b, hours=hours, seed=seed)
    rows = [{"flex_share": f, **min_staffing(a, b, f, stream, flex_aht_penalty)} for f in shares]
    out = pd.DataFrame(rows)
    out["saving_vs_dedicated"] = 1 - out["total"] / out.loc[out["flex_share"] == 0, "total"].iloc[0]
    out["erlang_c_separate"] = int(required_agents(a.calls, a.aht_s)[0] + required_agents(b.calls, b.aht_s)[0])
    return out


def day_comparison(
    day: pd.DataFrame,
    patience_a: float,
    patience_b: float,
    shares: tuple[float, ...] = FLEX_SHARES,
    flex_aht_penalty: float = 0.05,
    step: int = 2,
    hours: float = 4.0,
) -> pd.DataFrame:
    """Agent-hours for each cross-skilled share over a day, simulating every ``step``-th interval.

    Args:
        day: One row per interval with columns interval, calls_a, aht_a, calls_b, aht_b.
        patience_a: Mean caller patience on queue A.
        patience_b: Mean caller patience on queue B.
        shares: Cross-skilled shares to compare.
        flex_aht_penalty: Extra handle time when a flexible agent takes a queue-B call.
        step: Simulate every ``step``-th interval and let it represent ``step`` intervals.
        hours: Simulated hours per interval.
    """
    rows = []
    for rec in day.iloc[::step].itertuples():
        a = QueueLoad(rec.calls_a, rec.aht_a, patience_a)
        b = QueueLoad(rec.calls_b, rec.aht_b, patience_b)
        cmp = compare_shares(a, b, shares, flex_aht_penalty, hours=hours, seed=int(rec.interval))
        cmp["interval"] = rec.interval
        cmp["weight_intervals"] = step
        rows.append(cmp)
    out = pd.concat(rows, ignore_index=True)
    out["agent_hours"] = out["total"] * out["weight_intervals"] * INTERVAL_SECONDS / 3600
    return out


def summarise_day(detail: pd.DataFrame) -> pd.DataFrame:
    """Total agent-hours per cross-skilled share and saving against fully dedicated teams."""
    total = detail.groupby("flex_share")["agent_hours"].sum().reset_index()
    base = total.loc[total["flex_share"] == 0, "agent_hours"].iloc[0]
    total["saving"] = 1 - total["agent_hours"] / base
    total["fte_saved"] = (base - total["agent_hours"]) * 7 / 40  # same day-shape for a 7-day week, 40h FTE
    return total


def pooling_gain_bound(a: QueueLoad, b: QueueLoad) -> float:
    """Erlang C saving from fully pooling the two queues (upper bound on the cross-skilling gain)."""
    separate = int(required_agents(a.calls, a.aht_s)[0] + required_agents(b.calls, b.aht_s)[0])
    pooled_calls = a.calls + b.calls
    pooled = int(required_agents(pooled_calls, (a.calls * a.aht_s + b.calls * b.aht_s) / pooled_calls)[0])
    return 1 - pooled / separate if separate else 0.0

