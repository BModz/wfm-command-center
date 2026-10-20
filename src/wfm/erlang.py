"""Queueing models for contact-centre staffing: Erlang C and Erlang A.

All functions are vectorised: pass scalars or equal-length arrays (one entry per
interval) and get arrays back.

Erlang C (M/M/N) assumes callers wait forever. Erlang A (M/M/N+M) adds
exponentially distributed caller patience, so callers abandon and the queue
cannot explode. Erlang A matches real contact centres better, and Erlang C
therefore tends to overstate the agents needed.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from wfm.config import INTERVAL_SECONDS, MAX_OCCUPANCY, SL_TARGET, SL_THRESHOLD_S

ArrayLike = float | int | np.ndarray


def traffic_intensity(calls: ArrayLike, aht_s: ArrayLike, interval_s: float = INTERVAL_SECONDS) -> np.ndarray:
    """Offered load in Erlangs: A = calls x AHT / interval length."""
    return np.asarray(calls, dtype=float) * np.asarray(aht_s, dtype=float) / interval_s


def erlang_b(agents: ArrayLike, load: ArrayLike) -> np.ndarray:
    """Erlang B blocking probability via the stable recursion B(n) = A B(n-1) / (n + A B(n-1))."""
    n = np.atleast_1d(np.asarray(agents, dtype=int))
    a = np.atleast_1d(np.asarray(load, dtype=float))
    n, a = np.broadcast_arrays(n, a)
    b = np.ones_like(a)
    out = np.where(n <= 0, 1.0, 0.0)
    for k in range(1, int(n.max(initial=0)) + 1):
        b = a * b / (k + a * b)
        out = np.where(n == k, b, out)
    return out


def erlang_c(agents: ArrayLike, load: ArrayLike) -> np.ndarray:
    """Erlang C probability that an arriving call has to wait (1 when agents <= load)."""
    n = np.atleast_1d(np.asarray(agents, dtype=float))
    a = np.atleast_1d(np.asarray(load, dtype=float))
    n, a = np.broadcast_arrays(n, a)
    b = erlang_b(n.astype(int), a)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = n * b / (n - a * (1 - b))
    return np.where((n > a) & (a > 0), c, np.where(a > 0, 1.0, 0.0))


def erlang_c_metrics(
    calls: ArrayLike,
    aht_s: ArrayLike,
    agents: ArrayLike,
    threshold_s: float = SL_THRESHOLD_S,
    interval_s: float = INTERVAL_SECONDS,
) -> dict[str, np.ndarray]:
    """Erlang C service level, ASA, probability of wait and occupancy.

    Args:
        calls: Offered calls per interval.
        aht_s: Average handle time in seconds.
        agents: Agents logged in and available to take calls (net of shrinkage).
        threshold_s: Service-level answer threshold T (e.g. 20 for 80/20).
        interval_s: Interval length in seconds.

    Returns:
        Dict with ``p_wait``, ``service_level``, ``asa_s`` and ``occupancy`` arrays.
        When agents <= load the queue is unstable: SL is 0 and ASA infinite.
    """
    a = traffic_intensity(calls, aht_s, interval_s)
    n = np.asarray(agents, dtype=float)
    aht = np.asarray(aht_s, dtype=float)
    a, n, aht = np.broadcast_arrays(np.atleast_1d(a), np.atleast_1d(n), np.atleast_1d(aht))
    pw = erlang_c(n, a)
    stable = n > a
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sl = np.where(stable, 1 - pw * np.exp(-(n - a) * threshold_s / aht), 0.0)
        asa = np.where(stable, pw * aht / (n - a), np.inf)
        occ = np.where(n > 0, np.minimum(a / n, 1.0), 0.0)
    sl = np.where(a == 0, 1.0, sl)
    asa = np.where(a == 0, 0.0, asa)
    return {"p_wait": pw, "service_level": sl, "asa_s": asa, "occupancy": occ}


def required_agents(
    calls: ArrayLike,
    aht_s: ArrayLike,
    sl_target: float = SL_TARGET,
    threshold_s: float = SL_THRESHOLD_S,
    max_occupancy: float = MAX_OCCUPANCY,
    interval_s: float = INTERVAL_SECONDS,
) -> np.ndarray:
    """Minimum agents so Erlang C SL >= target and occupancy <= the occupancy cap.

    Searches upward from the smallest stable agent count, updating Erlang B
    incrementally so thousands of intervals are solved at once.
    """
    a = np.atleast_1d(traffic_intensity(calls, aht_s, interval_s))
    aht = np.broadcast_to(np.atleast_1d(np.asarray(aht_s, dtype=float)), a.shape)
    n = np.maximum(np.floor(a) + 1, np.ceil(a / max_occupancy)).astype(int)
    b = erlang_b(n, a)
    result = np.where(a <= 0, 0, -1)
    for _ in range(1000):
        pending = result < 0
        if not pending.any():
            break
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pw = n * b / (n - a * (1 - b))
            sl = 1 - pw * np.exp(-(n - a) * threshold_s / aht)
        result = np.where(pending & (sl >= sl_target), n, result)
        n = n + 1
        b = a * b / (n + a * b)
    return result


def erlang_a_metrics(
    calls: ArrayLike,
    aht_s: ArrayLike,
    agents: ArrayLike,
    patience_s: ArrayLike,
    threshold_s: float = SL_THRESHOLD_S,
    interval_s: float = INTERVAL_SECONDS,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Erlang A (M/M/N+M) metrics from the birth-death steady state.

    The system state n (calls in system) has arrival rate lambda and departure rate
    mu*min(n, N) + theta*max(n - N, 0), where theta = 1 / mean patience. A caller who
    finds j callers queued ahead is answered only if each stage completes before
    they abandon; the waiting time to answer is the sum of those exponential
    stages. The answer-within-T probability is computed by uniformisation.

    Service level is defined as calls answered within T divided by calls offered,
    so abandoned calls count against it.

    Returns:
        Dict with ``p_wait``, ``abandon_rate``, ``service_level``, ``asa_s``
        (average wait of answered calls), ``avg_queue`` and ``occupancy``.
    """
    calls, aht, n_agents, patience = np.broadcast_arrays(
        np.atleast_1d(np.asarray(calls, dtype=float)),
        np.atleast_1d(np.asarray(aht_s, dtype=float)),
        np.atleast_1d(np.asarray(agents, dtype=float)).round().astype(int),
        np.atleast_1d(np.asarray(patience_s, dtype=float)),
    )
    size = calls.size
    out = {k: np.zeros(size) for k in ("p_wait", "abandon_rate", "service_level", "asa_s", "avg_queue", "occupancy")}
    out["service_level"][:] = 1.0
    lam = calls / interval_s
    mu = 1.0 / aht
    theta = 1.0 / patience
    active = np.flatnonzero(calls > 0)
    # Queue positions needed: mean overload queue plus a generous tail. Sorting by it keeps chunks compact.
    q_need = np.ceil(np.maximum(lam - n_agents * mu, 0) / theta + 8 * np.sqrt(lam / theta) + 20).astype(int)
    active = active[np.argsort(q_need[active], kind="stable")]
    for start in range(0, active.size, chunk_size):
        idx = active[start : start + chunk_size]
        res = _erlang_a_chunk(lam[idx], mu[idx], theta[idx], n_agents[idx], int(q_need[idx].max()), threshold_s)
        for key, values in res.items():
            out[key][idx] = values
    return out


def _erlang_a_chunk(
    lam: np.ndarray, mu: np.ndarray, theta: np.ndarray, n: np.ndarray, q: int, t: float
) -> dict[str, np.ndarray]:
    """Erlang A metrics for one chunk of intervals sharing a queue-length truncation q."""
    rows = lam.size
    k_max = int(n.max()) + q
    k = np.arange(1, k_max + 1)
    death = mu[:, None] * np.minimum(k, n[:, None]) + theta[:, None] * np.maximum(k - n[:, None], 0)
    log_p = np.concatenate([np.zeros((rows, 1)), np.cumsum(np.log(lam[:, None]) - np.log(death), axis=1)], axis=1)
    log_p -= logsumexp(log_p, axis=1, keepdims=True)
    p = np.exp(log_p)

    states = np.arange(k_max + 1)
    p_no_wait = np.where(states[None, :] < n[:, None], p, 0.0).sum(axis=1)
    queue_len = np.maximum(states[None, :] - n[:, None], 0)
    avg_queue = (p * queue_len).sum(axis=1)

    # pi[j]: probability an arrival finds j callers queued ahead (PASTA).
    j = np.arange(q)
    pi = np.take_along_axis(p, np.minimum(n[:, None] + j[None, :], k_max), axis=1)
    succ = n[:, None] * mu[:, None] + j[None, :] * theta[:, None]
    total = succ + theta[:, None]
    p_served_j = np.cumprod(succ / total, axis=1)
    wait_served_j = np.cumsum(1.0 / total, axis=1)

    # Uniformisation: u[j] = P(answered within m jumps | j ahead); weight by Poisson(m; Lambda t).
    big_lambda = total[:, -1]
    lt = big_lambda * t
    m_max = int(np.ceil((lt + 8 * np.sqrt(lt) + 10).max()))
    step_succ = succ / big_lambda[:, None]
    step_stay = 1.0 - total / big_lambda[:, None]
    u = np.zeros((rows, q))
    within_t = np.zeros((rows, q))
    weights = poisson.pmf(np.arange(m_max + 1)[None, :], lt[:, None])
    for m in range(1, m_max + 1):
        ahead = np.concatenate([np.ones((rows, 1)), u[:, :-1]], axis=1)
        u = step_succ * ahead + step_stay * u
        within_t += weights[:, m : m + 1] * u

    served_q = (pi * p_served_j).sum(axis=1)
    p_served = p_no_wait + served_q
    with np.errstate(divide="ignore", invalid="ignore"):
        asa = np.where(p_served > 0, (pi * p_served_j * wait_served_j).sum(axis=1) / p_served, 0.0)
        occupancy = np.where(n > 0, np.minimum(lam * p_served / (n * mu), 1.0), 0.0)
    return {
        "p_wait": 1.0 - p_no_wait,
        "abandon_rate": np.clip(1.0 - p_served, 0.0, 1.0),
        "service_level": np.clip(p_no_wait + (pi * within_t).sum(axis=1), 0.0, 1.0),
        "asa_s": asa,
        "avg_queue": avg_queue,
        "occupancy": occupancy,
    }


def required_agents_erlang_a(
    calls: ArrayLike,
    aht_s: ArrayLike,
    patience_s: ArrayLike,
    sl_target: float = SL_TARGET,
    threshold_s: float = SL_THRESHOLD_S,
    max_occupancy: float = MAX_OCCUPANCY,
    interval_s: float = INTERVAL_SECONDS,
) -> np.ndarray:
    """Minimum agents so the Erlang A service level meets target under the occupancy cap."""
    calls = np.atleast_1d(np.asarray(calls, dtype=float))
    aht = np.broadcast_to(np.atleast_1d(np.asarray(aht_s, dtype=float)), calls.shape)
    patience = np.broadcast_to(np.atleast_1d(np.asarray(patience_s, dtype=float)), calls.shape)
    a = traffic_intensity(calls, aht, interval_s)
    n = np.maximum(np.ceil(a / max_occupancy), np.floor(0.6 * a)).astype(int)
    result = np.where(calls <= 0, 0, -1)
    for _ in range(1000):
        pending = np.flatnonzero(result < 0)
        if pending.size == 0:
            break
        sl = erlang_a_metrics(calls[pending], aht[pending], n[pending], patience[pending], threshold_s, interval_s)[
            "service_level"
        ]
        met = sl >= sl_target
        result[pending[met]] = n[pending[met]]
        n[pending[~met]] += 1
    return result
