"""Erlang C against textbook values and Erlang A sanity checks."""

import numpy as np
import pytest

from wfm.erlang import (
    erlang_a_metrics,
    erlang_b,
    erlang_c,
    erlang_c_metrics,
    required_agents,
    required_agents_erlang_a,
    traffic_intensity,
)

# Textbook case: 100 calls in 30 minutes, 180 s AHT -> 10 Erlangs.
CALLS, AHT = 100, 180


def test_traffic_intensity():
    assert traffic_intensity(CALLS, AHT)[()] == pytest.approx(10.0)


def test_erlang_b_known_value():
    # B(10, 10) = 0.2146 (standard Erlang B table).
    assert erlang_b(10, 10.0)[0] == pytest.approx(0.2146, abs=1e-4)


@pytest.mark.parametrize(
    "agents, p_wait, sl, asa",
    [(12, 0.4494, 0.6402, 40.44), (13, 0.2853, 0.7956, 17.12), (14, 0.1741, 0.8884, 7.84), (15, 0.1020, 0.9415, 3.67)],
)
def test_erlang_c_textbook(agents, p_wait, sl, asa):
    m = erlang_c_metrics(CALLS, AHT, agents)
    assert m["p_wait"][0] == pytest.approx(p_wait, abs=1e-4)
    assert m["service_level"][0] == pytest.approx(sl, abs=1e-4)
    assert m["asa_s"][0] == pytest.approx(asa, abs=0.01)
    assert m["occupancy"][0] == pytest.approx(10 / agents)


def test_required_agents_80_20():
    # 13 agents gives 79.6% (just misses), so the answer is 14.
    assert required_agents(CALLS, AHT)[0] == 14


def test_required_agents_respects_occupancy_cap():
    # At high load the occupancy cap, not the SL target, binds.
    calls, aht = 2000, 300
    load = traffic_intensity(calls, aht)[()]
    n = required_agents(calls, aht, max_occupancy=0.88)[0]
    assert load / n <= 0.88
    assert required_agents(calls, aht, max_occupancy=0.99)[0] < n


def test_required_agents_vectorised_and_zero():
    out = required_agents(np.array([0, 100, 100]), np.array([180, 180, 360]))
    assert out.tolist()[0] == 0
    assert out[2] > out[1]


def test_erlang_c_unstable_queue():
    assert erlang_c(9, 10.0)[0] == 1.0
    assert erlang_c_metrics(CALLS, AHT, 9)["service_level"][0] == 0.0


def test_erlang_a_converges_to_erlang_c_with_infinite_patience():
    a = erlang_a_metrics(CALLS, AHT, [12, 13, 14], patience_s=1e8)
    c = erlang_c_metrics(CALLS, AHT, [12, 13, 14])
    np.testing.assert_allclose(a["service_level"], c["service_level"], atol=1e-4)
    np.testing.assert_allclose(a["asa_s"], c["asa_s"], rtol=1e-3)
    assert np.all(a["abandon_rate"] < 1e-4)


def test_erlang_a_littles_law_for_abandonment():
    # Abandonment rate = theta * E[queue length] / lambda.
    patience = 120.0
    m = erlang_a_metrics(CALLS, AHT, np.arange(6, 16), patience)
    np.testing.assert_allclose(m["abandon_rate"], m["avg_queue"] / patience / (CALLS / 1800), rtol=1e-6)


def test_erlang_a_monotonic_in_agents():
    m = erlang_a_metrics(CALLS, AHT, np.arange(5, 20), 120)
    assert np.all(np.diff(m["service_level"]) > 0)
    assert np.all(np.diff(m["abandon_rate"]) < 0)


def test_erlang_a_handles_overload_and_zero_agents():
    m = erlang_a_metrics([300, 50, 0], [400, 400, 400], [10, 0, 5], 120)
    assert 0.5 < m["abandon_rate"][0] < 1.0  # heavy overload: most callers abandon
    assert m["abandon_rate"][1] == pytest.approx(1.0)  # nobody logged in
    assert m["service_level"][2] == 1.0  # no calls offered


def test_erlang_c_overstaffs_vs_erlang_a():
    calls = np.array([50, 100, 200, 400])
    c = required_agents(calls, 300)
    a = required_agents_erlang_a(calls, 300, patience_s=180)
    assert np.all(a <= c)
    assert (c - a).sum() > 0
