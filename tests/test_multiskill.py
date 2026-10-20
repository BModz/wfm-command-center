"""Cross-skilling simulation: validation against Erlang A and pooling behaviour."""

import numpy as np
import pytest

from wfm.erlang import erlang_a_metrics
from wfm.multiskill import QueueLoad, compare_shares, draw_calls, simulate

BILLING = QueueLoad(120, 270, 300)
TECH = QueueLoad(100, 410, 360)


def test_dedicated_simulation_matches_erlang_a():
    stream = draw_calls(BILLING, TECH, hours=200, seed=5)
    sim = simulate(stream, 20, 25, 0)
    exact = erlang_a_metrics([120, 100], [270, 410], [20, 25], [300, 360])
    assert sim["sl_a"] == pytest.approx(exact["service_level"][0], abs=0.025)
    assert sim["sl_b"] == pytest.approx(exact["service_level"][1], abs=0.025)
    assert sim["abandon_a"] == pytest.approx(exact["abandon_rate"][0], abs=0.01)


def test_simulation_is_reproducible():
    a = simulate(draw_calls(BILLING, TECH, hours=4, seed=9), 18, 22, 4)
    b = simulate(draw_calls(BILLING, TECH, hours=4, seed=9), 18, 22, 4)
    assert a == b


def test_flexible_agents_help_the_busier_queue():
    stream = draw_calls(BILLING, TECH, hours=40, seed=2)
    dedicated = simulate(stream, 20, 22, 0)
    with_flex = simulate(stream, 18, 20, 4)  # same headcount, four of them cross-skilled
    assert with_flex["sl_b"] > dedicated["sl_b"]


def test_cross_skilling_needs_no_more_agents():
    small_a, small_b = QueueLoad(20, 270, 300), QueueLoad(15, 410, 360)
    table = compare_shares(small_a, small_b, shares=(0.0, 0.3, 1.0), hours=40)
    totals = table.set_index("flex_share")["total"]
    assert totals[0.3] <= totals[0.0] and totals[1.0] <= totals[0.0]
    assert np.all(table[["sl_a", "sl_b"]].to_numpy() >= 0.8)
