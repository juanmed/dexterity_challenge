from __future__ import annotations

import pytest

from dexterity.algorithms.naive import NaiveAlgorithm
from dexterity.models import Box, TruckDims


def make_truck() -> TruckDims:
    return TruckDims(depth=2.0, width=2.6, height=2.75)


def make_box(bid: str = "b1") -> Box:
    return Box(id=bid, dimensions=(1.0, 1.0, 1.0), weight=1.0)


def test_naive_algo_returns_within_truck_bounds():
    algo = NaiveAlgorithm()
    truck = make_truck()
    algo.setup(truck)

    for i in range(30):
        box = make_box(f"b{i}")
        decision = algo.decide(box, [], 100 - i, 0.0)
        x, y, z = decision.position
        # Position must be non-negative and within truck (approximately)
        assert x >= 0
        assert y >= 0
        assert z >= 0
        assert x <= truck.depth
        assert y <= truck.width
        assert z <= truck.height


def test_naive_algo_orientation_is_unit_quaternion():
    algo = NaiveAlgorithm()
    algo.setup(make_truck())
    decision = algo.decide(make_box(), [], 10, 0.0)
    w, x, y, z = decision.orientation_wxyz
    magnitude = (w**2 + x**2 + y**2 + z**2) ** 0.5
    assert abs(magnitude - 1.0) < 1e-9


def test_naive_algo_no_stop_signal():
    algo = NaiveAlgorithm()
    algo.setup(make_truck())
    decision = algo.decide(make_box(), [], 10, 0.0)
    assert decision.stop is False
