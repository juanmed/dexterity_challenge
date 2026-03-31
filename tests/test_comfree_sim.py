from __future__ import annotations

import pytest

from dexterity.models import Box, TruckDims
from dexterity.sim import ComFreeSimulator, SimConfig


wp = pytest.importorskip("warp")


def test_comfree_sim_basic_settle():
    if not wp.get_device().is_cuda:
        pytest.skip("CUDA device not available for comfree_warp")

    sim = ComFreeSimulator(SimConfig(n_candidates=2, max_settle_steps=10))
    truck = TruckDims(depth=2.0, width=2.0, height=2.0)
    scene = sim.build_scene(truck, placed_boxes=[])
    box = Box(id="c", dimensions=(0.2, 0.2, 0.2), weight=1.0)

    positions = [[0.5, 0.5, 0.5], [1.0, 1.0, 0.5]]
    orientations = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    result = sim.evaluate_batch(scene, box, positions, orientations, n_settle_steps=10)

    assert result.settled_positions.shape == (2, 3)
    assert result.settled_orientations.shape == (2, 4)
    assert result.density.shape == (2,)
