from __future__ import annotations

import mujoco

from dexterity.models import Box, PlacedBox, TruckDims
from dexterity.sim.mjcf_builder import build_truck_model, get_candidate_qpos_offset, get_frozen_indices


def _make_box(idx: int) -> PlacedBox:
    return PlacedBox(
        id=f"b{idx}",
        position=(0.1 * idx, 0.1, 0.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        dimensions=(0.2, 0.2, 0.2),
    )


def test_build_truck_model_sparse_jacobian():
    truck = TruckDims(depth=2.0, width=2.0, height=2.0)
    placed = [_make_box(i) for i in range(11)]
    candidate = Box(id="c", dimensions=(0.2, 0.2, 0.2), weight=1.0)

    mjm, mjd = build_truck_model(truck, placed, candidate)
    assert mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE


def test_offsets_and_frozen_indices():
    assert get_candidate_qpos_offset(3) == 21
    assert get_frozen_indices(5, [3, 4]) == [0, 1, 2]
