from __future__ import annotations

import numpy as np

from dexterity.models import PlacedBox
from dexterity.sim.scoring import rotated_half_extents, compute_density, check_stability_single


def test_rotated_half_extents_identity():
    dims = (2.0, 4.0, 6.0)
    hx, hy, hz = rotated_half_extents(dims, (1.0, 0.0, 0.0, 0.0))
    assert np.allclose([hx, hy, hz], [1.0, 2.0, 3.0])


def test_compute_density_simple():
    placed = [
        PlacedBox(
            id="p1",
            position=(0.5, 0.5, 0.5),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            dimensions=(1.0, 1.0, 1.0),
        )
    ]
    density = compute_density(
        placed_boxes=placed,
        candidate_pos=np.array([1.5, 0.5, 0.5]),
        candidate_orient=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_dims=(1.0, 1.0, 1.0),
        truck_width=2.0,
        truck_height=2.0,
    )
    assert np.isclose(density, 2.0 / (2.0 * 2.0 * 2.0))


def test_check_stability_floor_support():
    stable, ratio = check_stability_single(
        box_pos=np.array([0.5, 0.5, 0.5]),
        box_orient=np.array([1.0, 0.0, 0.0, 0.0]),
        box_dims=(1.0, 1.0, 1.0),
        support_boxes=[],
    )
    assert stable is True
    assert np.isclose(ratio, 1.0)
