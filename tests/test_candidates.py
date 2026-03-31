from __future__ import annotations

from dexterity.models import Box, PlacedBox, TruckDims
from dexterity.search import CandidateGenerator
from dexterity.sim.scoring import rotated_half_extents


def test_candidate_generation_bounds_and_collisions():
    truck = TruckDims(depth=2.0, width=2.0, height=2.0)
    placed = [
        PlacedBox(
            id="p1",
            position=(0.5, 0.5, 0.5),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            dimensions=(1.0, 1.0, 1.0),
        )
    ]
    box = Box(id="c", dimensions=(1.0, 1.0, 1.0), weight=1.0)

    gen = CandidateGenerator(grid_spacing=1.0)
    candidates = gen.generate(box, placed, truck)

    assert candidates, "Expected at least one candidate placement"

    for cand in candidates:
        hx, hy, hz = rotated_half_extents(box.dimensions, cand.orientation)
        x, y, z = cand.position
        assert 0.0 <= x - hx <= truck.depth
        assert 0.0 <= y - hy <= truck.width
        assert z + hz <= truck.height + 1e-6

        # Ensure no collision with placed box AABB
        phx, phy, phz = rotated_half_extents(placed[0].dimensions, placed[0].orientation_wxyz)
        px, py, pz = placed[0].position
        overlap_x = min(x + hx, px + phx) - max(x - hx, px - phx)
        overlap_y = min(y + hy, py + phy) - max(y - hy, py - phy)
        overlap_z = min(z + hz, pz + phz) - max(z - hz, pz - phz)
        assert not (overlap_x > 1e-6 and overlap_y > 1e-6 and overlap_z > 1e-6)
