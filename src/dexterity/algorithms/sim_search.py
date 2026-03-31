from __future__ import annotations

import numpy as np

from ..models import Box, PlacedBox, PlacementDecision, TruckDims
from ..search import CandidateGenerator
from ..sim import ComFreeSimulator, PhysicsSim, SimConfig
from .base import Algorithm


class SimSearchAlgorithm(Algorithm):
    def __init__(self, sim_config: SimConfig | None = None) -> None:
        self._config = sim_config or SimConfig()
        self._sim: PhysicsSim = ComFreeSimulator(self._config)
        self._candidates = CandidateGenerator(grid_spacing=self._config.grid_spacing)
        self._truck: TruckDims | None = None

    def setup(self, truck: TruckDims) -> None:
        self._truck = truck
        self._sim.build_scene(truck, placed_boxes=[])

    def teardown(self, final_density: float, termination_reason: str | None) -> None:
        close = getattr(self._sim, "close", None)
        if callable(close):
            close()

    def decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        assert self._truck is not None, "setup() must be called before decide()"

        scene = self._sim.build_scene(self._truck, placed_boxes)
        candidates = self._candidates.generate(current_box, placed_boxes, self._truck)
        if not candidates:
            hx, hy, hz = (d / 2.0 for d in current_box.dimensions)
            return PlacementDecision(
                position=(hx, hy, hz),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        positions = np.array([c.position for c in candidates], dtype=np.float32)
        orientations = np.array([c.orientation for c in candidates], dtype=np.float32)
        result = self._sim.evaluate_batch(scene, current_box, positions, orientations)

        finite_mask = (
            np.isfinite(result.settled_positions).all(axis=1)
            & np.isfinite(result.settled_orientations).all(axis=1)
            & np.isfinite(result.density)
        )
        stable_mask = result.is_stable & finite_mask

        if stable_mask.any():
            stable_scores = np.where(stable_mask, result.density, -np.inf)
            best_idx = int(stable_scores.argmax())
        elif finite_mask.any():
            displaced = np.where(finite_mask, result.n_displaced, np.iinfo(result.n_displaced.dtype).max)
            best_idx = int(displaced.argmin())
        else:
            # Simulator produced only invalid candidates (NaN/Inf); fall back to a safe default.
            return PlacementDecision(
                position=(0.0, 0.0, current_box.dimensions[2] / 2.0),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        chosen = candidates[best_idx]
        # Submit the *candidate* pose (what we control), not the post-settle pose
        # from the simulator (which may drift slightly and can go out-of-bounds).
        return PlacementDecision(position=chosen.position, orientation_wxyz=chosen.orientation)
