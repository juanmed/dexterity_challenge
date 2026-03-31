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
            return PlacementDecision(
                position=(0.0, 0.0, current_box.dimensions[2] / 2.0),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        positions = np.array([c.position for c in candidates], dtype=np.float32)
        orientations = np.array([c.orientation for c in candidates], dtype=np.float32)
        result = self._sim.evaluate_batch(scene, current_box, positions, orientations)

        valid_mask = result.is_stable
        if not valid_mask.any():
            best_idx = int(result.n_displaced.argmin())
        else:
            valid_scores = np.where(valid_mask, result.density, -np.inf)
            best_idx = int(valid_scores.argmax())

        return PlacementDecision(
            position=tuple(result.settled_positions[best_idx]),
            orientation_wxyz=tuple(result.settled_orientations[best_idx]),
        )
