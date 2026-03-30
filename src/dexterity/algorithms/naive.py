from __future__ import annotations

from ..models import Box, PlacedBox, PlacementDecision, TruckDims
from .base import Algorithm


class NaiveAlgorithm(Algorithm):
    """Places boxes at the origin with identity orientation."""

    def __init__(self):
        self._truck: TruckDims | None = None
        self._step = 0

    def setup(self, truck: TruckDims) -> None:
        self._truck = truck
        self._step = 0

    def decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        assert self._truck is not None, "setup() must be called before decide()"

        # Simple stack: place along the depth axis based on step count
        # Each box is placed at a fixed grid position (naive, ignores collisions)
        bx, by, bz = current_box.dimensions
        x = (self._step % 5) * bx
        y = (self._step // 5 % 5) * by
        z = (self._step // 25) * bz

        # Clamp to truck bounds
        x = min(x, self._truck.depth - bx / 2)
        y = min(y, self._truck.width - by / 2)
        z = min(z, self._truck.height - bz / 2)

        self._step += 1

        return PlacementDecision(
            position=(x, y, z),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
