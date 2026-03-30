from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Box, PlacedBox, PlacementDecision, TruckDims


class Algorithm(ABC):
    def setup(self, truck: TruckDims) -> None:
        """Called once per game with truck dims from /start response."""

    @abstractmethod
    def decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        """Return a placement decision. May be sync or async."""

    def teardown(self, final_density: float, termination_reason: str | None) -> None:
        """Called once after game ends."""
