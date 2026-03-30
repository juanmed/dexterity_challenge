from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from ..models import Box, PlacedBox, PlacementDecision, TruckDims
from .base import Algorithm

logger = logging.getLogger(__name__)


class CppSoAlgorithm(Algorithm):
    """Adapter for a pybind11 .so that exposes setup/decide/teardown."""

    def __init__(self, so_path: str | Path, *, use_executor: bool = False):
        so_path = Path(so_path)
        if not so_path.exists():
            raise FileNotFoundError(f"C++ .so not found: {so_path}")

        spec = importlib.util.spec_from_file_location("dex_cpp_algo", so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._impl = module.Algorithm()
        self._use_executor = use_executor

    def setup(self, truck: TruckDims) -> None:
        self._impl.setup(truck.depth, truck.width, truck.height)

    def decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        placed_dicts = [
            {
                "id": pb.id,
                "dimensions": list(pb.dimensions),
                "position": list(pb.position),
                "orientation_wxyz": list(pb.orientation_wxyz),
            }
            for pb in placed_boxes
        ]
        result = self._impl.decide(
            {"id": current_box.id, "dimensions": list(current_box.dimensions), "weight": current_box.weight},
            placed_dicts,
            boxes_remaining,
            density,
        )
        return PlacementDecision(
            position=tuple(result["position"]),
            orientation_wxyz=tuple(result["orientation_wxyz"]),
            stop=result.get("stop", False),
        )

    def teardown(self, final_density: float, termination_reason: str | None) -> None:
        if hasattr(self._impl, "teardown"):
            self._impl.teardown(final_density, termination_reason or "")
