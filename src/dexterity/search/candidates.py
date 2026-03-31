from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Box, PlacedBox, TruckDims
from ..sim.scoring import rotated_half_extents


@dataclass(frozen=True)
class CandidatePlacement:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


class CandidateGenerator:
    def __init__(self, grid_spacing: float = 0.05) -> None:
        self._grid_spacing = grid_spacing

    def generate(
        self,
        box: Box,
        placed_boxes: list[PlacedBox],
        truck: TruckDims,
    ) -> list[CandidatePlacement]:
        orientations = self._axis_aligned_orientations(box.dimensions)
        candidates: list[CandidatePlacement] = []

        for orient in orientations:
            hx, hy, hz = rotated_half_extents(box.dimensions, orient)
            x_positions = self._axis_grid(hx, truck.depth)
            y_positions = self._axis_grid(hy, truck.width)

            for x in x_positions:
                for y in y_positions:
                    z = self._support_height(
                        x,
                        y,
                        hx,
                        hy,
                        hz,
                        placed_boxes,
                    )
                    if z + hz > truck.height + 1e-6:
                        continue
                    candidate = CandidatePlacement(
                        position=(x, y, z),
                        orientation=orient,
                    )
                    if self._collides(candidate, box.dimensions, placed_boxes):
                        continue
                    candidates.append(candidate)

        # Prioritize back wall packing and lower placements first
        candidates.sort(key=lambda c: (c.position[0], c.position[2], c.position[1]))
        return candidates

    def _axis_grid(self, half_extent: float, limit: float) -> np.ndarray:
        start = half_extent
        end = max(half_extent, limit - half_extent)
        if end < start + 1e-6:
            return np.array([limit / 2.0], dtype=np.float32)
        return np.arange(start, end + 1e-6, self._grid_spacing, dtype=np.float32)

    def _support_height(
        self,
        x: float,
        y: float,
        hx: float,
        hy: float,
        hz: float,
        placed_boxes: list[PlacedBox],
    ) -> float:
        bottom_z = hz
        max_top = 0.0
        for pb in placed_boxes:
            phx, phy, phz = rotated_half_extents(pb.dimensions, pb.orientation_wxyz)
            px, py, pz = pb.position
            if not self._overlap_1d(x - hx, x + hx, px - phx, px + phx):
                continue
            if not self._overlap_1d(y - hy, y + hy, py - phy, py + phy):
                continue
            max_top = max(max_top, pz + phz)
        return max(bottom_z, max_top + hz)

    @staticmethod
    def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
        return min(a1, b1) - max(a0, b0) > 1e-6

    def _collides(
        self,
        candidate: CandidatePlacement,
        dimensions: tuple[float, float, float],
        placed_boxes: list[PlacedBox],
    ) -> bool:
        hx, hy, hz = rotated_half_extents(dimensions, candidate.orientation)
        cx, cy, cz = candidate.position
        for pb in placed_boxes:
            phx, phy, phz = rotated_half_extents(pb.dimensions, pb.orientation_wxyz)
            px, py, pz = pb.position
            if self._overlap_1d(cx - hx, cx + hx, px - phx, px + phx) and \
               self._overlap_1d(cy - hy, cy + hy, py - phy, py + phy) and \
               self._overlap_1d(cz - hz, cz + hz, pz - phz, pz + phz):
                return True
        return False

    @staticmethod
    def _axis_aligned_orientations(
        dimensions: tuple[float, float, float]
    ) -> list[tuple[float, float, float, float]]:
        # Six axis-aligned orientations.
        base = [
            (1.0, 0.0, 0.0, 0.0),
            (0.70710678, 0.0, 0.0, 0.70710678),
            (0.70710678, 0.0, 0.70710678, 0.0),
            (0.70710678, 0.70710678, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]

        # Deduplicate by resulting half extents (handles duplicate dimensions).
        seen: set[tuple[int, int, int]] = set()
        unique: list[tuple[float, float, float, float]] = []
        for quat in base:
            hx, hy, hz = rotated_half_extents(dimensions, quat)
            key = (round(hx * 1000), round(hy * 1000), round(hz * 1000))
            if key in seen:
                continue
            seen.add(key)
            unique.append(quat)
        return unique
