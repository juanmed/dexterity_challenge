from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..models import Box, PlacedBox, TruckDims


@dataclass
class SimResult:
    settled_positions: np.ndarray     # (N, 3)
    settled_orientations: np.ndarray  # (N, 4) wxyz
    n_displaced: np.ndarray           # (N,) int
    is_stable: np.ndarray             # (N,) bool
    density: np.ndarray               # (N,) float
    steps_to_settle: np.ndarray       # (N,) int


@dataclass
class SimConfig:
    timestep: float = 0.002
    comfree_stiffness: float = 0.2
    comfree_damping: float = 0.01
    box_friction: float = 1.0
    box_condim: int = 3
    n_candidates: int = 64
    # Warp MuJoCo needs enough headroom for contacts/constraints; too-low values
    # can produce "nefc overflow" warnings, NaNs, and even segfaults in native code.
    nconmax_per_world: int = 256
    njmax_per_world: int = 2048
    max_settle_steps: int = 200
    settle_check_interval: int = 10
    settle_ke_threshold: float = 1e-4
    n_recent_dynamic: int = 5
    full_sweep_interval: int = 10
    displacement_alpha: float = 0.1
    unsettled_penalty: float = 0.5
    grid_spacing: float = 0.05
    warp_device: str = "cuda:0"
    visualize: bool = False
    visualize_every: int = 5
    stream_host: str = "127.0.0.1"
    stream_port: int = 0


class PhysicsSim(Protocol):
    def build_scene(self, truck: TruckDims, placed_boxes: list[PlacedBox]) -> Any: ...
    def evaluate_batch(
        self, scene: Any, candidate_box: Box,
        positions: np.ndarray,
        orientations: np.ndarray,
        n_settle_steps: int = 200,
    ) -> SimResult: ...
    def max_batch_size(self) -> int: ...
