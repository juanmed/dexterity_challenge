from __future__ import annotations

import time

import numpy as np

from dexterity.models import Box, TruckDims
from dexterity.sim import ComFreeSimulator, SimConfig


def main() -> None:
    config = SimConfig(n_candidates=64, max_settle_steps=50)
    sim = ComFreeSimulator(config)
    truck = TruckDims(depth=2.0, width=2.0, height=2.0)
    scene = sim.build_scene(truck, placed_boxes=[])
    box = Box(id="c", dimensions=(0.2, 0.2, 0.2), weight=1.0)

    positions = np.zeros((config.n_candidates, 3), dtype=np.float32)
    orientations = np.zeros((config.n_candidates, 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    positions[:, 0] = np.linspace(0.3, 1.7, config.n_candidates)
    positions[:, 1] = 1.0
    positions[:, 2] = 0.5

    start = time.time()
    result = sim.evaluate_batch(scene, box, positions, orientations, n_settle_steps=50)
    elapsed = time.time() - start
    print(f"Simulated {config.n_candidates} candidates in {elapsed:.3f}s")
    print(f"Mean density: {result.density.mean():.4f}")


if __name__ == "__main__":
    main()
