from __future__ import annotations
import numpy as np


def freeze_boxes_cpu(
    qpos: np.ndarray,          # (nworld, nq) — mutable
    qvel: np.ndarray,          # (nworld, nv) — mutable
    frozen_qpos_offsets: list[int],   # qpos offset for each frozen box
    frozen_qvel_offsets: list[int],   # qvel offset for each frozen box
    frozen_positions: np.ndarray,     # (n_frozen, 7) — target qpos values
) -> None:
    """Zero velocity and restore position of frozen boxes across all worlds."""
    for i, (qpos_off, qvel_off) in enumerate(zip(frozen_qpos_offsets, frozen_qvel_offsets)):
        qpos[:, qpos_off:qpos_off + 7] = frozen_positions[i]
        qvel[:, qvel_off:qvel_off + 6] = 0.0


def compute_frozen_offsets(
    n_placed: int,
    frozen_indices: list[int],
) -> tuple[list[int], list[int], np.ndarray]:
    """Compute qpos/qvel offsets and target positions for frozen boxes.

    In MuJoCo with freejoints, each body's freejoint contributes:
    - 7 qpos entries (x, y, z, qw, qx, qy, qz)
    - 6 qvel entries (vx, vy, vz, wx, wy, wz)

    Bodies are ordered: placed_box_0, placed_box_1, ..., candidate_box

    Returns (qpos_offsets, qvel_offsets) for the frozen boxes.
    """
    qpos_offsets = [idx * 7 for idx in frozen_indices]
    qvel_offsets = [idx * 6 for idx in frozen_indices]
    return qpos_offsets, qvel_offsets


def check_settled_cpu(
    qvel: np.ndarray,           # (nworld, nv)
    dynamic_qvel_offsets: list[int],
    threshold: float = 1e-4,
) -> np.ndarray:
    """Check if all dynamic boxes have settled in each world.
    Returns (nworld,) bool array."""
    nworld = qvel.shape[0]
    ke = np.zeros(nworld)
    for off in dynamic_qvel_offsets:
        ke += np.sum(qvel[:, off:off + 6] ** 2, axis=1)
    return ke < threshold
