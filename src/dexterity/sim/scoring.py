from __future__ import annotations
import numpy as np
from ..models import PlacedBox


def rotated_half_extents(dimensions: tuple[float, float, float], orientation_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """Compute AABB half-extents of a box after rotation. Returns [hx, hy, hz]."""
    w, x, y, z = orientation_wxyz
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    half_dims = np.array(dimensions) / 2.0
    return np.abs(R) @ half_dims


def compute_density(
    placed_boxes: list[PlacedBox],
    candidate_pos: np.ndarray,        # (3,)
    candidate_orient: np.ndarray,     # (4,) wxyz
    candidate_dims: tuple[float, float, float],
    truck_width: float,
    truck_height: float,
) -> float:
    """Compute packing density including the candidate box."""
    total_volume = sum(d[0]*d[1]*d[2] for pb in placed_boxes for d in [pb.dimensions])
    total_volume += candidate_dims[0] * candidate_dims[1] * candidate_dims[2]

    max_x = 0.0
    for pb in placed_boxes:
        hx = rotated_half_extents(pb.dimensions, pb.orientation_wxyz)[0]
        max_x = max(max_x, pb.position[0] + hx)

    cand_hx = rotated_half_extents(candidate_dims, tuple(candidate_orient))[0]
    max_x = max(max_x, candidate_pos[0] + cand_hx)

    if max_x <= 0.0:
        return 0.0
    return total_volume / (truck_width * truck_height * max_x)


def compute_density_batch(
    placed_boxes: list[PlacedBox],
    candidate_positions: np.ndarray,       # (N, 3)
    candidate_orientations: np.ndarray,    # (N, 4)
    candidate_dims: tuple[float, float, float],
    truck_width: float,
    truck_height: float,
) -> np.ndarray:
    """Compute density for N candidates. Returns (N,) float array."""
    N = candidate_positions.shape[0]
    densities = np.zeros(N)
    for i in range(N):
        densities[i] = compute_density(
            placed_boxes, candidate_positions[i], candidate_orientations[i],
            candidate_dims, truck_width, truck_height,
        )
    return densities


def check_stability_single(
    box_pos: np.ndarray,
    box_orient: np.ndarray,
    box_dims: tuple[float, float, float],
    support_boxes: list[PlacedBox],
    threshold: float = 0.25,
) -> tuple[bool, float]:
    """Check if a box is stable (has >= threshold footprint support).
    Returns (is_stable, support_ratio)."""
    hx, hy, hz = rotated_half_extents(box_dims, tuple(box_orient))
    cx, cy, cz = box_pos
    bottom_z = cz - hz
    footprint_area = max((2 * hx) * (2 * hy), 1e-9)

    # Floor support
    if bottom_z <= 1e-6:
        return True, 1.0

    support_area = 0.0
    for sb in support_boxes:
        shx, shy, shz = rotated_half_extents(sb.dimensions, sb.orientation_wxyz)
        sx, sy, sz = sb.position
        top_z = sz + shz
        if abs(top_z - bottom_z) > 1e-4:
            continue
        overlap_x = max(0.0, min(cx + hx, sx + shx) - max(cx - hx, sx - shx))
        overlap_y = max(0.0, min(cy + hy, sy + shy) - max(cy - hy, sy - shy))
        support_area += overlap_x * overlap_y

    support_area = min(support_area, footprint_area)
    ratio = support_area / footprint_area
    return ratio >= threshold, ratio
