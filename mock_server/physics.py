import numpy as np


def normalize_quaternion(q: list[float]) -> list[float]:
    arr = np.array(q, dtype=float)
    mag = np.linalg.norm(arr)
    if mag < 1e-6:
        raise ValueError("Degenerate quaternion: near-zero magnitude")
    return (arr / mag).tolist()


def quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotated_half_extents(dims: list[float], q: list[float]) -> np.ndarray:
    R = quaternion_to_rotation_matrix(q)
    half = np.array(dims, dtype=float) / 2.0
    return np.abs(R) @ half


def _xy_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _highest_support_surface(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    placed_boxes: list[dict],
) -> float:
    x_min = cx - hx
    x_max = cx + hx
    y_min = cy - hy
    y_max = cy + hy
    highest = 0.0
    for b in placed_boxes:
        dims = b["dimensions"]
        orientation = b["orientation_wxyz"]
        pb_center = b["position"]
        pb_hx, pb_hy, pb_hz = rotated_half_extents(dims, orientation)
        pb_x, pb_y, pb_z = pb_center
        x_min_b = pb_x - pb_hx
        x_max_b = pb_x + pb_hx
        y_min_b = pb_y - pb_hy
        y_max_b = pb_y + pb_hy
        overlap_x = _xy_overlap(x_min, x_max, x_min_b, x_max_b)
        overlap_y = _xy_overlap(y_min, y_max, y_min_b, y_max_b)
        if overlap_x > 0 and overlap_y > 0:
            highest = max(highest, pb_z + pb_hz)
    return highest


def dev_place(
    position: list[float],
    orientation_wxyz: list[float],
) -> tuple[list[float], list[float]]:
    q = normalize_quaternion(orientation_wxyz)
    return list(position), q


class PhysicsSimulator:
    def place(
        self,
        mode: str,
        dims: list[float],
        position: list[float],
        orientation_wxyz: list[float],
        placed_boxes: list[dict],
    ) -> tuple[list[float], list[float], int]:
        normalized = normalize_quaternion(orientation_wxyz)
        if mode == "dev":
            settled_position, settled_orientation = dev_place(position, normalized)
            return settled_position, settled_orientation, 0
        return self._compete_place(dims, position, normalized, placed_boxes)

    def _compete_place(
        self,
        dims: list[float],
        position: list[float],
        orientation_wxyz: list[float],
        placed_boxes: list[dict],
    ) -> tuple[list[float], list[float], int]:
        hx, hy, hz = rotated_half_extents(dims, orientation_wxyz)
        cx, cy, _cz = position
        support_top = _highest_support_surface(cx, cy, hx, hy, placed_boxes)
        settled_z = support_top + hz
        settled_position = [cx, cy, settled_z]
        return settled_position, orientation_wxyz, 0
