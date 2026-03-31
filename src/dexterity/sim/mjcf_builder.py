import math
import mujoco

from ..models import Box, PlacedBox, TruckDims


def build_truck_model(
    truck: TruckDims,
    placed_boxes: list[PlacedBox],
    candidate_box: Box,
    dynamic_indices: list[int] | None = None,
    timestep: float = 0.002,
    friction: float = 1.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_truck_spec(
        truck=truck,
        placed_boxes=placed_boxes,
        candidate_box=candidate_box,
        dynamic_indices=dynamic_indices,
        timestep=timestep,
        friction=friction,
    )

    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    return (mjm, mjd)


def build_truck_spec(
    truck: TruckDims,
    placed_boxes: list[PlacedBox],
    candidate_box: Box,
    dynamic_indices: list[int] | None = None,
    timestep: float = 0.002,
    friction: float = 1.0,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    spec.option.timestep = timestep
    spec.option.gravity = [0, 0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

    total_freejoints = len(placed_boxes) + 1
    if total_freejoints * 6 > 60:
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    else:
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

    worldbody = spec.worldbody

    walls = [
        {"pos": [truck.depth / 2, truck.width / 2, 0], "zaxis": [0, 0, 1]},
        {"pos": [truck.depth / 2, truck.width / 2, truck.height], "zaxis": [0, 0, -1]},
        {"pos": [0, truck.width / 2, truck.height / 2], "zaxis": [1, 0, 0]},
        {
            "pos": [truck.depth, truck.width / 2, truck.height / 2],
            "zaxis": [-1, 0, 0],
        },
        {"pos": [truck.depth / 2, 0, truck.height / 2], "zaxis": [0, 1, 0]},
        {
            "pos": [truck.depth / 2, truck.width, truck.height / 2],
            "zaxis": [0, -1, 0],
        },
    ]

    for wall in walls:
        geom = worldbody.add_geom()
        geom.pos = wall["pos"]
        geom.quat = _quat_from_zaxis(wall["zaxis"])
        geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        geom.size = [10, 10, 0.01]
        geom.contype = 1
        geom.conaffinity = 1

    for placed_box in placed_boxes:
        body = worldbody.add_body()
        body.pos = placed_box.position
        body.quat = placed_box.orientation_wxyz

        body.add_freejoint()

        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [d / 2 for d in placed_box.dimensions]
        geom.mass = 1.0
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 3
        geom.friction = [friction, 0.05, 0.01]

    candidate_body = worldbody.add_body()
    candidate_body.pos = [0, 0, 0]

    candidate_body.add_freejoint()

    candidate_geom = candidate_body.add_geom()
    candidate_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    candidate_geom.size = [d / 2 for d in candidate_box.dimensions]
    candidate_geom.mass = candidate_box.weight
    candidate_geom.contype = 1
    candidate_geom.conaffinity = 1
    candidate_geom.condim = 3
    candidate_geom.friction = [friction, 0.05, 0.01]

    return spec


def _quat_from_zaxis(zaxis: list[float]) -> list[float]:
    """Return quaternion (w, x, y, z) rotating +Z onto the given axis."""
    zx, zy, zz = zaxis
    norm = math.sqrt(zx * zx + zy * zy + zz * zz)
    if norm <= 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    vx, vy, vz = zx / norm, zy / norm, zz / norm
    dot = vz  # dot([0,0,1], v) = vz
    if dot > 0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -0.999999:
        # 180-degree rotation about X axis
        return [0.0, 1.0, 0.0, 0.0]
    # axis = cross([0,0,1], v) = (-vy, vx, 0)
    ax, ay, az = -vy, vx, 0.0
    axis_norm = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / axis_norm, ay / axis_norm, az / axis_norm
    angle = math.acos(max(-1.0, min(1.0, dot)))
    half = 0.5 * angle
    s = math.sin(half)
    return [math.cos(half), ax * s, ay * s, az * s]


def get_candidate_qpos_offset(placed_box_count: int) -> int:
    return placed_box_count * 7


def get_frozen_indices(
    placed_box_count: int,
    dynamic_indices: list[int] | None,
) -> list[int]:
    if dynamic_indices is None:
        return []
    all_indices = set(range(placed_box_count))
    return sorted(all_indices - set(dynamic_indices))
