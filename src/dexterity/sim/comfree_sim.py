from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import comfree_warp as cfwarp
import mujoco

if not hasattr(cfwarp, "put_model"):
    import comfree_warp.comfree_warp as cfwarp

from ..models import Box, PlacedBox, TruckDims
from .freeze import compute_frozen_offsets
from .mjcf_builder import (
    build_truck_model,
    build_truck_spec,
    get_candidate_qpos_offset,
    get_frozen_indices,
)
from .protocol import PhysicsSim, SimConfig, SimResult
from .scoring import check_stability_single, compute_density_batch

try:
    import warp as wp
except Exception:  # pragma: no cover - warp is a required dependency
    wp = None


if wp is not None:
    @wp.kernel
    def _freeze_boxes_kernel(
        qpos: wp.array2d(dtype=wp.float32),
        qvel: wp.array2d(dtype=wp.float32),
        frozen_qpos: wp.array2d(dtype=wp.float32),
        qpos_offsets: wp.array(dtype=wp.int32),
        qvel_offsets: wp.array(dtype=wp.int32),
        n_frozen: int,
    ):
        world_id, frozen_idx = wp.tid()
        if frozen_idx >= n_frozen:
            return
        qpos_off = qpos_offsets[frozen_idx]
        qvel_off = qvel_offsets[frozen_idx]
        for i in range(7):
            qpos[world_id, qpos_off + i] = frozen_qpos[frozen_idx, i]
        for i in range(6):
            qvel[world_id, qvel_off + i] = wp.float32(0.0)


    @wp.kernel
    def _kinetic_energy_kernel(
        qvel: wp.array2d(dtype=wp.float32),
        qvel_offsets: wp.array(dtype=wp.int32),
        n_offsets: int,
        out_ke: wp.array(dtype=wp.float32),
    ):
        world_id = wp.tid()
        total = wp.float32(0.0)
        for i in range(n_offsets):
            off = qvel_offsets[i]
            for j in range(6):
                v = qvel[world_id, off + j]
                total += v * v
        out_ke[world_id] = total


@dataclass
class _Scene:
    truck: TruckDims
    placed_boxes: list[PlacedBox]
    dynamic_indices: list[int]
    frozen_indices: list[int]


class ComFreeSimulator(PhysicsSim):
    def __init__(self, config: SimConfig | None = None) -> None:
        if wp is None:
            raise RuntimeError("warp is required for ComFreeSimulator")
        self._config = config or SimConfig()
        try:
            wp.set_device(self._config.warp_device)
        except Exception:
            pass
        self._viewer = None
        self._streamer = None
        self._stream_model_path = None
        self._stream_model_nbody = None
        self._viewer_model_nbody = None

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._streamer is not None:
            try:
                self._streamer.stop_connection()
            except Exception:
                pass
            self._streamer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def max_batch_size(self) -> int:
        return self._config.n_candidates

    def build_scene(self, truck: TruckDims, placed_boxes: list[PlacedBox]) -> _Scene:
        n_recent = self._config.n_recent_dynamic
        n_placed = len(placed_boxes)
        if n_placed <= n_recent:
            dynamic_indices = list(range(n_placed))
        else:
            dynamic_indices = list(range(n_placed - n_recent, n_placed))
        frozen_indices = get_frozen_indices(n_placed, dynamic_indices)
        return _Scene(truck=truck, placed_boxes=list(placed_boxes), dynamic_indices=dynamic_indices, frozen_indices=frozen_indices)

    def evaluate_batch(
        self,
        scene: _Scene,
        candidate_box: Box,
        positions: np.ndarray,
        orientations: np.ndarray,
        n_settle_steps: int = 200,
    ) -> SimResult:
        positions = np.asarray(positions, dtype=np.float32)
        orientations = np.asarray(orientations, dtype=np.float32)
        if positions.shape[0] == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return SimResult(
                settled_positions=empty,
                settled_orientations=np.zeros((0, 4), dtype=np.float32),
                n_displaced=np.zeros((0,), dtype=np.int32),
                is_stable=np.zeros((0,), dtype=bool),
                density=np.zeros((0,), dtype=np.float32),
                steps_to_settle=np.zeros((0,), dtype=np.int32),
            )

        max_batch = self.max_batch_size()
        total = positions.shape[0]
        settled_positions = np.zeros((total, 3), dtype=np.float32)
        settled_orientations = np.zeros((total, 4), dtype=np.float32)
        n_displaced = np.zeros((total,), dtype=np.int32)
        is_stable = np.zeros((total,), dtype=bool)
        density = np.zeros((total,), dtype=np.float32)
        steps_to_settle = np.zeros((total,), dtype=np.int32)

        for start in range(0, total, max_batch):
            end = min(total, start + max_batch)
            batch_positions = positions[start:end]
            batch_orientations = orientations[start:end]
            batch_result = self._evaluate_batch_inner(
                scene, candidate_box, batch_positions, batch_orientations, n_settle_steps,
            )
            settled_positions[start:end] = batch_result.settled_positions
            settled_orientations[start:end] = batch_result.settled_orientations
            n_displaced[start:end] = batch_result.n_displaced
            is_stable[start:end] = batch_result.is_stable
            density[start:end] = batch_result.density
            steps_to_settle[start:end] = batch_result.steps_to_settle

        return SimResult(
            settled_positions=settled_positions,
            settled_orientations=settled_orientations,
            n_displaced=n_displaced,
            is_stable=is_stable,
            density=density,
            steps_to_settle=steps_to_settle,
        )

    def _evaluate_batch_inner(
        self,
        scene: _Scene,
        candidate_box: Box,
        positions: np.ndarray,
        orientations: np.ndarray,
        n_settle_steps: int,
    ) -> SimResult:
        config = self._config
        nworld = positions.shape[0]
        n_placed = len(scene.placed_boxes)
        candidate_qpos_offset = get_candidate_qpos_offset(n_placed)

        mjm, mjd = build_truck_model(
            scene.truck,
            scene.placed_boxes,
            candidate_box,
            dynamic_indices=scene.dynamic_indices,
            timestep=config.timestep,
            friction=config.box_friction,
        )
        spec = None
        if config.visualize or config.stream_port > 0:
            spec = build_truck_spec(
                truck=scene.truck,
                placed_boxes=scene.placed_boxes,
                candidate_box=candidate_box,
                dynamic_indices=scene.dynamic_indices,
                timestep=config.timestep,
                friction=config.box_friction,
            )
            self._ensure_visualization(mjm, mjd, spec)

        m = cfwarp.put_model(
            mjm,
            comfree_stiffness=config.comfree_stiffness,
            comfree_damping=config.comfree_damping,
        )
        d = cfwarp.put_data(
            mjm,
            mjd,
            nworld=nworld,
            nconmax=config.nconmax_per_world,
            njmax=config.njmax_per_world,
        )

        qpos_array = np.tile(mjd.qpos, (nworld, 1)).astype(np.float32)
        for i, (pos, orient) in enumerate(zip(positions, orientations)):
            qpos_array[i, candidate_qpos_offset:candidate_qpos_offset + 3] = pos
            qpos_array[i, candidate_qpos_offset + 3:candidate_qpos_offset + 7] = orient

        wp.copy(d.qpos, wp.array(qpos_array, dtype=wp.float32, device=wp.get_device()))
        wp.copy(d.qvel, wp.zeros(d.qvel.shape, dtype=wp.float32, device=wp.get_device()))

        frozen_indices = scene.frozen_indices
        frozen_qpos_offsets, frozen_qvel_offsets = compute_frozen_offsets(n_placed, frozen_indices)
        frozen_positions = np.zeros((len(frozen_indices), 7), dtype=np.float32)
        for i, idx in enumerate(frozen_indices):
            offset = idx * 7
            frozen_positions[i] = qpos_array[0, offset:offset + 7]

        dynamic_qvel_offsets = [idx * 6 for idx in scene.dynamic_indices]
        dynamic_qvel_offsets.append(n_placed * 6)  # candidate

        frozen_qpos_wp = wp.array(frozen_positions, dtype=wp.float32, device=wp.get_device()) if frozen_indices else None
        frozen_qpos_offsets_wp = wp.array(frozen_qpos_offsets, dtype=wp.int32, device=wp.get_device()) if frozen_indices else None
        frozen_qvel_offsets_wp = wp.array(frozen_qvel_offsets, dtype=wp.int32, device=wp.get_device()) if frozen_indices else None

        dyn_qvel_offsets_wp = wp.array(dynamic_qvel_offsets, dtype=wp.int32, device=wp.get_device())
        ke_out = wp.empty(nworld, dtype=wp.float32, device=wp.get_device())

        cfwarp.step(m, d)
        cfwarp.step(m, d)

        graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                cfwarp.step(m, d)
            graph = capture.graph

        max_steps = min(n_settle_steps, config.max_settle_steps)
        interval = max(1, config.settle_check_interval)

        settled = np.zeros((nworld,), dtype=bool)
        steps_to_settle = np.full((nworld,), max_steps, dtype=np.int32)
        total_steps = 0

        while total_steps < max_steps:
            step_batch = min(interval, max_steps - total_steps)
            for _ in range(step_batch):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    cfwarp.step(m, d)
            total_steps += step_batch

            if frozen_indices:
                wp.launch(
                    _freeze_boxes_kernel,
                    dim=(nworld, len(frozen_indices)),
                    inputs=[
                        d.qpos,
                        d.qvel,
                        frozen_qpos_wp,
                        frozen_qpos_offsets_wp,
                        frozen_qvel_offsets_wp,
                        len(frozen_indices),
                    ],
                )

            wp.launch(
                _kinetic_energy_kernel,
                dim=nworld,
                inputs=[d.qvel, dyn_qvel_offsets_wp, len(dynamic_qvel_offsets), ke_out],
            )
            wp.synchronize()
            settled_now = ke_out.numpy() < config.settle_ke_threshold
            newly_settled = (~settled) & settled_now
            steps_to_settle[newly_settled] = total_steps
            settled |= settled_now
            if config.visualize or config.stream_port > 0:
                if (total_steps // interval) % max(1, config.visualize_every) == 0:
                    self._update_visualization(mjm, mjd, d, world_id=0)
            if settled.all():
                break

        wp.synchronize()
        qpos_final = d.qpos.numpy()
        if config.visualize or config.stream_port > 0:
            self._update_visualization(mjm, mjd, d, world_id=0)

        settled_positions = qpos_final[:, candidate_qpos_offset:candidate_qpos_offset + 3]
        settled_orientations = qpos_final[:, candidate_qpos_offset + 3:candidate_qpos_offset + 7]
        finite_mask = (
            np.isfinite(settled_positions).all(axis=1)
            & np.isfinite(settled_orientations).all(axis=1)
        )

        n_displaced = np.zeros((nworld,), dtype=np.int32)
        if n_placed > 0:
            placed_qpos = qpos_final[:, :n_placed * 7].reshape(nworld, n_placed, 7)
            placed_orient = placed_qpos[:, :, 3:7]
            placed_pos = placed_qpos[:, :, :3]
        else:
            placed_orient = np.zeros((nworld, 0, 4), dtype=np.float32)
            placed_pos = np.zeros((nworld, 0, 3), dtype=np.float32)

        for world_idx in range(nworld):
            if not finite_mask[world_idx]:
                n_displaced[world_idx] = n_placed + 1
                continue
            support_boxes: list[PlacedBox] = []
            for i, pb in enumerate(scene.placed_boxes):
                support_boxes.append(
                    PlacedBox(
                        id=pb.id,
                        position=tuple(placed_pos[world_idx, i]),
                        orientation_wxyz=tuple(placed_orient[world_idx, i]),
                        dimensions=pb.dimensions,
                    )
                )
            support_boxes.append(
                PlacedBox(
                    id="candidate",
                    position=tuple(settled_positions[world_idx]),
                    orientation_wxyz=tuple(settled_orientations[world_idx]),
                    dimensions=candidate_box.dimensions,
                )
            )

            displaced_count = 0
            for box_idx, box in enumerate(support_boxes):
                others = support_boxes[:box_idx] + support_boxes[box_idx + 1:]
                stable, ratio = check_stability_single(
                    np.array(box.position),
                    np.array(box.orientation_wxyz),
                    box.dimensions,
                    others,
                )
                if not stable:
                    displaced_count += 1
            n_displaced[world_idx] = displaced_count

        density = compute_density_batch(
            scene.placed_boxes,
            settled_positions,
            settled_orientations,
            candidate_box.dimensions,
            scene.truck.width,
            scene.truck.height,
        )
        density = density.astype(np.float32)
        density[~finite_mask] = -np.inf

        is_stable = (n_displaced < 3) & finite_mask

        return SimResult(
            settled_positions=settled_positions,
            settled_orientations=settled_orientations,
            n_displaced=n_displaced,
            is_stable=is_stable,
            density=density,
            steps_to_settle=steps_to_settle,
        )

    def _ensure_visualization(self, mjm, mjd, spec) -> None:
        config = self._config
        if config.visualize and (self._viewer is None or self._viewer_model_nbody != mjm.nbody):
            import mujoco.viewer
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = mujoco.viewer.launch_passive(mjm, mjd)
            self._viewer_model_nbody = mjm.nbody
        if config.stream_port > 0 and (self._streamer is None or self._stream_model_nbody != mjm.nbody):
            from comfree_warp.test_headless.streaming import StreamServer
            import tempfile

            if self._streamer is not None:
                self._streamer.stop_connection()
            xml = spec.to_xml()
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
            temp.write(xml.encode("utf-8"))
            temp.flush()
            self._stream_model_path = temp.name
            self._streamer = StreamServer(
                model_path=self._stream_model_path,
                host=config.stream_host,
                port=config.stream_port,
            )
            self._streamer.start()
            self._stream_model_nbody = mjm.nbody

    def _update_visualization(self, mjm, mjd, data, world_id: int = 0) -> None:
        if self._viewer is None and self._streamer is None:
            return
        qpos = data.qpos.numpy()[world_id]
        qvel = data.qvel.numpy()[world_id]
        np.copyto(mjd.qpos, qpos)
        np.copyto(mjd.qvel, qvel)
        mujoco.mj_forward(mjm, mjd)
        if self._viewer is not None:
            self._viewer.sync()
        if self._streamer is not None:
            self._streamer.send_state(mjd)
