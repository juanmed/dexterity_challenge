from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from .models import Box, PlacementDecision

logger = logging.getLogger(__name__)

FLUSH_EVERY = 1000

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False
    logger.warning("pyarrow not available; falling back to .jsonl storage")


_STEPS_SCHEMA = None
_GAMES_SCHEMA = None

if _ARROW_AVAILABLE:
    import pyarrow as pa

    _STEPS_SCHEMA = pa.schema([
        pa.field("game_id", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("step", pa.int32()),
        pa.field("algorithm_name", pa.string()),
        pa.field("box_id", pa.string()),
        pa.field("box_dim_x", pa.float64()),
        pa.field("box_dim_y", pa.float64()),
        pa.field("box_dim_z", pa.float64()),
        pa.field("box_weight", pa.float64()),
        pa.field("decision_pos_x", pa.float64()),
        pa.field("decision_pos_y", pa.float64()),
        pa.field("decision_pos_z", pa.float64()),
        pa.field("decision_quat_w", pa.float64()),
        pa.field("decision_quat_x", pa.float64()),
        pa.field("decision_quat_y", pa.float64()),
        pa.field("decision_quat_z", pa.float64()),
        pa.field("density_before", pa.float64()),
        pa.field("boxes_remaining", pa.int32()),
        pa.field("wall_time_ms", pa.float64()),
    ])

    _GAMES_SCHEMA = pa.schema([
        pa.field("game_id", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("algorithm_name", pa.string()),
        pa.field("n_steps", pa.int32()),
        pa.field("final_density", pa.float64()),
        pa.field("termination_reason", pa.string()),
        pa.field("start_time", pa.float64()),
        pa.field("end_time", pa.float64()),
        pa.field("total_wall_ms", pa.float64()),
    ])


class ArrowWriter:
    def __init__(
        self,
        output_dir: str | Path,
        mode: str = "dev",
        algorithm_name: str = "unknown",
        buffer_size: int = 10_000,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._algorithm_name = algorithm_name
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._task: asyncio.Task | None = None
        self._start_time = time.time()
        self._step_counts: dict[str, int] = {}
        self._game_start_times: dict[str, float] = {}

    async def start(self):
        self._task = asyncio.create_task(self._drain())

    async def stop(self):
        await self._queue.join()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue_step(
        self,
        game_id: str,
        box: Box,
        decision: PlacementDecision,
        density_before: float,
    ):
        if game_id not in self._game_start_times:
            self._game_start_times[game_id] = time.time()
        step = self._step_counts.get(game_id, 0)
        self._step_counts[game_id] = step + 1

        record = {
            "type": "step",
            "game_id": game_id,
            "mode": self._mode,
            "step": step,
            "algorithm_name": self._algorithm_name,
            "box_id": box.id,
            "box_dim_x": box.dimensions[0],
            "box_dim_y": box.dimensions[1],
            "box_dim_z": box.dimensions[2],
            "box_weight": box.weight,
            "decision_pos_x": decision.position[0],
            "decision_pos_y": decision.position[1],
            "decision_pos_z": decision.position[2],
            "decision_quat_w": decision.orientation_wxyz[0],
            "decision_quat_x": decision.orientation_wxyz[1],
            "decision_quat_y": decision.orientation_wxyz[2],
            "decision_quat_z": decision.orientation_wxyz[3],
            "density_before": density_before,
            "boxes_remaining": 0,
            "wall_time_ms": (time.time() - self._start_time) * 1000,
        }
        await self._queue.put(record)

    async def enqueue_game_end(
        self,
        game_id: str,
        final_density: float,
        termination_reason: str | None,
    ):
        end_time = time.time()
        start_time = self._game_start_times.pop(game_id, end_time)
        n_steps = self._step_counts.pop(game_id, 0)

        record = {
            "type": "game",
            "game_id": game_id,
            "mode": self._mode,
            "algorithm_name": self._algorithm_name,
            "n_steps": n_steps,
            "final_density": final_density,
            "termination_reason": termination_reason or "",
            "start_time": start_time,
            "end_time": end_time,
            "total_wall_ms": (end_time - start_time) * 1000,
        }
        await self._queue.put(record)

    async def _drain(self):
        step_buffer: list[dict] = []
        game_buffer: list[dict] = []

        while True:
            try:
                item = await self._queue.get()
                if item["type"] == "step":
                    step_buffer.append(item)
                else:
                    game_buffer.append(item)

                if len(step_buffer) >= FLUSH_EVERY:
                    self._flush_steps(step_buffer)
                    step_buffer.clear()
                if len(game_buffer) >= FLUSH_EVERY // 10:
                    self._flush_games(game_buffer)
                    game_buffer.clear()

                self._queue.task_done()
            except asyncio.CancelledError:
                # Flush remaining
                if step_buffer:
                    self._flush_steps(step_buffer)
                if game_buffer:
                    self._flush_games(game_buffer)
                raise

    def _flush_steps(self, buffer: list[dict]):
        if not buffer:
            return
        ts = int(time.time() * 1000)
        if _ARROW_AVAILABLE:
            self._write_parquet(buffer, self._output_dir / f"steps_{ts}.parquet", _STEPS_SCHEMA)
        else:
            self._write_jsonl(buffer, self._output_dir / f"steps_{ts}.jsonl")

    def _flush_games(self, buffer: list[dict]):
        if not buffer:
            return
        ts = int(time.time() * 1000)
        if _ARROW_AVAILABLE:
            self._write_parquet(buffer, self._output_dir / f"games_{ts}.parquet", _GAMES_SCHEMA)
        else:
            self._write_jsonl(buffer, self._output_dir / f"games_{ts}.jsonl")

    def _write_parquet(self, buffer: list[dict], path: Path, schema):
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols: dict[str, list] = {field.name: [] for field in schema}
        for row in buffer:
            for field in schema:
                cols[field.name].append(row.get(field.name))

        table = pa.table(cols, schema=schema)
        pq.write_table(table, str(path))
        logger.debug("Flushed %d rows to %s", len(buffer), path)

    def _write_jsonl(self, buffer: list[dict], path: Path):
        import json
        with open(path, "w") as f:
            for row in buffer:
                f.write(json.dumps(row) + "\n")
        logger.debug("Flushed %d rows (jsonl) to %s", len(buffer), path)


class NullWriter:
    """No-op writer for testing or when storage is not needed."""

    async def start(self):
        pass

    async def stop(self):
        pass

    async def enqueue_step(self, *args, **kwargs):
        pass

    async def enqueue_game_end(self, *args, **kwargs):
        pass
