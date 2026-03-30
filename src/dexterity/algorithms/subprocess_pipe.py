from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..models import Box, PlacedBox, PlacementDecision, TruckDims
from .base import Algorithm

logger = logging.getLogger(__name__)


class SubprocessPipeAlgorithm(Algorithm):
    """Adapter: persistent subprocess communicating via line-delimited JSON."""

    def __init__(self, binary_path: str | Path, *, call_timeout: float = 10.0):
        self._binary = str(Path(binary_path))
        self._call_timeout = call_timeout
        self._proc: asyncio.subprocess.Process | None = None

    async def _ensure_started(self):
        if self._proc is None or self._proc.returncode is not None:
            self._proc = await asyncio.create_subprocess_exec(
                self._binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self):
        assert self._proc and self._proc.stderr
        async for line in self._proc.stderr:
            logger.warning("[cpp-subprocess stderr] %s", line.decode().rstrip())

    async def _send(self, msg: dict) -> dict:
        await self._ensure_started()
        assert self._proc and self._proc.stdin and self._proc.stdout
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

        try:
            raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=self._call_timeout)
        except asyncio.TimeoutError:
            logger.error("Subprocess timeout; killing process")
            self._proc.kill()
            await self._proc.wait()
            self._proc = None
            raise

        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from subprocess: {raw!r}") from e

    def setup(self, truck: TruckDims) -> None:
        # setup is called synchronously from runner; schedule coroutine
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        loop.run_until_complete(self._async_setup(truck))

    async def _async_setup(self, truck: TruckDims) -> None:
        await self._send({"type": "setup", "truck": {"depth": truck.depth, "width": truck.width, "height": truck.height}})

    def decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        return loop.run_until_complete(self._async_decide(current_box, placed_boxes, boxes_remaining, density))

    async def _async_decide(
        self,
        current_box: Box,
        placed_boxes: list[PlacedBox],
        boxes_remaining: int,
        density: float,
    ) -> PlacementDecision:
        msg = {
            "type": "decide",
            "current_box": {
                "id": current_box.id,
                "dimensions": list(current_box.dimensions),
                "weight": current_box.weight,
            },
            "placed_boxes": [
                {
                    "id": pb.id,
                    "dimensions": list(pb.dimensions),
                    "position": list(pb.position),
                    "orientation_wxyz": list(pb.orientation_wxyz),
                }
                for pb in placed_boxes
            ],
            "boxes_remaining": boxes_remaining,
            "density": density,
        }
        result = await self._send(msg)
        return PlacementDecision(
            position=tuple(result["position"]),
            orientation_wxyz=tuple(result["orientation_wxyz"]),
            stop=result.get("stop", False),
        )

    def teardown(self, final_density: float, termination_reason: str | None) -> None:
        if self._proc and self._proc.returncode is None:
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            loop.run_until_complete(self._async_teardown(final_density, termination_reason))

    async def _async_teardown(self, final_density: float, termination_reason: str | None) -> None:
        try:
            await self._send({"type": "teardown", "final_density": final_density, "termination_reason": termination_reason or ""})
        except Exception:
            pass
        if self._proc:
            self._proc.stdin.close()
            await self._proc.wait()
            self._proc = None
