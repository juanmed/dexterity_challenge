from __future__ import annotations

import inspect
import logging
import time

from .algorithms.base import Algorithm
from .client import DexterityClient
from .models import GameResult, PlacedBox, PlaceResponse, StartResponse

logger = logging.getLogger(__name__)


class GameRunner:
    def __init__(
        self,
        client: DexterityClient,
        algorithm: Algorithm,
        writer,
        mode: str = "dev",
    ):
        self._client = client
        self._algorithm = algorithm
        self._writer = writer
        self._mode = mode

    async def run(self) -> GameResult:
        start_resp: StartResponse = await self._client.start(mode=self._mode)
        self._algorithm.setup(truck=start_resp.truck)

        game_id = start_resp.game_id
        current_box = start_resp.current_box
        placed_boxes: list[PlacedBox] = []
        density = 0.0
        place_resp: PlaceResponse | None = None
        termination_reason: str | None = None

        try:
            while current_box is not None:
                t0 = time.monotonic()
                decision = await self._call_algorithm(current_box, placed_boxes, density)
                algo_ms = (time.monotonic() - t0) * 1000

                if decision.stop:
                    await self._client.stop(game_id)
                    break

                await self._writer.enqueue_step(game_id, current_box, decision, density)

                t1 = time.monotonic()
                place_resp = await self._client.place(
                    game_id,
                    current_box.id,
                    decision.position,
                    decision.orientation_wxyz,
                )
                latency_ms = (time.monotonic() - t1) * 1000
                logger.debug(
                    "game=%s step placed box=%s density=%.4f latency=%.0fms",
                    game_id,
                    current_box.id,
                    place_resp.density,
                    latency_ms,
                )

                placed_boxes = place_resp.placed_boxes
                density = place_resp.density
                current_box = place_resp.current_box

                if place_resp.game_status == "completed":
                    break

            termination_reason = place_resp.termination_reason if place_resp else None
            await self._writer.enqueue_game_end(game_id, density, termination_reason)

            logger.info("game=%s finished density=%.4f reason=%s", game_id, density, termination_reason)
            return GameResult(game_id=game_id, density=density, termination_reason=termination_reason)
        finally:
            try:
                self._algorithm.teardown(density, termination_reason)
            except Exception:
                logger.exception("algorithm teardown failed")

    async def resume(self, game_id: str) -> GameResult:
        """Reconnect to an in-progress game and continue from current state."""
        state = await self._client.status(game_id)
        if state.game_status == "completed":
            return GameResult(game_id=game_id, density=state.density, termination_reason=None)

        from .models import PlacedBox
        from .models import TruckDims
        # We don't have truck dims in status; use a zero-dims truck as placeholder
        # (algorithm should not depend on truck dims if resuming)
        self._algorithm.setup(truck=TruckDims(depth=0, width=0, height=0))

        current_box = state.current_box
        placed_boxes = state.placed_boxes
        density = state.density
        place_resp = None
        termination_reason: str | None = None

        try:
            while current_box is not None:
                decision = await self._call_algorithm(current_box, placed_boxes, density)

                if decision.stop:
                    await self._client.stop(game_id)
                    break

                await self._writer.enqueue_step(game_id, current_box, decision, density)

                place_resp = await self._client.place(
                    game_id,
                    current_box.id,
                    decision.position,
                    decision.orientation_wxyz,
                )
                placed_boxes = place_resp.placed_boxes
                density = place_resp.density
                current_box = place_resp.current_box

                if place_resp.game_status == "completed":
                    break

            termination_reason = place_resp.termination_reason if place_resp else None
            await self._writer.enqueue_game_end(game_id, density, termination_reason)
            return GameResult(game_id=game_id, density=density, termination_reason=termination_reason)
        finally:
            try:
                self._algorithm.teardown(density, termination_reason)
            except Exception:
                logger.exception("algorithm teardown failed")

    async def _call_algorithm(self, current_box, placed_boxes, density):
        result = self._algorithm.decide(current_box, placed_boxes, len(placed_boxes), density)
        if inspect.isawaitable(result):
            return await result
        return result
