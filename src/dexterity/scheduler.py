from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from .algorithms.base import Algorithm
from .client import DexterityClient
from .models import GameResult
from .runner import GameRunner

logger = logging.getLogger(__name__)


class GameScheduler:
    def __init__(
        self,
        client: DexterityClient,
        algorithm_factory: Callable[[], Algorithm],
        writer,
        *,
        mode: str = "dev",
        max_concurrent: int = 500,
    ):
        self._client = client
        self._algorithm_factory = algorithm_factory
        self._writer = writer
        self._mode = mode
        self._max_concurrent = max_concurrent

    async def run_games(self, n_games: int) -> AsyncIterator[GameResult]:
        semaphore = asyncio.Semaphore(self._max_concurrent)
        queue: asyncio.Queue[GameResult | Exception] = asyncio.Queue()
        tasks: set[asyncio.Task] = set()

        async def run_one():
            async with semaphore:
                algorithm = self._algorithm_factory()
                runner = GameRunner(self._client, algorithm, self._writer, self._mode)
                try:
                    result = await runner.run()
                    await queue.put(result)
                except Exception as e:
                    logger.error("Game failed: %s", e, exc_info=True)
                    await queue.put(e)

        for _ in range(n_games):
            t = asyncio.create_task(run_one())
            tasks.add(t)
            t.add_done_callback(tasks.discard)

        for _ in range(n_games):
            result = await queue.get()
            if isinstance(result, Exception):
                pass  # already logged in run_one
            else:
                yield result

    # Make run_games usable as an async generator
    def __aiter__(self):
        raise TypeError("Call run_games(n) to get an async iterator")
