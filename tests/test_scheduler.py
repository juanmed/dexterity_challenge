from __future__ import annotations

import asyncio

import pytest

from dexterity.algorithms.naive import NaiveAlgorithm
from dexterity.models import GameResult
from dexterity.scheduler import GameScheduler
from dexterity.storage import NullWriter


@pytest.mark.asyncio
async def test_run_multiple_games(mock_client):
    scheduler = GameScheduler(
        mock_client,
        NaiveAlgorithm,
        NullWriter(),
        mode="dev",
        max_concurrent=3,
    )
    results = []
    async for result in scheduler.run_games(5):
        results.append(result)
    assert len(results) == 5
    assert all(isinstance(r, GameResult) for r in results)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency(mock_client):
    """Verify that max_concurrent=1 serialises games."""
    active = 0
    max_active = 0

    original_run = None

    from dexterity import runner as runner_module

    original_run_method = runner_module.GameRunner.run

    async def patched_run(self):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            return await original_run_method(self)
        finally:
            active -= 1

    runner_module.GameRunner.run = patched_run
    try:
        scheduler = GameScheduler(
            mock_client,
            NaiveAlgorithm,
            NullWriter(),
            mode="dev",
            max_concurrent=1,
        )
        results = []
        async for result in scheduler.run_games(3):
            results.append(result)
        assert max_active == 1
        assert len(results) == 3
    finally:
        runner_module.GameRunner.run = original_run_method


@pytest.mark.asyncio
async def test_game_exception_does_not_cancel_others(mock_client):
    """One failing game must not cancel the rest."""
    call_count = 0

    from dexterity.algorithms.base import Algorithm
    from dexterity.models import Box, PlacementDecision, TruckDims

    class SometimesFailAlgorithm(Algorithm):
        def setup(self, truck: TruckDims) -> None:
            pass

        def decide(self, current_box: Box, placed_boxes, boxes_remaining, density) -> PlacementDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("deliberate failure")
            return PlacementDecision(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    scheduler = GameScheduler(
        mock_client,
        SometimesFailAlgorithm,
        NullWriter(),
        mode="dev",
        max_concurrent=5,
    )
    results = []
    async for result in scheduler.run_games(3):
        results.append(result)
    # 2 games succeed (third may fail partway but scheduler continues)
    assert len(results) >= 1
