from __future__ import annotations

import pytest

from dexterity.algorithms.naive import NaiveAlgorithm
from dexterity.models import GameResult
from dexterity.runner import GameRunner
from dexterity.storage import NullWriter


@pytest.mark.asyncio
async def test_full_game_run(mock_client):
    runner = GameRunner(mock_client, NaiveAlgorithm(), NullWriter(), mode="dev")
    result = await runner.run()
    assert isinstance(result, GameResult)
    assert result.game_id
    assert 0.0 <= result.density <= 1.0


@pytest.mark.asyncio
async def test_game_completes_all_boxes(mock_client):
    runner = GameRunner(mock_client, NaiveAlgorithm(), NullWriter(), mode="dev")
    result = await runner.run()
    # All 5 boxes placed → density == 1.0 in mock
    assert result.density == 1.0


@pytest.mark.asyncio
async def test_stop_decision(mock_client):
    """If algorithm signals stop, runner calls /stop instead of /place."""
    from dexterity.algorithms.base import Algorithm
    from dexterity.models import Box, PlacedBox, PlacementDecision, TruckDims

    class StopOnFirstAlgorithm(Algorithm):
        def setup(self, truck: TruckDims) -> None:
            pass

        def decide(self, current_box: Box, placed_boxes, boxes_remaining, density) -> PlacementDecision:
            return PlacementDecision(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0), stop=True)

    runner = GameRunner(mock_client, StopOnFirstAlgorithm(), NullWriter(), mode="dev")
    result = await runner.run()
    assert result.density == 0.0  # stopped before any placement


@pytest.mark.asyncio
async def test_async_algorithm(mock_client):
    """Runner correctly awaits async decide() methods."""
    import asyncio
    from dexterity.algorithms.base import Algorithm
    from dexterity.models import Box, PlacementDecision, TruckDims

    class AsyncAlgorithm(Algorithm):
        def setup(self, truck: TruckDims) -> None:
            pass

        async def decide(self, current_box: Box, placed_boxes, boxes_remaining, density) -> PlacementDecision:
            await asyncio.sleep(0)
            return PlacementDecision(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    runner = GameRunner(mock_client, AsyncAlgorithm(), NullWriter(), mode="dev")
    result = await runner.run()
    assert result.density > 0.0
