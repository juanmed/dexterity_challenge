from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from dexterity.models import Box, PlacementDecision
from dexterity.storage import ArrowWriter


@pytest.mark.asyncio
async def test_jsonl_fallback(tmp_path, monkeypatch):
    """Writer falls back to .jsonl if pyarrow is mocked as unavailable."""
    import dexterity.storage as storage_module

    monkeypatch.setattr(storage_module, "_ARROW_AVAILABLE", False)

    writer = ArrowWriter(tmp_path, mode="dev", algorithm_name="naive", buffer_size=100)
    await writer.start()

    box = Box(id="b1", dimensions=(1.0, 1.0, 1.0), weight=1.0)
    decision = PlacementDecision(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    for i in range(5):
        await writer.enqueue_step(f"game-{i}", box, decision, 0.0)
    await writer.enqueue_game_end("game-0", 0.8, None)

    await writer.stop()

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) >= 1

    with open(jsonl_files[0]) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_parquet_write(tmp_path):
    pytest.importorskip("pyarrow")

    writer = ArrowWriter(tmp_path, mode="dev", algorithm_name="naive", buffer_size=2)
    await writer.start()

    box = Box(id="b1", dimensions=(1.0, 1.0, 1.0), weight=1.0)
    decision = PlacementDecision(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    for i in range(5):
        await writer.enqueue_step("game-1", box, decision, float(i) * 0.1)
    await writer.enqueue_game_end("game-1", 0.9, None)

    await writer.stop()

    import pyarrow.parquet as pq
    parquet_files = list(tmp_path.glob("steps_*.parquet"))
    assert len(parquet_files) >= 1
    table = pq.read_table(str(parquet_files[0]))
    assert table.num_rows >= 1
    assert "game_id" in table.schema.names
