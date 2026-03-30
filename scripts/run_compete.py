#!/usr/bin/env python
"""Run compete-mode games (rate-limited to 50/day)."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dexterity.algorithms.naive import NaiveAlgorithm
from dexterity.client import DexterityClient
from dexterity.scheduler import GameScheduler
from dexterity.storage import ArrowWriter, NullWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    key = os.environ.get("DEXTERITY_API_KEY")
    if not key:
        raise EnvironmentError(
            "DEXTERITY_API_KEY environment variable is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    return key


async def main():
    parser = argparse.ArgumentParser(description="Run compete-mode games")
    parser.add_argument("n_games", type=int, help="Number of games (max 50/day)")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--output-dir", default="data/compete")
    parser.add_argument("--algorithm", default="naive", choices=["naive", "cpp_so"])
    parser.add_argument("--so-path")
    args = parser.parse_args()

    if args.n_games > 50:
        logger.warning("n_games=%d exceeds daily limit of 50; server will rate-limit", args.n_games)

    api_key = get_api_key()

    def algorithm_factory():
        if args.algorithm == "naive":
            return NaiveAlgorithm()
        elif args.algorithm == "cpp_so":
            from dexterity.algorithms.cpp_so import CppSoAlgorithm
            return CppSoAlgorithm(args.so_path)
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    writer = ArrowWriter(args.output_dir, mode="compete", algorithm_name=args.algorithm)

    async with DexterityClient(api_key=api_key, mode="compete") as client:
        await writer.start()
        try:
            scheduler = GameScheduler(
                client,
                algorithm_factory,
                writer,
                mode="compete",
                max_concurrent=args.max_concurrent,
            )
            total = 0
            async for result in scheduler.run_games(args.n_games):
                total += 1
                logger.info(
                    "[%d/%d] game=%s density=%.4f reason=%s",
                    total,
                    args.n_games,
                    result.game_id,
                    result.density,
                    result.termination_reason,
                )
        finally:
            await writer.stop()


if __name__ == "__main__":
    asyncio.run(main())
