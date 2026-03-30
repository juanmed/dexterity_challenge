#!/usr/bin/env python
"""Run N games in dev mode."""
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
from dexterity.runner import GameRunner
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
    parser = argparse.ArgumentParser(description="Run dev-mode games")
    parser.add_argument("n_games", type=int, help="Number of games to run")
    parser.add_argument("--max-concurrent", type=int, default=500)
    parser.add_argument("--output-dir", default="data/dev", help="Parquet output directory")
    parser.add_argument("--no-storage", action="store_true", help="Disable storage")
    parser.add_argument("--algorithm", default="naive", choices=["naive", "cpp_so"])
    parser.add_argument("--so-path", help="Path to .so for cpp_so algorithm")
    parser.add_argument("--resume-game-id", help="Resume an in-progress game")
    parser.add_argument("--base-url", default=None, help="Override API base URL (e.g. http://localhost:8000/challenge/api)")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.no_storage:
        writer = NullWriter()
    else:
        writer = ArrowWriter(args.output_dir, mode="dev", algorithm_name=args.algorithm)

    def algorithm_factory():
        if args.algorithm == "naive":
            return NaiveAlgorithm()
        elif args.algorithm == "cpp_so":
            from dexterity.algorithms.cpp_so import CppSoAlgorithm
            if not args.so_path:
                raise ValueError("--so-path required for cpp_so algorithm")
            return CppSoAlgorithm(args.so_path)
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    async with DexterityClient(api_key=api_key, mode="dev", base_url=args.base_url) as client:
        await writer.start()
        try:
            if args.resume_game_id:
                algo = algorithm_factory()
                runner = GameRunner(client, algo, writer, mode="dev")
                result = await runner.resume(args.resume_game_id)
                logger.info("Resumed game %s: density=%.4f", result.game_id, result.density)
            else:
                scheduler = GameScheduler(
                    client,
                    algorithm_factory,
                    writer,
                    mode="dev",
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
