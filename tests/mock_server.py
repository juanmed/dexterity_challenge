"""Stateful in-process mock Dexterity API server (FastAPI)."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def make_app(*, require_api_key: str = "test-key", n_boxes: int = 5) -> FastAPI:
    app = FastAPI()

    games: dict[str, dict] = {}

    def _make_box(box_id: str | None = None) -> dict:
        bid = box_id or f"box-{uuid.uuid4().hex[:6]}"
        return {"id": bid, "dimensions": [1.0, 1.0, 1.0], "weight": 1.0}

    def _err(error: str, message: str, status: int = 400):
        return JSONResponse(
            status_code=status,
            content={"error": error, "message": message},
        )

    @app.post("/start")
    async def start(body: dict):
        if body.get("api_key") != require_api_key:
            return _err("invalid_api_key", "bad key", 401)
        game_id = str(uuid.uuid4())
        boxes = [_make_box() for _ in range(n_boxes)]
        games[game_id] = {
            "game_id": game_id,
            "mode": body.get("mode", "dev"),
            "boxes": boxes,
            "placed": [],
            "step": 0,
            "status": "in_progress",
        }
        return {
            "game_id": game_id,
            "truck": {"depth": 2.0, "width": 2.6, "height": 2.75},
            "current_box": boxes[0],
            "boxes_remaining": len(boxes) - 1,
            "mode": body.get("mode", "dev"),
        }

    @app.post("/place")
    async def place(body: dict):
        game_id = body.get("game_id")
        if game_id not in games:
            return _err("invalid_game_id", "no such game", 404)
        game = games[game_id]
        if game["status"] != "in_progress":
            return _err("invalid_game_id", "game already ended", 404)

        box_id = body.get("box_id")
        step = game["step"]
        current_box = game["boxes"][step] if step < len(game["boxes"]) else None
        if current_box is None or current_box["id"] != box_id:
            return _err("invalid_box_id", f"expected {current_box['id'] if current_box else 'none'}, got {box_id}", 400)

        placed_box = {
            "id": current_box["id"],
            "position": body.get("position", [0.0, 0.0, 0.0]),
            "orientation_wxyz": body.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0]),
            "dimensions": current_box["dimensions"],
        }
        game["placed"].append(placed_box)
        game["step"] += 1

        next_step = game["step"]
        next_box = game["boxes"][next_step] if next_step < len(game["boxes"]) else None
        boxes_remaining = max(0, len(game["boxes"]) - next_step - 1)
        density = len(game["placed"]) / n_boxes

        if next_box is None:
            game["status"] = "completed"

        return {
            "status": "ok",
            "placed_boxes": game["placed"],
            "current_box": next_box,
            "boxes_remaining": boxes_remaining,
            "density": density,
            "game_status": game["status"],
            "termination_reason": None,
        }

    @app.get("/status/{game_id}")
    async def status(game_id: str):
        if game_id not in games:
            raise HTTPException(status_code=404, detail={"error": "game_not_found", "message": "no such game"})
        game = games[game_id]
        step = game["step"]
        current_box = game["boxes"][step] if step < len(game["boxes"]) else None
        boxes_remaining = max(0, len(game["boxes"]) - step - 1)
        density = len(game["placed"]) / n_boxes
        return {
            "game_id": game_id,
            "game_status": game["status"],
            "mode": game["mode"],
            "boxes_placed": len(game["placed"]),
            "boxes_remaining": boxes_remaining,
            "density": density,
            "placed_boxes": game["placed"],
            "current_box": current_box,
        }

    @app.post("/stop")
    async def stop(body: dict):
        if body.get("api_key") != require_api_key:
            return _err("invalid_api_key", "bad key", 401)
        game_id = body.get("game_id")
        if game_id not in games:
            return _err("game_not_found", "no such game", 404)
        games[game_id]["status"] = "completed"
        games[game_id]["termination_reason"] = "player_stop"
        return {"status": "ok"}

    @app.get("/my-games")
    async def my_games(api_key: str = Query(...), mode: str | None = Query(None), status: str | None = Query(None)):
        if api_key != require_api_key:
            return _err("invalid_api_key", "bad key", 401)
        result = list(games.values())
        if mode:
            result = [g for g in result if g["mode"] == mode]
        if status:
            result = [g for g in result if g["status"] == status]
        return {"games": result}

    return app
