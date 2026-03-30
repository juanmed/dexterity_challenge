import asyncio
import os

from fastapi import FastAPI, HTTPException

from .box_catalog import BoxCatalog
from .engine import GameEngine, TRUCK
from .models import (
    Box,
    HealthResponse,
    MyGamesResponse,
    PlaceRequest,
    PlaceResponse,
    StartRequest,
    StartResponse,
    StatusResponse,
    StopRequest,
    StopResponse,
    TruckDims,
    PlacedBox,
)
from .physics import PhysicsSimulator

app = FastAPI(title="Dexterity Challenge Mock Server")

_ERROR_MAP = {
    "invalid_api_key": (401, "API key is missing or invalid."),
    "invalid_game_id": (404, "Game not found or already completed."),
    "invalid_box_id": (400, "box_id does not match current box."),
    "rate_limited": (429, "Daily compete-mode game limit reached."),
    "invalid_mode": (400, "mode must be 'dev' or 'compete'."),
    "validation_error": (422, "Invalid input."),
}


def _raise_from_engine_error(
    error_code: str,
    context: dict | None = None,
    message_override: str | None = None,
) -> None:
    http_status, message = _ERROR_MAP.get(error_code, (500, "Internal error."))
    if message_override:
        message = message_override
    if error_code == "invalid_box_id" and context:
        expected = context.get("expected")
        received = context.get("received")
        if expected is not None and received is not None:
            message = f"Expected box_id '{expected}', got '{received}'"
    raise HTTPException(
        status_code=http_status,
        detail={"error": error_code, "message": message},
    )


MOCK_SEED = int(os.getenv("MOCK_SEED", "42"))
COMPETE_LIMIT = int(os.getenv("MOCK_COMPETE_LIMIT", "50"))

catalog = BoxCatalog(seed=MOCK_SEED)
physics = PhysicsSimulator()
engine = GameEngine(catalog, physics, master_seed=MOCK_SEED, compete_limit=COMPETE_LIMIT)
lock = asyncio.Lock()


@app.get("/challenge/api/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post("/challenge/api/start", response_model=StartResponse)
async def start_game(req: StartRequest):
    async with lock:
        try:
            game = engine.start_game(req.api_key, req.mode)
        except ValueError as exc:
            _raise_from_engine_error(str(exc))
    current_box = Box(**game.box_queue[0])
    return StartResponse(
        game_id=game.game_id,
        truck=TruckDims(**TRUCK),
        current_box=current_box,
        boxes_remaining=max(len(game.box_queue) - 1, 0),
        mode=game.mode,
    )


@app.post("/challenge/api/place", response_model=PlaceResponse)
async def place_box(req: PlaceRequest):
    async with lock:
        try:
            game = engine.place_box(
                req.game_id, req.box_id, req.position, req.orientation_wxyz
            )
        except ValueError as exc:
            error_code = str(exc)
            if error_code.startswith("invalid_box_id:"):
                _, expected, received = error_code.split(":", 2)
                _raise_from_engine_error(
                    "invalid_box_id",
                    context={"expected": expected, "received": received},
                )
            _raise_from_engine_error(error_code)
    placed_boxes = [PlacedBox(**box) for box in game.placed_boxes]
    current_box = Box(**game.box_queue[0]) if game.box_queue else None
    status_text = "ok" if game.game_status == "in_progress" else "terminated"
    boxes_remaining = len(game.box_queue)
    return PlaceResponse(
        status=status_text,
        placed_boxes=placed_boxes,
        current_box=current_box,
        boxes_remaining=boxes_remaining,
        density=game.density,
        game_status=game.game_status,
        termination_reason=game.termination_reason,
    )


@app.get("/challenge/api/status/{game_id}", response_model=StatusResponse)
async def status(game_id: str):
    async with lock:
        try:
            game = engine.get_status(game_id)
        except ValueError:
            _raise_from_engine_error(
                "invalid_game_id", message_override="Game not found."
            )
    return StatusResponse(
        game_id=game.game_id,
        game_status=game.game_status,
        mode=game.mode,
        boxes_placed=len(game.placed_boxes),
        boxes_remaining=len(game.box_queue),
        density=game.density,
        placed_boxes=[PlacedBox(**box) for box in game.placed_boxes],
        current_box=Box(**game.box_queue[0]) if game.box_queue else None,
    )


@app.post("/challenge/api/stop", response_model=StopResponse)
async def stop_game(req: StopRequest):
    async with lock:
        try:
            game = engine.stop_game(req.api_key, req.game_id)
        except ValueError:
            _raise_from_engine_error("invalid_game_id")
    return StopResponse(
        status="ok",
        density=game.density,
        game_status=game.game_status,
        termination_reason=game.termination_reason,
    )


@app.get("/challenge/api/my-games", response_model=MyGamesResponse)
async def my_games(api_key: str, mode: str | None = None, status: str | None = None):
    async with lock:
        try:
            payload = engine.get_my_games(api_key, mode, status)
        except ValueError as exc:
            _raise_from_engine_error(str(exc))
    return MyGamesResponse(**payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mock_server.server:app",
        host=os.getenv("MOCK_HOST", "0.0.0.0"),
        port=int(os.getenv("MOCK_PORT", "8000")),
        log_level=os.getenv("MOCK_LOG_LEVEL", "info"),
    )
