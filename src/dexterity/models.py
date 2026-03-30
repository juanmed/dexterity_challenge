from __future__ import annotations
from pydantic import BaseModel


class Box(BaseModel):
    id: str
    dimensions: tuple[float, float, float]
    weight: float


class PlacedBox(BaseModel):
    id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    dimensions: tuple[float, float, float]


class TruckDims(BaseModel):
    depth: float
    width: float
    height: float


class StartResponse(BaseModel):
    game_id: str
    truck: TruckDims
    current_box: Box
    boxes_remaining: int
    mode: str


class PlaceResponse(BaseModel):
    status: str
    placed_boxes: list[PlacedBox]
    current_box: Box | None
    boxes_remaining: int
    density: float
    game_status: str
    termination_reason: str | None


class StatusResponse(BaseModel):
    game_id: str
    game_status: str
    mode: str
    boxes_placed: int
    boxes_remaining: int
    density: float
    placed_boxes: list[PlacedBox]
    current_box: Box | None


class PlacementDecision(BaseModel):
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    stop: bool = False


class GameResult(BaseModel):
    game_id: str
    density: float
    termination_reason: str | None = None
