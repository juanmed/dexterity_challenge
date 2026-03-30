from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


def ensure_length(value: list[float], expected: int, name: str) -> list[float]:
    if len(value) != expected:
        raise ValueError(f"{name} must have length {expected}")
    return value


class Box(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    dimensions: list[float]
    weight: float

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 3, "dimensions")


class PlacedBox(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    dimensions: list[float]
    position: list[float]
    orientation_wxyz: list[float]

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 3, "dimensions")

    @field_validator("position")
    @classmethod
    def _check_position(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 3, "position")

    @field_validator("orientation_wxyz")
    @classmethod
    def _check_orientation(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 4, "orientation_wxyz")


class TruckDims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth: float
    width: float
    height: float


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str
    mode: str = "compete"


class PlaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str
    box_id: str
    position: list[float]
    orientation_wxyz: list[float]

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 3, "position")

    @field_validator("orientation_wxyz")
    @classmethod
    def _validate_orientation(cls, v: list[float]) -> list[float]:
        return ensure_length(v, 4, "orientation_wxyz")


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str
    game_id: str


class StartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str
    truck: TruckDims
    current_box: Box
    boxes_remaining: int
    mode: str


class PlaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    placed_boxes: list[PlacedBox]
    current_box: Box | None
    boxes_remaining: int
    density: float
    game_status: str
    termination_reason: str | None


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str
    game_status: str
    mode: str
    boxes_placed: int
    boxes_remaining: int
    density: float
    placed_boxes: list[PlacedBox]
    current_box: Box | None


class StopResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    density: float
    game_status: str
    termination_reason: str


class MyGamesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str
    display_name: str | None
    summary: dict
    games: list[dict]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
