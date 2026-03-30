import datetime
import time
import uuid

from dataclasses import dataclass
from typing import Iterable

from .box_catalog import BoxCatalog
from .physics import rotated_half_extents, PhysicsSimulator

TRUCK = {"depth": 2.0, "width": 2.6, "height": 2.75}


@dataclass
class GameRecord:
    game_id: str
    api_key: str
    mode: str
    box_queue: list[dict]
    placed_boxes: list[dict]
    game_status: str
    termination_reason: str | None
    density: float
    created_at: float
    updated_at: float


class GameEngine:
    ALLOWED_MODES = {"dev", "compete"}

    def __init__(
        self,
        catalog: BoxCatalog,
        physics: PhysicsSimulator,
        master_seed: int = 0,
        compete_limit: int = 50,
    ):
        self._catalog = catalog
        self._physics = physics
        self._master_seed = master_seed
        self._compete_limit = compete_limit
        self._games: dict[str, GameRecord] = {}
        self._compete_starts: dict[str, list[float]] = {}

    def start_game(self, api_key: str, mode: str) -> GameRecord:
        self._validate_api_key(api_key)
        if mode not in self.ALLOWED_MODES:
            raise ValueError("invalid_mode")
        if mode == "compete":
            self._compete_limit_check(api_key)
        game_id = f"g_{uuid.uuid4().hex[:12]}"
        seed = int(game_id.split("_", 1)[1], 16) ^ self._master_seed
        box_queue = list(self._catalog.generate(seed))
        now = time.time()
        game = GameRecord(
            game_id=game_id,
            api_key=api_key,
            mode=mode,
            box_queue=box_queue.copy(),
            placed_boxes=[],
            game_status="in_progress",
            termination_reason=None,
            density=0.0,
            created_at=now,
            updated_at=now,
        )
        self._games[game_id] = game
        return game

    def place_box(
        self,
        game_id: str,
        box_id: str,
        position: list[float],
        orientation_wxyz: list[float],
    ) -> GameRecord:
        if len(position) != 3 or len(orientation_wxyz) != 4:
            raise ValueError("validation_error")
        game = self._games.get(game_id)
        if not game or game.game_status != "in_progress":
            raise ValueError("invalid_game_id")
        if not game.box_queue:
            raise ValueError("invalid_game_id")
        current_box = game.box_queue[0]
        if current_box["id"] != box_id:
            raise ValueError(f"invalid_box_id:{current_box['id']}:{box_id}")
        try:
            settled_position, settled_orientation, _ = self._physics.place(
                game.mode,
                current_box["dimensions"],
                position,
                orientation_wxyz,
                list(game.placed_boxes),
            )
        except ValueError:
            raise ValueError("validation_error")
        placed = {
            "id": current_box["id"],
            "dimensions": current_box["dimensions"],
            "position": settled_position,
            "orientation_wxyz": settled_orientation,
            "displaced": False,
        }
        game.placed_boxes.append(placed)
        game.box_queue.pop(0)
        newly_displaced = self._evaluate_displacements(game)
        game.density = self._compute_density(game)
        self._check_termination(game, newly_displaced)
        game.updated_at = time.time()
        return game

    def get_status(self, game_id: str) -> GameRecord:
        game = self._games.get(game_id)
        if not game:
            raise ValueError("invalid_game_id")
        return game

    def stop_game(self, api_key: str, game_id: str) -> GameRecord:
        self._validate_api_key(api_key)
        game = self._games.get(game_id)
        if not game or game.game_status == "completed":
            raise ValueError("invalid_game_id")
        game.game_status = "completed"
        game.termination_reason = "player_stop"
        game.box_queue.clear()
        game.updated_at = time.time()
        return game

    def get_my_games(
        self, api_key: str, mode: str | None, status: str | None
    ) -> dict:
        self._validate_api_key(api_key)
        games = [g for g in self._games.values() if g.api_key == api_key]
        daily_games = self._compete_starts.get(api_key, [])
        today_start = self._utc_day_start(time.time())
        games_today = sum(1 for ts in daily_games if ts >= today_start)
        summary = self._compute_summary(games, games_today)
        filtered = games
        if mode:
            filtered = [g for g in filtered if g.mode == mode]
        if status:
            filtered = [g for g in filtered if g.game_status == status]
        filtered = sorted(filtered, key=lambda g: g.created_at, reverse=True)
        games_list = [self._game_to_dict(g) for g in filtered]
        return {
            "api_key": api_key,
            "display_name": None,
            "summary": summary,
            "games": games_list,
        }

    def _compute_summary(self, games: Iterable[GameRecord], games_today: int) -> dict:
        games = list(games)
        completed = [g for g in games if g.game_status == "completed"]
        completed_compete = [
            g for g in completed if g.mode == "compete"
        ]
        avg_compete = (
            sum(g.density for g in completed_compete) / len(completed_compete)
            if completed_compete
            else None
        )
        best_compete = (
            max((g.density for g in completed_compete), default=None)
            if completed_compete
            else None
        )
        return {
            "total_games": len(games),
            "completed_games": len(completed),
            "in_progress_games": len([g for g in games if g.game_status == "in_progress"]),
            "avg_compete_density": avg_compete,
            "best_compete_density": best_compete,
            "games_today": games_today,
            "daily_limit": self._compete_limit,
        }

    def _game_to_dict(self, game: GameRecord) -> dict:
        return {
            "game_id": game.game_id,
            "mode": game.mode,
            "status": game.game_status,
            "density": game.density,
            "boxes_placed": len(game.placed_boxes),
            "total_boxes": len(game.placed_boxes) + len(game.box_queue),
            "termination_reason": game.termination_reason,
            "created_at": self._isoformat(game.created_at),
            "updated_at": self._isoformat(game.updated_at),
        }

    def _isoformat(self, ts: float) -> str:
        return (
            datetime.datetime.fromtimestamp(
                ts, datetime.timezone.utc
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )

    def _evaluate_displacements(self, game: GameRecord) -> int:
        footprints = [self._build_footprint(box) for box in game.placed_boxes]
        newly_displaced = 0
        for idx, box in enumerate(game.placed_boxes):
            fp = footprints[idx]
            support = self._support_area(fp, footprints, idx)
            ratio = support / fp["area"] if fp["area"] > 0 else 1.0
            box["_support_ratio"] = ratio
            if ratio < 0.25 and not box.get("displaced", False):
                box["displaced"] = True
                newly_displaced += 1
        return newly_displaced

    def _build_footprint(self, box: dict) -> dict[str, float]:
        hx, hy, hz = rotated_half_extents(
            box["dimensions"], box["orientation_wxyz"]
        )
        cx, cy, cz = box["position"]
        area = max((2 * hx) * (2 * hy), 1e-9)
        return {
            "x_min": cx - hx,
            "x_max": cx + hx,
            "y_min": cy - hy,
            "y_max": cy + hy,
            "bottom_z": cz - hz,
            "top_z": cz + hz,
            "area": area,
        }

    def _support_area(
        self, target: dict[str, float], footprints: list[dict[str, float]], idx: int
    ) -> float:
        bottom = target["bottom_z"]
        if bottom <= 1e-6:
            return target["area"]
        area = 0.0
        for j, other in enumerate(footprints):
            if j == idx:
                continue
            if other["top_z"] <= bottom + 1e-6:
                overlap_x = max(0.0, min(target["x_max"], other["x_max"]) - max(target["x_min"], other["x_min"]))
                overlap_y = max(0.0, min(target["y_max"], other["y_max"]) - max(target["y_min"], other["y_min"]))
                area += overlap_x * overlap_y
        return min(area, target["area"])

    def _compute_density(self, game: GameRecord) -> float:
        if not game.placed_boxes:
            return 0.0
        total_volume = 0.0
        max_x = 0.0
        for box in game.placed_boxes:
            dims = box["dimensions"]
            total_volume += dims[0] * dims[1] * dims[2]
            hx = rotated_half_extents(dims, box["orientation_wxyz"])[0]
            max_x = max(max_x, box["position"][0] + hx)
        if max_x <= 0.0:
            return 0.0
        return total_volume / (TRUCK["width"] * TRUCK["height"] * max_x)

    def _check_termination(self, game: GameRecord, newly_displaced: int) -> None:
        if newly_displaced >= 3:
            game.game_status = "completed"
            game.termination_reason = "unstable"
            game.box_queue.clear()
            return
        if not game.box_queue:
            game.game_status = "completed"
            game.termination_reason = None
        else:
            game.game_status = "in_progress"
            game.termination_reason = None

    def _validate_api_key(self, api_key: str) -> None:
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("invalid_api_key")

    def _compete_limit_check(self, api_key: str) -> None:
        now = time.time()
        today = self._utc_day_start(now)
        timestamps = [ts for ts in self._compete_starts.get(api_key, []) if ts >= today]
        if len(timestamps) >= self._compete_limit:
            raise ValueError("rate_limited")
        timestamps.append(now)
        self._compete_starts[api_key] = timestamps

    def _utc_day_start(self, ts: float) -> float:
        dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
        day = datetime.datetime(dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc)
        return day.timestamp()
