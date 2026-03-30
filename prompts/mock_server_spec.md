# Mock REST Server — Implementation Specification

## Purpose

Build a self-contained Python REST server that faithfully emulates the Dexterity AI Foresight
Challenge API at `https://dexterity.ai/challenge/api`. The server runs locally (default
`http://localhost:8000`) and allows the agent client to run full game sessions without needing
a real API key or internet connection.

The server has two layers:
- **GameEngine** — pure Python class, no HTTP, that owns all game state and simulation logic.
- **FastAPI frontend** — thin HTTP layer that validates requests, calls GameEngine, and formats responses.

---

## Technology Stack

- **Python 3.12**
- **FastAPI** — HTTP framework (matches the real server; its Pydantic v2 validation error format is already known from real server probing)
- **Uvicorn** — ASGI server to run the app
- **Pydantic v2** — request/response validation (same version as the real server)
- **NumPy** — for quaternion math and 3D geometry in the physics simulation
- **uv** for dependency management

Dependencies to declare in `pyproject.toml`:
```toml
[project.optional-dependencies]
mock-server = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "numpy>=1.26",
]
```

Run with:
```bash
uv run uvicorn mock_server.server:app --host 0.0.0.0 --port 8000 --reload
```

---

## File Layout

```
mock_server/
├── __init__.py
├── server.py       # FastAPI app, all route handlers
├── engine.py       # GameEngine class — all game logic
├── models.py       # Pydantic request/response models
├── box_catalog.py  # Box sequence generator
└── physics.py      # Dev/compete mode placement simulation
```

---

## Coordinate System (must match the real API exactly)

```
        Z (up, height: 2.75 m)
        ^
        |
        +---------> Y (width: 2.6 m)
       /
      /
     v  X (depth: 2.0 m, loading direction)
```

- Origin at front-bottom-left corner of truck interior.
- **position** is the **centre** of the box.
- A box flush on the floor at the front-left corner with dims `[lx, ly, lz]` has
  position `[lx/2, ly/2, lz/2]`.
- Truck bounds: X ∈ [0, 2.0], Y ∈ [0, 2.6], Z ∈ [0, 2.75].

---

## Pydantic Models (`mock_server/models.py`)

All models use **Pydantic v2**. Every float is stored as `float` (Python). All lists/tuples
of coordinates are `list[float]` with exactly 3 or 4 elements enforced by a validator.

```python
# ---------- shared primitives ----------

class Box(BaseModel):
    id: str
    dimensions: list[float]        # exactly [length_x, width_y, height_z]
    weight: float

class PlacedBox(BaseModel):
    id: str
    dimensions: list[float]        # [lx, ly, lz]
    position: list[float]          # [cx, cy, cz] centre, post-settle
    orientation_wxyz: list[float]  # [w, x, y, z] unit quaternion, post-settle

class TruckDims(BaseModel):
    depth: float   # X axis, always 2.0
    width: float   # Y axis, always 2.6
    height: float  # Z axis, always 2.75

# ---------- request bodies ----------

class StartRequest(BaseModel):
    api_key: str
    mode: str = "compete"   # "dev" | "compete"

class PlaceRequest(BaseModel):
    game_id: str
    box_id: str
    position: list[float]          # must be length 3
    orientation_wxyz: list[float]  # must be length 4; auto-normalised

class StopRequest(BaseModel):
    api_key: str
    game_id: str

# ---------- success responses ----------

class StartResponse(BaseModel):
    game_id: str
    truck: TruckDims
    current_box: Box
    boxes_remaining: int
    mode: str

class PlaceResponse(BaseModel):
    status: str                    # "ok" | "terminated"
    placed_boxes: list[PlacedBox]
    current_box: Box | None
    boxes_remaining: int
    density: float
    game_status: str               # "in_progress" | "completed"
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

class StopResponse(BaseModel):
    status: str        # "ok"
    density: float
    game_status: str   # "completed"
    termination_reason: str  # "player_stop"

class MyGamesResponse(BaseModel):
    api_key: str
    display_name: str | None
    summary: dict       # see /my-games section below
    games: list[dict]

class HealthResponse(BaseModel):
    status: str   # "ok"
```

---

## Error Response Format

**All errors** use the `{"detail": ...}` envelope that the real FastAPI server returns.

Application errors (401, 404, 400, 429):
```json
{
  "detail": {
    "error": "<error_code>",
    "message": "<human readable>"
  }
}
```

FastAPI/Pydantic validation errors (422) are generated automatically by FastAPI and look like:
```json
{
  "detail": [
    {
      "type": "string_type",
      "loc": ["body", "game_id"],
      "msg": "Input should be a valid string",
      "input": null
    }
  ]
}
```

Do **not** manually generate 422 errors — let FastAPI's built-in request validation produce them.

### Error codes

| HTTP | error field | Trigger |
|---|---|---|
| 401 | `invalid_api_key` | api_key not in the server's accepted key list |
| 404 | `invalid_game_id` | game_id not found, or game already completed |
| 400 | `invalid_box_id` | box_id does not match current_box.id |
| 429 | `rate_limited` | compete mode game limit exceeded (configurable, default 50/day) |
| 422 | (FastAPI auto) | missing field, wrong type, wrong list length |

Helper to raise application errors consistently:
```python
from fastapi import HTTPException

def app_error(http_status: int, error: str, message: str):
    raise HTTPException(
        status_code=http_status,
        detail={"error": error, "message": message}
    )
```

---

## Box Catalog (`mock_server/box_catalog.py`)

The server must generate realistic box sequences. Real-world parcel dimensions are used.

```python
class BoxCatalog:
    """
    Generates a repeatable sequence of boxes for a game.
    Each game gets the same sequence for a given seed, ensuring reproducibility.
    """

    # Representative parcel size pool (length_x, width_y, height_z in metres, weight in kg)
    BOX_POOL = [
        (0.600, 0.400, 0.400, 10.0),
        (0.500, 0.400, 0.300,  8.0),
        (0.400, 0.300, 0.200,  4.0),
        (0.350, 0.280, 0.190,  3.5),
        (0.406, 0.310, 0.211,  3.1),
        (0.300, 0.200, 0.150,  2.0),
        (0.250, 0.200, 0.150,  1.5),
        (0.200, 0.150, 0.100,  1.0),
        (0.450, 0.350, 0.250,  6.0),
        (0.550, 0.400, 0.300,  9.0),
        (0.380, 0.280, 0.200,  4.5),
        (0.320, 0.240, 0.180,  3.0),
    ]

    GAME_LENGTH = 120   # number of boxes per game (matches real API)

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, game_seed: int) -> list[dict]:
        """
        Returns a list of GAME_LENGTH box dicts, each with keys:
          id (str), dimensions ([lx, ly, lz]), weight (float)
        Box IDs are unique 6-digit strings derived from the game seed.
        """
        rng = random.Random(game_seed)
        boxes = []
        for i in range(self.GAME_LENGTH):
            lx, ly, lz, w = rng.choice(self.BOX_POOL)
            # Add ±10% uniform noise to each dimension for variety
            lx = round(lx * rng.uniform(0.9, 1.1), 3)
            ly = round(ly * rng.uniform(0.9, 1.1), 3)
            lz = round(lz * rng.uniform(0.9, 1.1), 3)
            w  = round(w  * rng.uniform(0.9, 1.1), 2)
            box_id = str(rng.randint(100000, 999999))
            boxes.append({"id": box_id, "dimensions": [lx, ly, lz], "weight": w})
        return boxes
```

---

## Physics Simulation (`mock_server/physics.py`)

Two modes with different behaviour:

### Dev mode physics

Boxes land **exactly** where the agent places them. No settling, no collision detection.
The placed position and orientation are stored as-is.

```python
def dev_place(
    position: list[float],
    orientation_wxyz: list[float],
) -> tuple[list[float], list[float]]:
    """Returns (settled_position, settled_orientation) identical to input."""
    q = normalize_quaternion(orientation_wxyz)
    return (list(position), q)
```

### Compete mode physics (simplified simulation)

Compete mode simulates gravity settling: a placed box falls straight down (−Z) from the
requested position until it rests on either the truck floor (Z = 0) or the top surface of
any already-placed box directly below it.

The simulation does **not** need to be physically accurate — it needs to be plausible and
fast. The following simplified algorithm is sufficient:

```
1. Rotate the box by the requested quaternion to get its axis-aligned bounding box (AABB).
2. Find the highest Z surface directly below the box's XY footprint among all placed boxes.
3. The box settles so its bottom face rests on that surface (Z_settle = surface_z + half_height_after_rotation).
4. X and Y positions remain as requested (no lateral drift).
5. Orientation remains as requested (no rotational settling).
6. After placement, check stability: if the box's AABB footprint has less than 50% overlap
   with support surfaces below, count it as potentially unstable.
```

**Stability check** (triggers game termination if threshold exceeded):
- After each placement, recheck all previously placed boxes.
- A box is "displaced" if it has moved more than 10 cm (0.1 m) from its originally placed position.
- If 3 or more boxes are displaced in a single placement step, set termination_reason = "unstable".
- In the simplified simulation: a box is marked displaced if it has no support beneath at
  least 25% of its XY footprint area. Accumulate displaced count across the game.

**Quaternion helpers** (implement in `physics.py`):

```python
def normalize_quaternion(q: list[float]) -> list[float]:
    """Normalise [w, x, y, z] to unit length. Raise ValueError if near-zero magnitude."""
    arr = np.array(q, dtype=float)
    mag = np.linalg.norm(arr)
    if mag < 1e-6:
        raise ValueError("Degenerate quaternion: near-zero magnitude")
    return (arr / mag).tolist()

def quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def rotated_half_extents(dims: list[float], q: list[float]) -> np.ndarray:
    """
    Given box dimensions [lx, ly, lz] and quaternion [w,x,y,z],
    return the half-extents of the axis-aligned bounding box after rotation.
    Result is a 3-vector [hx, hy, hz].
    """
    R = quaternion_to_rotation_matrix(q)
    half = np.array(dims) / 2.0
    return np.abs(R) @ half
```

---

## GameEngine Class (`mock_server/engine.py`)

`GameEngine` is the authoritative source of all game state. It is **not** async — it is a
pure synchronous Python class. Thread safety: the FastAPI app uses a single `threading.Lock`
around every `GameEngine` method call (or, since FastAPI with uvicorn is single-threaded by
default for async routes, a single `asyncio.Lock` is also acceptable).

```python
import uuid, time, random
from dataclasses import dataclass, field

TRUCK = {"depth": 2.0, "width": 2.6, "height": 2.75}
COMPETE_DAILY_LIMIT = 50


@dataclass
class GameRecord:
    game_id: str
    api_key: str
    mode: str                        # "dev" | "compete"
    box_queue: list[dict]            # remaining boxes to place, index 0 = current
    placed_boxes: list[dict]         # post-settle PlacedBox dicts
    game_status: str                 # "in_progress" | "completed"
    termination_reason: str | None
    density: float
    created_at: float                # time.time()
    updated_at: float


class GameEngine:
    """
    Manages all game state and simulation logic.
    All public methods are synchronous.
    Instantiate once and share across all requests.
    """

    def __init__(self, catalog: BoxCatalog, physics: PhysicsSimulator, master_seed: int = 0):
        self._catalog = catalog
        self._physics = physics
        self._games: dict[str, GameRecord] = {}
        self._master_seed = master_seed
        # compete-mode rate limiting: api_key → list of game start timestamps today
        self._compete_starts: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API (called by FastAPI route handlers)
    # ------------------------------------------------------------------

    def start_game(self, api_key: str, mode: str) -> GameRecord:
        """
        Create a new game session.

        - Generates a unique game_id: "g_" + uuid4().hex[:12]
        - Generates a box queue from BoxCatalog using seed derived from game_id
        - Enforces compete-mode daily limit (COMPETE_DAILY_LIMIT games per api_key per UTC day)
        - Returns the new GameRecord
        - Raises ValueError("rate_limited") if compete limit exceeded
        """

    def place_box(
        self,
        game_id: str,
        box_id: str,
        position: list[float],
        orientation_wxyz: list[float],
    ) -> GameRecord:
        """
        Place the current box.

        Validation (raises ValueError with specific error code strings):
          - game_id must exist and game_status must be "in_progress" → "invalid_game_id"
          - box_id must match game.box_queue[0]["id"] → "invalid_box_id"
          - position must be a list of exactly 3 floats → "validation_error"
          - orientation_wxyz must be a list of exactly 4 floats, non-degenerate → "validation_error"

        Simulation steps:
          1. Normalise the quaternion.
          2. Call physics.place(mode, position, orientation_wxyz, already_placed_boxes)
             which returns (settled_position, settled_orientation, displaced_count).
          3. Build a PlacedBox dict from settled pose.
          4. Append to game.placed_boxes.
          5. Remove the placed box from game.box_queue.
          6. Recompute density (see Density Calculation section).
          7. Check termination conditions (see Termination section).
          8. Update game.updated_at.
          9. Return updated GameRecord.
        """

    def get_status(self, game_id: str) -> GameRecord:
        """
        Return the current GameRecord.
        Raises ValueError("invalid_game_id") if not found.
        Note: completed games ARE returned (unlike /place which rejects them).
        """

    def stop_game(self, api_key: str, game_id: str) -> GameRecord:
        """
        End a game early.
        - Sets game_status = "completed", termination_reason = "player_stop".
        - Raises ValueError("invalid_game_id") if game not found or already completed.
        - api_key is validated but not checked against the game owner in the mock
          (the real server may check this; the mock accepts any valid key).
        """

    def get_my_games(self, api_key: str, mode: str | None, status: str | None) -> dict:
        """
        Return summary + filtered game list for the given api_key.
        Filters: mode ("dev"|"compete"|None), status ("in_progress"|"completed"|None).
        Returns a dict matching the MyGamesResponse schema (see /my-games section).
        """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_density(self, game: GameRecord) -> float:
        """
        density = sum(box_volumes) / (max_x_reached * truck_width * truck_height)

        max_x_reached = max over all placed boxes of (centre_x + half_extent_x_after_rotation)
        truck_width = 2.6, truck_height = 2.75

        If no boxes placed yet, density = 0.0.
        """

    def _check_termination(self, game: GameRecord, newly_displaced: int) -> None:
        """
        Sets game_status = "completed" and termination_reason if:
          - newly_displaced >= 3 → termination_reason = "unstable"
          - len(game.box_queue) == 0 → termination_reason = "completed" (all boxes placed)
        """

    def _validate_api_key(self, api_key: str) -> None:
        """
        The mock accepts ANY non-empty string as a valid API key.
        This lets the client tests run without a real key.
        Raises ValueError("invalid_api_key") if api_key is empty or None.
        """

    def _compete_limit_check(self, api_key: str) -> None:
        """
        Count compete-mode game starts for api_key today (UTC).
        Raises ValueError("rate_limited") if count >= COMPETE_DAILY_LIMIT.
        """
```

---

## FastAPI Routes (`mock_server/server.py`)

The FastAPI app is created as:
```python
app = FastAPI(title="Dexterity Challenge Mock Server")
```

A single `GameEngine` instance is created at module level and shared across all requests:
```python
catalog = BoxCatalog(seed=42)
physics = PhysicsSimulator()
engine = GameEngine(catalog, physics)
lock = asyncio.Lock()
```

Every route handler acquires `lock` before calling any `engine` method to ensure thread safety
under concurrent load.

---

### `GET /challenge/api/health`

No request body or parameters.

**Success response** — HTTP 200:
```json
{ "status": "ok" }
```

Implementation:
```python
@app.get("/challenge/api/health")
async def health():
    return {"status": "ok"}
```

---

### `POST /challenge/api/start`

**Request body** (JSON):
```json
{
  "api_key": "any_non_empty_string",
  "mode": "dev"
}
```

`mode` defaults to `"compete"` if omitted. Valid values: `"dev"`, `"compete"`.

**Success response** — HTTP 200:
```json
{
  "game_id": "g_a1b2c3d4e5f6",
  "truck": {
    "depth": 2.0,
    "width": 2.6,
    "height": 2.75
  },
  "current_box": {
    "id": "382947",
    "dimensions": [0.406, 0.310, 0.211],
    "weight": 3.13
  },
  "boxes_remaining": 120,
  "mode": "dev"
}
```

`boxes_remaining` = total boxes in queue - 1 (because current_box is already "in hand").
Wait — re-reading the API: `boxes_remaining` in `/start` = total queue length (120), and
`current_box` is the first box. Then after each `/place`, `boxes_remaining` decrements.
Use: `boxes_remaining = len(game.box_queue) - 1` (current box is index 0, it is not yet placed).

**Error responses**:
- HTTP 401: `api_key` empty → `{"detail": {"error": "invalid_api_key", "message": "API key is missing or invalid."}}`
- HTTP 400: `mode` not "dev" or "compete" → `{"detail": {"error": "invalid_mode", "message": "mode must be 'dev' or 'compete'"}}`
- HTTP 429: compete daily limit reached → `{"detail": {"error": "rate_limited", "message": "Daily compete-mode game limit reached."}}`

Implementation sketch:
```python
@app.post("/challenge/api/start")
async def start_game(req: StartRequest):
    async with lock:
        try:
            game = engine.start_game(req.api_key, req.mode)
        except ValueError as e:
            _raise_from_engine_error(str(e))
    current_box = game.box_queue[0]
    return StartResponse(
        game_id=game.game_id,
        truck=TruckDims(depth=2.0, width=2.6, height=2.75),
        current_box=Box(**current_box),
        boxes_remaining=len(game.box_queue) - 1,
        mode=game.mode,
    )
```

---

### `POST /challenge/api/place`

**Request body** (JSON):
```json
{
  "game_id": "g_a1b2c3d4e5f6",
  "box_id": "382947",
  "position": [0.203, 0.155, 0.106],
  "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

All four fields are required. `position` must have exactly 3 elements. `orientation_wxyz`
must have exactly 4 elements. The server auto-normalises the quaternion before simulation.

**Success response (game in progress)** — HTTP 200:
```json
{
  "status": "ok",
  "placed_boxes": [
    {
      "id": "382947",
      "dimensions": [0.406, 0.310, 0.211],
      "position": [0.203, 0.155, 0.106],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  ],
  "current_box": {
    "id": "519204",
    "dimensions": [0.350, 0.280, 0.190],
    "weight": 2.45
  },
  "boxes_remaining": 119,
  "density": 0.0031,
  "game_status": "in_progress",
  "termination_reason": null
}
```

**Success response (game terminated by unstable or all boxes placed)** — HTTP 200:
```json
{
  "status": "terminated",
  "placed_boxes": [ ... ],
  "current_box": null,
  "boxes_remaining": 0,
  "density": 0.4821,
  "game_status": "completed",
  "termination_reason": "unstable"
}
```

`termination_reason` values: `"unstable"`, `"player_stop"`, `null` (when last box placed normally).

`boxes_remaining` = number of boxes left in queue after this placement (0 when game over).

`placed_boxes` = full array of all placed boxes with their post-settle positions and orientations.

**Error responses**:
- HTTP 404: game not found or already completed → `{"detail": {"error": "invalid_game_id", "message": "Game not found or already completed."}}`
- HTTP 400: box_id mismatch → `{"detail": {"error": "invalid_box_id", "message": "Expected box_id '519204', got '382947'"}}`
- HTTP 422: missing field / wrong type / degenerate quaternion (FastAPI auto-generated or manually raised)

Implementation sketch:
```python
@app.post("/challenge/api/place")
async def place_box(req: PlaceRequest):
    async with lock:
        try:
            game = engine.place_box(
                req.game_id, req.box_id, req.position, req.orientation_wxyz
            )
        except ValueError as e:
            _raise_from_engine_error(str(e), req)
    next_box = game.box_queue[0] if game.box_queue else None
    return PlaceResponse(
        status="ok" if game.game_status == "in_progress" else "terminated",
        placed_boxes=[PlacedBox(**b) for b in game.placed_boxes],
        current_box=Box(**next_box) if next_box else None,
        boxes_remaining=len(game.box_queue) - (1 if game.box_queue else 0),
        density=game.density,
        game_status=game.game_status,
        termination_reason=game.termination_reason,
    )
```

---

### `GET /challenge/api/status/{game_id}`

**Path parameter**: `game_id` (string).

No request body. No api_key required.

**Success response** — HTTP 200:
```json
{
  "game_id": "g_a1b2c3d4e5f6",
  "game_status": "in_progress",
  "mode": "dev",
  "boxes_placed": 1,
  "boxes_remaining": 119,
  "density": 0.0031,
  "placed_boxes": [
    {
      "id": "382947",
      "dimensions": [0.406, 0.310, 0.211],
      "position": [0.203, 0.155, 0.106],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  ],
  "current_box": {
    "id": "519204",
    "dimensions": [0.350, 0.280, 0.190],
    "weight": 2.45
  }
}
```

`boxes_placed` = number of boxes in `placed_boxes` array.
`boxes_remaining` = boxes left in queue (not counting the current box in hand).
`current_box` is `null` if game is completed.

**Error responses**:
- HTTP 404: game not found → `{"detail": {"error": "invalid_game_id", "message": "Game not found."}}`

Note: unlike `/place`, this endpoint returns completed games (it does not raise 404 for them).

---

### `POST /challenge/api/stop`

**Request body** (JSON):
```json
{
  "api_key": "any_non_empty_string",
  "game_id": "g_a1b2c3d4e5f6"
}
```

**Success response** — HTTP 200:
```json
{
  "status": "ok",
  "density": 0.3241,
  "game_status": "completed",
  "termination_reason": "player_stop"
}
```

**Error responses**:
- HTTP 401: invalid api_key → `{"detail": {"error": "invalid_api_key", "message": "API key is missing or invalid."}}`
- HTTP 404: game not found or already completed → `{"detail": {"error": "invalid_game_id", "message": "Game not found or already completed."}}`
- HTTP 422: missing field (FastAPI auto-generated)

---

### `GET /challenge/api/my-games`

**Query parameters**:
- `api_key` (required, string)
- `mode` (optional, string: `"dev"` | `"compete"`)
- `status` (optional, string: `"in_progress"` | `"completed"`)

Example: `GET /challenge/api/my-games?api_key=dk_xxx&mode=dev`

**Success response** — HTTP 200:
```json
{
  "api_key": "dk_xxx",
  "display_name": null,
  "summary": {
    "total_games": 5,
    "completed_games": 4,
    "in_progress_games": 1,
    "avg_compete_density": 0.312,
    "best_compete_density": 0.421,
    "games_today": 3,
    "daily_limit": 50
  },
  "games": [
    {
      "game_id": "g_a1b2c3d4e5f6",
      "mode": "dev",
      "status": "completed",
      "density": 0.3241,
      "boxes_placed": 47,
      "total_boxes": 120,
      "termination_reason": "player_stop",
      "created_at": "2026-03-30T14:22:01Z",
      "updated_at": "2026-03-30T14:23:45Z"
    }
  ]
}
```

Games are returned **newest first** (sorted by `created_at` descending).

`summary` computation:
- `total_games` = count of all games for this api_key (ignoring mode/status filters — summary is always global)
- `completed_games` = count where `game_status == "completed"`
- `in_progress_games` = count where `game_status == "in_progress"`
- `avg_compete_density` = mean of `density` over completed compete-mode games (null if none)
- `best_compete_density` = max of `density` over completed compete-mode games (null if none)
- `games_today` = compete-mode games started today (UTC)
- `daily_limit` = 50

The `games` array IS filtered by the query parameters.

**Error responses**:
- HTTP 401: missing or empty api_key → `{"detail": {"error": "invalid_api_key", "message": "API key is missing or invalid."}}`

---

## Density Calculation (implement in `GameEngine._compute_density`)

```
density = sum_of_box_volumes / (max_x_reached * truck_width * truck_height)
```

Where:
- `sum_of_box_volumes` = sum of `lx * ly * lz` for all placed boxes (using original dimensions, not AABB)
- `max_x_reached` = maximum value of `centre_x + half_extent_x` over all placed boxes, where `half_extent_x` is the half-extent of the AABB along X after applying the settled rotation
- `truck_width` = 2.6 m (constant)
- `truck_height` = 2.75 m (constant)
- If no boxes placed: density = 0.0
- If `max_x_reached` == 0: density = 0.0 (guard against division by zero)

---

## Termination Conditions (implement in `GameEngine._check_termination`)

After each `/place` call:

1. **Unstable**: if the number of newly displaced boxes in this step >= 3, set
   `game_status = "completed"`, `termination_reason = "unstable"`. Clear `box_queue`.

2. **All boxes placed**: if `box_queue` is now empty after removing the just-placed box,
   set `game_status = "completed"`, `termination_reason = null`.

3. **In progress**: otherwise, leave `game_status = "in_progress"`, `termination_reason = null`.

"Displaced" definition for the mock (simplified):
In dev mode: no box is ever displaced (settled = requested).
In compete mode: a box is considered displaced if the Z component of its settled position
differs by more than 0.1 m from the requested Z position. This naturally happens when a box
is placed floating in the air and the physics drops it.

---

## Configuration and Startup

The server should accept the following environment variables or CLI arguments:

| Variable | Default | Description |
|---|---|---|
| `MOCK_HOST` | `0.0.0.0` | Bind address |
| `MOCK_PORT` | `8000` | Bind port |
| `MOCK_SEED` | `42` | Master RNG seed for box sequences |
| `MOCK_COMPETE_LIMIT` | `50` | Daily compete-mode game limit |
| `MOCK_LOG_LEVEL` | `info` | Uvicorn log level |

Entry point in `mock_server/server.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mock_server.server:app",
        host=os.getenv("MOCK_HOST", "0.0.0.0"),
        port=int(os.getenv("MOCK_PORT", "8000")),
        log_level=os.getenv("MOCK_LOG_LEVEL", "info"),
    )
```

---

## Error Dispatcher Helper

A single helper function converts `ValueError` error codes from `GameEngine` to FastAPI
`HTTPException` with the correct HTTP status and detail envelope:

```python
_ERROR_MAP = {
    "invalid_api_key":  (401, "API key is missing or invalid."),
    "invalid_game_id":  (404, "Game not found or already completed."),
    "invalid_box_id":   (400, "box_id does not match current box."),  # message overridden with specifics
    "rate_limited":     (429, "Daily compete-mode game limit reached."),
    "invalid_mode":     (400, "mode must be 'dev' or 'compete'."),
    "validation_error": (422, "Invalid input."),
}

def _raise_from_engine_error(error_code: str, context=None):
    http_status, message = _ERROR_MAP.get(error_code, (500, "Internal error."))
    raise HTTPException(
        status_code=http_status,
        detail={"error": error_code, "message": message}
    )
```

For `invalid_box_id`, override the message to include the expected and received IDs:
`f"Expected box_id '{expected}', got '{received}'"`.

---

## Testing the Mock Server

Once running, the existing `testAPI.py` script should work against it with no code changes
except setting:
```python
BASE = "http://localhost:8000/challenge/api"
API_KEY = "any_string"   # mock accepts any non-empty key
```

All endpoints must produce responses that are structurally identical to the real server
(same field names, same types, same HTTP status codes) so that the agent client code
developed against the mock will work against the real server without modification.
