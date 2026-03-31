# Physics Simulation Backend: ComFree-Warp Box Packing Solver

## Implementation Plan for the Dexterity AI Foresight Challenge

**Status**: Final (includes Codex gpt-5.1-codex-mini review feedback)
**Date**: 2026-03-31

### Executive Summary

This plan connects ComFree-Warp (GPU-accelerated complementarity-free contact physics) to the Dexterity box packing challenge. The simulator evaluates candidate box placements in parallel using `nworld` batched GPU simulation. All placed boxes are modeled as freejoints (not static geoms) to detect cascading instability, with old/deep boxes frozen at runtime via velocity zeroing and position locking. Scoring uses the competition's density formula and stability threshold internally. The architecture separates scene building (MJCFBuilder), physics evaluation (ComFreeSimulator), and candidate generation into independent modules behind a `PhysicsSim` protocol.

---

## 1. Simulator Choice: ComFree-Warp (Primary)

**Decision: Start with ComFree-Warp.** Reasons:

- Already integrated as a git submodule and workspace dependency
- Native GPU parallelism via `nworld` parameter (1024+ parallel environments on one GPU)
- Analytical (complementarity-free) contact model — deterministic, no iterative solver convergence issues
- Built on NVIDIA Warp (JIT-compiled GPU kernels) + MuJoCo 3.6.0 for model loading
- `wp.ScopedCapture` graph compilation eliminates per-step launch overhead
- Benchmark infrastructure already exists (`test_local/test_throuput_hand.py`)

**MuJoCo (CPU)**: Keep as a debugging fallback for when no GPU is available.

**AVBD (Augmented Vertex Block Descent)**: Defer as a future option. SIGGRAPH 2025 research shows 110K+ rigid blocks at 3.5ms/frame on RTX 4090 with only 4 iterations. However, it's a research prototype without a packaged library — would require implementing a new simulator wrapper. Consider as a drop-in replacement if ComFree-Warp performance is insufficient.

**Risk — server physics mismatch**: The competition server uses its own physics engine. Our simulator is an approximation for internal evaluation of candidate placements. Mitigate by tuning contact parameters (`comfree_stiffness`, `comfree_damping`, friction) to match server behavior empirically. The server-reported `PlacedBox` positions from `/place` responses are ground truth — use them to validate and calibrate our simulator.

---

## 2. Architecture Overview

```
+---------------------------------------------------+
|                   GameRunner                      |
|        (queries API, gets current_box,            |
|                 calls algorithm)                  |
+---------------------------------------------------+
                        |
                        ▼
+---------------------------------------------------+
|               Algorithm.decide()                  |
|         (search strategy — out of scope           |
|                  for this plan)                   |
|                                                   |
|  1. Generates N candidate (position,              |
|     orientation) pairs                            |
|  2. Calls PhysicsSim.evaluate_batch(N candidates) |
|  3. Receives SimResult (settled_pos, stability,   |
|     density)                                      |
|  4. Picks best candidate                          |
+---------------------------------------------------+
                        |
                        ▼
+---------------------------------------------------+
|               PhysicsSim (Protocol)               |
|                                                   |
|  build_scene(truck, placed_boxes) → scene_handle  |
|  evaluate_batch(scene, box, positions,            |
|    orientations)                                  |
|    → SimResult (settled_pos, n_displaced,         |
|      density)                                     |
|                                                   |
|  Implementations:                                 |
|    ComFreeSimulator (GPU, nworld parallel)        |
|    MuJoCoSimulator  (CPU, single-world fallback)  |
+---------------------------------------------------+
                        |
                        ▼
+---------------------------------------------------+
|                   MJCFBuilder                     |
|  build_truck_model(truck, placed_boxes,           |
|    candidate_box)                                 |
|    → (MjModel, MjData)                            |
|                                                   |
|  Truck = floor + ceiling + 4 walls (static        |
|    planes)                                        |
|  All boxes = free bodies (freejoint)              |
|  Old/deep boxes = marked static (frozen DOF)      |
|  Candidate box = free body with freejoint         |
+---------------------------------------------------+
```

---

## 3. Component Design

### 3.1 MJCFBuilder (`src/dexterity/sim/mjcf_builder.py`)

Programmatically builds MuJoCo models using `mujoco.MjSpec` (not XML string templating):

**Truck representation:**

- Floor plane at z=0
- Ceiling plane at z=2.75 (optional, prevents upward escape)
- Back wall at x=0 (prevents backward sliding)
- Front wall at x=2.0 (truck depth limit)
- Left wall at y=0
- Right wall at y=2.6

**Box representation — all placed boxes as freejoints:**

- **Every placed box is a free body** with `<freejoint>`. Each box has 7 qpos (pos xyz + quat wxyz) + 6 qvel DOF in the simulation state.
- **Old/deep boxes are frozen at runtime** by zeroing their velocity and locking qpos after each step (not by making them static geoms).
- **Candidate box**: One additional free body with `<freejoint>`.

**Why freejoints for all boxes (not static geoms for old ones):**

- Static geoms cannot be displaced — making old boxes static creates a permanent blind spot for cascading instability.
- With freejoints, we can selectively "unfreeze" any box for a full-dynamic stability sweep (e.g., every K placements).
- Enables detection of cascading collapses where a new box disturbs an old one, which then disturbs another.
- The displacement threshold should be adaptive per box size: `displaced = |delta_pos| > alpha * min(box_dims)`

**Which boxes to freeze vs leave dynamic:**

- Default: freeze all boxes except the most recent K (K=5-10, tunable) and the candidate.
- Additionally, identify "critical" boxes — those near overhangs, edges, or with low support ratios — and keep them dynamic regardless of age.
- Periodically run a full-dynamic sweep (all boxes unfrozen) to catch cascading instabilities.
- When the number of freejoints × 6 > 60 DOFs, use **sparse jacobian** mode (no nv_max limit). Dense jacobian is capped at nv_max=60 in `mujoco_warp/_src/io.py:136-138`.

```python
def build_truck_model(
    truck: TruckDims,
    placed_boxes: list[PlacedBox],
    candidate_box: Box,
    dynamic_indices: list[int] | None = None,  # which placed boxes remain dynamic
    timestep: float = 0.002,
    friction: float = 1.0,
    comfree_stiffness: float = 0.2,
    comfree_damping: float = 0.01,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Build a MuJoCo model for the truck packing scene.
    
    If dynamic_indices is None, all placed boxes are dynamic.
    Otherwise, only the specified indices are dynamic; the rest are frozen
    at runtime (still have freejoints but velocity zeroed each step).
    
    Uses sparse jacobian when total DOFs > 60.
    """
```

**Quaternion convention:** Both the challenge API and MuJoCo use wxyz. No conversion needed.

**Physics settings:**

```
timestep: 0.002s (small enough for stable box stacking)
integrator: EULER (fastest, sufficient for settling)
gravity: [0, 0, -9.81]
cone: pyramidal (condim=3 for tangent + normal friction)
friction: ~1.0 (high friction prevents sliding)
comfree_stiffness: 0.2 (default, tune to match server)
comfree_damping: 0.01 (10× default for faster settling)
jacobian: sparse (when nv > 60, i.e., >10 dynamic boxes)
```

### 3.2 PhysicsSim Protocol (`src/dexterity/sim/protocol.py`)

```python
@dataclass
class SimResult:
    settled_positions: np.ndarray     # (N, 3) — candidate box final position per world
    settled_orientations: np.ndarray  # (N, 4) wxyz
    n_displaced: np.ndarray           # (N,) int — boxes moved beyond threshold per world
    is_stable: np.ndarray             # (N,) bool — n_displaced < 3
    density: np.ndarray               # (N,) float — projected density for this placement
    steps_to_settle: np.ndarray       # (N,) int — how many steps until convergence

class PhysicsSim(Protocol):
    def build_scene(self, truck: TruckDims, placed_boxes: list[PlacedBox]) -> Any: ...
    def evaluate_batch(
        self, scene: Any, candidate_box: Box,
        positions: np.ndarray,       # (N, 3)
        orientations: np.ndarray,    # (N, 4) wxyz
        n_settle_steps: int = 200,
    ) -> SimResult: ...
    def max_batch_size(self) -> int: ...
```

**Scoring — density + stability only (internal heuristic, not from mock server):**

- `density`: `sum_box_volumes / (max_x_extent × truck_width × truck_height)` — matches the competition's density formula exactly. `max_x_extent` is the furthest x-coordinate of any box's forward face (position_x + rotated_half_extent_x).
- `is_stable`: `n_displaced < 3` — mirrors the competition's termination condition (3+ displaced boxes = game over with "unstable").
- Displacement: a box is displaced when its footprint support ratio drops below 0.25 (same as competition).
- Candidates where `is_stable == False` are immediately disqualified. Among stable candidates, maximize `density`.

### 3.3 ComFreeSimulator (`src/dexterity/sim/comfree_sim.py`)

Core GPU implementation. Workflow for `evaluate_batch`:

1. **Build model** via MJCFBuilder → `(mjm, mjd)` with candidate at dummy position

2. **Upload to GPU**:
   ```python
   m = cfwarp.put_model(mjm, comfree_stiffness=0.2, comfree_damping=0.01)
   ```

3. **Create N parallel worlds**:
   ```python
   d = cfwarp.put_data(mjm, mjd, nworld=N, nconmax=64, njmax=256)
   ```

4. **Set per-world initial qpos**: Write each candidate's `(pos, orient)` into `d.qpos[i, candidate_offset:candidate_offset+7]` for each world i. All other boxes (placed) share the same initial qpos across worlds.
   ```python
   qpos_array = np.tile(mjd.qpos, (N, 1)).astype(np.float32)
   for i, (pos, orient) in enumerate(zip(positions, orientations)):
       qpos_array[i, candidate_qpos_offset:candidate_qpos_offset+3] = pos
       qpos_array[i, candidate_qpos_offset+3:candidate_qpos_offset+7] = orient
   wp.copy(d.qpos, wp.array(qpos_array, dtype=wp.float32))
   ```

5. **Compile step graph**:
   ```python
   cfwarp.step(m, d)  # warm-up (triggers Warp JIT compilation)
   cfwarp.step(m, d)  # second warm-up
   with wp.ScopedCapture() as capture:
       cfwarp.step(m, d)
   graph = capture.graph
   ```

6. **Settle with adaptive early termination**:
   ```python
   for batch in range(20):  # 20 batches × 10 steps = 200 max
       for _ in range(10):
           wp.capture_launch(graph)
       wp.synchronize()
       # Freeze old/deep boxes: zero their velocity in d.qvel, restore their qpos
       _freeze_static_boxes(d, frozen_indices, frozen_qpos)
       # Check settling via kinetic energy
       if _all_settled(d, threshold=1e-4):
           break
   ```

7. **Read back**: Extract `d.qpos.numpy()` → settled positions/orientations for each world

8. **Detect displacement**: Compare all dynamic boxes' final positions against initial. Threshold is adaptive per box: `displaced = |delta_pos| > alpha * min(box_dims)`.

9. **Compute density**: For each world, calculate `sum_volumes / (max_x_extent × width × height)`.

**Freezing boxes at runtime (instead of static geoms):**

After each step batch, for frozen boxes: zero their `d.qvel` entries and restore their `d.qpos` entries to the known-good positions. This is done via a Warp kernel operating on `d.qvel` and `d.qpos` arrays, which is cheap since it's a simple memory write across all worlds.

```python
@wp.kernel
def freeze_boxes_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    frozen_qpos: wp.array2d(dtype=float),  # (n_frozen, 7) target positions
    frozen_offsets: wp.array(dtype=int),     # qpos offset for each frozen box
    n_frozen: int,
):
    worldid, frozen_idx = wp.tid()
    if frozen_idx >= n_frozen:
        return
    offset = frozen_offsets[frozen_idx]
    # Restore position
    for i in range(7):
        qpos[worldid, offset + i] = frozen_qpos[frozen_idx, i]
    # Zero velocity (6 DOFs per freejoint)
    vel_offset = offset  # qvel offset (assumes ordering matches)
    for i in range(6):
        qvel[worldid, vel_offset + i] = 0.0
```

**Batching for N > max_batch_size:** Split candidates into batches. Between batches, use `cfwarp.reset_data(m, d, reset=per_world_mask)` for selective per-world reset.

**Key API signatures (verified from codebase):**

- `cfwarp.put_model(mjm, comfree_stiffness=..., comfree_damping=...)` → `Model`
- `cfwarp.put_data(mjm, mjd, nworld=N, nconmax=64, njmax=1000)` → `Data`
- `cfwarp.reset_data(m, d, reset: wp.array(dtype=bool))` — per-world selective reset
- `cfwarp.step(m, d)` — single physics step across all worlds
- `d.qpos` — shape `(nworld, nq)`, `wp.array2d(dtype=float)`
- `d.qvel` — shape `(nworld, nv)`, `wp.array2d(dtype=float)`

**Contact budget sizing:**

For box packing with K dynamic boxes + 1 candidate:
- Each box has 6 faces, each face can contact floor/walls/other boxes
- Worst case: ~4 contacts per dynamic box = ~(K+1)×4 contacts per world
- With K=9 dynamic + 1 candidate: 40 contacts × condim=3 = 120 constraint rows
- `nconmax = 64` per world is safe
- `njmax = 256` per world provides margin

### 3.4 CandidateGenerator (`src/dexterity/search/candidates.py`)

Discretizes the placement space:

**Orientations:** A box with 3 distinct dimensions has 6 unique axis-aligned orientations (mapped to quaternions):

| Rotation     | Quaternion [w, x, y, z] | Effect               |
|--------------|-------------------------|----------------------|
| Identity     | [1, 0, 0, 0]            | No rotation          |
| 90° around Z | [0.707, 0, 0, 0.707]    | Swap length ↔ width  |
| 90° around Y | [0.707, 0, 0.707, 0]    | Swap length ↔ height |
| 90° around X | [0.707, 0.707, 0, 0]    | Swap width ↔ height  |
| 180° around Z| [0, 0, 0, 1]            | Flip length & width  |
| 90° X then Z | [0.5, 0.5, 0.5, 0.5]    | Full axis permutation|

For boxes with duplicate dimensions, some orientations are equivalent — deduplicate.

**Positions:** For each orientation:

1. Compute rotated AABB half-extents
2. Generate a grid of (x, y) positions with configurable spacing (default 5cm)
3. For each (x, y), compute z from highest support surface below (analytical, no sim needed)
4. Filter: remove positions that collide with existing boxes (AABB overlap check) or exceed truck bounds

**Expected candidate count:** ~500-2000 before pruning. With 6 orientations × ~300 valid positions per orientation. After AABB pruning: ~100-500.

**Heuristic pruning (fast, CPU):**

- Discard positions with no support (floating in air)
- Discard positions overlapping existing boxes (AABB intersection)
- Discard positions extending beyond truck wall bounds
- Prioritize positions that minimize wasted space (pack toward back wall first)

---

## 4. Parallelism Architecture

### Within a single game step (deciding one box):

```
CPU Thread                              GPU
----------                              ---
Generate candidates (grid + prune)
          |
          |--- Build MjSpec, compile --> put_model + put_data(nworld=N)
          |                              Compile step graph (ScopedCapture)
          |
          |--- Set per-world qpos -----> wp.copy(d.qpos, ...)
          |
          |                              Launch graph × 200 steps
          |                              (with freeze-box kernel between batches)
          |                              wp.synchronize()
          |
          |--- Read results <---------- d.qpos.numpy()
          |
          Compute density + stability per world (CPU or GPU)
          Return SimResult to caller
```

### Across multiple games (data collection):

- **Single GPU, sequential per game**: Each `decide()` blocks until GPU finishes. Multiple games run sequentially on same GPU.
- **Multi-GPU**: Assign different games to different GPUs via `wp.set_device("cuda:N")`. One `ComFreeSimulator` instance per GPU.

**Recommendation**: Start single-GPU. Multi-GPU when collecting training data at scale.

### Depth search parallelism (for future search algorithm integration):

For depth search (simulating future placements on top of current candidates):
- After settling depth-0 candidates, read back settled qpos for top-K worlds
- Rebuild model with the chosen candidate box now frozen as a placed box
- Re-run `evaluate_batch` for the next box's candidates
- Each depth level requires model rebuild (~20ms overhead: MjSpec build + compile + put_model + put_data + graph capture)
- With 2-second time budget per `decide()`, this allows ~100 evaluations across depth levels

---

## 5. Static/Frozen Body Strategy

### 5.1 Tiered Freeze Policy

```
Tier 0 — FROZEN (velocity zeroed, position locked each step batch):
  - All boxes placed more than K steps ago
  - Still have freejoints in the model (can be unfrozen)
  - Zero computational cost for dynamics but still participate in collision detection
  
Tier 1 — DYNAMIC (full physics simulation):
  - The candidate box being evaluated
  - The most recently placed K boxes (K=5-10, tunable)
  - Any "critical" box flagged as potentially unstable (low support ratio, near overhang)
  
Full-dynamic sweep:
  - Every M placements (M=10), run one evaluation with ALL boxes unfrozen
  - Catches cascading instabilities that frozen-box runs miss
  - Higher cost but provides ground-truth stability assessment
```

### 5.2 Freeze Implementation

After every settling sub-batch (e.g., every 10 physics steps):

1. Launch `freeze_boxes_kernel` to zero velocities and restore positions of frozen boxes
2. This prevents frozen boxes from drifting due to numerical noise
3. Frozen boxes still generate contacts (they have geoms) and support dynamic boxes

### 5.3 Sparse vs Dense Jacobian

- **Dense jacobian**: nv_max=60 limit (`io.py:136`). With 6 DOFs per freejoint, this allows max 10 dynamic boxes.
- **Sparse jacobian**: No nv limit. Required when total freejoints × 6 > 60.
- **Strategy**: With 120 boxes all as freejoints, nv=726. Must use sparse jacobian.
- **Alternative dense strategy**: If freezing is truly rigid (qpos locked, qvel zeroed), we could use `mujoco.mj_disable` flags or custom constraint to effectively reduce active DOFs. But the Warp kernels still iterate over all nv — sparse is the cleaner approach.

### 5.4 GPU Memory Considerations

With all 120 boxes as freejoints:
- qpos: (nworld, 847) float32 — 121 bodies × 7 qpos each
- qvel: (nworld, 726) float32 — 121 bodies × 6 qvel each
- For N=256 worlds: ~800KB for qpos + qvel alone
- Contact arrays (`nconmax=64`, `njmax=256`): additional ~256KB per world
- Total for 256 worlds: ~250MB — fits comfortably on modern GPUs (8GB+)

---

## 6. Settling Detection

### 6.1 Kinetic Energy Threshold

After each sub-batch of steps, check if all worlds have settled:

```python
@wp.kernel
def check_settled(
    qvel: wp.array2d(dtype=float),
    dynamic_offsets: wp.array(dtype=int),  # velocity offsets of dynamic boxes
    n_dynamic: int,
    threshold: float,  # e.g., 1e-4
    settled_out: wp.array(dtype=bool),     # (nworld,)
):
    worldid = wp.tid()
    ke = float(0.0)
    for d in range(n_dynamic):
        offset = dynamic_offsets[d]
        for i in range(6):  # 6 DOFs per freejoint
            v = qvel[worldid, offset + i]
            ke += v * v
    settled_out[worldid] = ke < threshold
```

### 6.2 Adaptive Stepping

```
Max budget: 200 steps (0.002s × 200 = 0.4s simulated time)
Check every 10 steps: 20 check points

for batch in range(20):
    for _ in range(10):
        wp.capture_launch(graph)
    wp.synchronize()
    freeze_static_boxes(...)
    if all_settled(d, threshold=1e-4):
        break
```

### 6.3 High Damping for Fast Settling

- `comfree_damping = 0.01` (10× default) to aggressively damp oscillations
- Consider per-joint `dof_damping` for additional viscous damping on freejoints
- Trades physical accuracy for speed — acceptable since we only need stable final positions

### 6.4 Timeout Handling

If settling not reached within step budget, score based on current state but apply a penalty multiplier (e.g., 0.5× density). This prevents the simulator from blocking the search algorithm indefinitely.

---

## 7. Key Technical Considerations

### 7.1 Model Recompilation Overhead

Each game step requires rebuilding the MjSpec (new box was placed) → `compile()` → `put_model()`. Estimated costs:

- `MjSpec.compile()`: ~1ms for 120 static geoms + 1-10 freejoints
- `cfwarp.put_model()`: ~5ms
- `cfwarp.put_data()`: ~2ms
- CUDA graph capture (2 warmup steps + capture): ~10ms

Total overhead per `decide()` call: ~20ms. Acceptable within a 2-second time budget.

**Optimization opportunities:**
- Cache compiled models by body count — if the model structure hasn't changed, reuse
- Pre-compile a model with max body count (121 bodies) and enable/disable via position tricks
- Profile each stage separately to identify bottlenecks

### 7.2 ComFree-Warp Contact Tuning

`comfree_stiffness` (default 0.2) and `comfree_damping` (default 0.001) control settling:

- Higher damping (0.01-0.05) → faster settling, fewer steps needed
- Higher stiffness → less interpenetration, but may cause oscillation if too high
- Run an **automated parameter sweep** against competition server outcomes
- Log divergences between simulated and server-reported positions to validate contact model
- The competition server reports settled positions via `PlaceResponse.placed_boxes` — compare these against simulator predictions

### 7.3 Displacement Detection

- Compare each dynamic box's final position against its initial position
- **Adaptive threshold per box size**: `displaced = |delta_pos| > alpha * min(box_dims)` where alpha ∈ [0.05, 0.1]
- Also check orientation change: `|delta_quat| > beta` (quaternion distance)
- Count displaced boxes per world; `is_stable = n_displaced < 3`
- For frozen boxes, displacement is zero by construction (they're locked)

### 7.4 CUDA Graph Constraints

CUDA graphs capture fixed memory addresses. Critical constraints:
- `nconmax` and `njmax` must be sized conservatively at capture time (cannot grow dynamically)
- Data arrays must not be reallocated between capture and launch
- Different model structures require new graph captures (one per `decide()` call)
- Writing to `d.qpos`/`d.qvel` between launches is fine (same memory, different values)

---

## 8. File Structure

```
src/dexterity/
  sim/
    __init__.py               # Public API exports
    mjcf_builder.py           # MjSpec scene construction (truck + boxes)
    protocol.py               # PhysicsSim Protocol, SimResult, SimConfig
    comfree_sim.py            # ComFreeSimulator — GPU implementation
    scoring.py                # Warp kernels: density, stability, settling detection
    freeze.py                 # Warp kernels: freeze/unfreeze box DOFs
  
  search/
    candidates.py             # CandidateGenerator: orientations, grid, pruning
  
  algorithms/
    sim_search.py             # Algorithm subclass using simulator
    base.py                   # (existing) Algorithm ABC
    naive.py                  # (existing) baseline

tests/
  test_mjcf_builder.py        # Scene construction unit tests
  test_comfree_sim.py          # GPU simulator integration tests
  test_scoring.py              # Scoring kernel tests
  test_candidates.py           # Candidate generation tests
  bench_simulator.py           # Performance benchmarks
```

---

## 9. Integration with Algorithm Interface

### SimSearchAlgorithm (`src/dexterity/algorithms/sim_search.py`)

```python
class SimSearchAlgorithm(Algorithm):
    def __init__(self, sim_config: SimConfig):
        self._sim: PhysicsSim = ComFreeSimulator(sim_config)
        self._candidates_gen = CandidateGenerator(
            grid_spacing=sim_config.grid_spacing,
        )
    
    def setup(self, truck: TruckDims) -> None:
        self._truck = truck
        self._sim.build_scene(truck, placed_boxes=[])
    
    def decide(
        self, current_box: Box, placed_boxes: list[PlacedBox],
        boxes_remaining: int, density: float,
    ) -> PlacementDecision:
        # 1. Rebuild scene with current placed boxes
        scene = self._sim.build_scene(self._truck, placed_boxes)
        
        # 2. Generate candidate placements
        candidates = self._candidates_gen.generate(
            current_box, placed_boxes, self._truck,
        )
        positions = np.array([c.position for c in candidates])
        orientations = np.array([c.orientation for c in candidates])
        
        # 3. Evaluate all candidates in parallel on GPU
        result = self._sim.evaluate_batch(
            scene, current_box, positions, orientations,
        )
        
        # 4. Filter unstable, pick best density
        valid_mask = result.is_stable
        if not valid_mask.any():
            # All candidates unstable — pick least bad
            best_idx = result.n_displaced.argmin()
        else:
            valid_densities = np.where(valid_mask, result.density, -np.inf)
            best_idx = valid_densities.argmax()
        
        return PlacementDecision(
            position=tuple(result.settled_positions[best_idx]),
            orientation_wxyz=tuple(result.settled_orientations[best_idx]),
        )
    
    def teardown(self, final_density: float, termination_reason: str | None) -> None:
        pass  # cleanup GPU resources if needed
```

---

## 10. Implementation Phases

### Phase 1: Minimal Viable Simulator
**Goal**: Single box drop on floor, verify settling.

1. `mjcf_builder.py` — truck walls + 1 dynamic candidate box (no placed boxes yet)
2. `comfree_sim.py` — single-world (nworld=1), no CUDA graph, manual step loop
3. `scoring.py` — CPU-side numpy density calculation
4. Verify: drop a box, check it settles at z = half_height

### Phase 2: Full Scene + Batched Evaluation
**Goal**: Evaluate N candidates in parallel.

5. `mjcf_builder.py` — add placed boxes as freejoints, freeze logic
6. `comfree_sim.py` — nworld=N, CUDA graph capture, per-world qpos init
7. `scoring.py` — GPU settling detection kernel
8. `freeze.py` — freeze/unfreeze Warp kernels
9. Verify: place 10 boxes, evaluate 64 candidates for box 11, check settled positions

### Phase 3: Scoring + Stability
**Goal**: Density and stability scoring match competition rules.

10. `scoring.py` — density kernel (sum_volumes / envelope), stability kernel (footprint support ratio)
11. `protocol.py` — complete SimResult with all fields
12. Validate against known outcomes from mock server runs

### Phase 4: Candidate Generation
**Goal**: Automated candidate placement generation.

13. `candidates.py` — 6 orientations, grid positions, AABB pruning
14. `sim_search.py` — full Algorithm integration
15. Run against mock server, compare density vs naive algorithm

### Phase 5: Tuning + Depth Search
**Goal**: Match server physics, enable lookahead.

16. Contact parameter sweep (stiffness, damping, friction) against server outcomes
17. State forking for depth search (model rebuild per depth level)
18. Performance profiling and optimization

---

## 11. Verification Approach

### Unit Tests

**Scene Builder** (`tests/test_mjcf_builder.py`):
- Truck walls at correct positions and orientations
- Box half-extents computed correctly from dimensions
- Model compiles for 0, 1, 50, 120 placed boxes + 1 candidate
- Sparse jacobian mode activated when nv > 60

**Scoring** (`tests/test_scoring.py`):
- Known density = hand-computed value for simple arrangements
- Box on floor = stable (support ratio 1.0)
- Box hanging off edge = unstable (support ratio < 0.25)
- Settling: zero velocity → settled, non-zero → not settled

### Integration Tests

**Simulator** (`tests/test_comfree_sim.py`):
- Single box drops to floor at z=half_height
- Box stacked on box settles at z=bottom_box_top + half_height
- Out-of-bounds placement: box falls, detected as displaced
- Batch consistency: same candidate in all N worlds → same result
- Freeze kernel: frozen box velocity stays zero after stepping

**End-to-End** (`tests/test_sim_integration.py`):
- Run SimSearchAlgorithm against mock server
- Verify no "unstable" termination (0 displaced boxes)
- Verify density improves over NaiveAlgorithm baseline

### Performance Benchmarks

**Latency** (`tests/bench_simulator.py`):
- `build_scene()` latency for 0..120 placed boxes
- `evaluate_batch()` latency for N=64, 128, 256, 512 candidates
- Steps-to-settle distribution across random placements
- Target: `build_scene()` < 30ms, `evaluate_batch(64)` < 100ms including settling

### Accuracy Validation

- Compare simulator predicted settled position vs server-reported `PlacedBox.position`
- Track position prediction error over full 120-box games
- Tune `comfree_stiffness`/`comfree_damping`/`friction` to minimize prediction error
- If error is consistently high, increase dynamic box count K

---

## 12. Configuration

```python
@dataclasses.dataclass
class SimConfig:
    # Physics
    timestep: float = 0.002
    comfree_stiffness: float = 0.2
    comfree_damping: float = 0.01    # 10× default for faster settling
    box_friction: float = 1.0
    box_condim: int = 3              # tangent + normal
    
    # Batching
    n_candidates: int = 64           # parallel worlds per evaluation
    nconmax_per_world: int = 64
    njmax_per_world: int = 256
    
    # Settling
    max_settle_steps: int = 200
    settle_check_interval: int = 10
    settle_ke_threshold: float = 1e-4
    
    # Freeze policy
    n_recent_dynamic: int = 5        # keep last K boxes dynamic
    full_sweep_interval: int = 10    # full-dynamic sweep every M placements
    displacement_alpha: float = 0.1  # adaptive threshold multiplier
    
    # Scoring
    unsettled_penalty: float = 0.5   # density multiplier for unsettled placements
    
    # Candidate generation
    grid_spacing: float = 0.05       # 5cm grid for position candidates
    
    # Device
    warp_device: str = "cuda:0"
```

---

## 13. Codex Feedback Integration

The following considerations were raised by Codex (gpt-5.1-codex-mini) review and integrated into this plan:

### 13.1 Freeze Kernel Must Be Graph-Capture-Safe

The freeze kernel (Section 3.3) writes to `d.qpos` and `d.qvel` between CUDA graph launches. This is safe because CUDA graphs capture kernel launch parameters and memory addresses, not memory contents — writing new values to the same arrays between `wp.capture_launch()` calls is a supported pattern. However, the freeze kernel itself must NOT be inside the captured graph (it runs between graph launches, not as part of the stepping graph). The frozen/unfrozen set must remain stable within a single `evaluate_batch()` call — it only changes between `decide()` calls when the model is rebuilt.

### 13.2 Broadphase Sync on Unfreeze

When running full-dynamic sweeps (all boxes unfrozen, Section 5.1), the broadphase collision detection must be re-synced. Frozen boxes may have accumulated positional error from numerical noise that was corrected by the freeze kernel. Before an unfreeze sweep, run `cfwarp.forward(m, d)` (which includes kinematics + collision) before stepping, to ensure the collision graph is consistent.

### 13.3 Jacobian Row Zeroing for Frozen DOFs

Simply zeroing velocity and restoring position may not fully decouple frozen boxes from the contact solver — constraint Jacobian rows referencing frozen DOFs can still inject forces. Two mitigation strategies:
1. **Preferred**: After the freeze kernel, also zero the Jacobian columns corresponding to frozen DOFs in `d.efc.J`. This prevents frozen boxes from receiving constraint forces.
2. **Alternative**: Accept minor force artifacts on frozen boxes since their positions are restored anyway. This is simpler and may be sufficient if frozen boxes vastly outnumber dynamic ones.

Start with approach (2) and upgrade to (1) if stability issues arise.

### 13.4 Debug Mode for Constraint Monitoring

Add an optional debug mode that logs per-world active constraint counts (`d.nefc`) and contact counts (`d.ncon`) after each settling batch. This helps verify that the sparse Jacobian remains lean and that `nconmax`/`njmax` budgets are not being exceeded silently.

### 13.5 Memory Layout: AoS vs SoA

ComFree-Warp uses SoA layout internally (separate `d.qpos`, `d.qvel`, `d.efc.J` arrays). This is already optimal for GPU coalesced reads. No layout changes needed. Contact buffers should remain compactly sized (`nconmax=64`) to avoid wasting shared memory bandwidth.

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sparse jacobian slower than dense | Reduced evaluation throughput | Profile; if too slow, limit dynamic boxes to ≤9 and use dense |
| Model rebuild too slow (>30ms) | Limits search breadth/depth | Cache compiled models; pre-build with max body count |
| CUDA graph invalidated by model changes | Must re-capture per round | Accept ~10ms re-capture cost per `decide()` call |
| ComFree settling oscillations | Boxes never converge | Increase `comfree_damping`; add per-joint damping; use KE threshold with patience |
| Contact overflow (nconmax exceeded) | Silent clipping or crash | Set generous `nconmax=64`; enable debug constraint monitoring (Section 13.4) |
| Simulator prediction ≠ server physics | Wrong placements chosen | Validate against server; tune contact parameters; use server positions as ground truth |
| 120 freejoints × 256 worlds = high GPU memory | OOM on smaller GPUs | Reduce nworld or switch old boxes to static geoms as fallback |
| Frozen DOF Jacobian leakage | Stray forces on frozen boxes | Start with position-restore approach; upgrade to J-column zeroing if needed (Section 13.3) |
| Broadphase desync after freeze | Stale collision pairs on unfreeze | Run `cfwarp.forward()` before full-dynamic sweeps (Section 13.2) |
