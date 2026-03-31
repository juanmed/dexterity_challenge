# Simulation Implementation Notes

Date: 2026-03-31

This document describes the concrete implementation of the physics simulation backend, candidate generation, and algorithm integration, plus the commands to run and test the system.

## Overview

The implementation follows the plan in `prompts/sim_connection.md` and provides:

- A GPU-backed ComFree-Warp simulator with batched evaluation and CUDA graph capture.
- A candidate generator that enumerates axis-aligned orientations and grid positions.
- A simulation-driven algorithm that picks the best stable placement by density.
- Tests and a simple benchmark script.

## Key Components

### 1. Simulator Backend

**File**: `src/dexterity/sim/comfree_sim.py`

**Class**: `ComFreeSimulator`

**Key behaviors**
- Builds MuJoCo models through `build_truck_model`.
- Uses ComFree-Warp to run parallel worlds (`nworld = N candidates`).
- Performs warmup steps and captures CUDA graph when on GPU.
- Freezes old boxes by restoring `qpos` and zeroing `qvel` after each settle interval.
- Settles via kinetic energy threshold on dynamic bodies.
- Computes stability using footprint support ratio for every box + candidate.
- Computes density using the competition formula.

**Notes**
- The workspace has a namespace package `comfree_warp`, so the simulator includes a fallback import for the actual module (`comfree_warp.comfree_warp`).
- Candidate box positions are injected directly into `d.qpos` per world.

### 2. MJCF Builder

**File**: `src/dexterity/sim/mjcf_builder.py`

**Key behaviors**
- Builds the truck scene with floor, ceiling, and walls as planes.
- All boxes (including placed ones) are free joints to allow displacement checks.
- Uses `quat` (not `zaxis`) on geoms because `MjSpec` geoms do not expose `zaxis`.
- Switches to sparse Jacobian when total DOFs exceed 60.

### 3. Scoring and Stability

**File**: `src/dexterity/sim/scoring.py`

**Key behaviors**
- `compute_density` matches competition formula.
- `check_stability_single` uses footprint overlap ratio (threshold 0.25).

### 4. Candidate Generation

**Files**:
- `src/dexterity/search/candidates.py`
- `src/dexterity/search/__init__.py`

**Key behaviors**
- Six axis-aligned orientations (deduplicated for symmetric boxes).
- Grid positions across the truck floor.
- Z computed from highest supporting surface below (AABB overlap).
- Filters out collisions and out-of-bounds placements.
- Sorts by `x`, then `z`, then `y` to pack toward the back wall.

### 5. Algorithm Integration

**Files**:
- `src/dexterity/algorithms/sim_search.py`
- `src/dexterity/algorithms/__init__.py`

**Class**: `SimSearchAlgorithm`

**Behavior**
- Uses `CandidateGenerator` to enumerate candidates.
- Calls `ComFreeSimulator.evaluate_batch`.
- Filters unstable candidates; picks highest density.

## File Index

- `src/dexterity/sim/comfree_sim.py`
- `src/dexterity/sim/mjcf_builder.py`
- `src/dexterity/sim/scoring.py`
- `src/dexterity/sim/freeze.py`
- `src/dexterity/sim/protocol.py`
- `src/dexterity/sim/__init__.py`
- `src/dexterity/search/candidates.py`
- `src/dexterity/search/__init__.py`
- `src/dexterity/algorithms/sim_search.py`
- `src/dexterity/algorithms/__init__.py`
- `tests/test_mjcf_builder.py`
- `tests/test_scoring.py`
- `tests/test_candidates.py`
- `tests/test_comfree_sim.py`
- `tests/bench_simulator.py`

## Commands

### Run all tests

```bash
uv run pytest
```

### Run simulator unit test only

```bash
uv run pytest tests/test_comfree_sim.py
```

### Run benchmark

```bash
uv run python tests/bench_simulator.py
```

### Optional: Activate existing virtual env

If you want to reuse the active environment instead of UV's `.venv`:

```bash
uv run --active pytest
```

## Notes / Caveats

- `tests/test_comfree_sim.py` requires CUDA; it will skip if no GPU is available.
- The simulator currently uses CPU stability checks for footprint support ratio; GPU kernels can be added later if needed.
- ComFree parameters (`comfree_stiffness`, `comfree_damping`, friction) remain configurable in `SimConfig` for tuning.

