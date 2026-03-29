# Plan 06-01 Summary: Transition Dataset and Replay Buffer

**Completed:** 2026-03-29
**Duration:** ~10 minutes
**Status:** ✅ All tasks complete

## What Was Built

### `src/mimic_sepsis_rl/datasets/transitions.py`
Core transition builder that converts the frozen state, action, and reward
surfaces into `(s_t, a_t, r_t, s_{t+1}, done)` tuples. Includes:
- `TransitionRow` frozen dataclass with episode metadata
- `TransitionDatasetMeta` for provenance tracking (spec versions, split labels)
- Episode and batch transition builders
- DataFrame export and Parquet persistence
- Metadata JSON serialisation

### `src/mimic_sepsis_rl/datasets/replay_buffer.py`
Episode-aware replay buffer wrapping the transition contract:
- `EpisodeBuffer` groups transitions by stay_id
- `ReplayBuffer` with episode iteration and flat access
- Structural validation (done semantics, state dim, action range)
- Parquet + JSON persistence

### `src/mimic_sepsis_rl/cli/build_transitions.py`
CLI entrypoint with `--dry-run` validation using synthetic data.

### `tests/datasets/test_transitions.py`
27 regression tests covering:
- Episode transition shape and done handling
- State vector null/NaN → 0.0 extraction
- Multi-episode batch transitions
- DataFrame export
- Metadata serialisation round-trip
- Replay buffer grouping and validation
- Persistence round-trips
- Determinism

## Verification

- ✅ `pytest -q tests/datasets/test_transitions.py` → 27 passed
- ✅ `python -m mimic_sepsis_rl.cli.build_transitions --dry-run` → PASSED
- ✅ Episode boundaries and done semantics verified
