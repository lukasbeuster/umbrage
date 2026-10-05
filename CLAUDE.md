# CLAUDE.md: umbrage

Lukas Beuster's general-purpose shade engine. MIT. Created 2026-10-05, before the NUS Shade x
Cardio contract commenced, deliberately as a separate repo so it stays his pre-existing IP.

Rules:
- Nothing project-specific in here: no Singapore paths, no NUS data, no client config.
  Project repos (e.g. ShadexCardio) depend on umbrage as a package.
- Keep numpy and torch paths numerically identical (tests/test_backends.py enforces it).
- `solweig` (GPL-3.0) stays an optional, lazily imported backend for cross-checks; do not
  copy code from it or from UMEP into umbrage.
- Every behaviour change gets a synthetic test with a known answer.

Candidates for later: horizon-angle precompute for many timesteps; GPU shadow mapping on
meshes (true 3D, overhangs); per-tile max height to cut steps; shaded-hours accumulator.
