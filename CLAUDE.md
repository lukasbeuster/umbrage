# CLAUDE.md: Umbrage

General-purpose shade engine by Lukas Beuster. MIT. Display name "Umbrage"; package, import and
repo name `umbrage` (lowercase).

Rules:
- Nothing project-specific in here: no site paths, no client data, no client config. Projects
  depend on umbrage as a package and pin a release tag.
- Keep numpy and torch paths numerically identical (tests/test_backends.py enforces it).
- `solweig` (GPL-3.0) stays an optional, lazily imported backend; do not copy code from it or
  from UMEP into umbrage.
- Every behaviour change gets a synthetic test with a known answer.
- Public repo: no client names, contracts or project data in code, docs, commit messages or figures.

Planned for 0.2 (breaking): `tz` required in `daylight_timesteps`, `year_table`, `zenith_dates`;
daylight window defaults to the whole day, filtered by sun elevation; no top-level `SGT` export.

To do (release):
- `CITATION.cff` and a Zenodo link so each release gets a DOI (not added yet; decide author
  metadata and ORCID first).
- Report the solweig GPU input-cache behaviour upstream (UMEP), with the fake-backend test as the
  reproduction.

Candidates for later: horizon-angle precompute for many timesteps; GPU shadow mapping on
meshes (true 3D, overhangs); per-tile max height to cut steps; shaded-hours accumulator.
