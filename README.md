# umbrage

*Umbrage: shade, from Latin* umbra.

Fast 2.5D shade from raster surface models. One small, tested kernel that runs on numpy,
on torch (CPU, Apple MPS, CUDA), or through UMEP's Rust `solweig` as a cross-check.
Built for national-scale binary shade at 1 m: many timesteps, buildings and vegetation,
no radiation budget.

```python
import numpy as np
from umbrage import cast_shadows, solar_position, SGT
from datetime import datetime

sun = solar_position(datetime(2026, 5, 15, 16, 0, tzinfo=SGT), 1.3521, 103.8198)
r = cast_shadows(dsm, res=1.0, azimuth=sun.azimuth, elevation=sun.elevation,
                 canopy_top=ct, canopy_base=cb, tau=0.03, device="mps")
r.building, r.vegetation, r.shade   # bool, bool, float32 in [0, 1]
```

## What is in here

| Module | Does |
|---|---|
| `umbrage.sun` | Solar position (NOAA algorithm, no dependencies; checked against pvlib to < 0.05 deg) |
| `umbrage.shade` | Shift-and-compare shadow casting (Ratti & Richens 2004), numpy or torch |
| `umbrage.tiling` | Tiles with halos for grids that do not fit in memory |
| `umbrage.backends.solweig_rs` | Optional wrapper around UMEP's Rust shadowing (`pip install solweig`, GPL-3.0) |

## Inputs and semantics

All heights absolute (m), same grid. `dsm` = terrain + opaque obstacles. `canopy_top`,
`canopy_base` = crown top and base, NaN where there is no vegetation; light passes crowns
with transmissivity `tau`. Outputs: `building` (shaded by the dsm), `vegetation` (ray
intercepted by a crown, independent of buildings), `shade` = 1 / 1 - tau / 0.

Differences from UMEP: SOLWEIG reports 1 = sunlit and its vegetation shadow excludes cells
already in building shadow. Against `solweig` 0.1.0b96 on a synthetic estate over a full day
(23 timesteps), building shade agrees on 99.93 to 100% of ground cells and combined shade on
99.64 to 100%; see `benchmarks/results/2026-10-05_synthetic_estate/`.

Edges: obstacles outside the array are invisible. Tile with a halo at least as long as the
longest shadow you model (`umbrage.tiling`), or cap shadow length with `max_distance_m`.

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,umep,dev]"
pytest
python benchmarks/bench_engines.py --size 4000    # seconds per km2 on this machine
```

## Provenance

Created 5 October 2026 by Lukas Beuster as a general-purpose tool, independent of any
client project, and released under the MIT licence. Projects that use it depend on it as
a package.
