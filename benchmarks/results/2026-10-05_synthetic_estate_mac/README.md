# Synthetic estate cross-check on a Mac, 2026-10-05

The same run as [`../2026-10-05_synthetic_estate/`](../2026-10-05_synthetic_estate/) (same scene,
date and 23 timesteps), repeated on an Apple-silicon Mac; both engines on the CPU (`solweig` with
`gpu=False`, as in the script).

`python benchmarks/synthetic_estate.py --out benchmarks/results/2026-10-05_synthetic_estate_mac`

- Agreement is the same as in the cloud run: building shade 99.93 to 100% of ground cells,
  combined 99.64 to 100%. The shade itself does not depend on the machine.
- Speed per run on 1.44 km²: umbrage numpy 0.09 to 1.29 s (two runs per timestep: buildings,
  then buildings + trees), solweig 0.05 to 0.90 s (one combined run). See `timesteps.csv`.

`scene.npz` and `shaded_hours.npz` are not in git.
