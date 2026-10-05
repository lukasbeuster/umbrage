# Synthetic estate cross-check, 2026-10-05

`python benchmarks/synthetic_estate.py --out benchmarks/results/2026-10-05_synthetic_estate`

A synthetic 1.2 x 1.2 km tropical public-housing estate at 1 m (60 slab blocks of 12 to 40
storeys, 12 point towers, 25 low-rise buildings, a 25 m hill, street trees every 14 m, a
dense park, covered linkways at 3 m). 15 May, every half hour from 07:30 to 18:30 with the sun
above 5 deg (23 timesteps). Engines: umbrage (numpy) and UMEP's Rust `solweig` 0.1.0b96 (CPU),
in a cloud container with no GPU. **Not real data; it tests the engines, not the inputs.**

## Results

- Building shade agrees on 99.93 to 100% of ground cells at every timestep.
- Combined (buildings + trees) agrees on 99.64 to 100%. The residue is a few isolated cells
  at crown and building edges (fig 2).
- Speed (CPU, this box): umbrage 0.15 to 2.1 s per run per 1.44 km², solweig 0.26 to 3.9 s
  (solweig also computes wall shadows). Times grow at low sun because shadows are long.
- Shaded hours, ground mean: 3.0 h from buildings, 5.0 h with trees; trees add 1.9 h.

## What changed in umbrage because of this test

The first comparison found a systematic one-pixel fringe at the far edge of crown shadows,
which brought combined agreement down to ~98.5% at high sun. Two fixes, both geometric:
1. A crown pixel has extent: test the ray *segment* between the previous and current step
   against the crown interval, not just the ray at the current step (the same idea as UMEP's
   current/previous-step flags).
2. Extend the search distance by one pixel so that segment test reaches the last crown pixel.
After the fix a single-tree test matches solweig pixel for pixel at 45 and 72 deg sun.

## Files

| File | What |
|---|---|
| `timesteps.csv` | per timestep: sun position, % ground shaded, agreement, run times |
| `fig1_shade_through_the_day.png` | combined shade at 08:00, 11:00, 13:00, 16:00, 18:00 |
| `fig2_engine_disagreement.png` | where umbrage and solweig differ |
| `fig3_shaded_hours.png` | shaded hours from buildings, with trees, and added by trees |
| `scene.npz`, `shaded_hours.npz` | inputs and outputs as arrays (not in git) |
