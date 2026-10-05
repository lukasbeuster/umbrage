"""Engine cross-check on a synthetic tropical public-housing estate.

A 1.2 x 1.2 km scene at 1 m, loosely modelled on a mature estate at 1.3 deg N (slab and point
blocks of 12 to 40 storeys, low-rise shops, street trees, a park, covered linkways, a hill).
Runs umbrage (numpy) and UMEP's Rust solweig for every half hour of one day and writes:

  scene.npz, timesteps.csv, fig1_shade_through_the_day.png,
  fig2_engine_disagreement.png, fig3_shaded_hours.png, README.md

Synthetic: nothing here is real Singapore data. It tests the engines, not the inputs.

    python benchmarks/synthetic_estate.py --out benchmarks/results/<date>_synthetic_estate
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import date
from pathlib import Path

import numpy as np

from umbrage import cast_shadows, daylight_timesteps

LAT, LON = 1.30, 103.80          # tropical site, 1.3 deg N
N = 1200                         # pixels = metres
STOREY = 2.9


def build_scene(seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N]
    terrain = 25.0 * np.exp(-(((x - 950) ** 2 + (y - 250) ** 2) / (2 * 220.0 ** 2)))  # a hill, NE
    bh = np.zeros((N, N), np.float32)                  # building height above ground
    roads = np.zeros((N, N), bool)
    for c in range(0, N, 300):                         # road grid every 300 m, 16 m wide
        roads[:, c:c + 16] = True
        roads[c:c + 16, :] = True
    park = (x > 620) & (x < 880) & (y > 650) & (y < 900)

    def free(r0, r1, c0, c1, pad=6):
        rr0, cc0 = max(r0 - pad, 0), max(c0 - pad, 0)
        return not (roads[rr0:r1 + pad, cc0:c1 + pad].any() or park[rr0:r1 + pad, cc0:c1 + pad].any()
                    or (bh[rr0:r1 + pad, cc0:c1 + pad] > 0).any())

    placed = 0
    while placed < 60:                                 # slab blocks, mixed orientation
        L, D = rng.integers(55, 90), rng.integers(12, 16)
        h, w = (D, L) if rng.random() < 0.6 else (L, D)
        r, c = rng.integers(20, N - 100, 2)
        if free(r, r + h, c, c + w, pad=18):
            bh[r:r + h, c:c + w] = STOREY * rng.integers(12, 41)
            placed += 1
    placed = 0
    while placed < 12:                                 # point towers
        r, c = rng.integers(20, N - 60, 2)
        if free(r, r + 25, c, c + 25, pad=20):
            bh[r:r + 25, c:c + 25] = STOREY * rng.integers(30, 46)
            placed += 1
    placed = 0
    while placed < 25:                                 # low-rise shops / community buildings
        r, c = rng.integers(20, N - 60, 2)
        if free(r, r + 20, c, c + 35, pad=8):
            bh[r:r + 20, c:c + 35] = STOREY * rng.integers(1, 4)
            placed += 1

    # covered linkways: 3 m wide, 3 m high, connecting block pairs (L-shaped)
    linkway = np.zeros((N, N), bool)
    blocks = np.argwhere(bh[::40, ::40] > 20) * 40
    for (r0, c0), (r1, c1) in zip(blocks[:-1:2], blocks[1::2]):
        if abs(r0 - r1) + abs(c0 - c1) > 250:
            continue
        linkway[min(r0, r1):max(r0, r1) + 3, c0:c0 + 3] = True
        linkway[r1:r1 + 3, min(c0, c1):max(c0, c1) + 3] = True
    linkway &= (bh == 0) & ~roads
    bh_struct = np.where(linkway, 3.0, bh)

    # vegetation: street trees along road edges, a dense park, scattered estate trees
    chm = np.zeros((N, N), np.float32)

    def tree(r, c, h):
        rad = max(3, int(0.35 * h))
        rr, cc = np.ogrid[-rad:rad + 1, -rad:rad + 1]
        disk = rr ** 2 + cc ** 2 <= rad ** 2
        r0, c0 = r - rad, c - rad
        if r0 < 0 or c0 < 0 or r0 + 2 * rad + 1 > N or c0 + 2 * rad + 1 > N:
            return
        win = chm[r0:r0 + 2 * rad + 1, c0:c0 + 2 * rad + 1]
        ok = disk & (bh_struct[r0:r0 + 2 * rad + 1, c0:c0 + 2 * rad + 1] == 0)
        win[ok] = np.maximum(win[ok], h)

    for c in range(0, N, 300):
        for r in range(10, N, 14):
            for off in (-5, 21):
                tree(r, c + off, rng.uniform(10, 18))
                tree(c + off, r, rng.uniform(10, 18))
    for _ in range(900):
        tree(rng.integers(660, 860), rng.integers(640, 860), rng.uniform(14, 25))
    for _ in range(500):
        tree(rng.integers(0, N), rng.integers(0, N), rng.uniform(6, 14))

    dsm = (terrain + bh_struct).astype(np.float32)
    has = chm > 0
    ct = np.where(has, terrain + chm, np.nan).astype(np.float32)
    cb = np.where(has, terrain + 0.25 * chm, np.nan).astype(np.float32)
    return dict(dsm=dsm, terrain=terrain.astype(np.float32), building_height=bh,
                linkway=linkway, chm=chm, canopy_top=ct, canopy_base=cb, roads=roads)


def run(out: Path, day: date):
    from umbrage.backends.solweig_rs import cast_shadows as rs_cast
    out.mkdir(parents=True, exist_ok=True)
    s = build_scene()
    np.savez_compressed(out / "scene.npz", **{k: v for k, v in s.items()})
    ground = (s["building_height"] == 0) & ~s["linkway"]       # receiving cells that matter
    steps = daylight_timesteps(day, LAT, LON, step_minutes=30, min_elevation=5.0)
    rows, keep = [], {}
    hours_built = np.zeros((N, N), np.float32)
    hours_comb = np.zeros((N, N), np.float32)
    for sp in steps:
        t0 = time.perf_counter()
        a_b = cast_shadows(s["dsm"], 1.0, sp.azimuth, sp.elevation, max_distance_m=1000)
        a_c = cast_shadows(s["dsm"], 1.0, sp.azimuth, sp.elevation, s["canopy_top"], s["canopy_base"],
                           max_distance_m=1000)
        t_um = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = rs_cast(s["dsm"], 1.0, sp.azimuth, sp.elevation, s["canopy_top"], s["canopy_base"],
                    max_distance_m=1000, gpu=False)
        t_rs = time.perf_counter() - t0
        um_comb = a_c.building | a_c.vegetation
        rs_comb = b.building | b.vegetation
        hours_built += 0.5 * a_b.building
        hours_comb += 0.5 * um_comb
        rows.append(dict(
            time_local=sp.when.strftime("%H:%M"), azimuth=round(sp.azimuth, 1),
            elevation=round(sp.elevation, 1),
            built_shade_ground_pct=round(100 * a_b.building[ground].mean(), 1),
            combined_shade_ground_pct=round(100 * um_comb[ground].mean(), 1),
            agree_building_pct=round(100 * (a_c.building == b.building)[ground].mean(), 3),
            agree_combined_pct=round(100 * (um_comb == rs_comb)[ground].mean(), 3),
            umbrage_s_two_runs=round(t_um, 2), solweig_s_one_run=round(t_rs, 2)))
        if sp.when.strftime("%H:%M") in ("08:00", "11:00", "13:00", "16:00", "18:00"):
            keep[sp.when.strftime("%H:%M")] = (um_comb, a_c.building, rs_comb, b.building, sp)
        print(rows[-1])
    with open(out / "timesteps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "shaded_hours.npz", built=hours_built, combined=hours_comb)
    figures(out, s, keep, hours_built, hours_comb, ground, day)
    return rows


def figures(out, s, keep, hours_built, hours_comb, ground, day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    bldg_col, link_col = "#3a3a38", "#eb6834"
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": muted,
                         "axes.edgecolor": "#d6d5d0", "figure.facecolor": surface})
    roof = s["building_height"] > 0

    def base(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.where(roof, 1, np.nan), cmap=ListedColormap([bldg_col]), interpolation="nearest")
        ax.contour(s["linkway"], levels=[0.5], colors=[link_col], linewidths=0.4)

    # fig 1: combined shade through the day
    shade_cmap = ListedColormap(["#f3efe2", "#2a78d6"])
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), constrained_layout=True)
    for ax, (t, (comb, _, _, _, sp)) in zip(axes, keep.items()):
        ax.imshow(np.where(roof, np.nan, comb), cmap=shade_cmap, vmin=0, vmax=1, interpolation="nearest")
        base(ax)
        ax.set_title(f"{t} local  (sun {sp.elevation:.0f}° at {sp.azimuth:.0f}°)\n"
                     f"{100 * comb[ground].mean():.0f}% of ground shaded", fontsize=9)
    fig.suptitle(f"Synthetic estate, {day:%d %b}: shade from buildings + trees (umbrage). "
                 "Blue = shade, cream = sun, dark = roofs, orange = covered linkways", fontsize=10)
    fig.savefig(out / "fig1_shade_through_the_day.png", dpi=130)
    plt.close(fig)

    # fig 2: where the engines disagree
    dis_cmap = ListedColormap(["#2a78d6", "#eb6834"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for ax, t in zip(axes, ["08:00", "13:00", "16:00"]):
        um, _, rsc, _, sp = keep[t]
        d = np.full(um.shape, np.nan)
        d[um & ~rsc] = 0
        d[~um & rsc] = 1
        d[roof] = np.nan
        ax.imshow(np.where(roof, np.nan, 0.0), cmap=ListedColormap(["#ecebe6"]), interpolation="nearest")
        ax.imshow(d, cmap=dis_cmap, vmin=0, vmax=1, interpolation="nearest")
        base(ax)
        agree = 100 * (um == rsc)[ground].mean()
        ax.set_title(f"{t}: engines agree on {agree:.2f}% of ground cells", fontsize=9)
    fig.suptitle("Disagreement, combined shade. Blue = shaded only in umbrage, "
                 "orange = shaded only in solweig (UMEP Rust)", fontsize=10)
    fig.savefig(out / "fig2_engine_disagreement.png", dpi=130)
    plt.close(fig)

    # fig 3: shaded hours, built vs combined, and the vegetation contribution
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    vmax = float(np.ceil(hours_comb.max()))
    for ax, data, title in [(axes[0], hours_built, "Buildings only"),
                            (axes[1], hours_comb, "Buildings + trees"),
                            (axes[2], hours_comb - hours_built, "Added by trees")]:
        im = ax.imshow(np.where(roof, np.nan, data), cmap="Blues", vmin=0,
                       vmax=vmax if title != "Added by trees" else None, interpolation="nearest")
        base(ax)
        ax.set_title(f"{title}: ground mean {data[ground].mean():.1f} h", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="hours of shade, 07:30 to 18:30")
    fig.suptitle("Shaded hours per day (half-hourly steps, sun above 5°)", fontsize=10)
    fig.savefig(out / "fig3_shaded_hours.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/synthetic_estate")
    ap.add_argument("--date", default="2026-05-15")
    a = ap.parse_args()
    run(Path(a.out), date.fromisoformat(a.date))
