"""2.5D shadow casting on raster surface models.

Algorithm: shift-and-compare (Ratti & Richens 2004), the family used by UMEP's
`shadowingfunction_20` and SOLWEIG. For each step k towards the sun the surface is shifted by
k pixels; a cell is shaded if the shifted surface rises above the sun ray through that cell.
Cost ~ cells x steps, with steps bounded by (relief / tan(elevation)) / resolution.

The same code runs on numpy (CPU) or torch (CPU, Apple MPS, CUDA).

Layers (all absolute heights in metres, same grid):
- dsm: terrain + opaque obstacles (buildings, walkway roofs, bridges)
- canopy_top / canopy_base: vegetation crown top and crown base; NaN where no vegetation.
  Light passes through crowns with transmissivity `tau`.

Semantics of the result (differs from UMEP, see docs/semantics.md):
- `building`: cell is shaded by the dsm
- `vegetation`: the ray from the cell is intercepted by a crown, regardless of buildings
- `shade`: 1 where building, else 1 - tau where vegetation, else 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ShadeResult:
    building: np.ndarray      # bool
    vegetation: np.ndarray    # bool
    shade: np.ndarray         # float32 in [0, 1]
    steps: int                # shift steps used (for benchmarking)


def sun_offsets(azimuth_deg: float, max_dist_px: float):
    """Integer (drow, dcol, distance_px) towards the sun, one pixel per step on the major axis."""
    az = math.radians(azimuth_deg)
    dx, dy = math.sin(az), math.cos(az)       # east, north components towards the sun
    major = max(abs(dx), abs(dy))
    k, seen = 1, set()
    while True:
        s = k / major
        if s > max_dist_px:
            break
        dr, dc = int(round(-dy * s)), int(round(dx * s))   # rows grow southward
        if (dr, dc) not in seen:
            seen.add((dr, dc))
            yield dr, dc, math.hypot(dr, dc)
        k += 1


def _slices(n: int, d: int):
    """Target and source slices so that target[i] pairs with source[i + d]."""
    t0, t1 = max(0, -d), n - max(0, d)
    return slice(t0, t1), slice(t0 + d, t1 + d)


def cast_shadows(dsm, res: float, azimuth: float, elevation: float,
                 canopy_top=None, canopy_base=None, tau: float = 0.03,
                 max_distance_m: float | None = None, device: str | None = None) -> ShadeResult:
    """Shadow mask for one sun position.

    res: pixel size (m). azimuth: deg clockwise from north. elevation: deg above horizon.
    device: None for numpy; "cpu", "mps" or "cuda" for torch.
    max_distance_m: cap on shadow length; default is relief / tan(elevation).
    Obstacles outside the array are invisible: tile with a halo >= max shadow length, then crop.
    """
    if elevation <= 0:
        raise ValueError("sun below horizon; handle upstream")
    use_veg = canopy_top is not None
    if use_veg and canopy_base is None:
        raise ValueError("canopy_base is required with canopy_top")

    z_np = np.asarray(dsm, dtype=np.float32)
    zmin = float(np.nanmin(z_np))
    zmax = float(np.nanmax(z_np))
    if use_veg:
        ct_np = np.asarray(canopy_top, dtype=np.float32)
        cb_np = np.asarray(canopy_base, dtype=np.float32)
        if np.isfinite(ct_np).any():
            zmax = max(zmax, float(np.nanmax(ct_np)))
        ct_np = np.where(np.isnan(ct_np), -np.inf, ct_np).astype(np.float32)   # no crown: never blocks
        cb_np = np.where(np.isnan(cb_np), np.inf, cb_np).astype(np.float32)

    if device is None:
        z = z_np
        b = np.zeros(z.shape, dtype=bool)
        ct, cb = (ct_np, cb_np) if use_veg else (None, None)
        v = np.zeros(z.shape, dtype=bool) if use_veg else None
    else:
        import torch
        dev = torch.device(device)
        z = torch.as_tensor(z_np, device=dev)
        b = torch.zeros(z.shape, dtype=torch.bool, device=dev)
        if use_veg:
            ct = torch.as_tensor(ct_np, device=dev)
            cb = torch.as_tensor(cb_np, device=dev)
            v = torch.zeros(z.shape, dtype=torch.bool, device=dev)
        else:
            ct = cb = v = None

    tan_e = math.tan(math.radians(elevation))
    max_px = (max(zmax - zmin, 0.0) / tan_e) / res + 1.0   # +1: pixels have extent (see segment test)
    if max_distance_m is not None:
        max_px = min(max_px, max_distance_m / res)

    H, W = z_np.shape
    steps = 0
    prev_px = 0.0
    for dr, dc, dist_px in sun_offsets(azimuth, max_px):
        if abs(dr) >= H or abs(dc) >= W:
            break
        tr, sr = _slices(H, dr)
        tc, sc = _slices(W, dc)
        ray = z[tr, tc] + dist_px * res * tan_e        # sun-ray height above each target cell
        b[tr, tc] |= z[sr, sc] > ray
        if use_veg:
            # A crown pixel has extent: the ray crosses it between the previous and this step,
            # so test the ray segment [ray_lo, ray] against the crown interval [base, top].
            # Same idea as UMEP's current/previous-step flags; removes a one-pixel fringe.
            ray_lo = z[tr, tc] + prev_px * res * tan_e
            v[tr, tc] |= (ct[sr, sc] > ray_lo) & (cb[sr, sc] < ray)
        prev_px = dist_px
        steps += 1

    if device is not None:
        b = b.cpu().numpy()
        v = v.cpu().numpy() if use_veg else None
    if v is None:
        v = np.zeros_like(b)

    shade = np.where(b, 1.0, np.where(v, 1.0 - tau, 0.0)).astype(np.float32)
    return ShadeResult(building=b, vegetation=v, shade=shade, steps=steps)
