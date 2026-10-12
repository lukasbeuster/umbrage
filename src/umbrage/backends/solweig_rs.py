"""Optional backend: UMEP's Rust shadowing via the `solweig` package (GPL-3.0).

`pip install solweig` (wheels for macOS arm64; GPU via WebGPU/Metal by default).
Imported lazily so umbrage itself stays MIT and dependency-light.

Uses the low-level `solweig.rustalgos.shadowing.calculate_shadows_wall_ht_25`, which needs
no weather data. The solweig API is pre-1.0 and may change; pin the version you benchmark.

Semantics are mapped onto umbrage's ShadeResult. In SOLWEIG, bldg_sh / veg_sh are 1 = sunlit.
Its veg_sh does not mark cells already in building shadow, so `vegetation` here is only
comparable with umbrage's where neither engine reports building shade.

GPU input cache (found 2026-10-08, solweig 0.1.0b96): the GPU path keeps the last uploaded
DSM / canopy arrays on the device and skips the upload when the next call passes arrays with
the same memory address and shape (`StaticShadowInputSig` in `rust/src/gpu/shadow_gpu.rs`;
solweig INVARIANTS.md no. 3). A caller that frees one DSM and allocates the next of the same
shape often gets the same address back from the allocator, and the GPU then shades the OLD
DSM: silently wrong output. This wrapper prevents it:
  1. it keeps a reference to the arrays it last passed, so their address cannot be reused by
     a different array while solweig may still treat them as uploaded;
  2. if the caller passes the same array object again, it checks a content fingerprint and
     passes a fresh copy when the contents changed in place.
The CPU path has no such cache. `reset_input_cache()` drops the held arrays.
"""
from __future__ import annotations

import numpy as np

from ..shade import ShadeResult

_HELD: dict[str, object] = {"arrays": (), "fingerprints": ()}


def _fingerprint(a: np.ndarray | None) -> tuple | None:
    """Cheap content fingerprint: position-weighted float64 sums of the row sums and column sums.
    An in-place change of any pixel changes them (barring exact cancellation). Element-wise
    products, not a matrix product: on macOS numpy's matmul goes through Accelerate, which raises
    spurious divide-by-zero / overflow / invalid warnings (2026-10-08 run: 2,000 per run)."""
    if a is None:
        return None
    x = np.nan_to_num(a.astype(np.float64, copy=False), nan=-1.0e9, posinf=1.0e12, neginf=-1.0e12)
    rows = x.sum(axis=1)
    cols = x.sum(axis=0)
    return (a.shape, float((rows * np.arange(1, rows.size + 1)).sum()),
            float((cols * np.arange(1, cols.size + 1)).sum()))


def _stable_inputs(arrays: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Arrays safe to pass to solweig's pointer-keyed GPU cache (see module docstring)."""
    held, fps = _HELD["arrays"], _HELD["fingerprints"]
    out = []
    for i, a in enumerate(arrays):
        if a is None:
            out.append(None)
            continue
        if i < len(held) and held[i] is a and _fingerprint(a) != fps[i]:
            a = a.copy()                     # same object mutated in place: new address, forced upload
        out.append(a)
    _HELD["arrays"] = tuple(out)             # hold them: their addresses cannot be reused meanwhile
    _HELD["fingerprints"] = tuple(_fingerprint(a) for a in out)
    return out


def reset_input_cache() -> None:
    _HELD["arrays"], _HELD["fingerprints"] = (), ()


def cast_shadows(dsm, res: float, azimuth: float, elevation: float,
                 canopy_top=None, canopy_base=None, tau: float = 0.03,
                 max_distance_m: float | None = None, gpu: bool = True) -> ShadeResult:
    from solweig.rustalgos import shadowing as S

    S.enable_gpu() if gpu else S.disable_gpu()
    z = np.ascontiguousarray(dsm, dtype=np.float32)
    zmax = float(np.nanmax(z))
    if canopy_top is not None:
        ct = np.nan_to_num(np.asarray(canopy_top, dtype=np.float32), nan=0.0)
        cb = np.nan_to_num(np.asarray(canopy_base, dtype=np.float32), nan=0.0)
        bush = np.zeros_like(z)
        z, ct, cb, bush = _stable_inputs([z, ct, cb, bush])
        zmax = max(zmax, float(ct.max()))
        out = S.calculate_shadows_wall_ht_25(azimuth, elevation, res, zmax, z, ct, cb, bush,
                                             None, None, None, None, None, max_distance_m)
        veg = np.asarray(out.veg_sh) < 0.5
    else:
        (z,) = _stable_inputs([z])
        out = S.calculate_shadows_wall_ht_25(azimuth, elevation, res, zmax, z, None, None, None,
                                             None, None, None, None, None, max_distance_m)
        veg = np.zeros(z.shape, dtype=bool)
    bldg = np.asarray(out.bldg_sh) < 0.5
    shade = np.where(bldg, 1.0, np.where(veg, 1.0 - tau, 0.0)).astype(np.float32)
    return ShadeResult(building=bldg, vegetation=veg, shade=shade, steps=-1)
