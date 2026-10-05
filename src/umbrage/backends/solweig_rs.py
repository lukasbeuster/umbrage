"""Optional backend: UMEP's Rust shadowing via the `solweig` package (GPL-3.0).

`pip install solweig` (wheels for macOS arm64; GPU via WebGPU/Metal by default).
Imported lazily so umbrage itself stays MIT and dependency-light.

Uses the low-level `solweig.rustalgos.shadowing.calculate_shadows_wall_ht_25`, which needs
no weather data. The solweig API is pre-1.0 and may change; pin the version you benchmark.

Semantics are mapped onto umbrage's ShadeResult. In SOLWEIG, bldg_sh / veg_sh are 1 = sunlit.
Its veg_sh does not mark cells already in building shadow, so `vegetation` here is only
comparable with umbrage's where neither engine reports building shade.
"""
from __future__ import annotations

import numpy as np

from ..shade import ShadeResult


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
        zmax = max(zmax, float(ct.max()))
        out = S.calculate_shadows_wall_ht_25(azimuth, elevation, res, zmax, z, ct, cb, bush,
                                             None, None, None, None, None, max_distance_m)
        veg = np.asarray(out.veg_sh) < 0.5
    else:
        out = S.calculate_shadows_wall_ht_25(azimuth, elevation, res, zmax, z, None, None, None,
                                             None, None, None, None, None, max_distance_m)
        veg = np.zeros(z.shape, dtype=bool)
    bldg = np.asarray(out.bldg_sh) < 0.5
    shade = np.where(bldg, 1.0, np.where(veg, 1.0 - tau, 0.0)).astype(np.float32)
    return ShadeResult(building=bldg, vegetation=veg, shade=shade, steps=-1)
