"""Cross-checks: torch vs numpy (identical), umbrage vs UMEP Rust solweig (near-identical)."""
import numpy as np
import pytest

from umbrage import cast_shadows
from umbrage.tiling import tiles


def synthetic_city(n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n), np.float32)
    for _ in range(40):
        r, c = rng.integers(0, n - 40, 2)
        z[r:r + 30, c:c + 20] = rng.uniform(10, 80)
    ct = np.full_like(z, np.nan)
    for _ in range(300):
        r, c = rng.integers(0, n - 8, 2)
        blk = ct[r:r + 7, c:c + 7]
        free = z[r:r + 7, c:c + 7] == 0
        blk[free] = rng.uniform(6, 18)
    cb = np.where(np.isnan(ct), np.nan, ct * 0.25)
    return z, ct, cb


def test_torch_cpu_matches_numpy():
    pytest.importorskip("torch")
    z, ct, cb = synthetic_city()
    a = cast_shadows(z, 1.0, 110.0, 30.0, ct, cb)
    b = cast_shadows(z, 1.0, 110.0, 30.0, ct, cb, device="cpu")
    assert (a.building == b.building).all() and (a.vegetation == b.vegetation).all()


def test_matches_umep_rust_buildings():
    pytest.importorskip("solweig")
    from umbrage.backends.solweig_rs import cast_shadows as rs
    z, ct, cb = synthetic_city()
    for az, el in [(100.0, 25.0), (200.0, 60.0), (300.0, 15.0)]:
        a = cast_shadows(z, 1.0, az, el)
        b = rs(z, 1.0, az, el, gpu=False)
        assert (a.building == b.building).mean() > 0.999


def test_tiles_cover_grid_once():
    seen = np.zeros((1050, 2300), int)
    for t in tiles(1050, 2300, 500, 100):
        r0, r1, c0, c1 = t.core
        seen[r0:r1, c0:c1] += 1
        R0, R1, C0, C1 = t.read
        cr0, cr1, cc0, cc1 = t.crop
        assert (cr1 - cr0, cc1 - cc0) == (r1 - r0, c1 - c0)
        assert R0 + cr0 == r0 and C0 + cc0 == c0
    assert (seen == 1).all()


def test_tiled_equals_untiled_with_enough_halo():
    z, _, _ = synthetic_city(n=600)
    full = cast_shadows(z, 1.0, 135.0, 30.0).building
    out = np.zeros_like(full)
    for t in tiles(600, 600, 200, 200):          # 80 m tall / tan(30) ~ 139 m < halo
        R0, R1, C0, C1 = t.read
        r = cast_shadows(z[R0:R1, C0:C1], 1.0, 135.0, 30.0, max_distance_m=200).building
        cr0, cr1, cc0, cc1 = t.crop
        r0, r1, c0, c1 = t.core
        out[r0:r1, c0:c1] = r[cr0:cr1, cc0:cc1]
    assert (out == full).all()
