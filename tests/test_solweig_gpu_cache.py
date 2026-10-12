"""solweig's GPU path skips re-uploading inputs whose memory address and shape match the last call
(2026-10-08). These tests (1) emulate that cache with a fake solweig to check the wrapper defeats it,
and (2) on a machine with a GPU, run a sequence of DIFFERENT same-shape DSMs through the real GPU path
and compare each with the CPU path. Test 2 is the one to run on the Mac after any solweig upgrade."""
import sys
import types

import numpy as np
import pytest

from umbrage import cast_shadows
from umbrage.backends import solweig_rs


def city(seed, n=300):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n), np.float32)
    for _ in range(25):
        r, c = rng.integers(0, n - 40, 2)
        z[r:r + 30, c:c + 20] = rng.uniform(10, 60)
    return z


class _FakeShadowing:
    """Mimics the GPU input cache: keeps the uploaded arrays while their (address, shape) are unchanged.
    Each input (DSM, canopy top, canopy base) is cached on its own, the worst case for the wrapper."""
    def __init__(self):
        self.sig, self.uploaded = None, None
        self.vsig, self.vuploaded = {}, {}

    def enable_gpu(self):
        pass

    def disable_gpu(self):
        pass

    def calculate_shadows_wall_ht_25(self, az, el, res, zmax, z, ct, cb, bush, *rest):
        sig = (z.__array_interface__["data"][0], z.shape)
        if sig != self.sig:
            self.sig, self.uploaded = sig, z.copy()
        if ct is None:
            b = cast_shadows(self.uploaded, res, az, el).building
            return types.SimpleNamespace(bldg_sh=(~b).astype(np.float32), veg_sh=np.ones_like(z))
        for k, a in (("ct", ct), ("cb", cb)):
            s = (a.__array_interface__["data"][0], a.shape)
            if s != self.vsig.get(k):
                self.vsig[k], self.vuploaded[k] = s, a.copy()
        top = np.where(self.vuploaded["ct"] > 0, self.vuploaded["ct"], np.nan)
        base = np.where(self.vuploaded["ct"] > 0, self.vuploaded["cb"], np.nan)
        r = cast_shadows(self.uploaded, res, az, el, top, base)
        return types.SimpleNamespace(bldg_sh=(~r.building).astype(np.float32),
                                     veg_sh=(~(r.vegetation & ~r.building)).astype(np.float32))


@pytest.fixture
def fake_solweig(monkeypatch):
    fake = _FakeShadowing()
    mod = types.ModuleType("solweig")
    ra = types.ModuleType("solweig.rustalgos")
    ra.shadowing = fake
    mod.rustalgos = ra
    monkeypatch.setitem(sys.modules, "solweig", mod)
    monkeypatch.setitem(sys.modules, "solweig.rustalgos", ra)
    solweig_rs.reset_input_cache()
    return fake


def test_fake_cache_is_stale_without_the_guard(fake_solweig):
    """The failure itself: same buffer, new contents -> the fake returns the old DSM's shade."""
    buf = city(1)
    first = fake_solweig.calculate_shadows_wall_ht_25(110.0, 30.0, 1.0, 60.0, buf, None, None, None)
    buf[:] = city(2)                                   # what an allocator reusing the address amounts to
    second = fake_solweig.calculate_shadows_wall_ht_25(110.0, 30.0, 1.0, 60.0, buf, None, None, None)
    assert (first.bldg_sh == second.bldg_sh).all()
    assert not (cast_shadows(city(2), 1.0, 110.0, 30.0).building == (second.bldg_sh < 0.5)).all()


def test_wrapper_defeats_stale_cache(fake_solweig):
    buf = np.empty((300, 300), np.float32)
    for seed in range(5):
        buf[:] = city(seed)                            # same object, new contents each time
        got = solweig_rs.cast_shadows(buf, 1.0, 110.0, 30.0).building
        assert (got == cast_shadows(city(seed), 1.0, 110.0, 30.0).building).all()
    for seed in range(5, 9):                           # new objects, freed and reallocated
        got = solweig_rs.cast_shadows(city(seed), 1.0, 250.0, 40.0).building
        assert (got == cast_shadows(city(seed), 1.0, 250.0, 40.0).building).all()


def trees(seed, z, n=300):
    """Canopy top / base (absolute, NaN = none) on open ground of z."""
    rng = np.random.default_rng(100 + seed)
    ct = np.full_like(z, np.nan)
    for _ in range(120):
        r, c = rng.integers(0, n - 8, 2)
        blk = ct[r:r + 7, c:c + 7]
        blk[z[r:r + 7, c:c + 7] == 0] = rng.uniform(6, 18)
    return ct, ct * 0.25


def test_wrapper_defeats_stale_cache_with_canopy(fake_solweig):
    """Combined runs: the same DSM with different canopies (e.g. CHMv2 then CHM v1), and canopy
    buffers reused with new contents, must each give their own vegetation shade."""
    z = city(0)
    ctb, cbb = np.empty_like(z), np.empty_like(z)
    for seed in range(4):
        ct, cb = trees(seed, z)
        ctb[:], cbb[:] = ct, cb                        # same objects, new contents
        got = solweig_rs.cast_shadows(z, 1.0, 110.0, 30.0, ctb, cbb)
        ref = cast_shadows(z, 1.0, 110.0, 30.0, ct, cb)
        assert ((got.vegetation & ~got.building) == (ref.vegetation & ~ref.building)).all()
    for seed in range(4, 8):                           # new canopy objects, same DSM object
        ct, cb = trees(seed, z)
        got = solweig_rs.cast_shadows(z, 1.0, 250.0, 40.0, ct, cb)
        ref = cast_shadows(z, 1.0, 250.0, 40.0, ct, cb)
        assert ((got.vegetation & ~got.building) == (ref.vegetation & ~ref.building)).all()


def test_real_gpu_sequence_matches_cpu():
    """On the Mac: different DSMs of one shape, one after another, GPU vs CPU."""
    pytest.importorskip("solweig")
    from solweig.rustalgos import shadowing as S
    S.enable_gpu()
    probe = solweig_rs.cast_shadows(city(0), 1.0, 110.0, 30.0, gpu=True)
    if "gpu_dispatch_count" in dir(S) and S.gpu_dispatch_count() == 0:
        pytest.skip("no GPU dispatch on this machine")
    del probe
    for seed in range(1, 7):
        z = city(seed)
        g = solweig_rs.cast_shadows(z, 1.0, 110.0, 30.0, gpu=True).building
        c = solweig_rs.cast_shadows(z, 1.0, 110.0, 30.0, gpu=False).building
        assert (g == c).mean() > 0.999, f"GPU differs from CPU for DSM {seed}: stale input?"


def test_fingerprint_is_quiet_and_sensitive():
    import warnings
    a = city(3)
    a[0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no RuntimeWarning may escape
        f1 = solweig_rs._fingerprint(a)
        b = a.copy()
        b[150, 150] += 0.5
        f2 = solweig_rs._fingerprint(b)
    assert all(np.isfinite(v) for v in f1[1:]) and f1 != f2


def test_real_gpu_combined_sequence_matches_cpu():
    """On the Mac: one DSM, different canopies, then different DSMs with canopy, GPU vs CPU."""
    pytest.importorskip("solweig")
    from solweig.rustalgos import shadowing as S
    S.enable_gpu()
    z0 = city(0)
    probe = solweig_rs.cast_shadows(z0, 1.0, 110.0, 30.0, *trees(0, z0), gpu=True)
    if "gpu_dispatch_count" in dir(S) and S.gpu_dispatch_count() == 0:
        pytest.skip("no GPU dispatch on this machine")
    del probe
    cases = [(z0, *trees(s, z0)) for s in range(1, 4)] + [(z, *trees(s, z)) for s, z in ((s, city(s)) for s in range(4, 7))]
    for i, (z, ct, cb) in enumerate(cases):
        g = solweig_rs.cast_shadows(z, 1.0, 110.0, 30.0, ct, cb, gpu=True)
        c = solweig_rs.cast_shadows(z, 1.0, 110.0, 30.0, ct, cb, gpu=False)
        assert (g.building == c.building).mean() > 0.999, f"case {i}: building shade differs: stale input?"
        assert (g.vegetation == c.vegetation).mean() > 0.999, f"case {i}: vegetation shade differs: stale canopy?"
