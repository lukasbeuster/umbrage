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
    """Mimics the GPU input cache: keeps the uploaded DSM while (address, shape) is unchanged."""
    def __init__(self):
        self.sig, self.uploaded = None, None

    def enable_gpu(self):
        pass

    def disable_gpu(self):
        pass

    def calculate_shadows_wall_ht_25(self, az, el, res, zmax, z, ct, cb, bush, *rest):
        sig = (z.__array_interface__["data"][0], z.shape)
        if sig != self.sig:
            self.sig, self.uploaded = sig, z.copy()
        b = cast_shadows(self.uploaded, res, az, el).building
        return types.SimpleNamespace(bldg_sh=(~b).astype(np.float32), veg_sh=np.ones_like(z))


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
